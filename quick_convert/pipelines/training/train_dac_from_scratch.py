"""Train a DAC neural audio codec from scratch.

Builds the 16 kHz DAC architecture with random weights and trains the full codec
(encoder + RVQ + decoder), then saves the weights. The trained *encoder* can be
plugged into the pipeline's `DACContentEncoder`.

Losses are in (`dac.nn.loss`), used exactly as in the
upstream recipe: multi-scale `MelSpectrogramLoss` + `MultiScaleSTFTLoss` +
waveform `L1Loss`, the VQ commitment/codebook losses from the model, and — with
`--gan` — the official `GANLoss` (LS-GAN + feature matching) over DAC's own
`Discriminator`. The generator loss is the official weighted sum using the
`lambdas` from conf/base.yml (mel 15, adv/feat 2, adv/gen 1, commitment 0.25,
codebook 1). As upstream, the stft/waveform terms are logged but not weighted in.

Reqs. `torch`, `torchaudio`, and `descript-audio-codec` (which brings `audiotools`).

Usage:
    python train_dac_from_scratch.py --data /path/to/wavs --steps 50000 --save dac16k.pt
    python train_dac_from_scratch.py --data /path/to/wavs --gan
    python train_dac_from_scratch.py                      # no --data: synthetic smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


# Official generator-loss weights (descript-audio-codec, conf/base.yml). The
# stft/waveform losses are computed for logging but, as upstream, are NOT here.
LAMBDAS = {
    "mel/loss": 15.0,
    "adv/feat_loss": 2.0,
    "adv/gen_loss": 1.0,
    "vq/commitment_loss": 0.25,
    "vq/codebook_loss": 1.0,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class AudioFolder(Dataset):
    """Random fixed-length crops from a folder of .wav/.flac files (mono, resampled)."""

    def __init__(self, root: str, sr: int, n_samples: int):
        self.files = [p for ext in ("*.wav", "*.flac") for p in Path(root).rglob(ext)]
        if not self.files:
            raise FileNotFoundError(f"No .wav/.flac under {root}")
        self.sr, self.n = sr, n_samples

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        wav, sr = torchaudio.load(str(self.files[i]))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.squeeze(0)
        if wav.numel() >= self.n:  # random crop
            start = torch.randint(0, wav.numel() - self.n + 1, (1,)).item()
            wav = wav[start : start + self.n]
        else:  # pad short clips
            wav = F.pad(wav, (0, self.n - wav.numel()))
        return wav


class SyntheticAudio(Dataset):
    """Filler noise so the script runs end-to-end without a dataset (smoke test only)."""

    def __init__(self, sr: int, n_samples: int, size: int = 512):
        self.n, self.size = n_samples, size

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return 0.1 * torch.randn(self.n)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="folder of wav/flac; omit for synthetic smoke test")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--save", default="dac16k_scratch.pt")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--gan", action="store_true", help="add official DAC discriminators (GANLoss)")
    p.add_argument(
        "--init",
        default="scratch",
        help="'scratch' = random init (no pretraining); or a DAC model_type to FINE-TUNE from, e.g. '16khz'",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    import dac
    from audiotools import AudioSignal
    from dac.nn.loss import GANLoss, L1Loss, MelSpectrogramLoss, MultiScaleSTFTLoss

    # --- model --------------------------------------------------------------
    if args.init == "scratch":
        # Random weights -> train the codec from zero. Pretraining is NOT used.
        model = dac.DAC(
            encoder_dim=64,
            encoder_rates=[2, 4, 5, 8],   # product 320 -> hop 320 -> 50 Hz
            decoder_dim=1536,
            decoder_rates=[8, 5, 4, 2],   # mirrors encoder so output length == input
            n_codebooks=12,
            codebook_size=1024,
            codebook_dim=8,
            sample_rate=args.sr,
        ).to(device)
        sr = args.sr
    else:
        # Warm-start: load Descript's PRETRAINED codec and continue training
        # (fine-tune). Architecture + sample rate come from the checkpoint, so
        # this inherits all of Descript's pretraining and adapts it to your data.
        model = dac.DAC.load(dac.utils.download(model_type=args.init)).to(device)
        sr = model.sample_rate
    print(f"init={args.init} | hop={model.hop_length} | frame_rate={sr / model.hop_length:.1f} Hz | latent={model.latent_dim}")

    n_samples = int(args.seconds * sr)

    # --- data -------------------------------------------------------------
    if args.data:
        ds = AudioFolder(args.data, sr, n_samples)
    else:
        print("[!] No --data given: using synthetic noise (smoke test, not real training).")
        ds = SyntheticAudio(sr, n_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # --- official DAC losses (configured as in conf/base.yml) -------------
    waveform_loss = L1Loss()
    stft_loss = MultiScaleSTFTLoss(window_lengths=[2048, 512])
    mel_loss = MelSpectrogramLoss(
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin=[0, 0, 0, 0, 0, 0, 0],
        mel_fmax=[None, None, None, None, None, None, None],
        pow=1.0,
        clamp_eps=1.0e-5,
        mag_weight=0.0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))

    if args.gan:
        from dac.model import Discriminator  # multi-period + multi-resolution STFT discs

        disc = Discriminator(sample_rate=sr).to(device)
        gan_loss = GANLoss(disc)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99))

    model.train()
    step = 0
    while step < args.steps:
        for wav in loader:
            wav = wav.unsqueeze(1).to(device)            # (B, 1, T)
            signal = AudioSignal(wav, sr)                # wrap for the official losses
            out = model(signal.audio_data, sr)
            recons = AudioSignal(out["audio"], sr)

            output = {}

            # (1) Discriminator step — GANLoss detaches `recons` internally.
            if args.gan:
                output["adv/disc_loss"] = gan_loss.discriminator_loss(recons, signal)
                opt_d.zero_grad()
                output["adv/disc_loss"].backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 10.0)
                opt_d.step()

            # (2) Generator step — official losses, weighted by LAMBDAS.
            output["mel/loss"] = mel_loss(recons, signal)
            output["stft/loss"] = stft_loss(recons, signal)        # logged only
            output["waveform/loss"] = waveform_loss(recons, signal)  # logged only
            if args.gan:
                output["adv/gen_loss"], output["adv/feat_loss"] = gan_loss.generator_loss(recons, signal)
            output["vq/commitment_loss"] = out["vq/commitment_loss"]
            output["vq/codebook_loss"] = out["vq/codebook_loss"]
            output["loss"] = sum(v * output[k] for k, v in LAMBDAS.items() if k in output)

            opt.zero_grad()
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
            opt.step()

            if step % args.log_every == 0:
                msg = f"step {step:>7} | loss {output['loss'].item():.3f} | mel {output['mel/loss'].item():.3f}"
                if args.gan:
                    msg += f" | disc {output['adv/disc_loss'].item():.3f}"
                print(msg)
            step += 1
            if step >= args.steps:
                break

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save)
    torch.save(model.encoder.state_dict(), args.save.replace(".pt", "_encoder.pt"))
    print(f"saved full codec -> {args.save}")
    print(f"saved encoder    -> {args.save.replace('.pt', '_encoder.pt')}")
    print(
        "\nUse the trained encoder in the pipeline:\n"
        "    enc = DACContentEncoder.from_scratch(trainable=True)\n"
        f"    enc.dac_encoder.load_state_dict(torch.load('{args.save.replace('.pt', '_encoder.pt')}'))"
    )


if __name__ == "__main__":
    main()
