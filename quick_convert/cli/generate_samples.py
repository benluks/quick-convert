"""Generate reconstruction audio from a trained controllable-RVQ checkpoint.

This reproduces what the model would have output at a given checkpoint —
nothing from past training needs to have been saved, because the audio is a
deterministic function of (weights + input features):
precomputed content -> encoder -> RVQ -> flow-matching decoder -> vocoder -> wav

It conditions on each utterance's own speaker embedding, so the result is a
*reconstruction* in the source voice (swap that embedding to anonymise).

This needs to be run from the repo root, in the training env, where the checkpoint, the precomputed
features, and the vocoder are available:

    UV_PROJECT_ENVIRONMENT=.venv-train uv run -m quick_convert.cli.generate_samples \
        --ckpt outputs/checkpoints/dac_v2_train-100+360+500/epoch=8-step=67000.ckpt \
        --n 4

Writes <out>/<utt>.wav (generated) and <out>/<utt>.orig.wav (source) for each
utterance. Run it on two checkpoints (e.g. step 5000 vs step 67000) to hear how
training changed the model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True, type=Path, help="Path to the Lightning .ckpt to load.")
    p.add_argument(
        "--config-name",
        default="run/train_controllable_rvq_dac_librispeech_v2",
        help="Hydra run config used for training (defines the architecture).",
    )
    p.add_argument("--out-root", default="outputs", help="out_root used to resolve config paths.")
    p.add_argument(
        "--out", type=Path, default=None, help="Output dir (default: samples_from_ckpt/<ckpt stem>/)."
    )
    p.add_argument("--n", type=int, default=4, help="Number of utterances to generate.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_checkpoint(module: torch.nn.Module, ckpt_path: Path) -> None:
    """Load a Lightning checkpoint into an (uncompiled) module.

    Strips the ``_orig_mod.`` prefix that ``torch.compile`` adds, and loads
    non-strictly: the AAM-Softmax classifier weights are built at train time and
    aren't needed for generation (the speaker *embedding* comes from the head's
    forward pass), so they legitimately show up as "unexpected".
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    result = module.load_state_dict(state_dict, strict=False)
    missing, unexpected = result.missing_keys, result.unexpected_keys
    print(f"loaded {ckpt_path.name}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if "global_step" in ckpt:
        print(f"  checkpoint: global_step={ckpt['global_step']} epoch={ckpt.get('epoch')}")
    enc_dec_missing = [k for k in missing if k.startswith(("encoder.", "decoder."))]
    if enc_dec_missing:
        print(
            f"  [!] {len(enc_dec_missing)} encoder/decoder keys are MISSING "
            f"(e.g. {enc_dec_missing[:3]}) — does --config-name match this checkpoint?"
        )


def main() -> None:
    args = parse_args()
    out_dir = args.out or Path("samples_from_ckpt") / args.ckpt.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name=args.config_name, overrides=[f"out_root={args.out_root}"])

    module = instantiate(cfg.trainer.module)
    load_checkpoint(module, args.ckpt)
    module = module.to(args.device).eval()

    val_dataset = instantiate(cfg.val_dataset)
    loader = val_dataset.make_dataloader(batch_size=1, shuffle=False)
    sr = int(module.decoder.vocoder.sampling_rate)

    n_done = 0
    for batch in loader:
        if n_done >= args.n:
            break
        sample = next(iter(batch))
        utt = sample.utt_id
        try:
            content = batch.resources["content"].values.to(args.device)
            lengths = batch.resources["content"].lengths.to(args.device)
            with torch.no_grad():
                enc_out = module.encoder(content, lengths)
                # encoder returns [z_q, z_quantized, text_q, spk_q, spk_output, emo_pros_q, (lengths)]
                text_q, spk_output, emo_pros_q = enc_out[2], enc_out[4], enc_out[5]
                gen_lengths = enc_out[6] if len(enc_out) > 6 else lengths
                mel, gen_audio = module._generate_media(
                    batch, text_q, emo_pros_q, spk_output, gen_lengths
                )

            wav = gen_audio[0].detach().cpu().float().reshape(1, -1)
            torchaudio.save(str(out_dir / f"{utt}.wav"), wav, sr)
            orig = sample.waveform.detach().cpu().float().reshape(1, -1)
            torchaudio.save(str(out_dir / f"{utt}.orig.wav"), orig, int(sample.sample_rate))
            print(f"  wrote {utt}.wav  ({wav.shape[-1] / sr:.1f}s)")
            n_done += 1
        except Exception as exc:  # skip e.g. utterances longer than the length cap
            print(f"  [skip] {utt}: {type(exc).__name__}: {exc}")

    print(f"\ndone -> {out_dir}  ({n_done} utterances)")


if __name__ == "__main__":
    main()
