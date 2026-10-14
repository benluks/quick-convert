from pathlib import Path

import torch
from torch.nn import functional as F

from ....data import AudioBatch
from .base import SpeakerEmbedding, SpeakerEncoder


class VoicePrivacyECAPAEncoder(SpeakerEncoder):
    """Frozen VoicePrivacy 2026 WavLM-Large + ECAPA attacker.

    Checkpoints are the official ``embedding_model.ckpt`` and
    ``WavLM-Large.pt`` files. Only load checkpoints from trusted sources.
    Batch items are encoded separately at their true length.
    """

    FEATURE_DIM = 192
    SAMPLE_RATE = 16_000

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def __init__(
        self, ecapa_checkpoint: str | Path, wavlm_checkpoint: str | Path, device: str | torch.device | None = None
    ):
        super().__init__(device=device)
        for path in (ecapa_checkpoint, wavlm_checkpoint):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization

        from ....external.voiceprivacy_asv.ecapa_model_wavlm_24layer import ECAPA_TDNN_test
        from ....external.voiceprivacy_asv.WavLM import WavLM, WavLMConfig

        self.ecapa = ECAPA_TDNN_test()
        checkpoint = torch.load(ecapa_checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        self.ecapa.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
        checkpoint = torch.load(wavlm_checkpoint, map_location="cpu", weights_only=False)
        self.wavlm = WavLM(WavLMConfig(checkpoint["cfg"]))
        self.wavlm.load_state_dict(checkpoint["model"], strict=True)
        if self.wavlm.cfg.encoder_layers != 24 or self.wavlm.cfg.encoder_embed_dim != 1024:
            raise ValueError("VoicePrivacy ECAPA requires WavLM-Large (24 layers, 1024 dimensions).")
        self.fbank = Fbank(n_mels=80, left_frames=0, right_frames=0, deltas=False)
        self.normalizer = InputNormalization(norm_type="sentence", std_norm=False)
        self.to(self.device)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # This inference adapter remains frozen even inside a training system.
        return super().train(False)

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sr: int) -> SpeakerEmbedding:
        if sr != self.sample_rate:
            raise ValueError(f"VoicePrivacy ECAPA expects {self.sample_rate} Hz audio, got {sr} Hz")
        if wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0)
        if wav.ndim != 1 or wav.numel() < 400:
            raise ValueError("Expected mono waveform [T] or [1, T] with at least 400 samples.")
        wav = wav.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(wav).all() or wav.abs().max() == 0:
            raise ValueError("VoicePrivacy ECAPA requires finite, non-silent audio.")
        # Official filterbank path peak-normalizes and trims exact edge zeros;
        # WavLM instead sees the full original waveform with cfg.normalize.
        nonzero = wav.nonzero().flatten()
        fbank_wav = (wav / wav.abs().max())[nonzero[0] : nonzero[-1] + 1]
        if fbank_wav.numel() < 400:
            raise ValueError("Non-silent waveform span must contain at least 400 samples.")
        feats = self.fbank(fbank_wav.unsqueeze(0))
        feats = self.normalizer(feats, torch.ones(1, device=self.device))
        ssl_wav = F.layer_norm(wav, wav.shape) if self.wavlm.cfg.normalize else wav
        outputs, _ = self.wavlm.extract_features(ssl_wav.unsqueeze(0), output_layer=24, ret_layer_results=True)
        layers = torch.stack([x for x, _ in outputs[1][1:]], dim=0).permute(2, 0, 1, 3)
        values = self._single_embedding(F.normalize(self.ecapa(feats, layers), dim=-1).squeeze(1))
        return SpeakerEmbedding(values, self.FEATURE_DIM, "voiceprivacy", "wavlm-large-ecapa-2026")

    @torch.inference_mode()
    def encode_batch(self, samples: AudioBatch) -> SpeakerEmbedding:
        if samples.waveforms is None or samples.lengths is None or samples.sample_rates is None:
            raise ValueError("Speaker encoding requires loaded audio.")
        values = []
        for wav, length, sr in zip(samples.waveforms, samples.lengths, samples.sample_rates, strict=True):
            if not 0 < int(length) <= wav.shape[-1]:
                raise ValueError("Invalid waveform length.")
            values.append(self.encode(wav[: int(length)], int(sr)).values)
        return SpeakerEmbedding(torch.stack(values), self.FEATURE_DIM, "voiceprivacy", "wavlm-large-ecapa-2026")

    def forward(self, samples: AudioBatch) -> SpeakerEmbedding:
        return self.encode_batch(samples)
