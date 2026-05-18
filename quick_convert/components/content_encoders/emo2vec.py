from __future__ import annotations
from pathlib import Path

import torch
import torchaudio


from funasr import AutoModel
from .base import ContentEncoder, ContentFeatures
 

# Source: https://github.com/ddlBoJack/emotion2vec/tree/main

class EmotionEncoder(ContentEncoder):
    """Content encoder backed by emotion2vec (iic/emotion2vec_plus_large).

    Extracts frame-level emotional representations from raw waveforms using
    the FunASR AutoModel interface.
    """

    def __init__(
        self,
        model_name: str = "iic/emotion2vec_plus_large",
        sample_rate: int = 16000,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Initialise the encoder and load the pretrained model.

        Args:
            model_name: HuggingFace / ModelScope model identifier.
            sample_rate: Expected input sample rate; audio is resampled to this
                value before encoding.
            device: Target device string. Auto-detected (CUDA > MPS > CPU) when
                ``None``.
            local_files_only: If ``True``, forbid downloading model weights.
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.local_files_only = local_files_only

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model = AutoModel(model=model_name).to(self.device)
        self.model.eval()

    def encode_file(self, path: str | Path) -> ContentFeatures:
        """Load an audio file from *path* and return its encoded features."""
        path = Path(path)
        wav, sr = torchaudio.load(path)

        if wav.dim() > 2:
            raise ValueError(
                f"Expected waveform of shape (channels, time), got {tuple(wav.shape)}"
            )

        if wav.dim() == 2 and wav.shape[0] > 1:  # Convert to mono if needed
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.squeeze(0).unsqueeze(0)

        return self.encode_waveforms(wav, sample_rate=sr)

    def _create_padding_mask(self, lengths: torch.Tensor) -> torch.Tensor:  # lengths: (B,)
        """Return a boolean mask of shape ``(B, T)`` where ``True`` marks padding."""
        if lengths.dim() != 2:
            lengths = lengths.unsqueeze(1)
        max_length = lengths.max()
        batch_size = lengths.shape[0]
        mask = torch.arange(max_length).expand(batch_size, max_length) >= lengths
        return mask.to(self.device)

    @torch.inference_mode()
    def encode_waveforms(
        self,
        wavs: torch.Tensor,
        lengths: torch.Tensor | None = None,
        sample_rate: int | None = None,
    ) -> ContentFeatures:
        """Encode a batch of waveforms and return frame-level features.

        Args:
            wavs: Float tensor of shape ``(batch, time)``.
            lengths: Optional absolute lengths in samples, shape ``(batch,)``.
                When ``None``, all frames are treated as valid.
            sample_rate: Sample rate of *wavs*. Resampled to ``self.sample_rate``
                when different. Defaults to ``self.sample_rate``.

        Returns:
            :class:`ContentFeatures` with ``values`` of shape
            ``(batch, frames, dim)``.
        """
        if wavs.dim() != 2:
            raise ValueError(
                f"Expected wavs with shape (batch, time), got {tuple(wavs.shape)}"
            )

        sample_rate = sample_rate or self.sample_rate

        if sample_rate != self.sample_rate:
            wavs = torchaudio.functional.resample(
                wavs,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )
            sample_rate = self.sample_rate

        wavs = wavs.detach().cpu()

        padding_mask = self._create_padding_mask(lengths)
        outputs = self.model.extract_features(
            x=wavs,
            padding_mask=padding_mask,
            remove_extra_tokens=True,
        )
        outputs = outputs['x']
        output_mask = outputs['padding_mask']
        output_lengths = (1 - output_mask).sum(dim=1)

        return ContentFeatures(
            values=outputs,
            lengths=output_lengths,
            feature_dim=outputs.shape[-1],
            representation_type="continuous",
            temporal_granularity="frame",
            backend="funasr",
            model_name=self.model_name,
            layer=self.layer,
        )
