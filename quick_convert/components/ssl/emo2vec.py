from __future__ import annotations
from pathlib import Path
from typing import Literal

import torch
import torchaudio


from funasr import AutoModel

from quick_convert.data.types import AudioBatch
from .base import ContentEncoder, ContentFeatures


# Source: https://github.com/ddlBoJack/emotion2vec/tree/main


class EmotionEncoder(ContentEncoder):
    FEATURE_DIM = 1024
    """Content encoder backed by emotion2vec (iic/emotion2vec_plus_large).

    Extracts frame-level emotional representations from raw waveforms using
    the FunASR AutoModel interface.
    """

    def __init__(
        self,
        model_name: str = "iic/emotion2vec_plus_large",
        sample_rate: int = 16000,
        layer: int = -1,
        granularity: Literal["frame", "utterance"] = "frame",
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(device=device)
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
        self.granularity = granularity
        self.layer = layer

        self.model = AutoModel(model=model_name, device=str(self.device))
        # technically unneessary, funasr does this under the hood
        self.model.model.eval()

    def encode_file(self, path: str | Path) -> ContentFeatures:
        """Load an audio file from *path* and return its encoded features."""
        path = Path(path)
        wav, sr = torchaudio.load(path)

        if wav.dim() > 2:
            raise ValueError(f"Expected waveform of shape (channels, time), got {tuple(wav.shape)}")

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

    def forward(self, batch: AudioBatch):
        if getattr(batch, "waveforms", None) is None:
            raise RuntimeError(
                f"{self.__class__.__name__} only works with loaded audio for now. Please set `load: true` in your dataset config"
            )
        if not (batch.sample_rates == self.sample_rate).all():
            raise RuntimeError(
                f"""Expected sample rates of input audio to be {self.sample_rate}, but got {batch.sample_rates}. 
                Batch resampling within {self.__class__.__name__} is not currently supported. 
                Please set `target_sr={self.sample_rate}` (with `load=True`) in the dataset section of your 
                config to perform resampling at the dataset level."""
            )

        content = self.encode_waveforms(batch.waveforms.to(self.device), batch.lengths.to(self.device))
        return content

    @torch.inference_mode()
    def encode_waveforms(
        self,
        waveforms: torch.Tensor,
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

        # padding_mask = self._create_padding_mask(lengths)
        waveforms_list = [waveforms[i, : lengths[i]] for i in range(waveforms.shape[0])]

        outputs = self.model.generate(input=waveforms_list, input_len=lengths, granularity=self.granularity)
        features = [torch.from_numpy(item["feats"]) for item in outputs]
        feature_lens = [len(feat) for feat in features]

        return ContentFeatures(
            values=features,
            lengths=feature_lens,
            feature_dim=self.feature_dim,
            representation_type="continuous",
            temporal_granularity=self.granularity,
            backend="funasr",
            model_name=self.model_name,
            layer=self.layer,
        )
