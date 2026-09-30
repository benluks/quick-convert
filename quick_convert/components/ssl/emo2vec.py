from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from quick_convert.data.types import AudioBatch

from ._waveforms import validate_waveforms
from .base import ContentEncoder, ContentFeatures


# Source: https://github.com/ddlBoJack/emotion2vec/tree/main


class EmotionEncoder(ContentEncoder):
    FEATURE_DIM = 1024
    SAMPLE_RATE = 16000
    FRAME_HZ = 50.0
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
        if sample_rate != 16000:
            raise ValueError("emotion2vec requires 16000 Hz audio.")
        if layer != -1:
            raise ValueError("FunASR emotion2vec exposes only the final layer; use layer=-1.")
        if granularity not in {"frame", "utterance"}:
            raise ValueError("Expected granularity='frame' or 'utterance'.")
        if local_files_only:
            raise ValueError("FunASR does not support local_files_only here; use a local model_name path instead.")
        """Initialise the encoder and load the pretrained model.

        Args:
            model_name: HuggingFace / ModelScope model identifier.
            sample_rate: Expected input sample rate; audio is resampled to this
                value for file inputs; waveform inputs must already match.
            device: Target device string. Auto-detected (CUDA > MPS > CPU) when
                ``None``.
            local_files_only: Unsupported by FunASR here; ``True`` raises.
        """
        self.model_name = model_name
        self.granularity = granularity
        self.layer = layer

        from funasr import AutoModel

        self.model = AutoModel(model=model_name, device=str(self.device))
        # technically unneessary, funasr does this under the hood
        self.model.model.eval()
        config = getattr(self.model.model, "cfg", {})
        self._feature_dim = int(config.get("embed_dim", self.FEATURE_DIM))

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def frame_hz(self) -> float | None:
        return self.FRAME_HZ if self.granularity == "frame" else None

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
            waveforms: Float tensor of shape ``(batch, time)``.
            lengths: Optional absolute lengths in samples, shape ``(batch,)``.
                When ``None``, all frames are treated as valid.
            sample_rate: Must match ``self.sample_rate`` when specified.

        Returns:
            :class:`ContentFeatures` with ``values`` of shape
            ``(batch, frames, dim)``.
        """

        lengths = validate_waveforms(waveforms, lengths, sample_rate, self.sample_rate)
        # FunASR's public API processes numpy audio; it owns backend device placement.
        waveform_list = [waveforms[i, : int(lengths[i])].detach().cpu().numpy().copy() for i in range(len(lengths))]
        outputs = self.model.generate(input=waveform_list, granularity=self.granularity, extract_embedding=True)
        if len(outputs) != len(waveform_list):
            raise RuntimeError("emotion2vec returned a different number of outputs than input waveforms.")
        features = [torch.as_tensor(item["feats"], device=self.device) for item in outputs]
        features = [feature.unsqueeze(0) if feature.ndim == 1 else feature for feature in features]
        feature_lens = torch.tensor([len(feature) for feature in features], dtype=torch.long, device=self.device)
        padded_features = pad_sequence(features, batch_first=True)

        return ContentFeatures(
            values=padded_features,
            lengths=feature_lens,
            feature_dim=padded_features.shape[-1],
            representation_type="continuous",
            temporal_granularity=self.granularity,
            backend="funasr",
            model_name=self.model_name,
            layer=self.layer,
            frame_hz=self.frame_hz,
        )
