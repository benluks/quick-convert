from __future__ import annotations
from pathlib import Path

import numpy as np

import torch
import torchaudio

from torch.nn.utils.rnn import pad_packed_sequence

from .base import ContentEncoder, ContentFeatures

# Source: https://github.com/MiniXC/masked_prosody_model/tree/main


class ProsodyEncoder(ContentEncoder):
    """Content encoder backed by Masked Prosody Model.

    Extracts frame-level prosody representations from raw waveforms
      using a transformers-like interface.
    """

    def __init__(
        self,
        model_name: str = "cdminix/masked_prosody_model",
        sample_rate: int = 22050,
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
        self.model_name = "cdminix/masked_prosody_model"
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.local_files_only = local_files_only

        from masked_prosody_model import MaskedProsodyModel

        self.model = MaskedProsodyModel.from_pretrained(model_id=model_name).to(self.device)
        self.model.eval()

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

    def process_tensor(self, audio: torch.Tensor, sr: int = 22050, layer: int = 7) -> torch.Tensor:
        """Process an audio file and extract model representations.

        Args:
                audio: Tensor containing the audio waveform
                sr: Sample rate of the audio
                layer: Which layer's representations to return

        Returns:
                Tensor containing the model's representations
        """

        # audio, sr = librosa.load(audio_path, sr=22050)
        # Convert to numpy, normalize, and window into 6s chunks
        audio = audio.detach().cpu().numpy()
        audio = audio / np.abs(audio).max()
        # window into 6s chunks
        windows = []
        for i in range(0, len(audio), sr * 6):
            windows.append(audio[i : i + sr * 6])
        results = []

        for i, window in enumerate(windows):
            pitch = self.pitch_measure(window, np.array([1000]))["measure"]
            energy = self.energy_measure(window, np.array([1000]))["measure"]
            vad = self.vad_measure(window, np.array([1000]))["measure"]
            pitch[np.isnan(pitch)] = -1000
            energy[np.isnan(energy)] = -1000
            vad[np.isnan(vad)] = -1000
            pitch = np.clip(
                pitch,
                self.args.pitch_min,
                self.args.pitch_max,
            ) / (self.args.pitch_max - self.args.pitch_min)
            energy = np.clip(
                energy,
                self.args.energy_min,
                self.args.energy_max,
            ) / (self.args.energy_max - self.args.energy_min)
            vad = np.clip(
                vad,
                self.args.vad_min,
                self.args.vad_max,
            ) / (self.args.vad_max - self.args.vad_min)
            pitch = torch.tensor(pitch)
            energy = torch.tensor(energy)
            vad = torch.tensor(vad)
            pitch = torch.bucketize(pitch, self.bins).long().unsqueeze(0)
            energy = torch.bucketize(energy, self.bins).long().unsqueeze(0)
            vad = torch.bucketize(vad, torch.linspace(0, 1, 2)).long().unsqueeze(0)
            all_features = torch.stack([pitch, energy, vad]).transpose(0, 1)
            result = self.mpm(all_features, return_layer=layer)
            results.append(result)
        # bring all representations together
        representations = []
        for result in results:
            representations.append(result["representations"].squeeze(0))
        representations = torch.cat(representations, dim=0)
        return representations

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
            raise ValueError(f"Expected wavs with shape (batch, time), got {tuple(wavs.shape)}")

        sample_rate = sample_rate or self.sample_rate

        if sample_rate != self.sample_rate:
            wavs = torchaudio.functional.resample(
                wavs,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )
            sample_rate = self.sample_rate

        wavs = wavs.detach().cpu()

        outputs = []
        for wav in wavs:
            output = self.model.process_tensor(wav.unsqueeze(0), sample_rate=sample_rate)
            outputs.append(output)

        # Pad outputs to the same length
        output, output_lengths = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0)

        # Subsample by a factor of 2 using average pooling
        output = torch.nn.functional.avg_pool1d(output.transpose(1, 2), kernel_size=3, stride=2, padding=1).transpose(
            1, 2
        )

        output_lengths = (output_lengths - 1) // 2 + 1

        return ContentFeatures(
            values=output,
            lengths=output_lengths,
            feature_dim=output.shape[-1],
            representation_type="continuous",
            temporal_granularity="frame",
            backend="funasr",
            model_name=self.model_name,
            layer=self.layer,
        )
