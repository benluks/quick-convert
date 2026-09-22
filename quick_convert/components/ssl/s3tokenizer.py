from __future__ import annotations

from os import PathLike
from pathlib import Path

import torch

from quick_convert.data.types import AudioBatch
from quick_convert.utils.audio import load_audio

from .base import ContentFeatures, DiscreteContentEncoder


class S3TokenizerContentEncoder(DiscreteContentEncoder):
    """S3Tokenizer V3 25 Hz content encoder with bottleneck introspection.

    layer=-1 returns all 12 encoder block outputs; layer=0..11 returns one
    block. The representation argument is reserved for stages after the
    encoder.

    The implementation uses S3Tokenizer's model and checkpoint loader rather
    than duplicating its architecture. Intermediate encoder states are captured
    with forward hooks, so the final captured layer is the exact tensor
    consumed by S3's quantizer.

    S3Tokenizer: https://github.com/xingchensong/S3Tokenizer
    Apache-2.0 licensed.
    """

    SAMPLE_RATE = 16_000
    FEATURE_DIM = 1280
    TIME_D = 1
    N_LAYERS = 12
    BOTTLENECK_DIM = 8
    REPRESENTATIONS = ("encoder", "pre_tanh", "post_tanh", "ternary", "tokens")

    def __init__(
        self,
        model_name: str = "speech_tokenizer_v3_25hz",
        representation: str = "encoder",
        layer: int = -1,
        device: str | None = None,
        download_root: str | None = None,
        max_length: int | None = None,
    ) -> None:
        super().__init__(device=device)
        self.validate_representation(representation)
        if model_name != "speech_tokenizer_v3_25hz":
            raise ValueError("S3TokenizerContentEncoder currently supports only 'speech_tokenizer_v3_25hz'.")
        if representation == "encoder" and layer != -1 and not 0 <= layer < self.N_LAYERS:
            raise ValueError(f"layer must be -1 or an integer in [0, {self.N_LAYERS - 1}].")

        self.model_name = model_name
        self.representation = representation
        self.layer = layer
        self.download_root = download_root
        self.max_length = max_length

        try:
            import s3tokenizer
        except ImportError as error:
            raise ImportError(
                "S3TokenizerContentEncoder requires the 's3tokenizer' package. "
                "Install quick-convert with the 's3tokenizer' extra."
            ) from error

        self._s3 = s3tokenizer
        self.model = s3tokenizer.load_model(model_name, download_root=download_root).to(self.device)
        self.model.freeze()
        self.model.eval()

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def feature_dim(self) -> int:
        if self.representation == "encoder":
            return self.FEATURE_DIM
        if self.representation == "tokens":
            return 1
        return self.BOTTLENECK_DIM

    def encode_file(self, path: PathLike) -> ContentFeatures:
        waveform, sample_rate = load_audio(
            Path(path),
            target_sr=self.sample_rate,
            mono=True,
            device="cpu",
        )
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)
        waveform = waveform.unsqueeze(0)
        lengths = torch.tensor([waveform.shape[-1]], dtype=torch.long)
        return self.encode_waveforms(waveform, lengths=lengths, sample_rate=sample_rate)

    def forward(self, batch: AudioBatch, **kwargs) -> ContentFeatures:
        if getattr(batch, "waveforms", None) is None:
            raise RuntimeError(f"{self.__class__.__name__} requires loaded audio. Set load=true in the dataset config.")
        if not (batch.sample_rates == self.sample_rate).all():
            raise RuntimeError(
                f"Expected {self.sample_rate} Hz audio, got {batch.sample_rates}. "
                f"Resample in the dataset with target_sr={self.sample_rate}."
            )
        return self.encode_waveforms(
            batch.waveforms,
            lengths=batch.lengths,
            sample_rate=self.sample_rate,
            **kwargs,
        )

    @torch.inference_mode()
    def encode_waveforms(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
        sample_rate: int | None = None,
        max_length: int | None = None,
    ) -> ContentFeatures:
        if waveforms.ndim != 2:
            raise ValueError(f"Expected waveforms with shape (batch, time), got {tuple(waveforms.shape)}")

        input_sample_rate = sample_rate or self.sample_rate
        if input_sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {input_sample_rate} Hz.")

        batch_size, padded_length = waveforms.shape
        if lengths is None:
            lengths = torch.full((batch_size,), padded_length, dtype=torch.long)
        else:
            lengths = lengths.to(dtype=torch.long)

        if lengths.shape != (batch_size,):
            raise ValueError(f"Expected lengths with shape ({batch_size},), got {tuple(lengths.shape)}")
        if torch.any(lengths <= 0):
            raise ValueError("All waveform lengths must be positive.")
        if torch.any(lengths > padded_length):
            raise ValueError("A waveform length exceeds the padded waveform size.")

        mels = []
        mel_lengths = []
        for index in range(batch_size):
            waveform = waveforms[index, : int(lengths[index])].to(self.device)
            mel = self._s3.log_mel_spectrogram(waveform, n_mels=128)
            mels.append(mel)
            mel_lengths.append(mel.shape[-1])

        max_mel_length = max(mel_lengths)
        padded_mels = [torch.nn.functional.pad(mel, (0, max_mel_length - mel.shape[-1])) for mel in mels]
        mel_batch = torch.stack(padded_mels)
        mel_lengths_tensor = torch.tensor(mel_lengths, dtype=torch.long, device=self.device)

        hidden, output_lengths, layers = self._encode_with_layers(mel_batch, mel_lengths_tensor)

        if self.representation == "encoder":
            if self.layer == -1:
                selected = torch.stack(layers, dim=2)
                layer_metadata: int | str | None = "all"
            else:
                selected = layers[self.layer]
                layer_metadata = self.layer
            representation_type = "continuous"
        else:
            pre_tanh = self.model.quantizer._codebook.project_down(hidden).float()
            if self.representation == "pre_tanh":
                selected = pre_tanh
                representation_type = "continuous"
            else:
                post_tanh = pre_tanh.tanh()
                if self.representation == "post_tanh":
                    selected = post_tanh
                    representation_type = "continuous"
                else:
                    ternary = self._ternary_factors(post_tanh)
                    if self.representation == "ternary":
                        selected = ternary
                        representation_type = "discrete_factorized"
                    else:
                        selected = self._pack_ternary(ternary).unsqueeze(-1)
                        representation_type = "discrete"
            layer_metadata = self.N_LAYERS - 1

        effective_max_length = max_length or self.max_length
        if effective_max_length is not None:
            selected = self._pad_or_trim_time(selected, effective_max_length)
            output_lengths = output_lengths.clamp_max(effective_max_length)

        return ContentFeatures(
            values=selected,
            lengths=output_lengths.to(selected.device),
            feature_dim=selected.shape[-1],
            representation_type=representation_type,
            temporal_granularity="frame",
            backend="s3tokenizer",
            model_name=self.model_name,
            layer=layer_metadata,
            frame_hz=25.0,
        )

    def _encode_with_layers(
        self,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        layers: list[torch.Tensor] = []

        def capture_layer(_module, _inputs, output):
            layers.append(output)

        handles = [block.register_forward_hook(capture_layer) for block in self.model.encoder.blocks]
        try:
            hidden, output_lengths = self.model.encoder(mel, mel_lengths)
        finally:
            for handle in handles:
                handle.remove()

        if len(layers) != self.N_LAYERS:
            raise RuntimeError(f"Expected {self.N_LAYERS} S3 encoder layers, captured {len(layers)}.")
        if layers[-1].data_ptr() != hidden.data_ptr() and not torch.equal(layers[-1], hidden):
            raise RuntimeError("Captured final S3 layer does not match encoder output.")
        return hidden, output_lengths, layers

    @staticmethod
    def _ternary_factors(post_tanh: torch.Tensor) -> torch.Tensor:
        """Return the eight FSQ coordinates in {-1, 0, 1}."""
        return (post_tanh * 0.9990000128746033).round().to(torch.int64)

    @staticmethod
    def _pack_ternary(ternary: torch.Tensor) -> torch.Tensor:
        """Pack eight ternary coordinates exactly as S3Tokenizer FSQ does."""
        if ternary.shape[-1] != 8:
            raise ValueError(f"Expected 8 ternary factors, got shape {tuple(ternary.shape)}.")
        powers = torch.pow(
            3,
            torch.arange(8, device=ternary.device, dtype=torch.int64),
        )
        return torch.sum((ternary + 1) * powers, dim=-1)
