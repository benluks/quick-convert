from __future__ import annotations

from os import PathLike
from pathlib import Path

import torch
import torch.nn.functional as F

from quick_convert.data.types import AudioBatch
from quick_convert.utils.audio import load_audio

from .base import ContentEncoder, ContentFeatures


class WavLMContentEncoder(ContentEncoder):
    FEATURE_DIM = 1024
    TIME_D = 1

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        sample_rate: int = 16000,
        layer: int | None = None,
        device: str | None = None,
        local_files_only: bool = False,
        downsample_factor: int = 0,
        max_length: int | None = None,
        do_normalize: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(device=device)

        self.model_name = model_name
        self.sample_rate = sample_rate
        self.layer = layer
        self.local_files_only = local_files_only
        self.downsample_factor = downsample_factor
        self.max_length = max_length

        if downsample_factor > 1 and layer is not None:
            raise ValueError("Layer-dimension downsampling requires all layers to be preserved. Set `layer=None`.")

        from transformers import AutoFeatureExtractor, AutoModel

        self.processor = AutoFeatureExtractor.from_pretrained(
            model_name, local_files_only=local_files_only, do_normalize=do_normalize
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            **kwargs,
        ).to(self.device)
        self.model.eval()

    def encode_file(self, path: PathLike) -> ContentFeatures:
        path = Path(path)
        waveform, sample_rate = load_audio(path, target_sr=self.sample_rate, mono=True, device="cpu")

        # if waveform.ndim == 2 and waveform.shape[0] > 1:
        #     waveform = waveform.mean(dim=0, keepdim=True)

        # if waveform.ndim != 2:
        #     raise ValueError(f"Expected waveform with shape (channels, time), got {tuple(waveform.shape)}")

        lengths = torch.tensor(
            [waveform.shape[-1]],
            dtype=torch.long,
            device=waveform.device,
        )

        return self.encode_waveforms(
            waveform,
            lengths=lengths,
            sample_rate=sample_rate,
        )

    def forward(self, batch: AudioBatch, **kwargs) -> ContentFeatures:
        if getattr(batch, "waveforms", None) is None:
            raise RuntimeError(
                f"{self.__class__.__name__} only works with loaded audio for "
                "now. Please set `load: true` in your dataset config."
            )

        if not (batch.sample_rates == self.sample_rate).all():
            raise RuntimeError(
                f"Expected input audio at {self.sample_rate} Hz, but got "
                f"{batch.sample_rates}. Batch resampling within "
                f"{self.__class__.__name__} is not currently supported. "
                f"Set `target_sr={self.sample_rate}` with `load=True` in the "
                "dataset config."
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
        """
        Encode a padded batch of waveforms with WavLM.

        Args:
            waveforms:
                Tensor with shape (batch, samples).
            lengths:
                Valid waveform lengths in samples, with shape (batch,).
                When omitted, every waveform is assumed to use the full padded
                length.
            sample_rate:
                Input sample rate. Must match ``self.sample_rate``.
            max_length:
                Optional fixed output length in WavLM frames.

        Returns:
            ContentFeatures whose values have shape:

            - ``(batch, frames, hidden_dim)`` when ``layer`` is an integer.
            - ``(batch, frames, layers, hidden_dim)`` when ``layer=None``.
        """
        if waveforms.ndim != 2:
            raise ValueError(f"Expected waveforms with shape (batch, time), got {tuple(waveforms.shape)}")

        input_sample_rate = sample_rate or self.sample_rate
        if input_sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {input_sample_rate} Hz.")

        batch_size, padded_length = waveforms.shape

        if lengths is None:
            lengths = torch.full(
                (batch_size,),
                padded_length,
                dtype=torch.long,
                device=waveforms.device,
            )
        else:
            lengths = lengths.to(dtype=torch.long)

        if lengths.shape != (batch_size,):
            raise ValueError(f"Expected lengths with shape ({batch_size},), got {tuple(lengths.shape)}")

        if torch.any(lengths <= 0):
            raise ValueError("All waveform lengths must be positive.")

        if torch.any(lengths > padded_length):
            raise ValueError(
                "A waveform length exceeds the padded waveform size: "
                f"maximum length is {padded_length}, got "
                f"{int(lengths.max())}."
            )

        # Hugging Face's audio processor accepts unpadded individual
        # waveforms, then pads them and constructs the sample-level mask.
        waveform_list = [waveforms[i, : int(lengths[i])].numpy() for i in range(batch_size)]

        inputs = self.processor(
            waveform_list,
            sampling_rate=input_sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(self.device)

        # inputs = {name: value.to(self.device) for name, value in inputs.items()}

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("WavLM did not return hidden states.")

        if self.layer is None:
            # hidden_states[0] is the projected convolutional representation.
            # hidden_states[1:] are the outputs of transformer layers 1..N.
            selected = torch.stack(
                hidden_states[1:],
                dim=2,
            )
        else:
            try:
                selected = hidden_states[self.layer]
            except IndexError as error:
                raise ValueError(
                    f"Invalid layer {self.layer}; WavLM returned "
                    f"{len(hidden_states)} hidden states. Valid non-negative "
                    f"indices are 0 through {len(hidden_states) - 1}."
                ) from error

        output_lengths = self._feature_output_lengths(lengths)
        output_lengths = output_lengths.to(selected.device)

        effective_max_length = max_length or self.max_length
        if effective_max_length is not None:
            selected = self._pad_or_trim_time(
                selected,
                effective_max_length,
            )
            output_lengths = output_lengths.clamp_max(effective_max_length)

        if self.downsample_factor > 1:
            selected = self._downsample_layers(
                selected,
                self.downsample_factor,
            )

        return ContentFeatures(
            values=selected,
            lengths=output_lengths,
            feature_dim=selected.shape[-1],
            representation_type="continuous",
            temporal_granularity="frame",
            backend="transformers",
            model_name=self.model_name,
            layer=self.layer,
        )

    def _feature_output_lengths(
        self,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate exact WavLM convolutional frontend output lengths.

        This follows the Conv1d output-size calculation used by WavLM's
        feature extractor:

            floor((length - kernel_size) / stride) + 1
        """
        output_lengths = input_lengths.clone()

        for kernel_size, stride in zip(
            self.model.config.conv_kernel,
            self.model.config.conv_stride,
            strict=True,
        ):
            output_lengths = (
                torch.div(
                    output_lengths - kernel_size,
                    stride,
                    rounding_mode="floor",
                )
                + 1
            )

        return output_lengths.clamp_min(0)

    @staticmethod
    def _downsample_layers(
        values: torch.Tensor,
        factor: int,
    ) -> torch.Tensor:
        """
        Average-pool across the layer dimension.

        Args:
            values:
                Tensor with shape (batch, frames, layers, hidden_dim).
            factor:
                Pooling kernel size and stride.
        """
        if values.ndim != 4:
            raise ValueError(
                "Layer downsampling expects values with shape "
                "(batch, frames, layers, hidden_dim), but got "
                f"{tuple(values.shape)}."
            )

        batch_size, frames, layers, hidden_dim = values.shape

        if factor > layers:
            raise ValueError(f"Downsample factor {factor} exceeds the number of layers ({layers}).")

        values = values.permute(0, 1, 3, 2)
        values = values.reshape(
            batch_size * frames,
            hidden_dim,
            layers,
        )

        values = F.avg_pool1d(
            values,
            kernel_size=factor,
            stride=factor,
        )

        pooled_layers = values.shape[-1]

        values = values.reshape(
            batch_size,
            frames,
            hidden_dim,
            pooled_layers,
        )

        return values.permute(0, 1, 3, 2)


if __name__ == "__main__":
    # import torch

    wavlm = WavLMContentEncoder()
    wavlm.encode_file("/Users/ben/librispeech/LibriSpeech/test-other/1688/142285/1688-142285-0004.flac")
