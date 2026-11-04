"""Frozen CosyVoice CAMPPlus features immediately before statistics pooling."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.data import AudioBatch
from quick_convert.utils.device import DeviceLike, configure_device


class CosyVoiceCAMPPlusFrameEncoder(nn.Module):
    """Expose the supported CosyVoice ONNX graph's 512-channel frame output.

    The graph is truncated in memory, never written back to the model cache.
    Each waveform is processed separately because CAMPPlus uses global context.
    ``encode_prefixes`` crops audio before normalization and ONNX inference.
    """

    FEATURE_DIM = 512
    SAMPLE_RATE = 16_000
    FRAME_HZ = 50

    def __init__(
        self,
        repo_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        filename: str = "campplus.onnx",
        revision: str | None = None,
        device: DeviceLike = None,
    ):
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0, device=configure_device(device)), persistent=False)
        import onnx
        import onnxruntime
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        model = onnx.load(path)
        pools = [node for node in model.graph.node if node.name == "/xvector/stats/ReduceMean"]
        if len(pools) != 1 or pools[0].op_type != "ReduceMean":
            raise ValueError("Unsupported CAMPPlus ONNX graph: cannot identify the pre-pooling output.")
        output = pools[0].input[0]
        producers = [i for i, node in enumerate(model.graph.node) if output in node.output]
        if len(producers) != 1 or len(model.graph.input) != 1:
            raise ValueError("Unsupported CAMPPlus ONNX graph: ambiguous frame output or inputs.")
        del model.graph.node[producers[0] + 1 :]
        del model.graph.output[:]
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, ["batch", self.FEATURE_DIM, "frames"])
        )
        used = {name for node in model.graph.node for name in node.input}
        weights = [value for value in model.graph.initializer if value.name in used]
        del model.graph.initializer[:]
        model.graph.initializer.extend(weights)
        onnx.checker.check_model(model)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    @property
    def device(self):
        return self._device_anchor.device

    @property
    def feature_dim(self):
        return self.FEATURE_DIM

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    @property
    def frame_hz(self):
        return self.FRAME_HZ

    def _encode(self, waveform, output_frames):
        import torchaudio.compliance.kaldi as kaldi

        # A 25 ms analysis window can exceed a one-frame (20 ms) prefix.
        # Zero padding never reads samples from the continuation.
        waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
        waveform = F.pad(waveform, (0, max(0, 400 - waveform.numel())))
        features = kaldi.fbank(waveform[None], num_mel_bins=80, sample_frequency=self.SAMPLE_RATE, dither=0)
        features = features - features.mean(dim=0, keepdim=True)
        values = self.session.run(None, {self.input_name: features[None].numpy()})[0]
        if values.ndim != 3 or values.shape[:2] != (1, self.FEATURE_DIM) or values.shape[2] == 0:
            raise ValueError("CAMPPlus ONNX frame output must have shape [1, 512, time].")
        values = torch.from_numpy(values.copy())
        if not torch.isfinite(values).all():
            raise ValueError("CAMPPlus returned nonfinite frame features.")
        # Kaldi snips boundary windows; map the prefix's complete feature sequence
        # onto the requested mel clock rather than padding with fictitious frames.
        values = F.interpolate(values, size=output_frames, mode="linear", align_corners=False)
        return values[0].transpose(0, 1).to(self.device)

    @torch.no_grad()
    def encode_prefixes(
        self, batch: AudioBatch, frame_lengths: torch.Tensor, frame_rate: float = 50
    ) -> ResolvedResource:
        """Encode only the requested waveform prefixes, returning [B, P, 512]."""
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("CAMPPlus frame encoding requires loaded audio and lengths.")
        if batch.waveforms.ndim != 2 or torch.any(batch.sample_rates != self.SAMPLE_RATE):
            raise ValueError("CAMPPlus frame encoding requires mono 16 kHz audio.")
        if frame_lengths.shape != (len(batch),) or frame_lengths.dtype != torch.long:
            raise ValueError("Prefix lengths must be int64 vectors with one entry per sample.")
        if not math.isfinite(frame_rate) or frame_rate <= 0 or torch.any(frame_lengths < 0):
            raise ValueError("Frame rate must be positive and finite; prefix lengths must be nonnegative.")
        values = []
        for waveform, samples, frames in zip(
            batch.waveforms, batch.lengths.tolist(), frame_lengths.tolist(), strict=True
        ):
            count = math.ceil(frames * self.SAMPLE_RATE / frame_rate)
            if samples < 0 or samples > waveform.numel() or count > samples:
                raise ValueError("Requested CAMPPlus prefix exceeds valid waveform storage.")
            if frames:
                values.append(self._encode(waveform[:count], frames))
            else:
                values.append(torch.empty(0, self.FEATURE_DIM, device=self.device))
        return ResolvedResource(pad_sequence(values, batch_first=True), frame_lengths.to(self.device).clone())

    @torch.no_grad()
    def forward(self, batch: AudioBatch) -> ResolvedResource:
        """Encode complete reference recordings on the native nominal 50 Hz clock."""
        if batch.lengths is None:
            raise ValueError("CAMPPlus frame encoding requires valid waveform lengths.")
        lengths = torch.div(batch.lengths * self.FRAME_HZ, self.SAMPLE_RATE, rounding_mode="floor")
        return self.encode_prefixes(batch, lengths, self.FRAME_HZ)
