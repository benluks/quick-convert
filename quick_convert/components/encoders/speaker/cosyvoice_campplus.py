from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from quick_convert.components.encoders.speaker.base import SpeakerEmbedding, SpeakerEncoder
from quick_convert.components.resources.resolution import ResolvedResource
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.utils.device import DeviceLike


class CosyVoiceCAMPPlusSpeakerEncoder(SpeakerEncoder):
    """CAMPPlus speaker features selected by ``output="pooled"`` or ``"frames"``.

    Pooled inference retains the pretrained backend's ten-second crop policy.
    Frame inference uses complete recordings or explicit prefixes, with valid
    lengths and alignment to the requested frame clock.
    """

    FEATURE_DIM = 192
    POOLED_DIM = 192
    FRAME_DIM = 512
    FRAME_HZ = 50
    SUPPORTED_OUTPUTS = frozenset({"pooled", "frames"})
    sample_rate = 16_000

    def __init__(
        self,
        repo_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        filename: str = "campplus.onnx",
        device: DeviceLike = None,
        output: str = "pooled",
        revision: str | None = None,
    ):
        super().__init__(device=device, output=output)
        self.register_buffer(
            "_device_anchor", torch.empty(0, device=self.__dict__.pop("_initial_device")), persistent=False
        )
        if output == "frames":
            self._initialize_frames(repo_id, filename, revision)
            return

        self.repo_id = repo_id
        self.filename = filename

        from huggingface_hub import hf_hub_download

        checkpoint_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
        )

        from quick_convert.external.cosyvoice.utils.onnx import EmbeddingExtractor

        self.model = EmbeddingExtractor(
            model_path=str(checkpoint_path),
        )

    def _encode_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(f"Expected waveform shape [T] or [1, T], got {tuple(waveform.shape)}")

        # CosyVoice's ONNX extractor runs on CPU internally.
        embedding = self.model.inference(
            waveform.cpu(),
        )

        return embedding.to(
            device=self.device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def encode(
        self,
        wav: torch.Tensor,
        sr: int,
    ) -> SpeakerEmbedding | ResolvedResource:
        if self.output == "frames":
            sample = AudioSample(utt_id="waveform", path=Path("waveform"), waveform=wav.reshape(1, -1), sample_rate=sr)
            return self.encode_batch(AudioBatch.from_samples([sample]))
        if sr != self.sample_rate:
            raise ValueError(f"CosyVoice CAMPPlus expects 16 kHz audio, got {sr} Hz")

        values = self._single_embedding(self._encode_waveform(wav))

        return SpeakerEmbedding(
            values=values,
            embedding_dim=self.feature_dim,
            backend="campplus",
            model_name="cosyvoice3",
        )

    @torch.inference_mode()
    def encode_batch(
        self,
        samples: AudioBatch,
    ) -> SpeakerEmbedding | ResolvedResource:
        if self.output == "frames":
            if samples.lengths is None:
                raise ValueError("CAMPPlus frame encoding requires valid waveform lengths.")
            lengths = torch.div(samples.lengths * self.FRAME_HZ, self.sample_rate, rounding_mode="floor")
            return self.encode_prefixes(samples, lengths, self.FRAME_HZ)
        if samples.waveforms is None or samples.lengths is None or samples.sample_rates is None:
            raise ValueError("Speaker encoding requires loaded audio.")
        if any(sr != self.sample_rate for sr in samples.sample_rates):
            raise ValueError("CosyVoice CAMPPlus expects 16 kHz audio")

        embeddings = []

        for waveform, length in zip(
            samples.waveforms,
            samples.lengths,
            strict=True,
        ):
            waveform = waveform[..., : int(length)]

            embedding = self._encode_waveform(waveform)
            embeddings.append(embedding)

        values = self._batch_embeddings(
            torch.stack(
                embeddings,
                dim=0,
            ),
            len(samples),
        )

        return SpeakerEmbedding(
            values=values,
            embedding_dim=self.feature_dim,
            backend="campplus",
            model_name="cosyvoice3",
        )

    def forward(
        self,
        samples: AudioBatch,
    ) -> SpeakerEmbedding | ResolvedResource:
        return self.encode_batch(samples)

    def _initialize_frames(self, repo_id, filename, revision):
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
            onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, ["batch", self.FRAME_DIM, "frames"])
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

    @device.setter
    def device(self, value):
        if "_device_anchor" in self._buffers:
            self._buffers["_device_anchor"] = self._buffers["_device_anchor"].to(value)
        else:
            self.__dict__["_initial_device"] = torch.device(value)

    @property
    def feature_dim(self):
        return self.FRAME_DIM if self.output == "frames" else self.POOLED_DIM

    @property
    def frame_hz(self):
        return self.FRAME_HZ if self.output == "frames" else None

    def _encode_frames(self, waveform, output_frames):
        import torchaudio.compliance.kaldi as kaldi

        # A 25 ms analysis window can exceed a one-frame (20 ms) prefix.
        # Zero padding never reads samples from the continuation.
        waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
        waveform = F.pad(waveform, (0, max(0, 400 - waveform.numel())))
        features = kaldi.fbank(waveform[None], num_mel_bins=80, sample_frequency=self.sample_rate, dither=0)
        features = features - features.mean(dim=0, keepdim=True)
        values = self.session.run(None, {self.input_name: features[None].numpy()})[0]
        if values.ndim != 3 or values.shape[:2] != (1, self.FRAME_DIM) or values.shape[2] == 0:
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
        if self.output != "frames":
            raise NotImplementedError("encode_prefixes requires output='frames'.")
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("CAMPPlus frame encoding requires loaded audio and lengths.")
        if batch.waveforms.ndim != 2 or torch.any(batch.sample_rates != self.sample_rate):
            raise ValueError("CAMPPlus frame encoding requires mono 16 kHz audio.")
        if frame_lengths.shape != (len(batch),) or frame_lengths.dtype != torch.long:
            raise ValueError("Prefix lengths must be int64 vectors with one entry per sample.")
        if not math.isfinite(frame_rate) or frame_rate <= 0 or torch.any(frame_lengths < 0):
            raise ValueError("Frame rate must be positive and finite; prefix lengths must be nonnegative.")
        values = []
        for waveform, samples, frames in zip(
            batch.waveforms, batch.lengths.tolist(), frame_lengths.tolist(), strict=True
        ):
            count = math.ceil(frames * self.sample_rate / frame_rate)
            if samples < 0 or samples > waveform.numel() or count > samples:
                raise ValueError("Requested CAMPPlus prefix exceeds valid waveform storage.")
            if frames:
                values.append(self._encode_frames(waveform[:count], frames))
            else:
                values.append(torch.empty(0, self.FRAME_DIM, device=self.device))
        return ResolvedResource(pad_sequence(values, batch_first=True), frame_lengths.to(self.device).clone())
