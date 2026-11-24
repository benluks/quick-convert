"""Direct waveform generation and identity target conditioning."""

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from quick_convert.data import GeneratedAudio
from quick_convert.systems.conversion.targets import PreparedTarget, SourceVoice


@dataclass
class WaveformGeneration:
    audio: GeneratedAudio


class WaveformReconstruction:
    def __init__(self, hop_size):
        if hop_size <= 0:
            raise ValueError("hop_size must be positive.")
        self.hop_size = hop_size

    def set_mode(self, system, mode):
        pass

    def requires_speaker(self, system):
        return False

    def input_frame_rate(self, system):
        return system.decoder.sample_rate / self.hop_size

    def speaker_condition(self, system, batch):
        return None

    def conditioning_streams(self, system, batch, target_lengths=None):
        return {}

    def decode_features(self, system, features, lengths, speaker_embedding=None, *, run_vocoder=True, duration_ratio=1.0):
        if speaker_embedding is not None or not run_vocoder or torch.any(torch.as_tensor(duration_ratio) != 1):
            raise ValueError("Direct waveform generation does not support these decoding options.")
        waves = [system.decoder(x[None, : int(n)])[0] for x, n in zip(features, lengths, strict=True)]
        return WaveformGeneration(
            GeneratedAudio(pad_sequence(waves, batch_first=True), lengths * self.hop_size, system.decoder.sample_rate)
        )

    def prepare_target(self, system, target):
        if system.training:
            raise RuntimeError("Call eval() before preparing targets.")
        if isinstance(target, SourceVoice):
            return PreparedTarget(None, system.target_space, "ssl-source-v1")
        if (
            isinstance(target, PreparedTarget)
            and target.space == system.target_space
            and target.representation == "ssl-source-v1"
            and target.conditioning is None
        ):
            return target
        raise ValueError("Only source reconstruction is currently supported.")

    @torch.inference_mode()
    def convert_batch(self, system, batch, *, target):
        self.prepare_target(system, target)
        return system(batch).generation.audio
