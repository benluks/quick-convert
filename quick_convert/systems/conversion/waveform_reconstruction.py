"""Direct frame-to-waveform reconstruction with shared resource resolution."""

import math
from uuid import uuid4

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from quick_convert.components.resources.resolution import OnlineResourceMixin
from quick_convert.data import GeneratedAudio
from quick_convert.systems._content import content_frame_rate, transform_content
from quick_convert.systems.base import InferenceSystem
from quick_convert.systems.conversion.targets import PreparedTarget, SourceVoice


class WaveformReconstructionSystem(OnlineResourceMixin, InferenceSystem):
    def __init__(self, decoder, hop_size, online_encoders=None, feature_transform=None):
        super().__init__()
        self.decoder = decoder
        self.hop_size = hop_size
        self.feature_transform = feature_transform or nn.Identity()
        self.configure_online_encoders(online_encoders)
        self.feature_transform.requires_grad_(False)
        self._target_space = str(uuid4())

    def train(self, mode=True):
        super().train(mode)
        self.online_encoders.eval()
        self.feature_transform.eval()
        return self

    def encode(self, batch):
        content = self.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("Content requires valid frame lengths.")
        rate = content_frame_rate(content.metadata, self.feature_transform)
        if rate is not None and not math.isclose(rate, self.decoder.sample_rate / self.hop_size):
            raise ValueError("Content frame rate does not match the waveform decoder clock.")
        with torch.no_grad():
            return transform_content(content.values, content.lengths, self.feature_transform, metadata=content.metadata)

    def forward(self, batch):
        features, lengths = self.encode(batch)
        waves = [self.decoder(x[None, : int(n)])[0] for x, n in zip(features, lengths, strict=True)]
        return GeneratedAudio(pad_sequence(waves, batch_first=True), lengths * self.hop_size, self.decoder.sample_rate)

    @property
    def target_space(self):
        return self._target_space

    def prepare_target(self, target):
        if isinstance(target, SourceVoice):
            return PreparedTarget(None, self.target_space, "waveform-source-v1")
        if (
            isinstance(target, PreparedTarget)
            and target.space == self.target_space
            and target.representation == "waveform-source-v1"
            and target.conditioning is None
        ):
            return target
        raise ValueError("Only source reconstruction is currently supported.")

    @torch.inference_mode()
    def convert_batch(self, batch, *, target):
        self.prepare_target(target)
        return self(batch)
