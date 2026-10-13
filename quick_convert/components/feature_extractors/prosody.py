"""Frame-level acoustic prosody targets, independent of learned embeddings."""

import torch

from quick_convert.components.prosody import ProsodyMeasurements

from .base import BaseFeatureExtractor


class OpenSmileProsodyExtractor(BaseFeatureExtractor):
    """Extract ComParE pitch, voicing, and smoothed RMS without normalizing audio.

    Outputs unpadded CPU tensors with columns given by ``COLUMNS``. Times are
    frame centers in seconds. Zero F0 denotes unvoiced frames, never a pitch
    regression target. ComParE's smoothing and voicing decision are retained.
    """

    COLUMNS = ProsodyMeasurements.COLUMNS

    def __init__(self):
        import opensmile

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

    @property
    def feature_name(self):
        return "prosody"

    def extract_batch(self, batch):
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("Prosody extraction requires loaded audio and valid lengths.")
        outputs = []
        for waveform, length, sample_rate in zip(batch.waveforms, batch.lengths, batch.sample_rates, strict=True):
            signal = waveform[: int(length)].detach().cpu().numpy().copy()
            if len(signal) == 0:
                raise ValueError("Cannot extract prosody from empty audio.")
            frame = self.smile.process_signal(signal, int(sample_rate))
            start = frame.index.get_level_values("start").total_seconds()
            end = frame.index.get_level_values("end").total_seconds()
            f0 = torch.as_tensor(frame["F0final_sma"].to_numpy().copy())
            rms = torch.as_tensor(frame["pcm_RMSenergy_sma"].to_numpy().copy())
            score = torch.as_tensor(frame["voicingFinalUnclipped_sma"].to_numpy().copy())
            times = torch.as_tensor((start + end).to_numpy() / 2)
            values = torch.stack((times, f0, (f0 > 0).to(f0.dtype), rms, score), dim=-1).float()
            outputs.append(ProsodyMeasurements(values).values)
        return outputs
