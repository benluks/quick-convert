from __future__ import annotations

import warnings

import numpy as np
import torch

from ..low_level_features.f0 import F0Extractor


class YAAPTF0Extractor(F0Extractor):
    """Frame-level F0 from pYAAPT (``amfm_decompy``) at a 10 ms hop.

    The waveform is zero-padded by half a frame on each side before analysis,
    and unvoiced frames are 0. Settings follow the speech-resynthesis HiFi-GAN
    recipe used by PSST1.

    Requires the ``yaapt`` extra.

    Args:
        frame_length: Analysis frame length in milliseconds.
        frame_space: Hop between frames in milliseconds.
        nccf_thresh1: pYAAPT NCCF voicing threshold.
        tda_frame_length: Time-domain analysis frame length in milliseconds.
        zero_on_failure: Return an all-unvoiced track, with a warning, when
            pYAAPT raises (as the research code does) instead of propagating.
    """

    def __init__(
        self,
        frame_length: float = 20.0,
        frame_space: float = 10.0,
        nccf_thresh1: float = 0.25,
        tda_frame_length: float = 25.0,
        zero_on_failure: bool = True,
    ):
        super().__init__()
        self.frame_length = frame_length
        self.frame_space = frame_space
        self.nccf_thresh1 = nccf_thresh1
        self.tda_frame_length = tda_frame_length
        self.zero_on_failure = zero_on_failure

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Return F0 in Hz with shape ``[frames]`` for a mono ``[T]`` waveform."""
        import amfm_decompy.basic_tools as basic
        import amfm_decompy.pYAAPT as pYAAPT

        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        if waveform.ndim != 1:
            raise ValueError(f"Expected a mono waveform with shape [T] or [1, T], got {tuple(waveform.shape)}")

        to_pad = int(self.frame_length / 1000 * sample_rate) // 2
        signal = np.pad(waveform.detach().cpu().double().numpy(), (to_pad, to_pad), "constant", constant_values=0)
        try:
            pitch = pYAAPT.yaapt(
                basic.SignalObj(signal, sample_rate),
                frame_length=self.frame_length,
                frame_space=self.frame_space,
                nccf_thresh1=self.nccf_thresh1,
                tda_frame_length=self.tda_frame_length,
            )
        except Exception as error:
            if not self.zero_on_failure:
                raise
            warnings.warn(f"pYAAPT failed ({error!r}); using an unvoiced F0 track.", stacklevel=2)
            hop = int(self.frame_space / 1000 * sample_rate)
            return torch.zeros(waveform.shape[-1] // hop)
        return torch.from_numpy(np.asarray(pitch.samp_values, dtype=np.float32))

    def extract_sample(self, sample):
        if sample.waveform is None or sample.sample_rate is None:
            raise ValueError(f"F0 extraction requires loaded audio for sample {sample.utt_id!r}.")
        return self.extract(sample.waveform, int(sample.sample_rate))
