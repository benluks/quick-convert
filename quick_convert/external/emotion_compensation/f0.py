# derived from
import numpy as np
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import torch

from quick_convert.components.feature_extractors.f0 import F0Extractor
from quick_convert.data.types import AudioBatch, AudioSample


def get_yaapt_f0(audio, rate=16000, interp=False):
    """
    Taken verbatim from https://github.com/xiaoxiaomiao323/emotion-compensation/blob/main/gen/adapted_from_facebookresearch/dataset_test.py
    """
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(
            signal,
            **{"frame_length": frame_length, "frame_space": 10.0, "nccf_thresh1": 0.25, "tda_frame_length": 25.0},
        )
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0


class PYAAPTF0Extractor(F0Extractor):
    """
    A wrapper I had to build to convert the tensor to numpy and possibly resolve the shape
    """

    def __init__(self):
        pass

    def _get_f0(self, audio, rate=16000, interp=False):
        return torch.FloatTensor(get_yaapt_f0(audio, rate=rate, interp=interp))

    def extract_sample(self, sample: AudioSample, rate=16000, interp=False):

        x = sample.waveform.cpu().numpy()
        # input_shape = x.shape
        return self._get_f0(x, rate=rate, interp=interp)

    def extract_batch(self, batch: AudioBatch, rate=16000, interp=False):

        x = batch.waveforms.cpu().numpy()
        # input_shape = x.shape
        return self._get_f0(x, rate=rate, interp=interp)
