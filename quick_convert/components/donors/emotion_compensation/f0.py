# derived from
import numpy as np
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from sympy import Union
import torch

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


class PYAAPTF0Extractor:
    """
    A wrapper I had to build to convert the tensor to numpy and possibly resolve the shape
    """

    def __init__(self):
        pass

    def __call__(self, x: torch.tensor, rate=16000, interp=False):

        x = x.cpu().numpy()
        # input_shape = x.shape
        f0 = get_yaapt_f0(x, rate=rate, interp=interp)
        return torch.FloatTensor(f0)
