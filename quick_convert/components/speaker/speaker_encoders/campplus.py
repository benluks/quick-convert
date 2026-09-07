# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)


from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchaudio.compliance.kaldi as Kaldi
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from quick_convert.components.layers.conv import FCM
from quick_convert.components.layers.stats_pool import StatsPool
from quick_convert.components.speaker.speaker_encoders.base import SpeakerEmbedding, SpeakerEncoder
from quick_convert.data.types import AudioBatch


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def extract_feature(audio):
    features = []
    feature_times = []
    feature_lengths = []
    for au in audio:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])
    # padding for batch inference
    features_padded = pad_list(features, pad_value=0)
    # features = torch.cat(features)
    return features_padded, feature_lengths, feature_times


def get_nonlinear(config_str, channels):
    nonlinear = torch.nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", torch.nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", torch.nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return nonlinear


class TDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, f"Expect equal paddings, but got even kernel size ({kernel_size})"
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(torch.nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super().__init__()
        self.linear_local = torch.nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = torch.nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"Expect equal paddings, but got even kernel size ({kernel_size})"
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = torch.nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(torch.nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module(f"tdnnd{i + 1}", layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super().__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


# @tables.register("model_classes", "CAMPPlus")
class CAMPPlus(torch.nn.Module):
    def __init__(
        self,
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
        **kwargs,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module(f"block{i + 1}", block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                f"transit{i + 1}",
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"))
        else:
            assert self.output_level == "frame", "`output_level` should be set to 'segment' or 'frame'. "

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == "frame":
            x = x.transpose(1, 2)
        return x

    def inference(self, audio_list):
        speech, *_ = extract_feature(audio_list)
        results = self.forward(speech.to(torch.float32))
        return results


class CAMPPlusSpeakerEncoder(SpeakerEncoder):
    FEATURE_DIM = 192

    def __init__(
        self,
        repo_id: str = "ResembleAI/chatterbox",
        filename: str = "s3gen.safetensors",
        prefix: str = "speaker_encoder.",
        device: str | torch.device | None = None,
    ):
        super().__init__(device=device)

        self.repo_id = repo_id
        self.filename = filename

        self.model = CAMPPlus(
            feat_dim=80,
            embedding_size=self.feature_dim,
            output_level="segment",
        )

        checkpoint_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
        )

        state_dict = load_file(checkpoint_path)

        speaker_state_dict = {
            key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)
        }

        if not speaker_state_dict:
            raise RuntimeError(f"No weights with prefix {prefix!r} found in {checkpoint_path}")

        self.model.load_state_dict(
            speaker_state_dict,
            strict=True,
        )

        self.model.requires_grad_(False)
        self.model.eval()

        self.to(self.device)

    def _extract_features(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        features = []

        for waveform, length in zip(waveforms, lengths):
            waveform = waveform[..., : int(length)].cpu()

            if waveform.ndim != 1:
                raise ValueError(f"Expected individual waveform to be 1-D, got {tuple(waveform.shape)}")

            feature = Kaldi.fbank(
                waveform.unsqueeze(0),
                num_mel_bins=80,
            )

            feature = feature - feature.mean(
                dim=0,
                keepdim=True,
            )

            features.append(feature)

        max_length = max(feature.shape[0] for feature in features)

        padded = features[0].new_zeros(
            len(features),
            max_length,
            features[0].shape[-1],
        )

        for i, feature in enumerate(features):
            padded[i, : feature.shape[0]] = feature

        return padded

    @torch.inference_mode()
    def encode(
        self,
        wav: torch.Tensor,
        sr: int,
    ) -> SpeakerEmbedding:
        if sr != 16_000:
            raise ValueError(f"CAMPPlus expects 16 kHz audio, got {sr} Hz")

        if wav.ndim == 2:
            if wav.shape[0] != 1:
                raise ValueError(f"Expected mono audio, got shape {tuple(wav.shape)}")
            wav = wav.squeeze(0)

        if wav.ndim != 1:
            raise ValueError(f"Expected waveform shape [T] or [1, T], got {tuple(wav.shape)}")

        features = self._extract_features(
            wav.unsqueeze(0),
            torch.tensor([wav.shape[0]]),
        )

        values = self.model(
            features.to(
                device=self.device,
                dtype=torch.float32,
            )
        )

        return SpeakerEmbedding(
            values=values.squeeze(0),
            embedding_dim=self.feature_dim,
            backend="campplus",
            model_name="chatterbox",
        )

    @torch.inference_mode()
    def encode_batch(
        self,
        samples: AudioBatch,
    ) -> SpeakerEmbedding:
        if any(sr != 16_000 for sr in samples.sample_rates):
            raise ValueError("CAMPPlus expects 16 kHz audio")

        features = self._extract_features(
            samples.waveforms,
            samples.lengths,
        )

        values = self.model(
            features.to(
                device=self.device,
                dtype=torch.float32,
            )
        )

        return SpeakerEmbedding(
            values=values,
            embedding_dim=self.feature_dim,
            backend="campplus",
            model_name="chatterbox",
        )

    def forward(
        self,
        samples: AudioBatch,
    ) -> SpeakerEmbedding:
        return self.encode_batch(samples)

    def train(
        self,
        mode: bool = True,
    ) -> "CAMPPlusSpeakerEncoder":
        super().train(mode)
        return self
