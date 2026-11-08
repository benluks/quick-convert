import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quick_convert.components.encoders.content import DACQuantizedContentEncoder, MimiQuantizedContentEncoder
from quick_convert.components.encoders.content.quantized import (
    QuantizedContentEncoder,
    SpeechTokenizerQuantizedContentEncoder,
)


class TinyCodec(QuantizedContentEncoder):
    BACKEND = "test"
    sample_rate = 16000
    n_codebooks = 2
    frame_hz = 50
    model_name = "test"

    def __init__(self):
        super().__init__("cpu")
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

    def _encode_layers(self, waveform):
        # Deliberately sensitive to padding; the adapter must trim before calling.
        return waveform.flatten()[:, None, None].expand(-1, 2, 3) * self.weight


def test_codec_does_not_expose_caller_storage_to_backend():
    class MutatingCodec(TinyCodec):
        def _encode_layers(self, waveform):
            waveform.zero_()
            return super()._encode_layers(waveform)

    waveform = torch.ones(1, 4)
    MutatingCodec().encode_waveforms(waveform)
    assert torch.all(waveform == 1)


def test_codec_batch_and_file_sample_rate_contract(tmp_path):
    import torchaudio

    from quick_convert.data import AudioBatch, AudioSample

    path = tmp_path / "stereo.wav"
    torchaudio.save(str(path), torch.ones(2, 4), 16000)
    codec = TinyCodec()
    assert codec.encode_file(path).lengths.tolist() == [4]
    batch = AudioBatch.from_samples([AudioSample("sample", path, waveform=torch.ones(4), sample_rate=16000)])
    assert codec(batch).values.shape == (1, 4, 2, 3)
    with pytest.raises(ValueError, match="16000"):
        codec.encode_waveforms(torch.ones(1, 4), sample_rate=8000)
    batch.sample_rates.fill_(8000)
    with pytest.raises(ValueError, match="Resample"):
        codec(batch)


@pytest.mark.parametrize("waveform", [torch.ones(1, 3, dtype=torch.long), torch.full((1, 3), float("nan"))])
def test_codec_rejects_invalid_samples(waveform):
    with pytest.raises(ValueError, match="finite floating-point"):
        TinyCodec().encode_waveforms(waveform)


def test_frozen_codec_trims_padding_and_preserves_inputs():
    codec = TinyCodec().train()
    waveform = torch.tensor([[1.0, 2.0, 3.0], [4.0, 99.0, 99.0]])
    before = waveform.clone()
    output = codec.encode_waveforms(waveform, torch.tensor([3, 1]))
    assert not codec.training
    assert output.values.shape == (2, 3, 2, 3)
    assert output.lengths.tolist() == [3, 1]
    assert output.values[1, 1:].count_nonzero() == 0
    assert torch.equal(waveform, before)
    assert not output.values.requires_grad
    nn.Linear(3, 1)(output.values).sum().backward()
    assert codec.weight.grad is None


@pytest.mark.parametrize("lengths", [torch.tensor([0]), torch.tensor([4]), torch.tensor([1.5])])
def test_codec_rejects_bad_lengths(lengths):
    with pytest.raises(ValueError, match="ength"):
        TinyCodec().encode_waveforms(torch.ones(1, 3), lengths)


def test_dac_keeps_original_rvq_and_uses_sequential_residuals(monkeypatch):
    class Quantizer(nn.Module):
        def forward(self, residual):
            return residual / 2, None, None, None, None

    encoder = nn.Conv1d(1, 3, 2, stride=2, bias=False)
    model = SimpleNamespace(encoder=encoder, quantizer=nn.Module())
    model.quantizer.quantizers = nn.ModuleList([Quantizer(), Quantizer()])
    model.sample_rate, model.hop_length, model.latent_dim = 16000, 2, 3
    monkeypatch.setitem(
        sys.modules,
        "dac",
        SimpleNamespace(
            DAC=SimpleNamespace(load=lambda _: model), utils=SimpleNamespace(download=lambda **_: "checkpoint")
        ),
    )
    codec = DACQuantizedContentEncoder(device="cpu").train()
    output = codec.encode_waveforms(torch.ones(2, 5), torch.tensor([5, 2]))
    assert output.lengths.tolist() == [3, 1]
    torch.testing.assert_close(output.values[0, :, 1], output.values[0, :, 0] / 2)
    assert codec.quantizer is model.quantizer
    assert not codec.encoder.training
    assert all(not p.requires_grad for p in codec.parameters())


@pytest.mark.parametrize("layer", [None, 0, 1])
def test_speechtokenizer_uses_native_layers(monkeypatch, layer):
    class Model(nn.Module):
        sample_rate, n_q, downsample_rate = 16000, 2, 320

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward_feature(self, waveform, layers):
            assert layers == [0, 1]
            return [waveform.expand(1, 3, -1) * (i + 1) for i in layers]

    monkeypatch.setitem(
        sys.modules,
        "speechtokenizer",
        SimpleNamespace(SpeechTokenizer=SimpleNamespace(load_from_checkpoint=lambda *_: Model())),
    )
    codec = SpeechTokenizerQuantizedContentEncoder("config", "weights", device="cpu", layer=layer).train()
    waveform = torch.tensor([[1.0, 2.0, 3.0], [4.0, 99.0, 99.0]], requires_grad=True)
    before = waveform.detach().clone()
    output = codec.encode_waveforms(waveform, torch.tensor([3, 1]), sample_rate=16000)
    assert output.lengths.tolist() == [3, 1]
    assert output.layer == ("rvq" if layer is None else layer)
    assert output.representation_type == "quantized"
    assert output.frame_hz == 50
    assert not codec.training and not codec.model.training
    assert all(not p.requires_grad for p in codec.parameters())
    assert not output.values.requires_grad
    assert output.values[1, 1:].count_nonzero() == 0
    torch.testing.assert_close(waveform, before)
    if layer is None:
        assert output.values.shape == (2, 3, 2, 3)
        torch.testing.assert_close(output.values[:, :, 1], output.values[:, :, 0] * 2)
    else:
        assert output.values.shape == (2, 3, 3)
        torch.testing.assert_close(output.values[0], before[0, :, None].expand(3, 3) * (layer + 1))
    nn.Linear(3, 1)(output.values).sum().backward()
    assert waveform.grad is None
    assert codec.model.weight.grad is None
    with pytest.raises(ValueError, match="codebooks"):
        SpeechTokenizerQuantizedContentEncoder("config", "weights", device="cpu", layer=2)


@pytest.mark.parametrize("layer", [-1, 0.5, True, "0"])
def test_speechtokenizer_rejects_invalid_layer_before_loading(layer):
    with pytest.raises(ValueError, match="layer"):
        SpeechTokenizerQuantizedContentEncoder("config", "weights", device="cpu", layer=layer)


@pytest.mark.parametrize("paths", [{"config_path": "config"}, {"checkpoint_path": "weights"}])
def test_speechtokenizer_requires_both_local_paths(paths):
    with pytest.raises(ValueError, match="both"):
        SpeechTokenizerQuantizedContentEncoder(device="cpu", **paths)


def test_speechtokenizer_downloads_default_model(monkeypatch):
    downloads, loaded = [], []

    def download(repo_id, filename):
        downloads.append((repo_id, filename))
        return f"/cache/{filename}"

    def load(config, checkpoint):
        loaded.append((config, checkpoint))
        model = nn.Linear(1, 1)
        model.n_q = 8
        return model

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(
        sys.modules, "speechtokenizer", SimpleNamespace(SpeechTokenizer=SimpleNamespace(load_from_checkpoint=load))
    )
    codec = SpeechTokenizerQuantizedContentEncoder(layer=0, device="cpu")
    assert downloads == [
        ("fnlp/SpeechTokenizer", "speechtokenizer_hubert_avg/config.json"),
        ("fnlp/SpeechTokenizer", "speechtokenizer_hubert_avg/SpeechTokenizer.pt"),
    ]
    assert loaded == [
        ("/cache/speechtokenizer_hubert_avg/config.json", "/cache/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    ]
    assert not codec.model.training
    assert all(not p.requires_grad for p in codec.parameters())


def test_speechtokenizer_content_config_selects_semantic_layer():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from quick_convert.dependencies import dependency_plan

    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parents[1] / "quick_convert/configs")):
        cfg = compose(
            config_name="run/precompute_content_speechtokenizer_librispeech",
            overrides=["device=cpu"],
        )
    assert cfg.encoder.layer == 0
    assert cfg.dataset.target_sr == 16000
    assert cfg.encoder._target_.endswith("SpeechTokenizerQuantizedContentEncoder")
    assert "speechtokenizer" in dependency_plan(cfg).extras


@pytest.mark.parametrize("decoder", ["cmdiff", "dit"])
def test_speechtokenizer_reconstruction_config_uses_single_layer_without_fusion(decoder):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from quick_convert.utils.resolvers import register_config_resolvers

    register_config_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parents[1] / "quick_convert/configs")):
        cfg = compose(config_name=f"run/train_sslr_speechtokenizer_{decoder}_campp_prefix", return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    try:
        job = OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"])
        OmegaConf.to_container(job, resolve=True, throw_on_missing=True)
    finally:
        HydraConfig.instance().cfg = None
    assert cfg.system.online_encoders.content.layer == 0
    assert cfg.system.feature_transform is None
    assert cfg.train_dataset.target_sr == cfg.val_dataset.target_sr == 16000
    assert cfg.data.target_sr == 16000
    assert cfg.system.decoder.feature_dim == 1024
    assert cfg.system.decoder.flow.input_frame_rate == 50
    assert cfg.system.decoder.flow.token_mel_ratio == 1
    if decoder == "cmdiff":
        assert cfg.system.decoder.flow.encoder.upsample_factor == 1
    else:
        assert cfg.system.decoder.flow._target_.endswith("CausalMaskedDiffWithDiT")
        assert "encoder" not in cfg.system.decoder.flow
    assert cfg.system.online_encoders.speaker is None
    assert cfg.system.online_encoders.frame_speaker._target_.endswith("CosyVoiceCAMPPlusSpeakerEncoder")
    dimensions = cfg.system.decoder.stream_conditioning.dimensions
    assert cfg.system.decoder.flow.decoder.estimator.cond_dim == dimensions.frame_speaker + 1
    assert cfg.system.decoder.flow.decoder.estimator.spk_dim is None


def test_mimi_contributions_reconstruct_native_quantizer(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from transformers import MimiConfig, MimiModel

    config = MimiConfig(
        hidden_size=16,
        num_filters=4,
        num_residual_layers=1,
        upsampling_ratios=[2, 2],
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        vector_quantization_hidden_dimension=8,
        num_quantizers=3,
        num_semantic_quantizers=1,
        codebook_size=8,
        codebook_dim=8,
        upsample_groups=16,
    )
    model = MimiModel(config).eval()
    for branch in (
        model.quantizer.semantic_residual_vector_quantizer,
        model.quantizer.acoustic_residual_vector_quantizer,
    ):
        for layer in branch.layers:
            layer.codebook.embed.normal_()
    monkeypatch.setattr(transformers.MimiModel, "from_pretrained", lambda _: model)
    codec = MimiQuantizedContentEncoder(device="cpu")
    waveform = torch.randn(1, 1, 128)
    with torch.no_grad():
        codes = model.encode(waveform, use_streaming=False).audio_codes
        expected = model.quantizer.decode(codes).transpose(1, 2)
    output = codec.encode_waveforms(waveform)
    torch.testing.assert_close(output.values.sum(2), expected)


def test_real_speechtokenizer_local_checkpoint_and_native_sum(tmp_path):
    import json

    SpeechTokenizer = pytest.importorskip("speechtokenizer").SpeechTokenizer
    config = dict(
        n_filters=2,
        dimension=8,
        semantic_dimension=8,
        strides=[2, 2],
        lstm_layers=0,
        bidirectional=False,
        dilation_base=2,
        residual_kernel_size=3,
        n_residual_layers=1,
        activation="ELU",
        sample_rate=16000,
        n_q=2,
        codebook_size=8,
    )
    model = SpeechTokenizer(config).eval()
    for layer in model.quantizer.vq.layers:
        layer._codebook.inited.fill_(True)
        layer._codebook.embed.normal_()
    config_path, checkpoint = tmp_path / "config.json", tmp_path / "model.pt"
    config_path.write_text(json.dumps(config))
    torch.save(model.state_dict(), checkpoint)
    codec = SpeechTokenizerQuantizedContentEncoder(str(config_path), str(checkpoint), device="cpu")
    before = {key: value.clone() for key, value in codec.state_dict().items()}
    waveform = torch.randn(1, 64)
    with torch.no_grad():
        quantized, *_ = model.quantizer(model.encoder(waveform[:, None]))
    output = codec.encode_waveforms(waveform)
    torch.testing.assert_close(output.values.sum(2), quantized.transpose(1, 2))
    for key, value in codec.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
