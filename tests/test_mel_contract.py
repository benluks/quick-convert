from dataclasses import asdict, replace

import pytest
import torch
import torchaudio
from torch import nn

from quick_convert.components.decoders.cosyvoice import CosyVoiceSpectrogramGenerator
from quick_convert.components.decoders.cosyvoice_config import read_cosyvoice_config
from quick_convert.components.feature_extractors.mel import MelSpectrogramConfig


SPEC = MelSpectrogramConfig(24000, 1920, 480, 1920, 80, 0, None)


class Flow(nn.Module):
    vocab_size = None
    input_size = 4
    output_size = 80
    input_frame_rate = 50
    token_mel_ratio = 1

    def forward(self, batch, device):
        self.batch = batch
        return {"loss": batch["speech_feat"].sum()}


def decoder(spec=SPEC):
    return CosyVoiceSpectrogramGenerator(Flow(), 4, device="cpu", mel_config=spec)


def tone(length):
    return 0.1 * torch.sin(2 * torch.pi * 440 * torch.arange(length) / 16000)


def test_resampled_targets_match_explicit_24k_extraction_and_preserve_input():
    wav = tone(16000)[None]
    original = wav.clone()
    module = decoder()
    actual, lengths = module._compute_mels(wav, torch.tensor([16000]), 16000)
    upsampled = torchaudio.functional.resample(wav, 16000, 24000)
    expected, expected_lengths = SPEC.extract(upsampled, torch.tensor([24000]), 24000)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(lengths, expected_lengths)
    assert lengths.tolist() == [50]
    torch.testing.assert_close(wav, original)


def test_shorter_sample_is_independent_of_batch_padding():
    short = tone(8000)
    batch = torch.stack([tone(16000), torch.nn.functional.pad(short, (0, 8000))])
    batched, lengths = SPEC.extract(batch, torch.tensor([16000, 8000]), 16000)
    single, single_lengths = SPEC.extract(short[None], torch.tensor([8000]), 16000)
    torch.testing.assert_close(batched[1, :, : lengths[1]], single[0])
    assert single_lengths.tolist() == [25]


def test_switching_spectral_contracts_has_no_shared_cache_collision():
    wav = tone(16000)[None]
    lengths = torch.tensor([16000])
    a, _ = SPEC.extract(wav, lengths, 16000)
    other = MelSpectrogramConfig(16000, 1280, 320, 1280, 80, 0, 8000)
    b, _ = other.extract(wav, lengths, 16000)
    again, _ = SPEC.extract(wav, lengths, 16000)
    assert a.shape == b.shape
    assert not torch.allclose(a, b)
    torch.testing.assert_close(a, again)


def test_training_uses_resolved_mel_targets_and_accepts_integer_sample_rate():
    module = decoder()
    wav = tone(16000)[None]
    module.compute_loss(torch.zeros(1, 50, 4), torch.tensor([50]), wav, torch.tensor([16000]), 16000, torch.zeros(1, 2))
    expected, _ = SPEC.extract(wav, torch.tensor([16000]), 16000)
    torch.testing.assert_close(module.flow.batch["speech_feat"], expected.transpose(1, 2))
    module.flow.input_frame_rate = 25
    with pytest.raises(ValueError, match="frame rate"):
        module.compute_loss(
            torch.zeros(1, 50, 4), torch.tensor([50]), wav, torch.tensor([16000]), 16000, torch.zeros(1, 2)
        )


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_training_preserves_content_and_mel_time_axes(ratio):
    module = decoder()
    module.flow.input_frame_rate = 50 / ratio
    module.flow.token_mel_ratio = ratio
    # The shorter waveform has one extra mel frame beyond a complete token.
    wav_lengths = torch.tensor([16000, 10560])
    content_lengths = torch.tensor([50 // ratio, 33 // ratio])
    original_lengths = content_lengths.clone()
    features = torch.randn(2, int(content_lengths.max()) + 3, 4)
    original_features = features.clone()
    wav = torch.stack([tone(16000), torch.nn.functional.pad(tone(10560), (0, 5440))])
    module.compute_loss(features, content_lengths, wav, wav_lengths, 16000, torch.zeros(2, 2))
    batch = module.flow.batch
    torch.testing.assert_close(batch["speech_token_len"], content_lengths)
    torch.testing.assert_close(batch["speech_feat_len"], content_lengths * ratio)
    assert batch["speech_token"].shape[1] == int(content_lengths.max())
    assert batch["speech_feat"].shape[1] == int(content_lengths.max()) * ratio
    torch.testing.assert_close(content_lengths, original_lengths)
    torch.testing.assert_close(features, original_features)


def test_training_rejects_large_duration_mismatch():
    module = decoder()
    module.flow.input_frame_rate = 25
    module.flow.token_mel_ratio = 2
    with pytest.raises(ValueError, match="durations differ"):
        module.compute_loss(
            torch.zeros(1, 20, 4),
            torch.tensor([20]),
            tone(16000)[None],
            torch.tensor([16000]),
            16000,
            torch.zeros(1, 2),
        )


def test_training_trims_padded_features_to_valid_mask_width(monkeypatch):
    module = decoder()
    monkeypatch.setattr(module, "_compute_mels", lambda *args: (torch.zeros(2, 80, 833), torch.tensor([833, 700])))
    lengths = torch.tensor([832, 700])
    module.compute_loss(
        torch.zeros(2, 833, 4), lengths, torch.zeros(2, 1), torch.ones(2, dtype=torch.long), 16000, torch.zeros(2, 2)
    )
    batch = module.flow.batch
    assert batch["speech_token"].shape[1] == int(batch["speech_token_len"].max()) == 832
    assert batch["speech_feat"].shape[1] == int(batch["speech_feat_len"].max()) == 832
    assert lengths.tolist() == [832, 700]


def test_training_projection_masks_padding_and_preserves_backward(monkeypatch):
    module = CosyVoiceSpectrogramGenerator(Flow(), 6, device="cpu", mel_config=SPEC)
    module.flow.input_frame_rate = 25
    module.flow.token_mel_ratio = 2

    def forward(batch, device):
        module.flow.batch = batch
        return {"loss": batch["speech_token"].square().mean()}

    monkeypatch.setattr(module.flow, "forward", forward)
    values = torch.randn(2, 30, 6, requires_grad=True)
    lengths = torch.tensor([25, 16])
    waveform = torch.stack([tone(16000), torch.nn.functional.pad(tone(10560), (0, 5440))])
    output = module.compute_loss(values, lengths, waveform, torch.tensor([16000, 10560]), 16000, torch.ones(2, 2))
    output.loss.backward()
    batch = module.flow.batch
    assert batch["speech_token"].shape == (2, 25, 4)
    assert torch.count_nonzero(batch["speech_token"][1, 16:]) == 0
    assert torch.count_nonzero(values.grad[1, 16:]) == 0
    assert module.input_projection.weight.grad.norm() > 0
    assert lengths.tolist() == [25, 16]


@pytest.mark.parametrize("lengths", [torch.tensor([0]), torch.tensor([51]), torch.tensor([50.0]), torch.tensor([[50]])])
def test_training_rejects_invalid_content_lengths(lengths):
    with pytest.raises(ValueError, match="Content"):
        decoder().compute_loss(
            torch.ones(1, 50, 4), lengths, tone(16000)[None], torch.tensor([16000]), 16000, torch.ones(1, 2)
        )


def test_mel_contract_survives_weights_only_restore_without_metadata_download(tmp_path):
    original = decoder()
    original._set_mel_metadata({"mel_config": asdict(SPEC), "revision": "pinned-commit"})
    path = tmp_path / "weights.pt"
    torch.save(original.state_dict(), path)
    restored = decoder(None)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.mel_config == SPEC
    assert restored._metadata()["revision"] == "pinned-commit"
    with pytest.raises(ValueError, match="differs"):
        decoder(replace(SPEC, fmax=8000)).load_state_dict(original.state_dict())
    legacy = {key: value for key, value in original.state_dict().items() if key != "_mel_metadata"}
    with pytest.raises(ValueError, match="legacy"):
        restored.load_state_dict(legacy)


def test_companion_yaml_is_read_as_data_with_references(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("""sample_rate: 24000
unused: !apply:os.system ["this must never execute"]
hift: !new:cosyvoice.hifigan.generator.CausalHiFTGenerator
  sampling_rate: !ref <sample_rate>
  in_channels: 80
  upsample_rates: [8, 5, 3]
  istft_params: {n_fft: 16, hop_len: 4}
  f0_predictor: !new:cosyvoice.hifigan.f0_predictor.CausalConvRNNF0Predictor
    num_class: 1
    in_channels: 80
    cond_channels: 512
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
  n_fft: 1920
  num_mels: 80
  sampling_rate: !ref <sample_rate>
  hop_size: 480
  win_size: 1920
  fmin: 0
  fmax: null
  center: false
""")
    config, f0, spec = read_cosyvoice_config(path)
    assert config["sampling_rate"] == 24000
    assert config["upsample_rates"] == [8, 5, 3]
    assert f0["in_channels"] == 80
    assert spec == SPEC
    original = path.read_text()
    path.write_text(original.replace("!name:matcha.utils.audio.mel_spectrogram", "!name:other.extractor"))
    with pytest.raises(ValueError, match="Unsupported companion component"):
        read_cosyvoice_config(path)
    path.write_text(original.replace("fmax: null", "fmax: !apply:os.system ['false']"))
    with pytest.raises(ValueError, match="tag"):
        read_cosyvoice_config(path)


def test_extraction_matches_checkpoint_matcha_normalization():
    pytest.importorskip("librosa")
    from quick_convert.external.matcha.utils import audio

    wav = torchaudio.functional.resample(tone(16000)[None], 16000, 24000)
    audio.mel_basis.clear()
    audio.hann_window.clear()
    try:
        expected = audio.mel_spectrogram(wav, 1920, 80, 24000, 480, 1920, 0, None)
        actual, _ = SPEC.extract(wav, torch.tensor([24000]), 24000)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    finally:
        audio.mel_basis.clear()
        audio.hann_window.clear()


def test_pretrained_adapter_uses_companion_architecture_and_pins_weights(tmp_path, monkeypatch):
    adapter = pytest.importorskip("quick_convert.components.decoders.hift_generator")
    from quick_convert.components.decoders import cosyvoice_config

    path = tmp_path / "snapshots" / "exact-commit" / "configs" / "model.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("unused")
    model_config = {"sampling_rate": 24000, "in_channels": 80, "base_channels": 64}
    f0_config = {"num_class": 1, "in_channels": 80, "cond_channels": 32}
    monkeypatch.setattr(cosyvoice_config, "read_cosyvoice_config", lambda _: (model_config, f0_config, SPEC))
    weights = tmp_path / "hift.pt"
    torch.save(nn.Linear(2, 2).state_dict(), weights)
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(path if kwargs["filename"] == "configs/model.yaml" else weights)

    monkeypatch.setattr(adapter, "hf_hub_download", download)

    class TinyHiFT(adapter.CosyVoiceHiFTDecoder):
        def __init__(self, config, f0_config, mel_config, device):
            nn.Module.__init__(self)
            self.config = config
            self.f0_config = f0_config
            self.mel_config = mel_config
            self.generator = nn.Linear(2, 2)

    model = TinyHiFT.from_pretrained("example/model", config_filename="configs/model.yaml", revision="branch")
    assert model.config.base_channels == 64
    assert model.f0_config.cond_channels == 32
    assert model.mel_config == SPEC
    assert downloads[0]["revision"] == "branch"
    assert downloads[1]["revision"] == "exact-commit"
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_saved_vocoder_architecture_restores_lazily_without_downloads(monkeypatch):
    class TinyVocoder(nn.Module):
        sample_rate = 24000
        samples_per_frame = 480
        mel_config = SPEC

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)

        def checkpoint_config(self):
            return {"config": {"sampling_rate": 24000}, "f0_config": {}, "mel_config": asdict(SPEC)}

    original = decoder()
    original.vocoder = TinyVocoder()
    state = original.state_dict()
    restored = decoder(None)
    initialized = []

    def initialize(metadata):
        initialized.append(metadata)
        restored.vocoder = TinyVocoder()

    monkeypatch.setattr(restored, "initialize_checkpoint_vocoder", initialize)
    restored.load_state_dict(state)
    assert initialized[0]["config"]["sampling_rate"] == 24000
    assert restored.mel_config == SPEC
    torch.testing.assert_close(original.vocoder.layer.weight, restored.vocoder.layer.weight)


def test_hydra_vocoder_metadata_is_serializable_and_restores_strictly(monkeypatch):
    import json

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    adapter = pytest.importorskip("quick_convert.components.decoders.hift_generator")
    monkeypatch.setattr(adapter, "CausalConvRNNF0Predictor", lambda **kwargs: nn.Linear(2, 2))
    monkeypatch.setattr(adapter, "CausalHiFTGenerator", lambda **kwargs: nn.Linear(2, 2))
    config = OmegaConf.create(
        {
            "_target_": "quick_convert.components.decoders.hift_generator.CosyVoiceHiFTDecoder",
            "config": {
                "_target_": "quick_convert.components.decoders.hift_generator.CosyVoiceHiFTConfig",
                "upsample_rates": [8, 5, 3],
                "istft_params": {"n_fft": 16, "hop_len": 4},
                "resblock_dilation_sizes": [[1, 3, 5]] * 3,
            },
            "device": "cpu",
            "mel_config": {
                "_target_": "quick_convert.components.feature_extractors.mel.MelSpectrogramConfig",
                **asdict(SPEC),
            },
        }
    )
    vocoder = instantiate(config)
    metadata = vocoder.checkpoint_config()
    assert json.loads(json.dumps(metadata)) == metadata
    assert type(metadata["config"]["upsample_rates"]) is list
    assert type(metadata["config"]["istft_params"]) is dict
    assert type(metadata["config"]["resblock_dilation_sizes"][0]) is list
    metadata["config"]["upsample_rates"][0] = 999
    assert vocoder.config.upsample_rates[0] == 8  # Snapshot does not mutate caller-owned configuration.

    original = decoder()
    original.vocoder = vocoder
    state = original.state_dict()  # Same metadata serialization that failed during inference loading.
    restored = decoder()
    restored.vocoder = instantiate(config)
    restored.load_state_dict(state, strict=True)
    assert restored._metadata()["config"]["upsample_rates"] == [8, 5, 3]
    torch.testing.assert_close(original.vocoder.generator.weight, restored.vocoder.generator.weight)

    from quick_convert.inference.source import InferenceState, restore_system

    # Excluded external state makes the loader inspect freshly instantiated
    # state before loading checkpoint weights, as in the reported traceback.
    restored_system = restore_system(
        InferenceState(
            system_config={
                "_target_": "quick_convert.components.decoders.cosyvoice.CosyVoiceSpectrogramGenerator",
                "flow": {"_target_": "test_mel_contract.Flow"},
                "feature_dim": 4,
                "device": "cpu",
                "vocoder": OmegaConf.to_container(config, resolve=True),
                "mel_config": OmegaConf.to_container(config.mel_config, resolve=True),
            },
            state_dict=state,
            excluded_state_prefixes=("flow.",),
        )
    )
    torch.testing.assert_close(original.vocoder.generator.weight, restored_system.vocoder.generator.weight)
    assert not restored_system.training
