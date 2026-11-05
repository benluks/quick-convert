"""Separate-speaker-reference data wiring and SSLR training without downloads."""

from pathlib import Path

import pytest
import soundfile as sf
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from test_tts_dit import decoder
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.systems.reconstruction import SSLReconstructionSystem
from quick_convert.utils.resolvers import register_config_resolvers


class ContentProbe(nn.Module):
    sample_rate = 16000

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, batch):
        self.calls.append(batch)
        lengths = batch.lengths // 320
        values = [wave[:1].expand(int(n), 4).clone() for wave, n in zip(batch.waveforms, lengths, strict=True)]
        return ResolvedResource(pad_sequence(values, batch_first=True), lengths)


class PrefixProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def encode_prefixes(self, batch, lengths, frame_rate):
        self.calls.append((batch, lengths.clone()))
        values = [waveform[:1].expand(int(n), 2).clone() for waveform, n in zip(batch.waveforms, lengths, strict=True)]
        return ResolvedResource(pad_sequence(values, batch_first=True), lengths)


@pytest.fixture
def paired_datasets(tmp_path):
    register_config_resolvers()
    tts_root = tmp_path / "tts"
    tts_root.mkdir()
    manifests = tmp_path / "speaker_split"
    manifests.mkdir()
    for split in ("train", "val"):
        rows = []
        for index, samples in enumerate((1280, 2560)):
            utt_id = f"{split}-{index}"
            audio_split = "train-clean-100" if split == "train" else "dev-clean"
            relative = Path(audio_split) / "19" / "198" / f"{utt_id}"
            path = tmp_path / "LibriSpeech" / relative.with_suffix(".wav")
            path.parent.mkdir(parents=True, exist_ok=True)
            tts_audio_path = tts_root / relative.with_suffix(".flac")
            tts_audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, torch.full((samples,), 0.2 * (index + 1)).numpy(), 16000, subtype="FLOAT")
            sf.write(tts_audio_path, torch.full(((6 - index * 3) * 320,), 0.7 + index * 0.1).numpy(), 16000)
            rows.append(f"{utt_id},{path},{split},same-speaker\n")
        (manifests / f"{split}_tts_checked.csv").write_text("utt_id,path,split,spkid\n" + "".join(rows))
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(
            config_name="run/train_sslr_tts_dit_campp_prefix",
            overrides=[
                f"tts_data_path={tts_root}",
                f"dataset_path={tmp_path}",
            ],
        )
        return instantiate(cfg.train_dataset), instantiate(cfg.val_dataset)


def module():
    SSLReconstructionTrainingModule = pytest.importorskip(
        "quick_convert.training.lightning.modules.ssl_reconstruction"
    ).SSLReconstructionTrainingModule
    model = decoder()
    model.stream_conditioning.prefix_fraction = (1, 1)
    probe = PrefixProbe()
    system = SSLReconstructionSystem(model, online_encoders={"speaker": probe, "content": ContentProbe()})
    return SSLReconstructionTrainingModule(
        system=system, optimization=None, reference_seconds=0.1, content_audio_resource="tts_path"
    ), probe


def test_separate_references_reach_training_and_validation_without_changing_targets(paired_datasets, monkeypatch):
    train, val = paired_datasets
    training, probe = module()
    monkeypatch.setattr(training, "log_dict", lambda *a, **kw: None)
    for dataset in (train, val):
        batch = next(iter(dataset.make_dataloader(batch_size=2)))
        assert batch.resources["reference_path"] == [str(batch.paths[1]), str(batch.paths[0])]
        original = batch.waveforms.clone()
        streams = training._conditioning_streams(batch)
        reference, prefix_lengths = probe.calls[-1]
        assert reference.lengths.tolist() == [1600, 1280]
        assert prefix_lengths.tolist() == [4, 3]
        assert streams["speaker"].is_prefix
        torch.testing.assert_close(reference.waveforms[:, 0], batch.waveforms.flip(0)[:, 0])
        assert reference.paths == list(reversed(batch.paths))
        output = training._shared_step(batch, "val")
        output.loss.backward()
        assert torch.isfinite(output.loss)
        encoded = training.system.online_encoders["content"].calls[-1]
        assert [str(path) for path in encoded.paths] == batch.resources["tts_path"]
        assert encoded.lengths.tolist() == [1920, 960]
        assert torch.all(encoded.waveforms[:, 0] > batch.waveforms[:, 0])
        assert not training.system.online_encoders["content"].training
        assert output.lengths.tolist() == [6, 3]
        torch.testing.assert_close(batch.waveforms, original)
        assert all(path.parent == batch.paths[0].parent for path in reference.paths)


def test_reference_validation_generation_uses_other_recording(paired_datasets, monkeypatch):
    training, probe = module()
    batch = next(iter(paired_datasets[1].make_dataloader(batch_size=2)))
    monkeypatch.setattr(training, "log_dict", lambda *a, **kw: None)
    output = training._shared_step(batch, "val")
    captured = {}

    def decode(*args, streams, **kwargs):
        captured.update(streams)
        # Stop after observing the validation generation inputs; no vocoder download.
        raise RuntimeError("observed-reference")

    monkeypatch.setattr(training.system, "decode_features", decode)
    with pytest.raises(RuntimeError, match="observed-reference"):
        training.log_validation_output(batch, output, 0)
    torch.testing.assert_close(captured["speaker"].values[:, 0, 0], batch.waveforms.flip(0)[:, 0])
    assert probe.calls[-1][0].paths == list(reversed(batch.paths))


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_reference_duration(seconds):
    SSLReconstructionTrainingModule = pytest.importorskip(
        "quick_convert.training.lightning.modules.ssl_reconstruction"
    ).SSLReconstructionTrainingModule
    with pytest.raises(ValueError, match="reference_seconds"):
        SSLReconstructionTrainingModule(
            system=SSLReconstructionSystem(decoder()), optimization=None, reference_seconds=seconds
        )


def test_mel_reference_training_and_validation_preserve_selected_reference(paired_datasets, monkeypatch):
    from quick_convert.components.decoders import StreamConditioning

    training, _ = module()
    training.decoder.stream_conditioning = StreamConditioning(
        {"mel": 2}, prefix_fraction=(1, 1), context_mask_fraction=(0, 0), stream_drop_probability=0
    )
    # This fixture's estimator already consumes two conditioning values and availability.
    training.system.online_encoders = nn.ModuleDict({"content": ContentProbe()})
    monkeypatch.setattr(training, "log_dict", lambda *a, **kw: None)
    batch = next(iter(paired_datasets[0].make_dataloader(batch_size=2)))
    reference_audio = torch.stack((batch.waveforms[1, :1600], batch.waveforms[0, :1280].new_zeros(1600)))
    reference_audio[1, :1280] = batch.waveforms[0, :1280]
    reference_mels, _ = training.decoder._compute_mels(reference_audio, torch.tensor([1600, 1280]), 16000)
    streams = training._conditioning_streams(batch)
    assert set(streams) == {"mel"}
    assert streams["mel"].is_prefix
    assert streams["mel"].lengths.tolist() == [4, 3]
    torch.testing.assert_close(streams["mel"].values[0], reference_mels[0, :, :4].T)
    original_compute = training.decoder.compute_loss
    observed = []

    def compute(*args, streams, **kwargs):
        observed.append(streams["mel"])
        return original_compute(*args, streams=streams, **kwargs)

    monkeypatch.setattr(training.decoder, "compute_loss", compute)
    output = training._shared_step(batch, "val")
    output.loss.backward()
    assert torch.isfinite(output.loss)
    torch.testing.assert_close(observed[0].values, streams["mel"].values)
    generated = training.system.generate_features(
        output.features.detach(),
        output.lengths,
        streams=streams,
        duration_ratio=torch.tensor([4 / 6, 8 / 3]),
        run_vocoder=False,
    ).generation
    assert generated.mel_lengths.tolist() == [4, 8]
    assert torch.isfinite(generated.mel).all()

    def decode(*args, streams, **kwargs):
        torch.testing.assert_close(streams["mel"].values, observed[0].values)
        raise RuntimeError("observed-mel-reference")

    monkeypatch.setattr(training.system, "decode_features", decode)
    with pytest.raises(RuntimeError, match="observed-mel-reference"):
        training.log_validation_output(batch, output, 0)


def test_mel_preset_replaces_speaker_stream_and_encoder():
    register_config_resolvers()
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(
            config_name="run/train_sslr_tts_dit_mel_prefix",
            overrides=["tts_data_path=/tmp/tts", "dataset_path=/tmp/original"],
        )
        conditioning = instantiate(cfg.system.decoder.stream_conditioning)
        assert conditioning.dimensions == {"mel": 80}
        assert cfg.system.decoder.flow.decoder.estimator.cond_dim == 81
        assert cfg.system.decoder.flow.decoder.estimator.spk_dim is None
        assert cfg.system.online_encoders.content.layer == -1
        assert cfg.system.online_encoders.speaker is None
        assert cfg.system.online_encoders.frame_speaker is None
        assert cfg.trainer.module.reference_seconds == 3.0


@pytest.mark.parametrize("failure", ["missing", "same-file", "nonexistent"])
def test_invalid_tts_pair_never_encodes_original(paired_datasets, failure):
    training, _ = module()
    batch = next(iter(paired_datasets[0].make_dataloader(batch_size=2)))
    if failure == "missing":
        batch.resources.pop("tts_path")
    elif failure == "same-file":
        batch.resources["tts_path"] = [str(path) for path in batch.paths]
    else:
        batch.resources["tts_path"] = ["/nonexistent/tts.flac"] * 2
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        training._content_resource(batch)
    assert training.system.online_encoders["content"].calls == []


def test_dac_tts_mel_preset_has_no_w2vbert_fields_or_dependencies():
    from quick_convert.dependencies import dependency_plan

    register_config_resolvers()
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(
            config_name="run/train_sslr_tts_dac_dit_mel_prefix",
            overrides=["tts_data_path=/tmp/tts", "dataset_path=/tmp/original", "device=cpu"],
        )
        content = cfg.system.online_encoders.content
        assert content._target_.endswith("DACContentEncoder")
        assert set(content) == {"_target_", "sample_rate", "device"}
        assert cfg.system.feature_transform.num_layers == 1
        assert cfg.system.decoder.feature_dim == cfg.system.decoder.content_encoder.input_size == 1024
        assert cfg.system.decoder.flow.input_frame_rate == 50
        assert cfg.system.decoder.flow.token_mel_ratio == 1
        assert instantiate(cfg.system.decoder.stream_conditioning).dimensions == {"mel": 80}
        assert cfg.trainer.module.content_audio_resource == "tts_path"
        plan = dependency_plan(cfg)
        assert {"dac", "cosyvoice", "training"} <= set(plan.extras)
        assert "transformers" not in plan.extras
        assert "campplus-frames" not in plan.extras


def test_dac_adapter_handles_raw_tts_flac_and_mel_prefix_training(paired_datasets, monkeypatch):
    from test_dac_online_sslr import encoder

    from quick_convert.components.decoders import StreamConditioning
    from quick_convert.components.layers.layer_fusion import LayerWeightedSum

    training, _ = module()
    dac = encoder()
    dac.hop_length = 320
    dac.dac_encoder = nn.Conv1d(1, 4, 320, stride=320).eval().requires_grad_(False)
    training.system.online_encoders = nn.ModuleDict({"content": dac})
    training.system.feature_transform = LayerWeightedSum(1)
    training.decoder.stream_conditioning = StreamConditioning(
        {"mel": 2}, prefix_fraction=(1, 1), context_mask_fraction=(0, 0), stream_drop_probability=0
    )
    monkeypatch.setattr(training, "log_dict", lambda *a, **kw: None)
    batch = next(iter(paired_datasets[0].make_dataloader(batch_size=2)))
    content = training._content_resource(batch)
    assert content.values.shape == (2, 6, 1, 4)
    assert content.lengths.tolist() == [6, 3]
    output = training._shared_step(batch, "val")
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert all(parameter.grad is None for parameter in dac.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in training.decoder.content_encoder.parameters()
    )
    generated = training.system.generate_features(
        content.values,
        content.lengths,
        streams=training._conditioning_streams(batch),
        duration_ratio=torch.tensor([4 / 6, 8 / 3]),
        run_vocoder=False,
    ).generation
    assert generated.mel_lengths.tolist() == [4, 8]
    assert torch.isfinite(generated.mel).all()


@pytest.mark.parametrize(
    "run", ["train_sslr_tts_dit_campp_prefix", "train_sslr_tts_dit_mel_prefix", "train_sslr_tts_dac_dit_mel_prefix"]
)
def test_tts_runs_use_checked_manifests_for_data_and_references(run):
    register_config_resolvers()
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(config_name=f"run/{run}", overrides=["tts_data_path=/tmp/tts", "dataset_path=/tmp/original"])
        for split, dataset in [("train", cfg.train_dataset), ("val", cfg.val_dataset)]:
            expected = f"/tmp/original/speaker_split/{split}_tts_checked.csv"
            assert dataset.manifest_path == expected
            assert dataset.resource_providers[1].manifest_path == expected
