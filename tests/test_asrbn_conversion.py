from dataclasses import replace

import pytest
import torch
from test_grouped_conversion import sample
from torch import nn

from quick_convert.data import AudioBatch, BaseDataset
from quick_convert.pipelines.conversion import SpeechConversionPipeline
from quick_convert.systems.conversion import ASRBNConversionSystem, ReferenceSpeech, SpeakerIdentity


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.targets = []

    def convert(self, waveform, *, target):
        self.targets.append(target)
        return waveform[..., : int(target) + 2] * self.weight


def make_system(monkeypatch):
    model = Backend()
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: model)
    return ASRBNConversionSystem(device="cpu")


def test_targets_interleave_without_mutable_default_and_preserve_inputs(monkeypatch):
    system = make_system(monkeypatch)
    first = system.prepare_target(SpeakerIdentity(0))
    second = system.prepare_target(SpeakerIdentity("7"))
    batch = AudioBatch.from_samples([sample("source", "a")])
    before = batch.waveforms.clone()
    for target, length in [(first, 2), (second, 9), (first, 2)]:
        generated = system.convert_batch(batch, target=target)
        assert generated.lengths.tolist() == [min(length, batch.lengths[0].item())]
    assert system.model.targets == ["0", "7", "0"]
    torch.testing.assert_close(before, batch.waveforms)
    assert not hasattr(system, "target")
    with pytest.raises(ValueError, match="incompatible"):
        system.convert_batch(batch, target=replace(first, space="other"))
    with pytest.raises(ValueError, match="identity"):
        system.prepare_target(replace(first, speaker_id="7"))
    with pytest.raises(TypeError, match="SpeakerIdentity"):
        system.prepare_target(ReferenceSpeech(["ref.wav"]))


def test_identity_pipeline_resumes_and_rejects_changed_target(tmp_path, monkeypatch):
    system = make_system(monkeypatch)
    dataset = BaseDataset(rows=[sample("source", "a")])
    SpeechConversionPipeline(system, dataset, SpeakerIdentity(0), tmp_path).run()
    system.model.targets.clear()
    SpeechConversionPipeline(system, dataset, SpeakerIdentity(0), tmp_path).run()
    assert not system.model.targets
    with pytest.raises(ValueError):
        SpeechConversionPipeline(system, dataset, SpeakerIdentity(7), tmp_path).run()


@pytest.mark.parametrize("run", ["convert_asrbn_clac", "convert_asrbn_stutter", "convert_knnvc_clac"])
def test_conversion_presets_wire_explicit_targets(run, tmp_path, monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from quick_convert.systems.conversion import KNNVCConversionSystem
    from quick_convert.utils.resolvers import register_config_resolvers

    register_config_resolvers()
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: Backend())
    root = tmp_path / "1069" / "chapter"
    root.mkdir(parents=True)
    (root / "ref.flac").touch()
    with initialize_config_dir(config_dir=str(Path("quick_convert/configs").resolve()), version_base=None):
        cfg = compose(config_name=f"run/{run}", overrides=["device=cpu", "target_speaker=1069"])
    system = instantiate(cfg.system)
    if "knnvc" in run:
        cfg.target_root = str(tmp_path)
        assert isinstance(system, KNNVCConversionSystem)
        assert len(instantiate(cfg.target).paths) == 1
    else:
        assert isinstance(system, ASRBNConversionSystem)
        assert instantiate(cfg.target) == SpeakerIdentity("1069")
    assert cfg.pipeline._target_ == "quick_convert.pipelines.conversion.SpeechConversionPipeline"
    assert OmegaConf.to_container(cfg.pipeline.system, resolve=True) == OmegaConf.to_container(cfg.system, resolve=True)
