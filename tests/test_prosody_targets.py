from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from quick_convert.components.prosody import ProsodyTransform


def measurements(pitch):
    pitch = torch.tensor(pitch, dtype=torch.float64)
    return torch.stack((torch.arange(len(pitch)), pitch, pitch > 0, torch.ones_like(pitch), pitch > 0), dim=-1)


def test_scaling_masks_roundtrip_and_ownership(tmp_path):
    a = measurements([100, 200, 0])
    b = measurements([100, 400, 0])
    original = a.clone()
    speaker = ProsodyTransform().fit([(a, "a"), (b, "b")])
    global_scale = ProsodyTransform(pitch_scale="global").fit([(a, "a"), (b, "b")])
    assert torch.allclose(speaker.transform(a, "a")["pitch"][:2], torch.tensor([-1.0, 1.0], dtype=a.dtype))
    assert not torch.allclose(global_scale.transform(a, "a")["pitch"], speaker.transform(a, "a")["pitch"])
    assert not speaker.transform(a, "a")["voiced"][-1]
    assert torch.equal(a, original)
    path = tmp_path / "stats.pt"
    speaker.save(path)
    assert torch.equal(ProsodyTransform.load(path).transform(a, "a")["pitch"], speaker.transform(a, "a")["pitch"])
    with pytest.raises(ValueError, match="No voiced"):
        speaker.transform(a, "unknown")


def test_unvoiced_and_tied_bins():
    targets = ProsodyTransform(bins="equal_width").fit([(measurements([100, 100]), "a")])
    result = targets.transform(measurements([0, 0]), "unknown")
    assert not result["voiced"].any()
    assert torch.isfinite(result["energy"]).all()
    assert len(targets.state["pitch_edges"]) == 1
    with pytest.raises(ValueError, match="voiced frames"):
        ProsodyTransform().fit([(measurements([0, 0]), "a")])


@pytest.mark.parametrize(
    "run", ["precompute_prosody_opensmile_librispeech", "fit_prosody_librispeech", "fit_speaker_index_librispeech"]
)
def test_configs_compose(run):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs"), version_base="1.3"
    ):
        cfg = compose(config_name="run/" + run)
    assert cfg.pipeline._target_


def test_fit_pipeline_uses_named_resources_without_audio(tmp_path):
    from quick_convert.data import ManifestDataset
    from quick_convert.data.resources import PathResourceProvider, collate_resources
    from quick_convert.pipelines.fit import FitPipeline

    torch.save(measurements([100, 200, 0]), tmp_path / "sample.pt")
    manifest = tmp_path / "train.csv"
    manifest.write_text("utt_id,path,split,spkid\nsample,missing.wav,train,a\n")
    dataset = ManifestDataset(
        manifest,
        resources={"spkid": {"column": "spkid", "kind": "text"}},
        resource_providers=[
            PathResourceProvider(
                name="prosody", kind="torch_tensor", path_template=str(tmp_path / "{sample.utt_id}.pt")
            )
        ],
        load=["prosody", "spkid"],
    )
    out = tmp_path / "stats.pt"
    FitPipeline(dataset, ProsodyTransform(), out, resources=["prosody", "spkid"]).run()
    assert "a" in ProsodyTransform.load(out).state["speakers"]
    resources = collate_resources([dataset[0]])
    assert resources["prosody"].lengths.tolist() == [3]
