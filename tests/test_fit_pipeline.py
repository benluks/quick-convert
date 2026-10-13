import pytest
import torch

from quick_convert.components.fitting import load_fitted_state
from quick_convert.components.prosody import ProsodyTargets, ProsodyTransform
from quick_convert.data import ManifestDataset
from quick_convert.data.indexing import Indexer
from quick_convert.pipelines.fit import FitPipeline


def test_generic_fit_pipeline_fits_and_restores_speaker_index(tmp_path):
    manifest = tmp_path / "train.csv"
    manifest.write_text("utt_id,path,spkid\na,missing-a.wav,b\nb,missing-b.wav,a\n")
    dataset = ManifestDataset(manifest, resources={"spkid": {"column": "spkid", "kind": "text"}}, load=["spkid"])
    template = "{sample.resources.spkid.value}"
    indexer = Indexer(template)
    out = tmp_path / "index" / "state.pt"
    FitPipeline(dataset, indexer, out).run()
    restored = load_fitted_state(Indexer(template), out)
    assert restored.transform(dataset[0]) == restored.encode("b") == 1
    assert restored.encode_tensor(["a", "b"]).tolist() == [0, 1]
    assert restored.decode_many([0, 1]) == ["a", "b"]
    assert indexer.value_to_idx == restored.value_to_idx
    with pytest.raises(ValueError, match="configured index"):
        load_fitted_state(Indexer("different"), out)


def test_prosody_record_contract_and_existing_state_format(tmp_path):
    values = torch.tensor([[0.0, 100.0, 1.0, 0.1, 0.9], [0.01, 200.0, 1.0, 0.2, 0.9]])
    transform = ProsodyTransform().fit([(values, "a")])
    expected = transform.transform(values, "a")
    assert torch.equal(transform.transform((values, "a"))["pitch"], expected["pitch"])
    out = tmp_path / "existing-stats.pt"
    # The unchanged version-1 format remains loadable through the old name.
    torch.save(transform.state, out)
    restored = ProsodyTargets.load(out)
    assert torch.equal(restored.transform((values, "a"))["pitch_bins"], expected["pitch_bins"])
    assert not hasattr(transform, "head_targets")
    with pytest.raises(ValueError, match="configured transform"):
        load_fitted_state(ProsodyTransform(pitch_scale="global"), out)


def test_generic_fit_selection_rejects_unloaded_resources(tmp_path):
    from types import SimpleNamespace

    record = SimpleNamespace(utt_id="a", resources={"prosody": SimpleNamespace(value=None)})
    with pytest.raises(ValueError, match="must be loaded"):
        FitPipeline([record], ProsodyTransform(), tmp_path / "state.pt", resources=["prosody"]).run()
