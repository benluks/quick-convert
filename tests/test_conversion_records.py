from dataclasses import replace
from pathlib import Path

import pytest
import torchaudio
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_grouped_conversion import System, sample, targets

from quick_convert.data import BaseDataset
from quick_convert.pipelines.conversion import SpeechConversionPipeline
from quick_convert.pipelines.conversion_records import conversion_output_dir, read_journal
from quick_convert.systems.conversion.targets import ReferenceSpeech
from quick_convert.utils.resolvers import register_config_resolvers


def corpus(tmp_path):
    root = tmp_path / "LibriSpeech"
    sources = []
    for index in range(3):
        item = sample(f"utt-{index}", "source", split="test-clean")
        path = root / "test-clean" / "source" / item.path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(path), item.waveform.unsqueeze(0), item.sample_rate)
        sources.append(replace(item, path=path))
    reference = root / "train-clean-100" / "target.wav"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"reference")
    return (
        sources,
        ReferenceSpeech([reference]),
        {
            "system": "speech-conversion",
            "dataset": "librispeech",
            "source_root": str(root),
            "speaker_resource": "identity",
            "target_speaker": "6081",
        },
    )


def load(path):
    return yaml.safe_load(path.read_text())


def test_changed_reference_prevents_resume(tmp_path):
    sources, target, metadata = corpus(tmp_path)
    output = tmp_path / "converted"
    SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, output, metadata=metadata).run()
    original = (output / "run.yaml").read_bytes()
    target.paths[0].write_bytes(b"changed reference recording")
    with pytest.raises(ValueError, match="target assignment differs"):
        SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, output, metadata=metadata).run()
    assert (output / "run.yaml").read_bytes() == original


def test_changed_guidance_prevents_resume_without_saved_config(tmp_path):
    from quick_convert.pipelines.conversion_records import ConversionRecords

    output = tmp_path / "output"
    with ConversionRecords(output, {"inference_cfg_rate": 0.0}, {"mode": "fixed"}):
        pass
    with pytest.raises(ValueError, match="target assignment differs"):
        with ConversionRecords(output, {"inference_cfg_rate": 1.0}, {"mode": "fixed"}):
            pass


def test_fixed_conversion_records_paths_identity_and_rebuild(tmp_path):
    sources, target, metadata = corpus(tmp_path)
    output = tmp_path / "converted"
    pipeline = SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, output, metadata=metadata)
    pipeline.write_config("system: {label: test-model}\npipeline: {batch_size: 1}\n")
    pipeline.run()
    manifest = load(output / "speechscope.yaml")
    assert len(manifest["utterances"]) == len(manifest["variants"]) == 3
    assert manifest["roots"]["librispeech"] == [metadata["source_root"]]
    assert manifest["utterances"][0]["audio"] == {"root": "librispeech", "path": "test-clean/source/utt-0.wav"}
    assert manifest["utterances"][0]["speaker_id"] == "source"
    variant = manifest["variants"][0]
    assert variant["audio"] == {"path": "test-clean/utt-0.wav"}
    assert variant["target_speaker"] == "6081"
    assert variant["metadata"]["num_samples"] == 3
    assert variant["metadata"]["duration"] == 3 / 16000
    assert variant["metadata"]["dataset"] == "librispeech"
    target_id = variant["settings"]["target_id"]
    assert load(output / "targets.yaml")["targets"][target_id]["references"][0]["root"] == "librispeech"
    run = load(output / "run.yaml")
    assert run["status"] == "complete"
    assert {"git_commit", "git_dirty", "package_version", "working_directory"} <= run["execution"].keys()
    assert len(list((output / "configs").glob("*.yaml"))) == 1

    for filename in ("speechscope.yaml", "targets.yaml", "config.yaml", "target_assignments.jsonl"):
        (output / filename).unlink()
    system = System()
    resumed = SpeechConversionPipeline(
        system, BaseDataset(rows=list(reversed(sources))), target, output, metadata=metadata, batch_size=2
    )
    resumed.write_config("system: {label: test-model}\npipeline: {batch_size: 2}\n")
    resumed.run()
    assert not system.prepared and not system.converted
    assert load(output / "speechscope.yaml") == manifest
    assert load(output / "targets.yaml")["targets"][target_id]
    assert len(list((output / "configs").glob("*.yaml"))) == 2
    assert load(output / "run.yaml")["execution"]["skipped"] == 3


def test_partial_failure_and_regeneration_keep_current_snapshots(tmp_path):
    sources, target, metadata = corpus(tmp_path)

    class FailingSystem(System):
        def convert_batch(self, batch, *, target):
            if self.converted:
                raise RuntimeError("inference failed")
            return super().convert_batch(batch, target=target)

    output = tmp_path / "converted"
    metadata["system"] = "speech-conversion"  # A test subclass does not change the declared model.
    with pytest.raises(RuntimeError, match="inference failed"):
        SpeechConversionPipeline(FailingSystem(), BaseDataset(rows=sources), target, output, metadata=metadata).run()
    assert load(output / "run.yaml")["status"] == "failed"
    assert len(load(output / "speechscope.yaml")["variants"]) == 1
    first_id = load(output / "speechscope.yaml")["variants"][0]["id"]
    (output / "test-clean/utt-0.wav").unlink()
    SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, output, metadata=metadata).run()
    manifest = load(output / "speechscope.yaml")
    assert len(manifest["variants"]) == 3
    assert manifest["variants"][0]["id"] == first_id
    assert load(output / "run.yaml")["status"] == "complete"
    assert len(read_journal(output / "conversions.jsonl")) == 4
    assert len((output / "target_assignments.jsonl").read_text().splitlines()) == 3


def test_legacy_unrecorded_outputs_are_not_silently_labelled(tmp_path):
    sources, target, metadata = corpus(tmp_path)
    output = tmp_path / "converted"
    destination = output / "test-clean/utt-0.wav"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"unknown generation")
    with pytest.raises(ValueError, match="record is missing"):
        SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, output, metadata=metadata).run()
    assert not load(output / "speechscope.yaml")["variants"]


def test_interrupted_publish_and_changed_audio_are_regenerated(tmp_path, monkeypatch):
    sources, target, metadata = corpus(tmp_path)
    output = tmp_path / "converted"
    replace_file = Path.replace

    def fail_publish(path, destination):
        if path.suffix == ".wav":
            raise OSError("interrupted publish")
        return replace_file(path, destination)

    monkeypatch.setattr(Path, "replace", fail_publish)
    with pytest.raises(OSError, match="interrupted publish"):
        SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, output, metadata=metadata).run()
    assert len(read_journal(output / "conversions.jsonl")) == 1
    assert not load(output / "speechscope.yaml")["variants"]
    assert not list(output.rglob("*.wav"))

    monkeypatch.setattr(Path, "replace", replace_file)
    SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, output, metadata=metadata).run()
    (output / "test-clean/utt-0.wav").write_bytes(b"incomplete audio")
    system = System()
    SpeechConversionPipeline(system, BaseDataset(rows=sources), target, output, metadata=metadata).run()
    assert len(system.converted) == 1
    assert len(load(output / "speechscope.yaml")["variants"]) == 3
    waveform, rate = torchaudio.load(str(output / "test-clean/utt-0.wav"))
    assert waveform.shape[-1] == 3 and rate == 16000


def test_changed_checkpoint_cannot_relabel_existing_outputs_or_config(tmp_path):
    sources, target, metadata = corpus(tmp_path)
    checkpoint = tmp_path / "run/checkpoints/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"first model")
    metadata.update(source=str(checkpoint.parent.parent))
    output = tmp_path / "converted"
    SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, output, metadata=metadata).run()
    original_run = (output / "run.yaml").read_text()
    original_config = (output / "config.yaml").read_text()
    checkpoint.write_bytes(b"replacement model weights")
    with pytest.raises(ValueError, match="model changed"):
        SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, output, metadata=metadata).run()
    assert (output / "run.yaml").read_text() == original_run
    assert (output / "config.yaml").read_text() == original_config


@pytest.mark.parametrize("direct_checkpoint", [False, True])
def test_default_layout_uses_training_experiment_and_retains_run_id(tmp_path, direct_checkpoint):
    run = tmp_path / "tezj8b44"
    checkpoint = run / "checkpoints/best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    (run / "config.yaml").write_text("exp_name: speech-conversion_w2vb_cmdiff_prosody_librispeech\n")
    metadata = {"system": "speech-conversion", "source": str(checkpoint if direct_checkpoint else run)}
    if not direct_checkpoint:
        metadata["checkpoint"] = "checkpoints/best.ckpt"
    output = Path(conversion_output_dir(tmp_path, metadata, {}))
    assert output.parts[-3] == "speech-conversion_w2vb_cmdiff_prosody_librispeech-tezj8b44-best"
    # The run ID survives even when experiment names are long or need sanitizing.
    (run / "config.yaml").write_text("exp_name: " + "long experiment " * 15 + "\n")
    assert Path(conversion_output_dir(tmp_path, metadata, {})).parts[-3].endswith("-tezj8b44-best")


@pytest.mark.parametrize("config", ["{}", "exp_name: '${hydra:runtime.choices.system}'", "exp_name: null"])
def test_default_layout_falls_back_without_resolved_experiment(tmp_path, config):
    run = tmp_path / "tezj8b44"
    run.mkdir()
    (run / "config.yaml").write_text(config)
    output = Path(conversion_output_dir(tmp_path, {"source": str(run)}, {}))
    assert output.parts[-3] == "tezj8b44"


def test_default_layout_does_not_read_artifact_weights(tmp_path):
    artifact = tmp_path / "exported-model"
    artifact.mkdir()
    (artifact / "artifact.yaml").write_text("weights: weights.pt\n")
    (artifact / "weights.pt").write_bytes(b"\x80binary model")
    output = Path(conversion_output_dir(tmp_path, {"source": str(artifact)}, {}))
    assert output.parts[-3] == "exported-model"


def test_default_layout_and_model_changes(tmp_path):
    run = tmp_path / "tezj8b44"
    checkpoint = run / "checkpoints/best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    metadata = {
        "system": "speech-conversion",
        "dataset": "librispeech",
        "source": str(run),
        "checkpoint": "checkpoints/best.ckpt",
        "target_speaker": "6081",
    }
    target = {"mode": "fixed", "references": ["ref.wav"]}
    first = conversion_output_dir(tmp_path / "outputs", metadata, target)
    assert Path(first).parts[-5:-1] == ("anonymized", "speech-conversion", "tezj8b44-best", "librispeech")
    assert Path(first).name.startswith("fixed-6081-")
    assert Path(conversion_output_dir(tmp_path, metadata, {"mode": "resynthesize"})).name.startswith("resynthesize-")
    assert conversion_output_dir(tmp_path / "outputs", metadata, target) == first
    assert conversion_output_dir(tmp_path / "outputs", {**metadata, "target_speaker": "7"}, target) != first
    checkpoint.write_bytes(b"different model")
    assert conversion_output_dir(tmp_path / "outputs", metadata, target) != first
    random = {"mode": "random", "seed": 42}
    assert "random-seed-42-" in conversion_output_dir(tmp_path, metadata, random)
    assert conversion_output_dir(tmp_path, metadata, random) != conversion_output_dir(
        tmp_path, metadata, {**random, "seed": 43}
    )


def test_independent_runs_have_distinct_variants_and_stable_source_ids(tmp_path):
    sources, target, metadata = corpus(tmp_path)
    variants = []
    for name in ("a", "b"):
        SpeechConversionPipeline(System(), BaseDataset(rows=sources), target, tmp_path / name, metadata=metadata).run()
        variants.append(load(tmp_path / name / "speechscope.yaml")["variants"][0])
    assert variants[0]["utterance_id"] == variants[1]["utterance_id"]
    assert variants[0]["id"] != variants[1]["id"]


def test_recover_torn_journal_without_accepting_committed_corruption(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"ok":1}\n{"broken"')
    assert read_journal(path) == [{"ok": 1}]
    assert path.read_bytes() == b'{"ok":1}\n'
    path.write_bytes(b'{"ok":1}')
    assert read_journal(path) == [{"ok": 1}]
    assert path.read_bytes().endswith(b"\n")
    path.write_bytes(b'{"broken"\n{"ok":1}\n')
    with pytest.raises(ValueError, match="journal row 1"):
        read_journal(path)


def test_random_groups_are_catalogued_as_targets(tmp_path):
    sources, _, metadata = corpus(tmp_path)
    metadata.pop("target_speaker")
    policy = targets(mode="random", seed=42)
    output = tmp_path / "converted"
    SpeechConversionPipeline(System(), BaseDataset(rows=sources), policy, output, metadata=metadata).run()
    manifest = load(output / "speechscope.yaml")
    catalogue = load(output / "targets.yaml")["targets"]
    for variant in manifest["variants"]:
        assert variant["target_speaker"] in {"a", "b", "c"}
        assert variant["seed"] == 42
        assert variant["settings"]["target_id"] in catalogue
        assert "references" not in variant["settings"]["target_selection"]


@pytest.mark.parametrize(
    "run_name", ["convert_reference_librispeech", "convert_reference_librispeech_random", "convert_reference_manifest"]
)
def test_canonical_default_paths_resolve_without_inference(run_name, tmp_path):
    register_config_resolvers()
    config_dir = Path(__file__).parents[1] / "quick_convert/configs"
    overrides = ["source=/runs/tezj8b44", "checkpoint=checkpoints/best.ckpt", f"out_root={tmp_path}"]
    if run_name == "convert_reference_librispeech":
        overrides.append("target_id=6081")
    if run_name == "convert_reference_manifest":
        overrides.extend(["manifest_path=/data/sources.csv", "target_paths=[/data/ref.wav]"])
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=f"run/{run_name}", overrides=overrides)
    rendered = OmegaConf.to_container(cfg, resolve=True)
    assert Path(rendered["pipeline"]["out_dir"]).is_relative_to(tmp_path / "anonymized/speech-conversion/tezj8b44-best")
    assert rendered["pipeline"]["metadata"]["checkpoint"] == "checkpoints/best.ckpt"


def test_prepared_prompt_records_exact_intervals_and_overlaps(tmp_path):
    from quick_convert.components.reference_prompt import PromptSegment

    sources, request, metadata = corpus(tmp_path)

    class PromptSystem(System):
        def prepare_target(self, target):
            prepared = super().prepare_target(target)
            return replace(
                prepared,
                prompt_segments=(PromptSegment(request.paths[0], 20, 70), PromptSegment(request.paths[0], 90, 150, 10)),
                prompt_sample_rate=1000,
                prompt_crossfade_seconds=0.01,
            )

    SpeechConversionPipeline(
        PromptSystem(), BaseDataset(rows=sources), request, tmp_path / "converted", metadata=metadata
    ).run()
    catalogue = load(tmp_path / "converted/targets.yaml")
    prompt = next(iter(catalogue["targets"].values()))["prompt"]
    assert prompt["seconds"] == 0.1
    assert prompt["num_samples"] == 100
    assert prompt["sample_rate"] == 1000
    assert prompt["crossfade_seconds"] == 0.01
    assert [segment["overlap_samples"] for segment in prompt["segments"]] == [0, 10]
    assert prompt["segments"][0]["audio"]["root"] == "librispeech"


def test_reference_prompt_builder_changes_output_identity():
    from quick_convert.components.reference_prompt import EnergyActivityDetector, SpeechPromptBuilder
    from quick_convert.pipelines.conversion import target_description

    prefix = ReferenceSpeech(["target.wav"], prompt_seconds=30)
    speech = ReferenceSpeech(["target.wav"], prompt_seconds=30, prompt_builder=SpeechPromptBuilder())
    louder = replace(speech, prompt_builder=SpeechPromptBuilder(EnergyActivityDetector(threshold_db=-30)))
    descriptions = [target_description(target) for target in (prefix, speech, louder)]
    paths = {conversion_output_dir("outputs", {}, description) for description in descriptions}
    assert len(paths) == 3
