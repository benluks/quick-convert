import csv

import pytest

from quick_convert.data import MetadataSample
from quick_convert.pipelines.build_manifest import BuildManifestPipeline


def test_build_manifest_writes_configured_columns(tmp_path):
    audio_path = tmp_path / "speaker" / "utterance.flac"
    sample = MetadataSample(utt_id="custom-id", path=audio_path, split="train")
    manifest_path = tmp_path / "manifests" / "data.csv"

    BuildManifestPipeline(
        dataset=[sample],
        out_path=manifest_path,
        columns={
            "id": "{sample.utt_id}",
            "audio": "{path}",
            "speaker": "{path.parent.name}",
        },
    ).run()

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        assert list(csv.DictReader(manifest_file)) == [
            {"id": "custom-id", "audio": str(audio_path), "speaker": "speaker"}
        ]


def test_build_manifest_requires_explicit_overwrite(tmp_path):
    manifest_path = tmp_path / "data.csv"
    manifest_path.write_text("keep me", encoding="utf-8")
    pipeline = BuildManifestPipeline(dataset=[], out_path=manifest_path, columns={"id": "{sample.utt_id}"})

    with pytest.raises(FileExistsError, match="overwrite=true"):
        pipeline.run()

    assert manifest_path.read_text(encoding="utf-8") == "keep me"
