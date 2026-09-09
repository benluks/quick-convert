import csv
from dataclasses import FrozenInstanceError

import pytest
import torch

from quick_convert.data import AudioSample, ManifestDataset
from quick_convert.data.resources import CSVAnnotationProvider, ResourceCollection, ResourceRef, collate_resources


def test_manifest_distinguishes_values_from_serialized_resources(tmp_path):
    token_path = tmp_path / "tokens.pt"
    torch.save([1, 2, 3], token_path)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=["utt_id", "path", "transcript", "token_ids"])
        writer.writeheader()
        writer.writerow(
            {
                "utt_id": "sample",
                "path": str(tmp_path / "sample.wav"),
                "transcript": "hello",
                "token_ids": str(token_path),
            }
        )

    dataset = ManifestDataset(
        manifest_path,
        resources={
            "transcript": {"column": "transcript", "kind": "text"},
            "token_ids": {"column": "token_ids", "kind": "token_ids"},
        },
        load=["token_ids"],
    )

    sample = dataset[0]
    assert sample.resources["transcript"].value == "hello"
    assert sample.resources["transcript"].path is None
    assert sample.resources["token_ids"].path == token_path
    assert sample.resources["token_ids"].value == [1, 2, 3]


def test_load_all_includes_resources_stored_on_manifest_rows(tmp_path, monkeypatch):
    tensor_path = tmp_path / "features.pt"
    torch.save(torch.ones(2, 3), tensor_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        f"utt_id,path,features\nsample,{tmp_path / 'sample.wav'},{tensor_path}\n",
        encoding="utf-8",
    )
    dataset = ManifestDataset(
        manifest_path,
        resources={"features": {"column": "features", "kind": "torch_tensor"}},
        load="all",
    )
    monkeypatch.setattr(dataset, "load_sample", lambda sample: sample)

    assert torch.equal(dataset[0].resources["features"].value, torch.ones(2, 3))


def test_resource_references_validate_kind_and_location():
    with pytest.raises(ValueError, match="Unsupported resource kind"):
        ResourceRef(name="features", kind="speaker_embedding", value=torch.ones(2))

    with pytest.raises(ValueError, match="must have a path or a value"):
        ResourceRef(name="features", kind="torch_tensor")


def test_resource_max_length_must_match_across_batch():
    samples = [
        AudioSample(
            utt_id=str(index),
            path=None,
            resources=ResourceCollection.from_refs(
                [ResourceRef(name="features", kind="torch_tensor", value=torch.ones(2, 3), max_length=max_length)]
            ),
        )
        for index, max_length in enumerate([4, 5])
    ]

    with pytest.raises(ValueError, match="inconsistent max_length"):
        collate_resources(samples)


def test_resource_max_length_extends_tensor_and_token_batches():
    tensor_refs = [
        ResourceRef(name="features", kind="torch_tensor", value=torch.ones(2, 3), max_length=4),
        ResourceRef(name="features", kind="torch_tensor", value=torch.ones(3, 3), max_length=4),
    ]
    token_refs = [
        ResourceRef(name="tokens", kind="token_ids", value=[1, 2], max_length=4),
        ResourceRef(name="tokens", kind="token_ids", value=[3], max_length=4),
    ]
    tensor_samples = [
        AudioSample(utt_id=str(index), path=None, resources=ResourceCollection.from_refs([ref]))
        for index, ref in enumerate(tensor_refs)
    ]
    token_samples = [
        AudioSample(utt_id=str(index), path=None, resources=ResourceCollection.from_refs([ref]))
        for index, ref in enumerate(token_refs)
    ]

    assert collate_resources(tensor_samples)["features"].values.shape == (2, 4, 3)
    assert collate_resources(token_samples)["tokens"].values.shape == (2, 4)


def test_csv_annotation_provider_is_semantically_generic(tmp_path):
    annotation_path = tmp_path / "labels.txt"
    annotation_path.write_text("sample happy\n", encoding="utf-8")
    sample = AudioSample(utt_id="sample", path=tmp_path / "sample.wav")
    provider = CSVAnnotationProvider(
        name="emotion",
        path_template=str(annotation_path),
        item_key="utt_id",
        delimiter=" ",
    )

    annotation = provider(sample)

    assert annotation.name == "emotion"
    assert annotation.value == "happy"
    assert annotation.kind == "text"


def test_resource_objects_have_no_public_in_place_mutation():
    ref = ResourceRef(name="speaker", kind="text", value="speaker-1")
    source = {"speaker": ref}
    resources = ResourceCollection(source)

    with pytest.raises(FrozenInstanceError):
        ref.value = "speaker-2"
    with pytest.raises(TypeError):
        resources["speaker"] = ResourceRef(name="speaker", kind="text", value="speaker-2")

    source.clear()
    assert resources["speaker"] == ref
