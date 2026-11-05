"""Resolve reference paths without adding conditioning behavior to datasets."""

import hashlib
from pathlib import Path

from .base import BaseResourceProvider, ResourceKind, ResourceRef


class SameSpeakerReferenceProvider(BaseResourceProvider):
    """Select a deterministic different recording from a split-local manifest.

    The manifest supplies ``path``, ``split``, ``utt_id`` and a speaker column.
    Candidates are grouped by split and speaker, excluding the source path.
    By default return a text path for existing consumers. With kind="audio",
    return a lazy audio reference for dataset loading and collation.
    sample_rate optionally selects the loaded audio rate; cropping remains consumer policy.
    """

    def __init__(
        self,
        manifest_path: str,
        speaker_column: str = "spkid",
        seed: int = 0,
        kind: ResourceKind = "text",
        sample_rate: int | None = None,
    ):
        super().__init__("reference_path")
        if kind not in {"text", "audio"}:
            raise ValueError("Reference resources must use kind='text' or kind='audio'.")
        if sample_rate is not None and (kind != "audio" or sample_rate <= 0):
            raise ValueError("sample_rate requires an audio reference and must be positive.")
        self.kind = kind
        self.sample_rate = sample_rate
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("Reference seed must be an integer.")
        from quick_convert.data.manifest_dataset import ManifestDataset

        rows = ManifestDataset(manifest_path, resources={"spkid": {"column": speaker_column, "kind": "text"}}).rows
        groups = {}
        for row in rows:
            key = (row.split, str(row.resources["spkid"].value))
            groups.setdefault(key, set()).add(row.path)
        groups = {key: sorted(paths) for key, paths in groups.items()}
        positions = {key: {path: i for i, path in enumerate(paths)} for key, paths in groups.items()}
        self.references = {}
        for row in rows:
            group = (row.split, str(row.resources["spkid"].value))
            candidates = groups[group]
            if len(candidates) < 2:
                raise ValueError(f"No different same-speaker reference for {row.utt_id!r}.")
            key = f"{seed}:{row.split}:{row.utt_id}".encode()
            index = int.from_bytes(hashlib.sha256(key).digest(), "big") % (len(candidates) - 1)
            index += index >= positions[group][row.path]
            self.references[(row.split, row.utt_id, row.path)] = candidates[index]

    def __call__(self, sample):
        key = (sample.split, sample.utt_id, Path(sample.path))
        try:
            path = self.references[key]
        except KeyError as error:
            raise KeyError(f"Sample {sample.utt_id!r} is absent from the reference manifest.") from error
        if Path(path).resolve() == Path(sample.path).resolve():
            raise ValueError("The reference recording must differ from the target recording.")
        if self.kind == "audio":
            return ResourceRef(self.name, "audio", path=Path(path), sample_rate=self.sample_rate)
        return ResourceRef(self.name, "text", value=str(path))
