"""Resolve reference paths without adding conditioning behavior to datasets."""

import hashlib
from pathlib import Path

from .base import BaseResourceProvider, ResourceRef


class SameSpeakerReferenceProvider(BaseResourceProvider):
    """Select a deterministic different recording from a split-local manifest.

    The manifest supplies ``path``, ``split``, ``utt_id`` and a speaker column.
    Candidates are grouped by split and speaker, excluding the source path.
    This provider resolves paths only; consumers own audio loading and cropping.
    """

    def __init__(self, manifest_path: str, speaker_column: str = "spkid", seed: int = 0):
        super().__init__("reference_path")
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
        return ResourceRef(self.name, "text", value=str(path))
