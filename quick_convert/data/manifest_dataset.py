import csv
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

from quick_convert.data.resources.base import ResourceCollection, ResourceKind, ResourceRef

from .base_dataset import BaseDataset
from .types import MetadataSample


class ManifestDataset(BaseDataset):
    """Dataset backed by one or more CSV manifests.

    Each CSV row becomes a :class:`MetadataSample`. At minimum, manifests
    normally contain an utterance ID and audio path; an optional split column
    is copied directly onto the sample.

    Additional manifest columns can be exposed as resources through
    ``resources``. This is useful for values already present in the manifest,
    while :class:`BaseResourceProvider` objects remain appropriate for
    resources resolved externally.

    Args:
        manifest_path:
            Path or paths to CSV manifest files.
        path_column:
            Column containing the audio path.
        utt_id_column:
            Column containing the utterance ID.
        split_column:
            Optional column containing the dataset split.
        resources:
            Mapping from resource name to a specification containing
            ``"column"`` and ``"kind"``. For example::

                resources={
                    "transcript": {
                        "column": "text",
                        "kind": "text",
                    },
                }

        **kwargs:
            Additional arguments forwarded to :class:`BaseDataset`.

    Example:
        Given::

            utt_id,path,split,text
            001,/data/001.wav,train,hello world

        construct::

            dataset = ManifestDataset(
                "manifest.csv",
                resources={
                    "transcript": {
                        "column": "text",
                        "kind": "text",
                    }
                },
            )

            dataset[0].resources.transcript.value
            # "hello world"
    """

    def __init__(
        self,
        manifest_path: PathLike | Iterable[PathLike],
        path_column: str = "path",
        utt_id_column: str = "utt_id",
        split_column: str = "split",
        resources: dict[str, dict[str, str | ResourceKind]] | None = None,
        **kwargs,
    ):

        resources = resources or {}
        rows = []

        if isinstance(manifest_path, (str, Path)):
            manifest_paths = [manifest_path]
        else:
            manifest_paths = list(manifest_path)

        for path in manifest_paths:
            with open(path, newline="") as f:
                try:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(
                            MetadataSample(
                                utt_id=row.get(utt_id_column),
                                path=Path(row[path_column]) if row.get(path_column) else None,
                                split=row.get(split_column),
                                # spk_id=row.get(spk_id_column),
                                resources=ResourceCollection.from_refs(
                                    [self._resource_from_cell(name, spec, row) for name, spec in resources.items()]
                                ),
                            )
                        )
                except csv.Error as error:
                    raise ValueError(
                        f"Failed to parse manifest file {path} as CSV. Please check the file format and delimiter."
                    ) from error

        super().__init__(rows=rows, **kwargs)

    @staticmethod
    def _resource_from_cell(name: str, spec: dict[str, str | ResourceKind], row: dict[str, str]) -> ResourceRef:
        kind = spec["kind"]
        cell = row[spec["column"]]
        if kind in {"torch_tensor", "token_ids"}:
            return ResourceRef(name=name, kind=kind, path=Path(cell))
        return ResourceRef(name=name, kind=kind, value=cell)
