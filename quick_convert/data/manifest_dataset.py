import csv
from os import PathLike
from pathlib import Path
from typing import Iterable

from .base_dataset import BaseDataset
from .types import MetadataSample


class ManifestDataset(BaseDataset):
    """
    Because you can add multiple manifests, you may choose to see different files as "splits". At least for the time being,
    that will only be reflected in the `split` field of the MetadataSample if you fill the `split` column in your manifest CSVs.

    We could potentially add a `default_split` argument to the constructor that fills in missing splits with a default value based
    on the manifest file they came from, if that would be helpful.
    """

    def __init__(
        self,
        manifest_path: PathLike | Iterable[PathLike],
        path_column: str = "path",
        utt_id_column: str = "utt_id",
        split_column: str = "split",
        spk_id_column: str = "spk_id",
        **kwargs,
    ):
        rows = []

        if isinstance(manifest_path, (str, Path)):
            manifest_paths = [manifest_path]
        else:
            manifest_paths = list(manifest_path)

        for path in manifest_paths:
            with open(path, newline="") as f:
                try:
                    reader = csv.DictReader(f)
                except csv.Error:
                    raise ValueError(
                        f"Failed to parse manifest file {path} as CSV. Please check the file format and delimiter."
                    )

            for row in reader:
                rows.append(
                    MetadataSample(
                        utt_id=row.get(utt_id_column),
                        path=Path(row[path_column]) if row.get(path_column) else None,
                        split=row.get(split_column),
                        spk_id=row.get(spk_id_column),
                    )
                )

        super().__init__(rows=rows, **kwargs)
