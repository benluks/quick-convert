# quick_convert/pipelines/manifest.py

from __future__ import annotations

import csv
from os import PathLike
from pathlib import Path

from tqdm import tqdm

from ..data.types import MetadataSample
from ..utils.paths import TemplateFormatter


class BuildManifestPipeline:
    def __init__(
        self,
        dataset,
        out_path: PathLike,
        columns: dict[str, str],
        overwrite: bool = False,
    ) -> None:
        self.dataset = dataset
        self.out_path = Path(out_path)
        self.columns = columns
        self.overwrite = overwrite

    def run(self) -> None:
        if self.out_path.exists() and not self.overwrite:
            raise FileExistsError(f"Manifest already exists: {self.out_path}. Set overwrite=true to replace it.")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with self.out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.columns.keys()))
            writer.writeheader()

            for sample in tqdm(self.dataset):
                writer.writerow(self.sample_to_row(sample))

    def sample_to_row(self, sample: MetadataSample) -> dict[str, str]:
        return {
            name: TemplateFormatter.format_str(template, sample=sample, path=sample.path)
            for name, template in self.columns.items()
        }
