# pipelines/precompute_features.py

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm


class PrecomputeFeaturesPipeline:
    def __init__(
        self,
        dataset,
        extractor,
        out_root: str | Path,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        self.dataset = dataset
        self.extractor = extractor
        self.out_root = Path(out_root)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def run(self) -> None:
        loader = self.dataset.make_dataloader(batch_size=self.batch_size, num_workers=self.num_workers)

        feature_dir = self.out_root / self.extractor.feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = feature_dir / "manifest.jsonl"

        with manifest_path.open("w") as manifest_file:
            for batch in tqdm(loader):
                outputs = self.extractor.extract_batch(batch)

                if len(outputs) != len(batch):
                    raise ValueError(f"Extractor returned {len(outputs)} outputs for batch of size {len(batch)}")

                # write samples
                for sample, output in zip(batch, outputs):
                    split = sample.split or ""
                    utt_id = sample.path.stem

                    out_dir = feature_dir / split
                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_path = out_dir / f"{utt_id}.pt"
                    torch.save(output, out_path)

                    row = {
                        "utt_id": utt_id,
                        "path": str(sample.path),
                        "split": sample.split,
                        "feature_path": str(out_path),
                    }

                    if getattr(sample, "spk_id", None) is not None:
                        row["spk_id"] = sample.spk_id

                    manifest_file.write(json.dumps(row) + "\n")
