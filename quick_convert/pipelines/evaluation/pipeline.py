from __future__ import annotations

import csv
import json
from os import PathLike
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from ...data.base_dataset import BaseDataset
from .metrics import Metric


class EvalPipeline:
    def __init__(
        self,
        dataset: BaseDataset,
        system,
        metrics: Iterable[Metric] | None,
        out_dir: PathLike,
        batch_size: int,
        num_workers: int = 0,
        ref_dataset: BaseDataset | None = None,  # optional argument
    ):
        self.dataset = dataset
        self.ref_dataset = ref_dataset  # Store the reference dataset
        self.system = system
        self.metrics = list(metrics or [])
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.ref_dataset is None:
            print("No reference dataset provided. Falling back to predictions for evaluation.")

    def run(self) -> dict:
        records = self.generate_records()

        for metric in self.metrics:
            self.add_per_utt_scores(records, metric)

        self.write_records_csv(records)

        aggregate_scores = {}
        for metric in self.metrics:
            refs = [record[metric.ref_key] for record in records]
            preds = [record[metric.pred_key] for record in records]
            aggregate_scores.update(metric.compute(refs, preds))

        results = {
            **aggregate_scores,
            "num_files": len(records),
            "predictions_path": str(self.out_dir / "predictions.csv"),
        }

        print(f"Results saved to {self.out_dir / 'results.json'}")
        print(json.dumps(results, indent=2))

        self.write_results_json(results)

        return results

    def generate_records(self) -> list[dict]:

        # Create DataLoaders for both datasets
        pred_loader = self.dataset.make_dataloader(
            batch_size=self.batch_size, num_workers=self.num_workers
        )

        ref_loader = (
            self.ref_dataset.make_dataloader(batch_size=self.batch_size, num_workers=self.num_workers)
            if self.ref_dataset
            else None
        )

        ref_iter = iter(ref_loader) if ref_loader else None

        records = []
        for pred_batch in tqdm(pred_loader, desc="Evaluating"):
            refs = {}
            preds = {}

            for metric in self.metrics:

                if ref_iter is not None:
                    ref_batch = next(ref_iter)
                    if len(ref_batch) != len(pred_batch):
                        raise ValueError("Mismatch between reference and prediction batches.")    
                    refs[metric.key] = metric.get_references(ref_batch)
                else:
                    # If no reference dataset is provided, use predictions as references
                    refs[metric.key] = metric.get_references(pred_batch)
                
                preds[metric.key] = self.system.get_labels(pred_batch)

            # Validate batch sizes
            for key, values in preds.items():
                if len(values) != len(pred_batch):
                    raise ValueError(
                        f"Anonymized dataset returned {len(values)} predictions for key {key!r}, but batch has size {len(pred_batch)}"
                    )
            for key, values in refs.items():
                if len(values) != len(ref_batch):
                    raise ValueError(
                        f"Original dataset returned {len(values)} references for key {key!r}, but batch has size {len(ref_batch)}"
                    )

            # Combine reference and prediction data into records
            for i, sample in enumerate(pred_batch):
                record = {
                    "utt_id": sample.utt_id,
                    "path": str(sample.path),
                    "split": sample.split,
                }

                if getattr(sample, "spk_id", None) is not None:
                    record["spk_id"] = sample.spk_id

                for key, values in refs.items():
                    record[f"ref_{key}"] = values[i]

                for key, values in preds.items():
                    record[f"pred_{key}"] = values[i]

                records.append(record)

        return records

    def add_per_utt_scores(self, records: list[dict], metric: Metric) -> None:
        for record in records:
            record.update(metric.compute(record[metric.ref_key], record[metric.pred_key]))

    def write_records_csv(self, records: list[dict]) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        path = self.out_dir / "predictions.csv"

        if not records:
            path.write_text("", encoding="utf-8")
            return path

        fieldnames = sorted({key for record in records for key in record})

        preferred = ["utt_id", "path", "split", "spk_id"]
        fieldnames = [
            *[key for key in preferred if key in fieldnames],
            *[key for key in fieldnames if key not in preferred],
        ]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        return path

    def write_results_json(self, results: dict) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        path = self.out_dir / "results.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return path
