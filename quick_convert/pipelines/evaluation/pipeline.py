from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

from tqdm import tqdm

from ...data.base_dataset import BaseDataset
from .metrics import Metric


class EvalPipeline:
    """Generate predictions and evaluate them against reference values.

    ``record_resources`` names sample resources to copy into each prediction
    record. This keeps metadata such as speaker identity generic and opt-in
    instead of assigning it a dedicated sample field.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        system,
        metrics: Iterable[Metric] | None,
        out_dir: PathLike,
        batch_size: int,
        num_workers: int = 0,
        ref_dataset: BaseDataset | None = None,  # optional argument
        record_resources: Iterable[str] | None = None,
    ):
        self.dataset = dataset
        self.ref_dataset = ref_dataset  # Store the reference dataset
        self.system = system
        self.metrics = list(metrics or [])
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.record_resources = list(dict.fromkeys(record_resources or []))
        reserved_columns = {"utt_id", "path", "split"}.intersection(self.record_resources)
        if reserved_columns:
            raise ValueError(f"Resource columns conflict with sample metadata: {sorted(reserved_columns)}")

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
        pred_loader = self.dataset.make_dataloader(batch_size=self.batch_size, num_workers=self.num_workers)

        ref_loader = (
            self.ref_dataset.make_dataloader(batch_size=self.batch_size, num_workers=self.num_workers)
            if self.ref_dataset
            else None
        )

        ref_iter = iter(ref_loader) if ref_loader else None

        records = []
        for pred_batch in tqdm(pred_loader, desc="Evaluating"):
            ref_batch = next(ref_iter) if ref_iter is not None else pred_batch
            if len(ref_batch) != len(pred_batch):
                raise ValueError("Mismatch between reference and prediction batches.")

            refs = {}
            predictions = self.system.get_labels(pred_batch) if self.metrics else []

            for metric in self.metrics:
                refs[metric.key] = metric.get_references(ref_batch)

            # Validate batch sizes
            if self.metrics and len(predictions) != len(pred_batch):
                raise ValueError(
                    f"System returned {len(predictions)} predictions for a batch of size {len(pred_batch)}"
                )
            for key, values in refs.items():
                if len(values) != len(pred_batch):
                    raise ValueError(
                        f"Reference data returned {len(values)} values for key {key!r}, "
                        f"but batch has size {len(pred_batch)}"
                    )

            # Combine reference and prediction data into records
            for i, sample in enumerate(pred_batch):
                record = {
                    "utt_id": sample.utt_id,
                    "path": str(sample.path),
                    "split": sample.split,
                }

                for name in self.record_resources:
                    try:
                        resource = sample.resources[name]
                    except KeyError as error:
                        raise ValueError(f"Sample {sample.utt_id!r} has no resource named {name!r}.") from error

                    if resource.value is not None:
                        record[name] = resource.value
                    elif resource.path is not None:
                        record[name] = str(resource.path)
                    else:
                        raise ValueError(f"Resource {name!r} for sample {sample.utt_id!r} has no value or path.")

                for key, values in refs.items():
                    record[f"ref_{key}"] = values[i]

                for metric in self.metrics:
                    record[metric.pred_key] = predictions[i]

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

        preferred = ["utt_id", "path", "split", *self.record_resources]
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
