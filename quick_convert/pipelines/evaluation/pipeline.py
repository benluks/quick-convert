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
    instead of assigning it a dedicated sample field. With a separate reference
    dataset, both datasets must have the same unique utterance IDs. Records
    follow prediction order. ``reference_systems`` maps metric keys to systems
    that generate reference values from the separate dataset.
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
        reference_systems: dict[str, object] | None = None,
    ):
        self.dataset = dataset
        self.ref_dataset = ref_dataset  # Store the reference dataset
        self.system = system
        self.metrics = list(metrics or [])
        self.reference_systems = dict(reference_systems or {})
        unknown_keys = self.reference_systems.keys() - {metric.key for metric in self.metrics}
        if unknown_keys:
            raise ValueError(f"Reference systems have no corresponding metric: {sorted(unknown_keys)}")
        if self.reference_systems and ref_dataset is None:
            raise ValueError("Reference systems require a separate reference dataset.")
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

    def _reference_values(self, batch) -> dict[str, list]:
        refs = {}
        for metric in self.metrics:
            reference_system = self.reference_systems.get(metric.key)
            values = metric.get_references(batch) if reference_system is None else reference_system.get_labels(batch)
            if len(values) != len(batch):
                raise ValueError(
                    f"Reference data returned {len(values)} values for key {metric.key!r}, "
                    f"but batch has size {len(batch)}"
                )
            refs[metric.key] = values
        return refs

    def generate_records(self) -> list[dict]:
        loader_kwargs = {"batch_size": self.batch_size, "num_workers": self.num_workers}
        pred_loader = self.dataset.make_dataloader(**loader_kwargs)

        references_by_id = None
        if self.ref_dataset is not None:
            references_by_id = {}
            for batch in self.ref_dataset.make_dataloader(**loader_kwargs):
                refs = self._reference_values(batch)
                for i, utt_id in enumerate(batch.utt_ids):
                    if utt_id in references_by_id:
                        raise ValueError(f"Duplicate reference utterance ID {utt_id!r}.")
                    references_by_id[utt_id] = {key: values[i] for key, values in refs.items()}

        records = []
        prediction_ids = set()
        for pred_batch in tqdm(pred_loader, desc="Evaluating"):
            if references_by_id is None:
                refs = self._reference_values(pred_batch)
            else:
                paired_refs = []
                for utt_id in pred_batch.utt_ids:
                    if utt_id in prediction_ids:
                        raise ValueError(f"Duplicate prediction utterance ID {utt_id!r}.")
                    if utt_id not in references_by_id:
                        raise ValueError(f"Missing reference for utterance ID {utt_id!r}.")
                    prediction_ids.add(utt_id)
                    paired_refs.append(references_by_id.pop(utt_id))
                refs = {metric.key: [values[metric.key] for values in paired_refs] for metric in self.metrics}

            predictions = self.system.get_labels(pred_batch) if self.metrics else []
            if self.metrics and len(predictions) != len(pred_batch):
                raise ValueError(
                    f"System returned {len(predictions)} predictions for a batch of size {len(pred_batch)}"
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

        if references_by_id:
            example = next(iter(references_by_id))
            raise ValueError(
                f"{len(references_by_id)} reference utterance(s) have no prediction, including {example!r}."
            )
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
