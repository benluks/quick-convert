"""Extract and persist features with identity-checked resumption."""

from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path

import torch
from tqdm import tqdm

from quick_convert.data import BaseDataset
from quick_convert.pipelines.provenance import execution_metadata, resolved_config
from quick_convert.utils.atomic_files import atomic_file


class PrecomputeFeaturesPipeline:
    def __init__(
        self,
        dataset: BaseDataset,
        extractor,
        out_dir: PathLike,
        skip_existing: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        cache_key: str | None = None,
    ):
        self.dataset = dataset
        self.extractor = extractor
        self.out_root = Path(out_dir)
        self.skip_existing = skip_existing
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_key = cache_key
        self.config = None

    def write_config(self, config: str) -> None:
        self.config = resolved_config(config)

    def _plan(self) -> tuple[list[dict], list[int], str]:
        rows = []
        pending = []
        destinations = set()
        inputs = hashlib.sha256()
        for index, sample in enumerate(self.dataset.rows):
            split = sample.split or ""
            utt_id = sample.utt_id
            for part in (utt_id, split):
                if part in (".", "..") or "/" in part or "\\" in part:
                    raise ValueError(f"Unsafe precompute output identifier: {part!r}")
            if not utt_id:
                raise ValueError("Precompute requires nonempty utterance IDs.")
            path = self.out_root / split / f"{utt_id}.pt"
            if path in destinations:
                raise ValueError(f"Duplicate feature output path: {path}")
            destinations.add(path)
            row = {"utt_id": utt_id, "path": str(sample.path), "split": sample.split}
            inputs.update((json.dumps(row) + "\n").encode())
            rows.append({**row, "feature_path": str(path)})
            if not self.skip_existing or not path.is_file():
                pending.append(index)
        return rows, pending, inputs.hexdigest()

    def _check_identity(self, inputs_digest: str) -> None:
        if self.skip_existing and self.config is None and not self.cache_key:
            raise ValueError(
                "Python precompute resume requires an explicit cache_key for the extractor and input revision."
            )
        pipeline_config = self.config["pipeline"] if self.config is not None else {}
        identity = {
            "cache_key": self.cache_key,
            "extractor_type": f"{type(self.extractor).__module__}.{type(self.extractor).__qualname__}",
            "extractor": pipeline_config.get("extractor"),
            "dataset": pipeline_config.get("dataset"),
            "inputs_sha256": inputs_digest,
        }
        cache_format = getattr(self.extractor, "cache_format", None)
        if cache_format is not None:
            identity["cache_format"] = cache_format
        path = self.out_root / "provenance.json"
        if not path.exists():
            if (self.out_root / "manifest.jsonl").exists() or any(self.out_root.rglob("*.pt")):
                raise ValueError(
                    "Existing feature files have no cache identity. Recompute into a fresh output directory."
                )
            try:
                with atomic_file(path, replace=False) as stream:
                    json.dump(
                        {"schema_version": 1, "identity": identity, "execution": execution_metadata()},
                        stream,
                        indent=2,
                        default=str,
                    )
            except FileExistsError:
                # Another writer may have published its identity first.
                pass
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("schema_version") != 1 or saved.get("identity") != identity:
            raise ValueError("Precompute cache identity changed. Use a fresh output directory.")

    def run(self) -> None:
        rows, pending, inputs_digest = self._plan()
        self._check_identity(inputs_digest)

        if pending:
            loader = self.dataset.make_dataloader(
                batch_size=self.batch_size, num_workers=self.num_workers, sampler=pending
            )
            pending_rows = iter(rows[index] for index in pending)
            for batch in tqdm(loader):
                outputs = self.extractor.extract_batch(batch)
                if len(outputs) != len(batch):
                    raise ValueError(f"Extractor returned {len(outputs)} outputs for batch of size {len(batch)}")
                for sample, output in zip(batch, outputs, strict=True):
                    row = next(pending_rows, None)
                    if row is None or (sample.utt_id, str(sample.path), sample.split) != (
                        row["utt_id"],
                        row["path"],
                        row["split"],
                    ):
                        raise ValueError("Precompute loader did not preserve the selected sample order.")
                    with atomic_file(Path(row["feature_path"]), "wb") as stream:
                        torch.save(output, stream)
            if next(pending_rows, None) is not None:
                raise ValueError("Precompute loader omitted selected samples.")

        with atomic_file(self.out_root / "manifest.jsonl") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
