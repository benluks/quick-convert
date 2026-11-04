"""Portable conversion output layout, provenance journal and SpeechScope export."""

import hashlib
import json
import logging
import os
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from uuid import uuid4

import yaml
from omegaconf import OmegaConf

from quick_convert.utils.paths import checkpoint_run_config


logger = logging.getLogger(__name__)


def audio_destination(out_dir: Path, sample) -> Path:
    split = sample.split or ""
    if (
        not sample.utt_id
        or sample.utt_id in {".", ".."}
        or Path(sample.utt_id).name != sample.utt_id
        or "\\" in sample.utt_id
        or split in {".", ".."}
        or (split and Path(split).name != split)
        or "\\" in split
    ):
        raise ValueError("Conversion requires plain utterance IDs and split names.")
    return out_dir / split / f"{sample.utt_id}.wav"


def plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def audio_digest(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_identity(path) -> dict:
    """Identify a local source cheaply, without reading multi-GB model weights."""
    path = Path(path).expanduser().resolve()
    result = {"path": str(path)}
    if path.is_file():
        stat = path.stat()
        result.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def model_identity(metadata) -> dict:
    metadata = plain(metadata) or {}
    source = metadata.get("source")
    result = {"system": metadata.get("system", "speech-conversion")}
    if not source:
        return result
    source = Path(source).expanduser().resolve()
    result["source"] = str(source)
    if source.is_file():
        files = [source]
        config = metadata.get("config")
        config = (
            (Path(config) if Path(config).is_absolute() else source.parent / config)
            if config
            else checkpoint_run_config(source)
        )
        if config is not None:
            files.append(config)
    elif (source / "artifact.yaml").is_file():
        artifact = OmegaConf.load(source / "artifact.yaml")
        files = [source / "artifact.yaml", source / artifact.weights]
        result["checkpoint"] = str(files[1])
    else:
        checkpoint = Path(metadata.get("checkpoint") or "checkpoints/last.ckpt")
        files = [checkpoint if checkpoint.is_absolute() else source / checkpoint]
        config = Path(metadata.get("config") or "config.yaml")
        files.append(config if config.is_absolute() else source / config)
    result["files"] = [file_identity(path) for path in files]
    result.setdefault("checkpoint", str(files[0]))
    return result


def slug(value) -> str:
    return re.sub(r"[^\w.-]+", "-", str(value)).strip(".-")[:100] or "unnamed"


def conversion_output_dir(root, metadata, target, suffix="") -> str:
    """Resolve a default directory without loading models or datasets.

    Kept pure apart from local file stat calls so Hydra can select its logging
    directory before instantiating the inference graph.
    """
    metadata, target = plain(metadata) or {}, plain(target) or {}
    model = model_identity(metadata)
    source = metadata.get("source")
    name = Path(str(source)).stem if source else "unversioned"
    if source and Path(source).is_file() and Path(source).parent.name == "checkpoints":
        name = Path(source).parent.parent.name + "-" + Path(source).stem
    checkpoint = metadata.get("checkpoint")
    if checkpoint:
        name += "-" + Path(str(checkpoint)).stem
    name = slug(name)
    # Read only the saved scalar; legacy configs may contain unavailable resolvers.
    for file in model.get("files", [])[1:]:
        config_path = Path(file["path"])
        if str(config_path) == model.get("checkpoint") or not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text())
        experiment = config.get("exp_name") if isinstance(config, Mapping) else None
        if isinstance(experiment, str) and experiment.strip() and "${" not in experiment:
            name = f"{slug(experiment)}-{name}"
            break
    if target.get("mode") == "resynthesize" or metadata.get("resynthesize"):
        policy = "resynthesize"
    elif target.get("mode") == "backend-default":
        policy = "backend-default"
    elif target.get("mode") == "random":
        policy = f"random-seed-{target.get('seed', 0)}"
    else:
        label = metadata.get("target_speaker", target.get("value"))
        policy = f"fixed-{label}" if label is not None else "reference"
    fingerprint = digest({"model": model, "target": target, "metadata": metadata})[:12]
    return str(
        Path(root)
        / "anonymized"
        / slug(model["system"])
        / name
        / slug(metadata.get("dataset", "dataset"))
        / f"{slug(policy)}-{fingerprint}{slug(suffix) if suffix else ''}"
    )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_yaml(path: Path, document) -> None:
    atomic_text(path, yaml.safe_dump(plain(document), sort_keys=False, allow_unicode=True))


def append_record(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_journal(path: Path) -> list[dict]:
    """Recover a torn final append, but reject corruption in committed rows."""
    if not path.exists():
        return []
    rows, offset = [], 0
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Invalid conversion journal row {index + 1}: {path}") from None
            logger.warning("Recovering incomplete final journal row in %s", path)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        rows.append(row)
        offset += len(line)
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return rows


def _code_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_dirty():
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=Path(__file__).parents[2],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


class ConversionRecords:
    """Durable per-output records with atomic, derived YAML snapshots.

    The JSONL journal is committed after each audio file. YAML views refresh
    every 50 outputs and when the run exits, including on an exception. A resume
    rebuilds views from the journal before doing inference.
    """

    def __init__(self, out_dir, metadata, target, *, config=None, overwrite=False):
        self.out_dir = Path(out_dir).resolve()
        self.metadata = plain(metadata) or {}
        self.target = plain(target)
        self.config = config or yaml.safe_dump(
            {
                "format": "quick-convert-runtime-description",
                "metadata": self.metadata,
                "target": plain(target),
            },
            sort_keys=False,
        )
        self.overwrite = overwrite
        self.model = model_identity(self.metadata)
        self.dataset = str(self.metadata.get("dataset", "dataset"))
        self.roots = dict(self.metadata.get("roots") or {})
        if self.metadata.get("source_root"):
            self.roots.setdefault(self.dataset, [str(Path(self.metadata["source_root"]).resolve())])
        if self.metadata.get("reference_root"):
            self.roots.setdefault(self.dataset + "_targets", [str(Path(self.metadata["reference_root"]).resolve())])
        self.roots = {
            alias: [str(Path(root).expanduser().resolve()) for root in ([roots] if isinstance(roots, str) else roots)]
            for alias, roots in self.roots.items()
        }
        self.identity = {"model": self.model, "dataset": self.dataset, "target": self.target}
        if self.metadata.get("inference_cfg_rate") is not None:
            self.identity["inference_cfg_rate"] = self.metadata["inference_cfg_rate"]
        if config:
            configured = yaml.safe_load(config)
            system = dict(configured.get("system") or configured.get("anonymizer") or {})
            # Device selection and loader batch sizes do not define a variant.
            system.pop("map_location", None)
            system.pop("device", None)
            self.identity["generation"] = {"system": system, "run": configured.get("run", {})}
        self.run_id = uuid4().hex
        self.records = {}
        self.targets = {}
        self.generated = self.skipped = 0
        self.code_revision = _code_revision()
        self.code_dirty = _code_dirty()
        self.created = datetime.now(UTC).isoformat()
        self.status = "running"
        self.config_hash = digest(yaml.safe_load(self.config))

    def __enter__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        run_path = self.out_dir / "run.yaml"
        if run_path.exists():
            previous = yaml.safe_load(run_path.read_text())
            if previous["identity"] != self.identity and not self.overwrite:
                raise ValueError(
                    "Saved target assignment differs or model changed; choose a new out_dir or explicitly overwrite."
                )
            if previous["identity"] == self.identity:
                self.created = previous["created_at"]
                self.run_id = previous["id"]
        self.records = {row["source_id"]: row for row in read_journal(self.out_dir / "conversions.jsonl")}
        self.targets = {row["id"]: row["target"] for row in read_journal(self.out_dir / "targets.jsonl")}
        for filename in ("conversions.jsonl", "targets.jsonl"):
            if not (self.out_dir / filename).exists():
                (self.out_dir / filename).touch()
        if self.config:
            atomic_yaml(self.out_dir / "config.yaml", yaml.safe_load(self.config))
            atomic_yaml(self.out_dir / "configs" / f"{self.config_hash}.yaml", yaml.safe_load(self.config))
        self.refresh()
        logger.info(
            "Conversion records: %s; SpeechScope manifest: %s",
            self.out_dir / "conversions.jsonl",
            self.out_dir / "speechscope.yaml",
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.status = "failed" if exc_type else "complete"
        self.refresh()

    def source_id(self, sample):
        return "/".join(quote(str(value), safe="") for value in (self.dataset, sample.split or "", sample.utt_id))

    def audio(self, path):
        path = Path(path).expanduser().resolve()
        for alias, roots in self.roots.items():
            for root in [roots] if isinstance(roots, str) else roots:
                try:
                    return {"root": alias, "path": str(path.relative_to(Path(root).expanduser().resolve()))}
                except ValueError:
                    pass
        # Absolute audio strings are supported by SpeechScope. Mappings require
        # relative paths, including when no alias is specified.
        return str(path)

    def should_skip(self, sample, destination, assignment):
        if not destination.exists() or self.overwrite:
            return False
        previous = self.records.get(self.source_id(sample))
        if (
            previous is None
            or previous["run_id"] != self.run_id
            or previous["assignment"] != assignment
            or previous["source_audio"] != self.audio(sample.path)
        ):
            raise ValueError(
                f"Saved target assignment differs or conversion record is missing for {sample.utt_id!r}; use a new output directory or overwrite."
            )
        if previous["output_sha256"] != audio_digest(destination):
            logger.info("Regenerating incomplete or changed tracked audio: %s", destination)
            return False
        if previous["source_file"] != file_identity(sample.path):
            logger.info("Regenerating output for changed source audio: %s", sample.path)
            return False
        self.skipped += 1
        return True

    def save(self, sample, destination, assignment, waveform, sample_rate, *, target_key=None):
        """Journal a completed temporary WAV, then atomically publish it."""
        import torchaudio

        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            suffix=".wav", prefix=f".{destination.stem}-", dir=destination.parent, delete=False
        ) as stream:
            staged = Path(stream.name)
        try:
            torchaudio.save(str(staged), waveform.unsqueeze(0).cpu(), sample_rate)
            with staged.open("rb") as stream:
                os.fsync(stream.fileno())
            self.commit(
                sample,
                destination,
                assignment,
                sample_rate,
                waveform.shape[-1],
                target_key=target_key,
                staged=staged,
            )
        finally:
            staged.unlink(missing_ok=True)

    def prepared(self, key, target):
        key = self.run_id + ":" + str(key)
        description = {
            "references": [self.audio(path) for path in target.references],
            "durations": list(target.durations),
            "aggregation": target.aggregation,
            "representation": target.representation,
            "space": target.space,
        }
        if target.prompt_segments:
            count = sum(
                segment.end_sample - segment.start_sample - segment.overlap_samples
                for segment in target.prompt_segments
            )
            description["prompt"] = {
                "sample_rate": target.prompt_sample_rate,
                "num_samples": count,
                "seconds": count / target.prompt_sample_rate,
                "crossfade_seconds": target.prompt_crossfade_seconds,
                "segments": [
                    {
                        "audio": self.audio(segment.path),
                        "start_sample": segment.start_sample,
                        "end_sample": segment.end_sample,
                        "overlap_samples": segment.overlap_samples,
                    }
                    for segment in target.prompt_segments
                ],
            }
        if self.targets.get(key) == description:
            return
        append_record(self.out_dir / "targets.jsonl", {"id": key, "target": description})
        self.targets[key] = description
        atomic_yaml(self.out_dir / "targets.yaml", {"roots": self.roots, "targets": self.targets})

    def commit(self, sample, destination, assignment, sample_rate, num_samples, *, target_key=None, staged):
        source_id = self.source_id(sample)
        previous = self.records.get(source_id)
        speaker = None
        resource_name = self.metadata.get("speaker_resource", "spkid")
        if resource_name in sample.resources.keys():
            ref = sample.resources[resource_name]
            if ref.kind == "text" and ref.value is not None:
                speaker = str(ref.value)
        row = {
            "source_id": source_id,
            "variant_id": previous["variant_id"] if previous else self.run_id + ":" + source_id,
            "target_id": self.run_id + ":" + str(target_key) if target_key is not None else None,
            "utt_id": sample.utt_id,
            "dataset": self.dataset,
            "split": sample.split,
            "speaker_id": speaker,
            "source_audio": self.audio(sample.path),
            "source_file": file_identity(sample.path),
            "source_sampling": self.metadata.get("source_sampling"),
            "output_audio": str(destination.resolve().relative_to(self.out_dir)),
            "output_sha256": audio_digest(staged),
            "output_size": staged.stat().st_size,
            "output_mtime_ns": staged.stat().st_mtime_ns,
            "assignment": assignment,
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration": num_samples / sample_rate,
            "run_id": self.run_id,
            "model": self.model,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": self.code_revision,
            "git_dirty": self.code_dirty,
            "config_sha256": self.config_hash,
        }
        append_record(self.out_dir / "conversions.jsonl", row)
        self.records[source_id] = row
        staged.replace(destination)
        self.generated += 1
        if self.generated % 50 == 0:
            self.refresh()

    def refresh(self):
        utterances, variants = [], []
        for row in self.records.values():
            if not self._live(row):
                continue
            metadata = {"dataset": row["dataset"], "split": row["split"], "utt_id": row["utt_id"]}
            utterances.append(
                {
                    "id": row["source_id"],
                    "audio": row["source_audio"],
                    "speaker_id": row["speaker_id"],
                    "metadata": metadata,
                }
            )
            assignment = row["assignment"]
            target_speaker = assignment.get("target_speaker", self.metadata.get("target_speaker"))
            variants.append(
                {
                    "id": row["variant_id"],
                    "utterance_id": row["source_id"],
                    "system": row["model"]["system"],
                    "target_speaker": str(target_speaker) if target_speaker is not None else None,
                    "checkpoint": row["model"].get("checkpoint"),
                    "seed": assignment.get("seed"),
                    "settings": {
                        "target_selection": {
                            key: value
                            for key, value in assignment.items()
                            if key not in {"references", "source_path", "utt_id", "split"}
                        },
                        "target_id": row["target_id"],
                        "target_catalogue": "targets.yaml",
                    },
                    "audio": {"path": row["output_audio"]},
                    "metadata": {
                        **metadata,
                        "model": row["model"],
                        "duration": row["duration"],
                        "sample_rate": row["sample_rate"],
                        "num_samples": row["num_samples"],
                        "run_id": row["run_id"],
                        "git_commit": row["git_commit"],
                        "git_dirty": row["git_dirty"],
                        "created_at": row["created_at"],
                        "config_sha256": row["config_sha256"],
                        "config_path": f"configs/{row['config_sha256']}.yaml",
                        "output_sha256": row["output_sha256"],
                        "source_file": row["source_file"],
                        "source_sampling": row.get("source_sampling"),
                    },
                }
            )
        atomic_yaml(
            self.out_dir / "speechscope.yaml", {"roots": self.roots, "utterances": utterances, "variants": variants}
        )
        atomic_yaml(self.out_dir / "targets.yaml", {"roots": self.roots, "targets": self.targets})
        atomic_text(
            self.out_dir / "target_assignments.jsonl",
            "".join(
                json.dumps(row["assignment"], ensure_ascii=False) + "\n"
                for row in self.records.values()
                if self._live(row)
            ),
        )

        atomic_yaml(
            self.out_dir / "run.yaml",
            {
                "format": "quick-convert-conversion",
                "version": 1,
                "id": self.run_id,
                "identity": self.identity,
                "metadata": self.metadata,
                "created_at": self.created,
                "updated_at": datetime.now(UTC).isoformat(),
                "status": self.status,
                "outputs": len(variants),
                "execution": {
                    "converted": self.generated,
                    "skipped": self.skipped,
                    "git_commit": self.code_revision,
                    "git_dirty": self.code_dirty,
                    "package_version": version("quick-convert"),
                    "working_directory": str(Path.cwd()),
                },
                "files": {
                    "config": "config.yaml" if self.config else None,
                    "journal": "conversions.jsonl",
                    "targets": "targets.yaml",
                    "target_journal": "targets.jsonl",
                    "config_history": "configs/",
                    "speechscope": "speechscope.yaml",
                },
            },
        )

    def _live(self, row):
        path = self.out_dir / row["output_audio"]
        if not path.is_file():
            return False
        stat = path.stat()
        if stat.st_size != row["output_size"]:
            return False
        return stat.st_mtime_ns == row["output_mtime_ns"] or audio_digest(path) == row["output_sha256"]
