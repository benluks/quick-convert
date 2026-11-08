"""Dataset voice conversion with explicit or dataset-selected targets."""

import logging
import math
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from quick_convert.data import AudioBatch, BaseDataset
from quick_convert.data.sampling import shuffled_sample_indices
from quick_convert.pipelines.conversion_records import (
    ConversionRecords,
    audio_destination,
    conversion_output_dir,
    file_identity,
)
from quick_convert.pipelines.targets import DatasetReferenceTargets, resource_label
from quick_convert.systems.conversion.base import SpeechConversionSystem
from quick_convert.systems.conversion.targets import (
    GeneratedVoice,
    PreparedTarget,
    ReferenceSpeech,
    SourceVoice,
    SpeakerIdentity,
)


if TYPE_CHECKING:
    from quick_convert.components.conditioning.reference_prompt import SpeechPromptBuilder
    from quick_convert.components.transforms.speaker_aggregation import SpeakerEmbeddingAggregation


logger = logging.getLogger(__name__)


def reference_speech_from_dataset(
    dataset: BaseDataset,
    speaker_id: str | int,
    *,
    speaker_resource: str = "spkid",
    aggregation: "SpeakerEmbeddingAggregation | None" = None,
    prompt_seconds: float | None = None,
    prompt_builder: "SpeechPromptBuilder | None" = None,
    allow_missing_streams: bool = False,
) -> ReferenceSpeech:
    """Select reference recordings through a dataset's speaker resource.

    Uses normal dataset access, including configured resource providers and
    loading. Configure the reference dataset to load only speaker metadata;
    the system will encode selected audio during target preparation.
    """
    paths = []
    logger.info("Selecting reference recordings for %s=%s", speaker_resource, speaker_id)
    for sample in tqdm(dataset, desc=f"Selecting {speaker_resource}={speaker_id}", unit="recording"):
        if resource_label(sample, speaker_resource) == str(speaker_id):
            if sample.path is None:
                raise ValueError(f"Selected sample {sample.utt_id!r} has no audio path.")
            paths.append(sample.path)
    if not paths:
        raise ValueError(f"No recordings found for speaker {speaker_id!r} in the reference dataset.")
    logger.info("Selected %d reference recordings for %s=%s", len(paths), speaker_resource, speaker_id)
    return ReferenceSpeech(
        paths=paths,
        aggregation=aggregation,
        prompt_seconds=prompt_seconds,
        prompt_builder=prompt_builder,
        allow_missing_streams=allow_missing_streams,
    )


def reference_speech_from_directory(
    root: PathLike, speaker_id: str | int, pattern: str = "*/*.flac"
) -> ReferenceSpeech:
    """Select a speaker's recordings at the workflow boundary."""
    return ReferenceSpeech(sorted((Path(root) / str(speaker_id)).glob(pattern)))


def target_description(target):
    """Describe selection and references without serializing conditioning tensors."""
    if isinstance(target, SpeakerIdentity):
        return {"mode": "identity", "speaker_id": target.speaker_id}
    if isinstance(target, GeneratedVoice):
        return {"mode": "generated", "seed": target.seed}
    if isinstance(target, SourceVoice):
        return {"mode": "resynthesize"}
    if isinstance(target, DatasetReferenceTargets):
        return {
            "mode": target.mode,
            "resource": target.resource,
            "value": target.value,
            "seed": target.seed,
            "exclude_source": target.exclude_source,
            "source_resource": target.source_resource,
            "catalogue_sha256": target.catalogue_digest,
            "prompt_seconds": next(iter(target.groups.values())).prompt_seconds,
            "prompt_builder": next(iter(target.groups.values())).prompt_builder.description()
            if next(iter(target.groups.values())).prompt_builder is not None
            else None,
            "allow_missing_streams": next(iter(target.groups.values())).allow_missing_streams,
            "aggregation": next(iter(target.groups.values())).aggregation.method
            if next(iter(target.groups.values())).aggregation
            else None,
        }
    if isinstance(target, PreparedTarget):
        return {
            "mode": "prepared",
            **({"speaker_id": target.speaker_id} if target.speaker_id is not None else {}),
            "space": target.space,
            "representation": target.representation,
            "references": [str(path.resolve()) for path in target.references],
            "reference_files": [file_identity(path) for path in target.references],
            "aggregation": target.aggregation,
            **(
                {
                    "prompt": {
                        "sample_rate": target.prompt_sample_rate,
                        "crossfade_seconds": target.prompt_crossfade_seconds,
                        "segments": [
                            {
                                "path": str(segment.path.resolve()),
                                "start_sample": segment.start_sample,
                                "end_sample": segment.end_sample,
                                "overlap_samples": segment.overlap_samples,
                            }
                            for segment in target.prompt_segments
                        ],
                    }
                }
                if target.prompt_segments
                else {}
            ),
        }
    return {
        "mode": "fixed",
        **({"allow_missing_streams": True} if target.allow_missing_streams else {}),
        "references": [str(path.resolve()) for path in target.paths],
        "reference_files": [file_identity(path) for path in target.paths],
        "aggregation": target.aggregation.method if target.aggregation else None,
        **({"prompt_builder": target.prompt_builder.description()} if target.prompt_builder is not None else {}),
        **({"prompt_seconds": target.prompt_seconds} if target.prompt_seconds is not None else {}),
    }


class SpeechConversionPipeline:
    """Convert audio and automatically maintain provenance and portable manifests.

    Explicit out_dir values preserve custom layouts. Otherwise the output path
    is derived from model/dataset/target metadata. Dataset groups use a bounded
    preparation cache and convert one sample at a time.
    """

    def __init__(
        self,
        system: SpeechConversionSystem,
        dataset: BaseDataset,
        target: ReferenceSpeech
        | SpeakerIdentity
        | SourceVoice
        | GeneratedVoice
        | PreparedTarget
        | DatasetReferenceTargets,
        out_dir: PathLike | None = None,
        *,
        batch_size: int = 1,
        num_workers: int = 0,
        overwrite: bool = False,
        max_prepared_targets: int = 16,
        metadata: dict | None = None,
        out_root: PathLike = "outputs",
        max_samples: int | None = None,
        sample_seed: int = 0,
        inference_cfg_rate: float | None = None,
    ):
        if not isinstance(
            target,
            (ReferenceSpeech, SpeakerIdentity, SourceVoice, GeneratedVoice, PreparedTarget, DatasetReferenceTargets),
        ):
            raise TypeError(
                "Conversion requires reference speech, a speaker identity, source voice, generated voice, a prepared target, or dataset reference groups."
            )
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
        if inference_cfg_rate is not None and (not math.isfinite(inference_cfg_rate) or inference_cfg_rate < 0):
            raise ValueError("inference_cfg_rate must be finite and nonnegative.")
        shuffled_sample_indices(0, max_samples, sample_seed)  # Validate without reading data.
        self.max_samples = max_samples
        self.sample_seed = sample_seed
        self.generation_kwargs = {} if inference_cfg_rate is None else {"inference_cfg_rate": inference_cfg_rate}
        self.system = system
        self.dataset = dataset
        self.target = target
        self.metadata = {
            "system": type(system).__name__,
            "dataset": Path(getattr(dataset, "root", None) or "dataset").name,
            **getattr(system, "_inference_source", {}),
            **dict(metadata or {}),
        }
        self.metadata.update(self.generation_kwargs)
        if getattr(dataset, "root", None):
            self.metadata.setdefault("source_root", str(dataset.root))
        if isinstance(target, (SpeakerIdentity, PreparedTarget)) and target.speaker_id is not None:
            self.metadata.setdefault("target_speaker", target.speaker_id)
        self.description = target_description(target)
        if self.metadata.get("target_speaker") is not None:
            self.description["target_speaker"] = str(self.metadata["target_speaker"])
        self.out_dir = (
            Path(out_dir)
            if out_dir is not None
            else Path(conversion_output_dir(out_root, self.metadata, self.description))
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.overwrite = overwrite
        self.config = None
        if (
            not isinstance(max_prepared_targets, int)
            or isinstance(max_prepared_targets, bool)
            or max_prepared_targets < 1
        ):
            raise ValueError("max_prepared_targets must be positive.")
        self.max_prepared_targets = max_prepared_targets

    def write_config(self, config: str) -> None:
        # Validation of existing output provenance precedes replacing any files.
        self.config = config

    def _assignment(self, sample):
        if isinstance(self.target, DatasetReferenceTargets):
            label = self.target.select(sample)
            assignment = self.target.assignment(sample, label)
            if self.target.resource == self.metadata.get("speaker_resource", "spkid"):
                assignment["target_speaker"] = label
            return label, assignment
        return "fixed", {
            "utt_id": sample.utt_id,
            "split": sample.split,
            "source_path": str(sample.path),
            **self.description,
            "target_speaker": self.metadata.get("target_speaker"),
        }

    def run(self) -> Path:
        logger.info(
            "Starting speech conversion: output=%s, batch_size=%d, workers=%d",
            self.out_dir,
            self.batch_size,
            self.num_workers,
        )
        cache = OrderedDict()
        destinations = set()
        preparations = 0
        grouped = isinstance(self.target, DatasetReferenceTargets)
        metadata = dict(self.metadata)
        loader_kwargs = {"batch_size": self.batch_size, "num_workers": self.num_workers}
        if self.max_samples is not None:
            indices = shuffled_sample_indices(len(self.dataset), self.max_samples, self.sample_seed)
            loader_kwargs["sampler"] = indices
            metadata["source_sampling"] = {
                "max_samples": self.max_samples,
                "seed": self.sample_seed,
                "population": len(self.dataset),
                "selected": len(indices),
            }
            logger.info(
                "Shuffled source subset: %d/%d samples, sample_seed=%d",
                len(indices),
                len(self.dataset),
                self.sample_seed,
            )
        with ConversionRecords(
            self.out_dir, metadata, self.description, config=self.config, overwrite=self.overwrite
        ) as records:
            loader = self.dataset.make_dataloader(**loader_kwargs)
            progress = tqdm(loader, desc="Converting speech", unit="batch")
            for batch in progress:
                pending = []
                for sample in batch:
                    destination = audio_destination(self.out_dir, sample)
                    if destination in destinations:
                        raise ValueError(f"Duplicate conversion output: {destination}")
                    destinations.add(destination)
                    label, assignment = self._assignment(sample)
                    if not records.should_skip(sample, destination, assignment):
                        pending.append((sample, destination, label, assignment))
                # Existing systems consume one target per batch. Fixed targets
                # retain batching; selected groups use singleton inference.
                chunks = [[item] for item in pending] if grouped else [pending]
                for chunk in chunks:
                    if not chunk:
                        continue
                    label = chunk[0][2]
                    if label not in cache:
                        if len(cache) >= self.max_prepared_targets:
                            evicted = next(iter(cache))
                            del cache[evicted]
                            logger.info(
                                "Evicting prepared target %s (cache limit %d)", evicted, self.max_prepared_targets
                            )
                        request = self.target.groups[label] if grouped else self.target
                        logger.info(
                            "Preparing target %s from %d recordings",
                            label,
                            len(request.paths)
                            if isinstance(request, ReferenceSpeech)
                            else len(getattr(request, "references", ())),
                        )
                        cache[label] = self.system.prepare_target(request)
                        records.prepared(label, cache[label])
                        preparations += 1
                        logger.info(
                            "Target ready: representation=%s, references=%d, aggregation=%s",
                            cache[label].representation,
                            len(cache[label].references),
                            cache[label].aggregation,
                        )
                    cache.move_to_end(label)
                    inputs = (
                        batch
                        if not grouped and len(chunk) == len(batch)
                        else AudioBatch.from_samples([item[0] for item in chunk])
                    )
                    generated = self.system.convert_batch(inputs, target=cache[label], **self.generation_kwargs)
                    if len(generated) != len(chunk):
                        raise ValueError("Conversion output count does not match the source batch.")
                    for index, (sample, destination, _, assignment) in enumerate(chunk):
                        waveform = generated.waveform(index).cpu()
                        records.save(
                            sample,
                            destination,
                            assignment,
                            waveform,
                            generated.sample_rate,
                            target_key=label,
                        )
                    progress.set_postfix(
                        converted=records.generated, skipped=records.skipped, cached=len(cache), target=label
                    )
                progress.set_postfix(converted=records.generated, skipped=records.skipped, cached=len(cache))
        logger.info(
            "Conversion complete: %d converted, %d skipped, %d target preparations; output=%s; SpeechScope manifest=%s",
            records.generated,
            records.skipped,
            preparations,
            self.out_dir,
            self.out_dir / "speechscope.yaml",
        )
        return self.out_dir
