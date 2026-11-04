"""Dataset reference groups and reproducible per-utterance selection."""

import hashlib
import json
import logging

from tqdm import tqdm

from quick_convert.components.speaker import SpeakerEmbeddingAggregation
from quick_convert.data import AudioSample, BaseDataset
from quick_convert.pipelines.conversion_records import file_identity
from quick_convert.systems.anonymization.targets import ReferenceSpeech


logger = logging.getLogger(__name__)


def resource_label(sample: AudioSample, name: str) -> str:
    """Read an explicitly loaded, scalar text annotation."""
    try:
        resource = sample.resources[name]
    except KeyError:
        raise ValueError(f"Sample {sample.utt_id!r} lacks speaker resource {name!r}.") from None
    if resource.kind != "text" or not isinstance(resource.value, (str, int)) or isinstance(resource.value, bool):
        raise ValueError(f"Speaker resource {name!r} must contain loaded text labels.")
    label = str(resource.value)
    if not label:
        raise ValueError(f"Sample {sample.utt_id!r} has an empty resource {name!r}.")
    return label


class DatasetReferenceTargets:
    """Group references by a resource, then select a fixed or random group.

    Random selection hashes seed, split and utterance ID over sorted eligible
    labels, so dataset order, batching and resumption do not change assignments.
    Keep the candidate dataset and policy unchanged when resuming.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        resource: str = "spkid",
        *,
        mode: str = "fixed",
        value: str | int | None = None,
        seed: int = 0,
        exclude_source: bool = False,
        source_resource: str | None = None,
        aggregation: SpeakerEmbeddingAggregation | None = None,
        prompt_seconds: float | None = None,
        prompt_builder=None,
        allow_missing_streams: bool = False,
    ):
        if mode not in {"fixed", "random"}:
            raise ValueError("Target mode must be 'fixed' or 'random'.")
        if mode == "fixed" and value is None:
            raise ValueError("Fixed selection requires a group value.")
        if mode == "random" and value is not None:
            raise ValueError("Random selection does not accept a fixed group value.")
        if mode == "fixed" and seed != 0:
            raise ValueError("A random seed is only used by random selection.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("Target seed must be an integer.")
        if source_resource is not None and not exclude_source:
            raise ValueError("source_resource requires exclude_source=True.")
        groups = {}
        logger.info("Grouping reference dataset by resource %s", resource)
        for sample in tqdm(dataset, desc=f"Grouping references by {resource}", unit="recording"):
            label = resource_label(sample, resource)
            if sample.path is None:
                raise ValueError(f"Reference sample {sample.utt_id!r} has no audio path.")
            groups.setdefault(label, set()).add(sample.path)
        if not groups:
            raise ValueError("Reference dataset has no groups.")
        self.groups = {
            label: ReferenceSpeech(
                sorted(paths),
                aggregation=aggregation,
                prompt_seconds=prompt_seconds,
                prompt_builder=prompt_builder,
                allow_missing_streams=allow_missing_streams,
            )
            for label, paths in sorted(groups.items())
        }
        self.resource = resource
        self.mode = mode
        self.value = str(value) if value is not None else None
        self.seed = seed
        self.exclude_source = exclude_source
        self.source_resource = source_resource or resource
        catalogue = {label: [file_identity(path) for path in request.paths] for label, request in self.groups.items()}
        self.catalogue_digest = hashlib.sha256(json.dumps(catalogue, sort_keys=True).encode()).hexdigest()
        logger.info(
            "Reference catalogue: %d groups, %d recordings; mode=%s, seed=%d, exclude_source=%s",
            len(self.groups),
            sum(len(request.paths) for request in self.groups.values()),
            mode,
            seed,
            exclude_source,
        )
        if self.value is not None and self.value not in self.groups:
            raise ValueError(f"No recordings found for group {self.value!r}.")

    def select(self, sample: AudioSample) -> str:
        excluded = resource_label(sample, self.source_resource) if self.exclude_source else None
        candidates = [label for label in self.groups if label != excluded]
        if self.mode == "fixed":
            if self.value == excluded:
                raise ValueError(f"Fixed target equals the excluded source group for {sample.utt_id!r}.")
            return self.value
        if not candidates:
            raise ValueError(f"No eligible target groups for {sample.utt_id!r}.")
        key = json.dumps([self.seed, sample.split, sample.utt_id], ensure_ascii=False).encode()
        index = int.from_bytes(hashlib.sha256(key).digest(), "big") % len(candidates)
        return candidates[index]

    def assignment(self, sample: AudioSample, label: str) -> dict:
        request = self.groups[label]
        return {
            "utt_id": sample.utt_id,
            "split": sample.split,
            "source_path": str(sample.path),
            "resource": self.resource,
            "target_value": label,
            "mode": self.mode,
            "seed": self.seed,
            "exclude_source": self.exclude_source,
            "source_resource": self.source_resource if self.exclude_source else None,
            "references": [str(path) for path in request.paths],
            "aggregation": request.aggregation.method if request.aggregation else None,
            "catalogue_sha256": self.catalogue_digest,
        }
