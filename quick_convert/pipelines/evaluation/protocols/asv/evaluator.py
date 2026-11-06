from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import torch
from tqdm import tqdm

from quick_convert.components.speaker import SpeakerEmbeddingAggregation, SpeakerEncoder
from quick_convert.components.speaker.aggregation import audio_durations
from quick_convert.data import BaseDataset

from ...provenance import describe_component, write_provenance
from .scoring import evaluate_trials
from .trials import ASVTrial


class ASVEvaluator:
    """Evaluate aggregated enrollment speaker vectors against test utterances.

    Cache keys are caller-managed identities for weights, preprocessing, and
    dataset contents. Caching is disabled unless explicitly requested.
    """

    def __init__(
        self,
        enroll_dataset: BaseDataset,
        test_dataset: BaseDataset,
        speaker_encoder: SpeakerEncoder,
        out_dir: PathLike,
        batch_size: int = 32,
        num_workers: int = 0,
        spkid_key: str = "spkid",
        cache_embeddings: bool = False,
        cache_key: str | None = None,
        cache_every: int = 100,
        enrollment_aggregation: SpeakerEmbeddingAggregation | None = None,
    ) -> None:
        self._resolved_config = None
        self.enroll_dataset = enroll_dataset
        self.test_dataset = test_dataset
        self.speaker_encoder = speaker_encoder
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.spkid_key = spkid_key
        self.cache_embeddings = cache_embeddings
        self.cache_key = cache_key
        if cache_embeddings and not cache_key:
            raise ValueError("Caching requires an explicit cache_key identifying the model and dataset revision.")
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative.")

        if cache_every < 1:
            raise ValueError("cache_every must be at least 1.")

        self.cache_every = cache_every
        if enrollment_aggregation is not None and not isinstance(enrollment_aggregation, SpeakerEmbeddingAggregation):
            raise TypeError("enrollment_aggregation must be a SpeakerEmbeddingAggregation policy.")
        self.enrollment_aggregation = (
            enrollment_aggregation if enrollment_aggregation is not None else SpeakerEmbeddingAggregation()
        )

    def write_config(self, config: str) -> None:
        """Capture the resolved configuration supplied by the CLI runner."""
        self._resolved_config = config

    def run(self) -> dict[str, float | int]:
        results = self.evaluate(
            enroll_dataset=self.enroll_dataset,
            test_dataset=self.test_dataset,
        )

        self.out_dir.mkdir(parents=True, exist_ok=True)

        with (self.out_dir / "results.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=2)

        write_provenance(
            self.out_dir,
            protocol="asv",
            config=self._resolved_config,
            pairing="all_enrollment_speakers_by_test_utterance",
            datasets={
                "enrollment": describe_component(self.enroll_dataset),
                "test": describe_component(self.test_dataset),
            },
            models={"speaker_encoder": describe_component(self.speaker_encoder)},
            metrics=[{"key": "eer", "type": "discrete_eer", "score": "cosine"}],
            enrollment_aggregation=self.enrollment_aggregation.method,
            speaker_id_resource=self.spkid_key,
            embedding_cache_key=self.cache_key if self.cache_embeddings else None,
        )
        return results

    @torch.inference_mode()
    def extract_embeddings(
        self,
        dataset: BaseDataset,
        *,
        desc: str = "Extracting embeddings",
        cache_name: str | None = None,
        durations: dict[str, float] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
        """Extract embeddings and optionally populate valid audio durations.

        The return value remains two mappings. When durations are requested,
        loaded length/rate metadata enriches legacy caches without re-encoding.
        """
        new_since_cache = 0
        metadata_since_cache = 0
        cache_path = (
            self.out_dir / "cache" / f"{cache_name}.pt" if self.cache_embeddings and cache_name is not None else None
        )

        if cache_path is not None and cache_path.exists():
            cache = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
            if cache.get("cache_key") != self.cache_key:
                raise ValueError("ASV cache_key changed; use a fresh output directory or remove the old cache.")
            embeddings = cache["embeddings"]
            speakers = cache["speakers"]
            cached_durations = dict(cache.get("durations", {}))
        else:
            embeddings = {}
            speakers = {}
            cached_durations = {}

        loader = dataset.make_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        self.speaker_encoder.eval()
        seen = set()

        for batch in tqdm(
            loader,
            desc=desc,
            unit="batch",
        ):
            for sample in batch:
                if sample.utt_id in seen:
                    raise ValueError(f"Duplicate utterance ID: {sample.utt_id!r}")
                seen.add(sample.utt_id)
            if (
                durations is not None
                and getattr(batch, "lengths", None) is not None
                and getattr(batch, "sample_rates", None) is not None
            ):
                for sample, duration in zip(batch, audio_durations(batch), strict=True):
                    uid = sample.utt_id
                    if uid in cached_durations and cached_durations[uid] != duration:
                        raise ValueError(
                            f"Recording duration changed for cached utterance {uid!r}; rebuild its embedding cache."
                        )
                    if uid not in cached_durations:
                        cached_durations[uid] = duration
                        if uid in embeddings:
                            metadata_since_cache += 1
            if all(sample.utt_id in embeddings for sample in batch):
                continue

            output = self.speaker_encoder.encode_batch(batch)
            values = output.values.detach().cpu()

            if values.ndim != 2 or not torch.isfinite(values).all() or torch.any(values.norm(dim=-1) == 0):
                raise ValueError("Speaker embeddings must be finite, nonzero vectors with shape [B, D].")
            if len(values) != len(batch):
                raise ValueError(
                    f"Speaker encoder returned {len(values)} embeddings for a batch of {len(batch)} samples."
                )

            num_new = 0

            for sample, embedding in zip(
                batch,
                values,
                strict=True,
            ):
                if sample.utt_id in embeddings:
                    continue

                if self.spkid_key not in sample.resources.keys():
                    raise ValueError(f"Sample {sample.utt_id!r} has no {self.spkid_key!r} resource.")

                spkid = sample.resources[self.spkid_key].value

                if spkid is None:
                    raise ValueError(f"Sample {sample.utt_id!r} has no speaker ID.")

                embeddings[sample.utt_id] = embedding
                speakers[sample.utt_id] = str(spkid)
                num_new += 1

            new_since_cache += num_new

            if cache_path is not None and new_since_cache + metadata_since_cache >= self.cache_every:
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "cache_key": self.cache_key,
                        "embeddings": embeddings,
                        "speakers": speakers,
                        "durations": cached_durations,
                    },
                    cache_path,
                )
                new_since_cache = 0
                metadata_since_cache = 0

        if cache_path is not None and new_since_cache + metadata_since_cache > 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "cache_key": self.cache_key,
                    "embeddings": embeddings,
                    "speakers": speakers,
                    "durations": cached_durations,
                },
                cache_path,
            )

        if durations is not None:
            durations.update({key: cached_durations[key] for key in sorted(seen) if key in cached_durations})
        return {key: embeddings[key] for key in sorted(seen)}, {key: speakers[key] for key in sorted(seen)}

    @staticmethod
    def aggregate_enrollment(
        embeddings: dict[str, torch.Tensor],
        speakers: dict[str, str],
        *,
        aggregation: SpeakerEmbeddingAggregation | None = None,
        durations: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Group speaker recordings and apply the shared embedding policy."""
        if aggregation is not None and not isinstance(aggregation, SpeakerEmbeddingAggregation):
            raise TypeError("aggregation must be a SpeakerEmbeddingAggregation policy.")
        aggregation = aggregation if aggregation is not None else SpeakerEmbeddingAggregation()
        grouped: dict[str, list[str]] = defaultdict(list)
        for utt_id in embeddings:
            grouped[speakers[utt_id]].append(utt_id)
        enrollment = {}
        for speaker, utt_ids in grouped.items():
            recording_durations = None
            if durations is not None and all(utt_id in durations for utt_id in utt_ids):
                recording_durations = [durations[utt_id] for utt_id in utt_ids]
            if aggregation.method == "duration_weighted" and recording_durations is None:
                raise ValueError(
                    f"Duration-weighted enrollment for speaker {speaker!r} requires durations for every recording. "
                    "Load enrollment audio to recover missing duration metadata."
                )
            enrollment[speaker] = aggregation(
                torch.stack([embeddings[utt_id] for utt_id in utt_ids]), durations=recording_durations
            )
        return enrollment

    def evaluate(
        self,
        enroll_dataset: BaseDataset,
        test_dataset: BaseDataset,
        trials: Iterable[ASVTrial] | None = None,
    ) -> dict[str, float | int]:
        enroll_durations = {} if self.enrollment_aggregation.method == "duration_weighted" else None
        enroll_utt_embeddings, enroll_speakers = self.extract_embeddings(
            enroll_dataset,
            desc="Extracting enrollment embeddings",
            cache_name="enroll",
            durations=enroll_durations,
        )

        test_embeddings, test_speakers = self.extract_embeddings(
            test_dataset,
            desc="Extracting test embeddings",
            cache_name="test",
        )

        if not enroll_utt_embeddings or not test_embeddings:
            raise ValueError("Enrollment and test datasets must both be nonempty.")

        enroll_embeddings = self.aggregate_enrollment(
            enroll_utt_embeddings,
            enroll_speakers,
            aggregation=self.enrollment_aggregation,
            durations=enroll_durations,
        )

        if trials is None:
            trials = self.make_trials(
                enroll_embeddings,
                test_speakers,
            )

        return evaluate_trials(
            enroll_embeddings,
            test_embeddings,
            tqdm(trials, desc="Scoring ASV trials", unit="trial"),
        )

    @staticmethod
    def make_trials(
        enrollment: dict[str, torch.Tensor],
        test_speakers: dict[str, str],
    ) -> list[ASVTrial]:
        trials = []

        for enroll_speaker in tqdm(
            enrollment,
            desc="Generating ASV trials",
            unit="speaker",
        ):
            for test_id, test_speaker in test_speakers.items():
                trials.append(
                    ASVTrial(
                        enroll_id=enroll_speaker,
                        test_id=test_id,
                        target=enroll_speaker == test_speaker,
                    )
                )

        return trials
