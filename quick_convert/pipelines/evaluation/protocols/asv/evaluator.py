from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import torch
from tqdm import tqdm

from quick_convert.components.speaker import SpeakerEncoder
from quick_convert.data import BaseDataset

from ...metrics.eer import compute_eer
from .scoring import cosine_score_matrix
from .trials import ASVTrial


class ASVEvaluator:
    """Evaluate mean enrollment speaker vectors against test utterances.

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
    ) -> None:
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

        return results

    @torch.inference_mode()
    def extract_embeddings(
        self,
        dataset: BaseDataset,
        *,
        desc: str = "Extracting embeddings",
        cache_name: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, str]]:

        new_since_cache = 0
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
        else:
            embeddings = {}
            speakers = {}

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

            if cache_path is not None and new_since_cache >= self.cache_every:
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "cache_key": self.cache_key,
                        "embeddings": embeddings,
                        "speakers": speakers,
                    },
                    cache_path,
                )
                new_since_cache = 0

        if cache_path is not None and new_since_cache > 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "cache_key": self.cache_key,
                    "embeddings": embeddings,
                    "speakers": speakers,
                },
                cache_path,
            )

        return {key: embeddings[key] for key in sorted(seen)}, {key: speakers[key] for key in sorted(seen)}

    @staticmethod
    def aggregate_enrollment(
        embeddings: dict[str, torch.Tensor],
        speakers: dict[str, str],
    ) -> dict[str, torch.Tensor]:
        grouped: dict[str, list[torch.Tensor]] = defaultdict(list)

        for utt_id, embedding in embeddings.items():
            grouped[speakers[utt_id]].append(embedding)

        return {speaker: torch.stack(values).mean(dim=0) for speaker, values in grouped.items()}

    def evaluate(
        self,
        enroll_dataset: BaseDataset,
        test_dataset: BaseDataset,
        trials: Iterable[ASVTrial] | None = None,
    ) -> dict[str, float | int]:
        enroll_utt_embeddings, enroll_speakers = self.extract_embeddings(
            enroll_dataset,
            desc="Extracting enrollment embeddings",
            cache_name="enroll",
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
        )

        if trials is None:
            trials = self.make_trials(
                enroll_embeddings,
                test_speakers,
            )

        trials = list(trials)

        enroll_ids = list(enroll_embeddings)
        test_ids = list(test_embeddings)

        enroll_index = {speaker: index for index, speaker in enumerate(enroll_ids)}

        test_index = {utt_id: index for index, utt_id in enumerate(test_ids)}

        enroll_matrix = torch.stack([enroll_embeddings[speaker] for speaker in enroll_ids])

        test_matrix = torch.stack([test_embeddings[utt_id] for utt_id in test_ids])

        scores = cosine_score_matrix(
            enroll_matrix,
            test_matrix,
        )

        target_scores = []
        non_target_scores = []

        for trial in tqdm(
            trials,
            desc="Scoring ASV trials",
            unit="trial",
        ):
            if trial.enroll_id not in enroll_index or trial.test_id not in test_index:
                raise ValueError(f"Unknown enrollment or test ID in trial: {trial}")
            score = scores[
                enroll_index[trial.enroll_id],
                test_index[trial.test_id],
            ]

            if trial.target:
                target_scores.append(score)
            else:
                non_target_scores.append(score)

        if not target_scores or not non_target_scores:
            raise ValueError("ASV evaluation requires both target and non-target trials.")
        target_scores = torch.stack(target_scores)
        non_target_scores = torch.stack(non_target_scores)

        eer, threshold = compute_eer(
            target_scores,
            non_target_scores,
        )

        return {
            "eer": eer,
            "eer_threshold": threshold,
            "num_target_trials": len(target_scores),
            "num_non_target_trials": len(non_target_scores),
        }

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
