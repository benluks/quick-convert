from dataclasses import replace
from os import PathLike
from pathlib import Path
from typing import Generic, Optional

import torchaudio
from tqdm import tqdm

from .targets import T_Target

from .base_anonymizer import BaseAnonymizer
from quick_convert.data.base_dataset import BaseDataset


class AnonymizationPipeline(Generic[T_Target]):
    def __init__(
        self,
        anonymizer: BaseAnonymizer,
        dataset: BaseDataset,
        target_speaker=None,
        out_dir: PathLike = None,
        suffix="",
        overwrite=False,
        feature_providers: Optional[list] = None,
        batch_size=1,
        **dataloader_kwargs,
    ):

        self.anonymizer = anonymizer
        self.dataset = dataset
        self.target_speaker = target_speaker
        self.out_dir = out_dir
        self.suffix = suffix
        self.overwrite = overwrite
        self.feature_providers = feature_providers or []
        self.is_batched = batch_size > 1

        if self.is_batched:
            self.dataloader = self.dataset.make_dataloader(batch_size, **dataloader_kwargs)

    def provide_features(self, sample_or_batch):
        features = dict(getattr(sample_or_batch, "features", {}) or {})
        provider_fn = "provide_batch" if self.is_batched else "provide_sample"

        for provider in self.feature_providers:
            # this shouldn't happen because feature extraction shouldn't exist at any other point in the pipeline
            if provider.key in features:
                continue
            features.update({provider.key: getattr(provider, provider_fn)(sample_or_batch)})

        return replace(sample_or_batch, features=features)

    def process_dir():
        pass

    def run(self, out_dir=None, target_speaker=None, suffix="", resynthesize=False, **kwargs):

        if not out_dir:
            out_dir = self.out_dir

        if resynthesize:
            anonymize_fn = self.anonymizer.resynthesize
        else:
            if not target_speaker:
                target_speaker = self.target_speaker
            if target_speaker is not None:
                self.anonymizer.set_target(target_speaker, **kwargs)
            anonymize_fn = self.anonymizer.anonymize

        out_dir = Path(out_dir)
        for split in self.dataset.splits or [""]:
            (out_dir / split).mkdir(parents=True, exist_ok=True)

        for sample in tqdm(
            self.dataset,
            desc=f"Anonymizing data from {self.dataset.root} into {str(out_dir)}",
        ):
            sample = self.provide_features(sample)
            split = sample.split or ""
            out_path = Path(out_dir) / split / f"{Path(sample.path).stem}{self.suffix}.wav"
            if out_path.exists() and not self.overwrite:
                continue

            wav_conv = anonymize_fn(sample)
            torchaudio.save(str(out_path), wav_conv, self.anonymizer.sr)
