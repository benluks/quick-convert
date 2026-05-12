from dataclasses import replace
from os import PathLike
from pathlib import Path
from typing import Generic

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
        **kwargs,
    ):

        self.anonymizer = anonymizer
        self.dataset = dataset
        self.target_speaker = target_speaker
        self.out_dir = out_dir
        self.suffix = suffix
        self.overwrite = overwrite

    def get_feature_providers(self):
        return [
            *getattr(self.anonymizer, "feature_providers", []),
        ]

    def provide_sample_features(self, sample):
        features = dict(getattr(sample, "features", {}) or {})

        for provider in self.get_feature_providers():
            if provider.key in features:
                continue
            features.update(provider.provide_sample(sample))

        return replace(sample, features=features)

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
            sample = self.provide_sample_features(sample)
            split = sample.split or ""
            out_path = Path(out_dir) / split / f"{Path(sample.path).stem}{self.suffix}.wav"
            if out_path.exists() and not self.overwrite:
                continue

            wav_conv = anonymize_fn(sample)
            torchaudio.save(str(out_path), wav_conv, self.anonymizer.sr)
