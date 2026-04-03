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
        **kwargs,
    ):

        self.anonymizer = anonymizer
        self.dataset = dataset
        self.target_speaker = target_speaker
        self.out_dir = out_dir
        self.suffix = suffix

    def process_dir():
        pass

    def run(
        self, out_dir=None, target_speaker=None, suffix="", resynthesize=False, **kwargs
    ):

        if not out_dir:
            out_dir = self.out_dir

        if resynthesize:
            anonymize_fn = self.anonymizer.resynthesize
        else:
            if not target_speaker:
                target_speaker = self.target_speaker
            self.anonymizer.set_target(target_speaker, **kwargs)
            anonymize_fn = self.anonymizer.convert

        out_dir = Path(out_dir)
        for split in self.dataset.splits or [""]:
            (out_dir / split).mkdir(parents=True, exist_ok=True)

        for row in tqdm(
            self.dataset.rows,
            desc=f"Anonymizing data from {self.dataset.root} into {str(out_dir)}",
        ):
            split = row.split or ""
            out_path = Path(out_dir) / split / f"{Path(row.path).stem}{self.suffix}.wav"
            wav_conv = anonymize_fn(row.path)
            torchaudio.save(str(out_path), wav_conv, self.anonymizer.sr)
