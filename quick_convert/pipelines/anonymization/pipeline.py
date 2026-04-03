from os import PathLike
from pathlib import Path

import torchaudio
from tqdm import tqdm

from .base_anonymizer import BaseAnonymizer
from quick_convert.data.base_dataset import BaseDataset


class AnonymizationPipeline:
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

    def run(self, out_dir=None, target_speaker=None, suffix="", **kwargs):

        if not out_dir:
            out_dir = self.out_dir
        if not target_speaker:
            target_speaker = self.target_speaker

        if kwargs.get("resynthesize", False):
            anonymize_fn = self.anonymizer.resynthesize
        else:
            self.anonymizer.set_target(target_speaker, **kwargs)
            anonymize_fn = self.anonymizer.convert

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for fpath in tqdm(
            self.dataset,
            desc=f"Anonymizing data from {self.dataset.root} into {str(out_dir)}",
        ):
            out_path = Path(out_dir) / f"{Path(fpath).stem}{self.suffix}.wav"
            wav_conv = anonymize_fn(fpath)
            torchaudio.save(str(out_path), wav_conv, self.anonymizer.sr)
