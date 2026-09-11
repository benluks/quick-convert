from os import PathLike
from pathlib import Path
from typing import Generic

import torchaudio
from tqdm import tqdm

from quick_convert.data.base_dataset import BaseDataset
from quick_convert.systems.anonymization import BaseAnonymizer
from quick_convert.systems.anonymization.targets import T_Target


class AnonymizationPipeline(Generic[T_Target]):
    def __init__(
        self,
        anonymizer: BaseAnonymizer,
        dataset: BaseDataset,
        target_speaker=None,
        out_dir: PathLike = None,
        suffix="",
        overwrite=False,
        batch_size=1,
        **dataloader_kwargs,
    ):

        self.anonymizer = anonymizer
        self.dataset = dataset
        self.target_speaker = target_speaker
        self.out_dir = out_dir
        self.suffix = suffix
        self.overwrite = overwrite
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs

    def run(self, out_dir=None, target_speaker=None, suffix="", resynthesize=False, **kwargs):

        if not out_dir:
            out_dir = self.out_dir

        if resynthesize:
            anonymize_batch = self.anonymizer.resynthesize_batch
        else:
            if not target_speaker:
                target_speaker = self.target_speaker
            if target_speaker is not None:
                self.anonymizer.set_target(target_speaker, **kwargs)
            anonymize_batch = self.anonymizer.anonymize_batch

        out_dir = Path(out_dir)
        for split in self.dataset.splits or [""]:
            (out_dir / split).mkdir(parents=True, exist_ok=True)

        loader = self.dataset.make_dataloader(batch_size=self.batch_size, **self.dataloader_kwargs)
        for batch in tqdm(
            loader,
            desc=f"Anonymizing data from {self.dataset.root} into {str(out_dir)}",
        ):
            generated = anonymize_batch(batch)
            if len(generated) != len(batch):
                raise ValueError(
                    f"Anonymizer returned {len(generated)} outputs for a batch of size {len(batch)}."
                )

            for index, sample in enumerate(batch):
                waveform = generated.waveform(index)
                split = sample.split or ""
                out_path = Path(out_dir) / split / f"{Path(sample.path).stem}{self.suffix}.wav"
                if out_path.exists() and not self.overwrite:
                    continue

                torchaudio.save(str(out_path), waveform.unsqueeze(0).cpu(), generated.sample_rate)
