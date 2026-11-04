from os import PathLike
from pathlib import Path
from typing import Generic

from tqdm import tqdm

from quick_convert.data.base_dataset import BaseDataset
from quick_convert.data.sampling import shuffled_sample_indices
from quick_convert.pipelines.conversion_records import ConversionRecords, audio_destination, conversion_output_dir
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
        metadata=None,
        out_root="outputs",
        max_samples=None,
        sample_seed=0,
        **dataloader_kwargs,
    ):

        shuffled_sample_indices(0, max_samples, sample_seed)
        self.max_samples = max_samples
        self.sample_seed = sample_seed
        self.anonymizer = anonymizer
        self.dataset = dataset
        self.target_speaker = target_speaker
        self.out_dir = out_dir
        self.suffix = suffix
        self.overwrite = overwrite
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs
        self.metadata = {
            "system": type(anonymizer).__name__,
            "dataset": Path(getattr(dataset, "root", None) or "dataset").name,
            **dict(metadata or {}),
        }
        if getattr(dataset, "root", None):
            self.metadata.setdefault("source_root", str(dataset.root))
        self.out_root = out_root
        self.config = None

    def write_config(self, config):
        self.config = config

    def run(self, out_dir=None, target_speaker=None, suffix="", resynthesize=False, **kwargs):

        if not out_dir:
            out_dir = self.out_dir

        if resynthesize:
            anonymize_batch = self.anonymizer.resynthesize_batch
        else:
            if target_speaker is None:
                target_speaker = self.target_speaker
            if target_speaker is not None:
                self.anonymizer.set_target(target_speaker, **kwargs)
            anonymize_batch = self.anonymizer.anonymize_batch

        description = {
            "mode": "resynthesize" if resynthesize else "fixed" if target_speaker is not None else "backend-default",
            "value": target_speaker,
            "suffix": suffix or self.suffix,
            "settings": {key: str(value) if isinstance(value, PathLike) else value for key, value in kwargs.items()},
        }
        metadata = {**self.metadata, "target_speaker": target_speaker}
        out_dir = Path(out_dir) if out_dir else Path(conversion_output_dir(self.out_root, metadata, description))
        destinations = set()
        loader_kwargs = dict(self.dataloader_kwargs)
        if self.max_samples is not None:
            if "sampler" in loader_kwargs or loader_kwargs.get("shuffle") or loader_kwargs.get("drop_last"):
                raise ValueError("max_samples cannot be combined with sampler, shuffle or drop_last overrides.")
            indices = shuffled_sample_indices(len(self.dataset), self.max_samples, self.sample_seed)
            loader_kwargs["sampler"] = indices
            metadata["source_sampling"] = {
                "max_samples": self.max_samples,
                "seed": self.sample_seed,
                "population": len(self.dataset),
                "selected": len(indices),
            }
        with ConversionRecords(out_dir, metadata, description, config=self.config, overwrite=self.overwrite) as records:
            loader = self.dataset.make_dataloader(batch_size=self.batch_size, **loader_kwargs)
            for batch in tqdm(loader, desc=f"Anonymizing data into {out_dir}"):
                pending = []
                for index, sample in enumerate(batch):
                    out_path = audio_destination(out_dir, sample)
                    out_path = out_path.with_name(out_path.stem + (suffix or self.suffix) + ".wav")
                    if out_path in destinations:
                        raise ValueError(f"Duplicate conversion output: {out_path}")
                    destinations.add(out_path)
                    assignment = {"utt_id": sample.utt_id, "split": sample.split, **description}
                    if not records.should_skip(sample, out_path, assignment):
                        pending.append((index, sample, out_path, assignment))
                if not pending:
                    continue
                generated = anonymize_batch(batch)
                if len(generated) != len(batch):
                    raise ValueError(f"Anonymizer returned {len(generated)} outputs for a batch of size {len(batch)}.")
                for index, sample, out_path, assignment in pending:
                    waveform = generated.waveform(index)
                    records.save(sample, out_path, assignment, waveform, generated.sample_rate)
        return out_dir
