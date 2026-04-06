from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from .train_speaker_embeddings import SpeakerBrain, dataio_prep
from .prepare_dataset import prepare_asv_csvs


def train_asv(
    hparams_file: str | Path,
    overrides: Sequence[str] | None = None,
) -> None:
    torch.backends.cudnn.benchmark = True

    hparams_file = str(Path(hparams_file).expanduser().resolve())
    overrides = list(overrides or [])

    # Match the structure SpeechBrain expects from sb.parse_arguments
    run_opts: dict = {}
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Use your own prep function here.
    run_on_main(
        prepare_asv_csvs,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams.get("splits"),
            "split_ratio": hparams.get("split_ratio", [90, 10]),
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams.get("skip_prep", False),
            "random_segment": hparams.get("random_chunk", False),
            "file_pattern": hparams.get("file_pattern", "*.wav"),
        },
    )

    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    train_data, valid_data, label_encoder = dataio_prep(hparams)

    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
