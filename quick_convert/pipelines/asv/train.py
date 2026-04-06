from __future__ import annotations

from pathlib import Path
from typing import Sequence

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml

from .train_speaker_embeddings import SpeakerBrain, dataio_prep


def train_asv(
    hparams_file: str | Path,
    overrides: Sequence[str] | None = None,
) -> None:
    torch.backends.cudnn.benchmark = True

    hparams_file = str(Path(hparams_file).expanduser().resolve())
    overrides = overrides or {}

    run_opts: dict = {}

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

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
