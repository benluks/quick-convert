from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Any

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.metric_stats import EER, minDCF

from .speaker_verification_cosine import (
    compute_embedding_loop,
    dataio_prep,
    get_verification_scores,
)


def eval_asv(
    hparams_file: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    hparams_file = str(Path(hparams_file).expanduser().resolve())
    overrides = overrides or {}

    run_opts = {"device": "cuda" if torch.cuda.is_available() else "cpu"}

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    train_dataloader, enrol_dataloader, test_dataloader = dataio_prep(hparams)

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()
    hparams["embedding_model"].eval()
    hparams["embedding_model"].to(run_opts["device"])

    enrol_dict = compute_embedding_loop(enrol_dataloader, hparams, run_opts)
    test_dict = compute_embedding_loop(test_dataloader, hparams, run_opts)

    train_dict = None
    if hparams["score_norm"]:
        train_dict = compute_embedding_loop(train_dataloader, hparams, run_opts)

    with open(hparams["verification_file"], encoding="utf-8") as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores, negative_scores = get_verification_scores(
        veri_test=veri_test,
        params=hparams,
        enrol_dict=enrol_dict,
        test_dict=test_dict,
        train_dict=train_dict,
    )

    positive_scores = torch.stack([s.detach().cpu() for s in positive_scores]).float()
    negative_scores = torch.stack([s.detach().cpu() for s in negative_scores]).float()

    print("positive_scores.shape:", positive_scores.shape)
    print("negative_scores.shape:", negative_scores.shape)

    eer, eer_th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    min_dcf, min_dcf_th = minDCF(
        torch.tensor(positive_scores),
        torch.tensor(negative_scores),
    )

    results = {
        "eer": float(eer),
        "eer_percent": float(eer * 100),
        "eer_threshold": float(eer_th),
        "min_dcf": float(min_dcf),
        "min_dcf_percent": float(min_dcf * 100),
        "min_dcf_threshold": float(min_dcf_th),
    }

    # write results
    output_dir = Path(hparams["output_folder"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "results.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to {metrics_path}")

    return results
