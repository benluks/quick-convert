from __future__ import annotations

from pathlib import Path
from typing import Optional

from speechbrain.utils.checkpoints import Checkpointer


def find_embedding_model_ckpt(
    save_folder: str | Path,
    *,
    min_key: str | None = None,
    max_key: str | None = None,
) -> Path | None:
    """
    Return the path to embedding_model.ckpt inside the chosen SpeechBrain
    checkpoint directory.

    Selection behavior:
    - if neither min_key nor max_key is provided: pick the most recent ckpt
    - if min_key is provided: pick checkpoint with lowest value for that key
    - if max_key is provided: pick checkpoint with highest value for that key
    """
    save_folder = Path(save_folder)

    # No need to provide real recoverables just to search the checkpoint dirs.
    checkpointer = Checkpointer(
        checkpoints_dir=str(save_folder),
        recoverables={},
    )

    ckpt = checkpointer.find_checkpoint(
        min_key=min_key,
        max_key=max_key,
    )

    if ckpt is None:
        return None

    emb_ckpt = Path(ckpt.path) / "embedding_model.ckpt"
    if not emb_ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint was found at {ckpt.path}, but embedding_model.ckpt "
            f"does not exist inside it."
        )

    return emb_ckpt
