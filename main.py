from pathlib import Path

import torch
import torchaudio

from quick_convert.pipelines.anonymization.pipeline import Pipeline

target_spk_m = sorted(
    map(
        str,
        Path("/cfs/collections/librispeech/LibriSpeech/train-clean-100/6081/").glob(
            "*/*.flac"
        ),
    )
)

target_spk_f = sorted(
    map(
        str,
        Path("/cfs/collections/librispeech/LibriSpeech/train-clean-100/1069/").glob(
            "*/*.flac"
        ),
    )
)


# pipeline = Pipeline("knnvc")
# pipeline.anonymize_dir("audio/stutter/f", )

