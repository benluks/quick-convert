from .asrbn import ASRBNAnonymizer
from .knnvc import KNNVCAnonymizer
from .nac import NACAnonymizer

from .pipeline import AnonymizationPipeline


__all__ = [
    "ASRBNAnonymizer",
    "KNNVCAnonymizer",
    "NACAnonymizer",
    "AnonymizationPipeline",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=str, default=ASRBNAnonymizer)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    anon = args.pipeline()