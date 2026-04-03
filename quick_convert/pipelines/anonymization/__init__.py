from .nac import NACAnonymizer
from .knnvc import KNNVCAnonymizer
from .pipeline import AnonymizationPipeline

__all__ = ["NACAnonymizer", "KNNVCAnonymizer", "AnonymizationPipeline"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=str, default="checkpoints/nac_models")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()
