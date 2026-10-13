"""Fit target statistics from an explicitly selected training dataset."""

from .fit import FitPipeline


class FitProsodyPipeline(FitPipeline):
    """Compatibility constructor for older run configs; use FitPipeline in new ones."""

    def __init__(self, dataset, targets, out_path):
        super().__init__(dataset, targets, out_path, resources=("prosody", "spkid"))
