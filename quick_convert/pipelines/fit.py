"""Reusable orchestration for fitting a transform from dataset records."""

from pathlib import Path

from quick_convert.components.fitting import FittedTransform, save_fitted_state


class FitPipeline:
    """Fit on selected records and save state, without domain-specific behavior.

    Without ``resources``, pass each dataset sample to the transform (as for an
    Indexer). With resources, pass a tuple of loaded values in the configured
    order (as for a prosody transform). Dataset load settings remain explicit.
    """

    def __init__(self, dataset, transform: FittedTransform, out_path, resources=None):
        self.dataset = dataset
        self.transform = transform
        self.out_path = Path(out_path)
        self.resources = tuple(resources or ())

    def run(self):
        self.transform.fit(_Records(self.dataset, self.resources))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        save_fitted_state(self.transform, self.out_path)


class _Records:
    """Re-iterable resource selection for multi-pass fitting without a cache."""

    def __init__(self, dataset, resources):
        self.dataset, self.resources = dataset, resources

    def __iter__(self):
        for sample in self.dataset:
            if not self.resources:
                yield sample
                continue
            values = tuple(sample.resources[name].value for name in self.resources)
            if any(value is None for value in values):
                raise ValueError(f"Fitting resources {self.resources} must be loaded for {sample.utt_id!r}.")
            yield values
