"""Training-fitted normalization and discretization of acoustic measurements."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import torch


@dataclass(frozen=True)
class ProsodyMeasurements:
    """Named view of existing unpadded acoustic sidecars; physical units remain intact.

    Construction validates the on-disk schema once. Views preserve the caller's
    tensor; no peak normalization, copying, or loss conventions are applied.
    """

    values: torch.Tensor
    COLUMNS: ClassVar[tuple[str, ...]] = ("time_seconds", "f0_hz", "voiced", "rms", "voicing_score")

    def __post_init__(self):
        values = self.values
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise TypeError("Prosody measurements must be a floating-point tensor.")
        if values.ndim != 2 or values.shape[1] != len(self.COLUMNS) or not len(values):
            raise ValueError("Expected nonempty [T, 5] prosody measurements.")
        if not torch.isfinite(values).all():
            raise ValueError("Prosody measurements must be finite.")
        if torch.any(self.timestamps < 0) or torch.any(self.timestamps[1:] <= self.timestamps[:-1]):
            raise ValueError("Prosody timestamps must be nonnegative and strictly increasing.")
        if torch.any((values[:, 2] != 0) & (values[:, 2] != 1)):
            raise ValueError("Voiced flags must be zero or one.")
        if torch.any(self.rms < 0) or torch.any(self.f0_hz < 0) or torch.any(self.f0_hz[self.voiced] <= 0):
            raise ValueError("RMS/F0 must be nonnegative and voiced F0 positive.")

    @property
    def timestamps(self):
        return self.values[:, 0]

    @property
    def f0_hz(self):
        return self.values[:, 1]

    @property
    def voiced(self):
        return self.values[:, 2].bool()

    @property
    def rms(self):
        return self.values[:, 3]

    @property
    def voicing_score(self):
        return self.values[:, 4]


class _Moments:
    """Merge frame statistics without retaining frames (population variance)."""

    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, values):
        count = values.numel()
        if not count:
            return
        mean = values.mean().item()
        m2 = ((values - mean).square().sum()).item()
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def statistics(self):
        if not self.count:
            raise ValueError("Cannot fit pitch statistics without voiced frames.")
        return torch.tensor([self.mean, max((self.m2 / self.count) ** 0.5, 1e-6)], dtype=torch.float64)


class _Bins:
    """Exact extrema and uniform frame sampling via bounded random priorities."""

    def __init__(self, capacity, generator):
        self.capacity, self.generator = capacity, generator
        self.values = torch.empty(0, dtype=torch.float64)
        self.priorities = torch.empty(0, dtype=torch.float64)
        self.minimum, self.maximum = float("inf"), -float("inf")
        self.count = 0

    def update(self, values):
        if not values.numel():
            return
        self.minimum = min(self.minimum, values.min().item())
        self.maximum = max(self.maximum, values.max().item())
        self.count += values.numel()
        if not self.capacity:
            return
        for chunk in values.split(65536):
            priorities = torch.rand(len(chunk), generator=self.generator, dtype=torch.float64)
            if len(self.values) == self.capacity:
                candidates = priorities < self.priorities.max()
                chunk, priorities = chunk[candidates], priorities[candidates]
                if not len(chunk):
                    continue
            values = torch.cat((self.values, chunk))
            priorities = torch.cat((self.priorities, priorities))
            if len(values) > self.capacity:
                indices = priorities.topk(self.capacity, largest=False).indices
                values, priorities = values[indices], priorities[indices]
            self.values, self.priorities = values, priorities

    def edges(self, strategy, count):
        if strategy == "quantile":
            edges = torch.quantile(self.values, torch.arange(1, count, dtype=torch.float64) / count)
        else:
            edges = torch.linspace(self.minimum, self.maximum, count + 1, dtype=torch.float64)[1:-1]
        return torch.unique_consecutive(edges)


class ProsodyTransform:
    """Fit and apply log-pitch/log-RMS targets; speaker statistics use voiced frames.

    ``fit`` consumes (measurement tensor, speaker ID) pairs from training only.
    All frames contribute equally. Unknown speakers raise rather than silently
    changing normalization. Statistics stream; quantiles use a bounded uniform sample.
    """

    def __init__(
        self,
        pitch_center="speaker",
        pitch_scale="speaker",
        energy_center="utterance",
        bins="quantile",
        n_bins=32,
        quantile_sample_size=100000,
        seed=0,
    ):
        if pitch_center not in {"global", "speaker"} or pitch_scale not in {"global", "speaker"}:
            raise ValueError("Pitch center and scale must be global or speaker.")
        if energy_center not in {"global", "utterance"}:
            raise ValueError("Energy center must be global or utterance.")
        if bins not in {"quantile", "equal_width"} or n_bins < 2:
            raise ValueError("Use quantile or equal_width bins and n_bins >= 2.")
        if not isinstance(quantile_sample_size, int) or quantile_sample_size < 1:
            raise ValueError("quantile_sample_size must be a positive integer.")
        self.config = dict(
            pitch_center=pitch_center,
            pitch_scale=pitch_scale,
            energy_center=energy_center,
            bins=bins,
            n_bins=n_bins,
            quantile_sample_size=quantile_sample_size,
            seed=seed,
        )
        self.state = None

    @staticmethod
    def _measurements(values):
        measurements = ProsodyMeasurements(values)
        return measurements.f0_hz.clamp_min(1e-8).log(), measurements.rms.clamp_min(1e-8).log(), measurements.voiced

    def fit(self, records):
        """Fit two CPU passes with bounded memory; iterators spool to temporary disk.

        Re-iterable inputs (including FitPipeline datasets) are read twice without
        copying the corpus. A one-shot iterator is preserved using temporary
        sidecars, automatically removed after fitting or failure.
        """
        with TemporaryDirectory(prefix="quick-convert-prosody-") as scratch:
            spool = Path(scratch) if isinstance(records, Iterator) else None
            pitch_stats, energy_stats, speakers = _Moments(), _Moments(), {}
            count = 0
            for count, (values, speaker) in enumerate(records, start=1):
                values = values.detach().cpu().double()
                speaker = str(speaker)
                pitch, energy, voiced = self._measurements(values)
                pitch = pitch[voiced]
                pitch_stats.update(pitch)
                energy_stats.update(energy)
                if len(pitch):
                    speakers.setdefault(speaker, _Moments()).update(pitch)
                if spool is not None:
                    torch.save((values, speaker), spool / f"{count}.pt")
            if not count:
                raise ValueError("Cannot fit empty training data.")
            self.state = {
                "version": 1,
                "config": self.config.copy(),
                "pitch": pitch_stats.statistics(),
                "energy": energy_stats.statistics(),
                "speakers": {speaker: moments.statistics() for speaker, moments in speakers.items()},
            }
            if spool is not None:
                records = (torch.load(spool / f"{i}.pt", weights_only=True) for i in range(1, count + 1))
            generator = torch.Generator().manual_seed(self.config["seed"])
            capacity = self.config["quantile_sample_size"] if self.config["bins"] == "quantile" else 0
            bins = {name: _Bins(capacity, generator) for name in ("pitch", "energy")}
            observed_records = 0
            for values, speaker in records:
                observed_records += 1
                result = self.transform(values.detach().cpu().double(), speaker)
                bins["pitch"].update(result["pitch"][result["voiced"]])
                bins["energy"].update(result["energy"])
            if observed_records != count:
                raise ValueError("Fitting records changed between passes.")
            for name, accumulator in bins.items():
                self.state[name + "_edges"] = accumulator.edges(self.config["bins"], self.config["n_bins"])
            self.state["fitting"] = {
                "records": count,
                "frames": {name: accumulator.count for name, accumulator in bins.items()},
                "sampled_frames": {name: len(accumulator.values) for name, accumulator in bins.items()},
            }
        return self

    def transform(self, values, speaker=None):
        """Return continuous targets, voicing mask, and fitted categorical targets."""
        if speaker is None:
            values, speaker = values
        if self.state is None:
            raise RuntimeError("Fit or load training statistics before transforming targets.")
        pitch, energy, voiced = self._measurements(values)
        global_pitch = self.state["pitch"].to(values)
        if "speaker" in (self.config["pitch_center"], self.config["pitch_scale"]) and voiced.any():
            if str(speaker) not in self.state["speakers"]:
                raise ValueError(f"No voiced training statistics for speaker {speaker!r}.")
            speaker_pitch = self.state["speakers"][str(speaker)].to(values)
        else:
            speaker_pitch = global_pitch
        center = speaker_pitch if self.config["pitch_center"] == "speaker" else global_pitch
        scale = speaker_pitch if self.config["pitch_scale"] == "speaker" else global_pitch
        pitch = torch.where(voiced, (pitch - center[0]) / scale[1], 0)
        energy_stats = self.state["energy"].to(values)
        energy_center = energy.mean() if self.config["energy_center"] == "utterance" else energy_stats[0]
        energy = (energy - energy_center) / energy_stats[1]
        result = dict(pitch=pitch, energy=energy, voiced=voiced)
        for name in ("pitch", "energy"):
            if name + "_edges" in self.state:
                labels = torch.bucketize(result[name].contiguous(), self.state[name + "_edges"].to(values))
                result[name + "_bins"] = labels
        return result

    def state_dict(self):
        if self.state is None:
            raise RuntimeError("No fitted statistics to save.")
        return self.state

    def load_state_dict(self, state):
        config = {"quantile_sample_size": 100000, "seed": 0, **state["config"]}
        if state["version"] != 1 or config != self.config:
            raise ValueError("Prosody state does not match the configured transform.")
        self.state = {**state, "config": config}
        return self

    def save(self, path):
        """Compatibility convenience; fitting workflows use shared persistence."""
        from .fitting import save_fitted_state

        save_fitted_state(self, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["version"] != 1:
            raise ValueError("Unsupported prosody statistics version.")
        result = cls(**state["config"])
        return result.load_state_dict(state)


# Existing Python imports and Hydra targets remain readable during migration.
ProsodyTargets = ProsodyTransform
