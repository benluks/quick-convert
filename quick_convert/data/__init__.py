"""Dataset and batching primitives for speech experiments.

The public data API is centered around:

- :class:`BaseDataset` for filesystem-backed audio datasets.
- :class:`ManifestDataset` for CSV-backed datasets.
- :class:`AudioSample` and :class:`AudioBatch` for model-facing data.
- ``quick_convert.data.resources`` for attaching arbitrary annotations,
  metadata, and precomputed or online features to samples.

Dataset classes deliberately remain agnostic to experiment-specific resources.
"""

from __future__ import annotations

from .base_dataset import BaseDataset
from .manifest_dataset import ManifestDataset
from .types import AudioBatch, AudioSample


__all__ = ["AudioBatch", "AudioSample", "BaseDataset", "ManifestDataset"]
