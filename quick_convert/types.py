from __future__ import annotations

from os import PathLike
from typing import TypeAlias

import torch


AudioInput: TypeAlias = str | PathLike[str] | torch.Tensor
"""A filesystem path or an in-memory waveform tensor."""
