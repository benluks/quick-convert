from dataclasses import replace

import torch

from ..audio import load_audio_resource
from ..base import ResourceRef


def load_torch(ref, device="cpu"):
    if ref.path is None:
        raise ValueError(f"Resource {ref.name!r} has no path to load.")
    return torch.load(ref.path, map_location=device, weights_only=True)


LOADER_REGISTRY = {"torch_tensor": load_torch, "token_ids": load_torch, "audio": load_audio_resource}


def load_resource(ref: ResourceRef, **kwargs):
    """Materialize a path-backed resource and return an updated reference.

    The built-in registry loads ``torch_tensor`` and ``token_ids`` resources
    with :func:`torch.load`. Text resources are expected to be supplied as
    in-memory values by their provider. Audio resources load as mono
    AudioSample objects, optionally resampled to the reference sample_rate.

    Raises:
        ValueError: If the resource kind has no registered loader.
    """
    if ref.kind not in LOADER_REGISTRY:
        raise ValueError(f"No loader registered for resource kind {ref.kind}")

    loader_fn = LOADER_REGISTRY[ref.kind]
    value = loader_fn(ref, **kwargs)
    return replace(ref, value=value)
