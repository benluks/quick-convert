<<<<<<< HEAD
"""Load path-backed resources into memory.

Resource providers resolve *what* belongs to a sample and where it is stored.
This module handles the separate concern of materializing a ``ResourceRef``
according to its ``kind``.
"""

=======
>>>>>>> 79ae067 (refactor data module)
from dataclasses import replace

import torch

from . import ResourceRef


def load_torch(ref, device="cpu"):
<<<<<<< HEAD
    """Load a torch-serialized resource from ``ref.path``."""
    return torch.load(ref.path, map_location=device)


# Resource kinds intentionally own loading behavior here rather than in
# providers, so the same provider can remain concerned only with resolution.
LOADER_REGISTRY = {
    "torch_tensor": load_torch,
    "token_ids": load_torch,
}


def load_resource(ref: ResourceRef, **kwargs):
    """Materialize a path-backed resource.

    The loader is selected from :data:`LOADER_REGISTRY` using ``ref.kind``.
    A new reference containing the loaded value is returned; the original
    reference is not mutated.

    Args:
        ref:
            Resource reference to load.
        **kwargs:
            Additional arguments passed to the registered loader.

    Returns:
        A copy of ``ref`` with ``value`` populated.

    Raises:
        ValueError:
            If no loader is registered for the resource kind.
    """
=======
    return torch.load(ref.path, map_location=device)


LOADER_REGISTRY = {"torch_tensor": load_torch, "token_ids": load_torch}


def load_resource(ref: ResourceRef, **kwargs):
>>>>>>> 79ae067 (refactor data module)
    if ref.kind not in LOADER_REGISTRY:
        raise ValueError(f"No loader registered for resource kind {ref.kind}")

    loader_fn = LOADER_REGISTRY[ref.kind]
    value = loader_fn(ref, **kwargs)
    return replace(ref, value=value)
