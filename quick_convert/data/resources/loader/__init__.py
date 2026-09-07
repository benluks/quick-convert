from dataclasses import replace

import torch

from .. import ResourceRef


def load_torch(ref, device="cpu"):
    return torch.load(ref.path, map_location=device)


LOADER_REGISTRY = {"torch_tensor": load_torch, "token_ids": load_torch}


def load_resource(ref: ResourceRef, **kwargs):
    if ref.kind not in LOADER_REGISTRY:
        raise ValueError(f"No loader registered for resource kind {ref.kind}")

    loader_fn = LOADER_REGISTRY[ref.kind]
    value = loader_fn(ref, **kwargs)
    return replace(ref, value=value)
