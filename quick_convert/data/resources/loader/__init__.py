from dataclasses import replace

import torch

from .. import ResourceRef


def load_torch_tensor(ref):
    return torch.load(ref.path)


LOADER_REGISTRY = {"torch_tensor": load_torch_tensor}


def load_resource(ref: ResourceRef):
    if ref.kind not in LOADER_REGISTRY:
        raise ValueError(f"No loader registered for resource kind {ref.kind}")

    loader_fn = LOADER_REGISTRY[ref.kind]
    value = loader_fn(ref)
    return replace(ref, value=value)
