from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from quick_convert.data.types import AudioBatch


@dataclass
class ResolvedResource:
    values: Any
    lengths: torch.Tensor | None = None


class OnlineResourceMixin:
    online_encoders: nn.ModuleDict
    trainable_online_encoders: frozenset[str] = frozenset()

    def configure_online_encoders(
        self,
        online_encoders: dict[str, nn.Module] | None,
        trainable_online_encoders: tuple[str, ...] = (),
    ) -> None:
        """Register online encoders and explicitly select which may train."""
        self.online_encoders = nn.ModuleDict(online_encoders or {})
        unknown = set(trainable_online_encoders) - set(self.online_encoders)
        if unknown:
            raise ValueError(f"Unknown trainable online encoders: {sorted(unknown)}.")

        self.trainable_online_encoders = frozenset(trainable_online_encoders)
        for name, encoder in self.online_encoders.items():
            trainable = name in self.trainable_online_encoders
            encoder.requires_grad_(trainable)
            if not trainable:
                encoder.eval()

    def get_resource(
        self,
        batch: AudioBatch,
        name: str,
    ) -> ResolvedResource:
        """
        Retrieve a batched resource if it is already loaded, otherwise compute
        it using an online resource encoder with the same name.
        """
        resource = batch.resources.get(name)

        if resource is not None:
            return self._normalize_resource(resource)

        encoder = self.online_encoders[name] if name in self.online_encoders else None

        if encoder is not None:
            trainable = name in self.trainable_online_encoders
            if trainable:
                resource = encoder(batch)
            else:
                with torch.inference_mode():
                    resource = encoder(batch)

            return self._normalize_resource(resource, detach=not trainable)

        raise RuntimeError(
            f"No resource named {name!r} was found in the batch and no online encoder with that name exists."
        )

    @staticmethod
    def _normalize_resource(resource: Any, *, detach: bool = True) -> ResolvedResource:
        if isinstance(resource, ResolvedResource):
            return ResolvedResource(
                values=OnlineResourceMixin._maybe_detach(resource.values, detach),
                lengths=resource.lengths,
            )

        # Tensors expose a callable ``values`` method for sparse operations;
        # they are already the resource value, not a resource wrapper.
        if isinstance(resource, torch.Tensor):
            return ResolvedResource(values=OnlineResourceMixin._maybe_detach(resource, detach), lengths=None)

        if hasattr(resource, "values"):
            return ResolvedResource(
                values=OnlineResourceMixin._maybe_detach(resource.values, detach),
                lengths=getattr(resource, "lengths", None),
            )

        if isinstance(resource, tuple):
            if len(resource) != 2:
                raise ValueError("Tuple resources must have the form `(values, lengths)`.")

            values, lengths = resource

            return ResolvedResource(
                values=OnlineResourceMixin._maybe_detach(values, detach),
                lengths=lengths,
            )

        return ResolvedResource(
            values=OnlineResourceMixin._maybe_detach(resource, detach),
            lengths=None,
        )

    @staticmethod
    def _maybe_detach(value: Any, detach: bool) -> Any:
        return value.detach() if detach and isinstance(value, torch.Tensor) else value
