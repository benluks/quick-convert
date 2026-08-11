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
            with torch.inference_mode():
                resource = encoder(batch)

            return self._normalize_resource(resource)

        raise RuntimeError(
            f"No resource named {name!r} was found in the batch and no online encoder with that name exists."
        )

    @staticmethod
    def _normalize_resource(resource: Any) -> ResolvedResource:
        if isinstance(resource, ResolvedResource):
            return ResolvedResource(
                values=OnlineResourceMixin._detach(resource.values),
                lengths=resource.lengths,
            )

        if hasattr(resource, "values"):
            return ResolvedResource(
                values=OnlineResourceMixin._detach(resource.values),
                lengths=getattr(resource, "lengths", None),
            )

        if isinstance(resource, tuple):
            if len(resource) != 2:
                raise ValueError("Tuple resources must have the form `(values, lengths)`.")

            values, lengths = resource

            return ResolvedResource(
                values=OnlineResourceMixin._detach(values),
                lengths=lengths,
            )

        return ResolvedResource(
            values=OnlineResourceMixin._detach(resource),
            lengths=None,
        )

    @staticmethod
    def _detach(value: Any) -> Any:
        return value.detach() if isinstance(value, torch.Tensor) else value
