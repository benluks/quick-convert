from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from quick_convert.components.ssl import ContentEncoder, ContentFeatures
from quick_convert.data.types import AudioBatch


class OnlineResourceMixin:
    online_encoders: nn.ModuleDict

    def get_resource(
        self,
        batch: AudioBatch,
        name: str,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve a resource from the batch if precomputed, otherwise compute
        it using an online encoder with the same name.

        Returns:
            Tuple of (values, lengths).
        """

        resource = batch.resources.get(name)

        if resource is not None:
            return resource

        encoder: ContentEncoder = self.online_encoders[name] if name in self.online_encoders else None
        if encoder is not None:
            with torch.inference_mode():
                features: ContentFeatures = encoder(batch)

            return features.values.detach(), features.lengths

        raise RuntimeError(
            f"No resource named '{name}' was found in the batch and no online encoder with that name exists."
        )
