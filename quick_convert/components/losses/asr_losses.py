from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CTCOutput:
    logits: torch.Tensor
    log_probs: torch.Tensor
    loss: torch.Tensor


class CTCLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        blank_id: int = 0,
    ):
        super().__init__()

        self.reduction = reduction
        self.blank_id = blank_id

    def forward(
        self,
        logits: torch.FloatTensor,
        linguistic_targets: torch.LongTensor,
        input_lengths: torch.LongTensor,
        target_lengths: torch.LongTensor,
    ) -> CTCOutput:
        log_probs = logits.log_softmax(dim=-1)

        loss = F.ctc_loss(
            log_probs,
            linguistic_targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.blank_id,
            reduction=self.reduction,
        )

        return CTCOutput(
            logits=logits,
            log_probs=log_probs,
            loss=loss,
        )
