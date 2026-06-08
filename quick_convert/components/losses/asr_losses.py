import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCLoss(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 output_dim: int,
                 reduction: str = 'mean'):
        super().__init__()

        self.ctc_l = torch.nn.Linear(hidden_dim, output_dim)
        self.reduction = reduction

    def forward(
        self,
        x: torch.FloatTensor,
        linguistic_targets: torch.LongTensor,
        input_lengths: torch.LongTensor,
        target_lengths: torch.LongTensor
    ) -> torch.FloatTensor:
        x = self.ctc_l(x)
        x = x.log_softmax(dim=-1)  # Log probabilities for CTC loss
        ctc_loss = F.ctc_loss(
            x,
            linguistic_targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=0,  # Assuming 0 is the blank token ID
            reduction=self.reduction,
        )
        return ctc_loss