import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_convert.components.layers import ConformerBlock
from quick_convert.components.losses import CTCLoss

class LinguisticCTCHead(nn.Module):
    def __init__(
            self, 
            hidden_dim: int, 
            loss: CTCLoss = CTCLoss(), # Can probably be made more general...
        ):
        super().__init__()
        """
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.conformer_block = ConformerBlock(
            embed_dim=hidden_dim,
            num_heads=4,
            ffn_dim=hidden_dim * 4,
            conv_kernel_size=31,
            dropout=0.1,
            bias=True,
        )
        """
        self.ln = nn.LayerNorm(hidden_dim)
        self.ctc_loss = loss

    def forward(self, x: torch.Tensor, *kwargs) -> torch.Tensor:
        x = self.ln(x)
        return x
    
    def compute_loss(
        self,
        x: torch.FloatTensor,
        linguistic_targets: torch.LongTensor,
        input_lengths: torch.LongTensor,
        target_lengths: torch.LongTensor,
    ) -> torch.Tensor:
        
        """
        Implementation assumes tokenization happens outside the model, 
        and that 0 is reserved for the CTC blank token. 
        """        
        x = self.forward(x)
        x = x.transpose(0, 1)  # (T, B, output_dim) for CTC loss
        ctc_loss = self.ctc_loss(x, linguistic_targets, input_lengths, target_lengths)
        return ctc_loss


class LinguisticConformerCTCHead(nn.Module):
    def __init__(
            self, 
            hidden_dim: int, 
            output_dim: int, 
            dropout_p: float = 0.1, 
            conv_kernel_size: int = 31, 
            bias: bool = True, 
            num_heads: int = 4, 
            ffn_dim: int = None,
            use_flash_attention: bool = True,
            loss: CTCLoss = CTCLoss(),
        ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4
        """
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.conformer_block = ConformerBlock(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout_p,
            bias=bias,
            use_flash_attention=use_flash_attention,
        )
        """
        self.ln = nn.LayerNorm(hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.conformer_block = ConformerBlock(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout_p,
            bias=bias,
            use_flash_attention=use_flash_attention,
        )
        self.ctc_loss = loss

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.ln(x)
        x = self.linear_1(x)
        x = self.conformer_block(x, padding_mask)
        return x
    
    def compute_loss(
        self,
        x: torch.FloatTensor,
        linguistic_targets: torch.LongTensor,
        padding_mask: torch.LongTensor | None,
        input_lengths: torch.LongTensor,
        target_lengths: torch.LongTensor,
    ) -> torch.Tensor:
        
        """
        Implementation assumes tokenization happens outside the model, 
        and that 0 is reserved for the CTC blank token. 
        """
        x = self.forward(x, padding_mask)
        x = x.transpose(0, 1)  # (T, B, output_dim) for CTC loss
        ctc_loss = self.ctc_loss(x, linguistic_targets, input_lengths, target_lengths)
        return ctc_loss

