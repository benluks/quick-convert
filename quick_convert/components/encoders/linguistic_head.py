import torch
from torch import nn
from torchaudio.models import Conformer

from quick_convert.components.layers import ConformerBlock
from quick_convert.components.losses import CTCLoss

from ..layers.heads import HeadOutput, HeadTarget, SupervisedHead


class LinguisticCTCHead(SupervisedHead):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        loss: CTCLoss,
        decoder: Conformer | nn.LSTM | None = None,
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
        self.ln = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.decoder = decoder
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.loss = loss

    def predict(
        self,
        features: torch.Tensor,
        *,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> HeadOutput:
        del lengths, padding_mask

        logits = self.forward(features)

        return HeadOutput(
            predictions=logits,
            states={"logits": logits},
        )

    def compute_loss(
        self,
        features,
        *,
        targets: HeadTarget,
        padding_mask,
        lengths=None,
    ) -> HeadOutput:
        """
        Implementation assumes tokenization happens outside the model,
        and that 0 is reserved for the CTC blank token.
        """

        if targets.lengths is None:
            raise ValueError("Linguistic CTC targets require target lengths.")

        logits = self.forward(features, lengths=lengths)
        logits = logits.transpose(0, 1)  # (T, B, output_dim) for CTC loss
        output = self.loss(logits, targets.values, lengths, targets.lengths)
        return HeadOutput(
            loss=output.loss,
            predictions=output.log_probs,
            states={"logits": output.logits, "log_probs": output.log_probs},
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = self.ln(x)
        x = self.input_proj(x)

        if self.decoder is not None:
            x, _ = self.decoder(x, lengths)

        return self.output_proj(x)


class LinguisticConformerCTCHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        loss: CTCLoss = CTCLoss,
        dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        bias: bool = True,
        num_heads: int = 4,
        ffn_dim: int = None,
        use_flash_attention: bool = True,
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
