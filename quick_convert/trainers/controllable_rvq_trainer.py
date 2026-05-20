from __future__ import annotations

from typing import Any, Optional

import torch
import lightning as L

from quick_convert.components.encoders import RVQDisentangler
from quick_convert.components.decoders.flow_matching.base import BASECFM
from quick_convert.data.types import AudioBatch


class ControllableRVQTrainer(L.LightningModule):
    """
    PyTorch Lightning trainer for the controllable RVQ disentanglement model.

    Training objective
    ------------------
    The model is trained end-to-end with four loss terms:

    * ``commitment_loss``  — encoder learns to commit to codebook entries.
    * ``codebook_loss``    — codebook entries move toward encoder outputs.
    * ``cfm_loss``         — conditional flow matching loss on the
                             reconstructed latent.
    * ``distillation_loss`` — distillation loss from several SSL teacher models.

    The SSL feature extractor (W2VBert) is frozen by default.

    Args:
        model:
            Instantiated :class:`~quick_convert.components.encoders.RVQDisentangler`.
        decoder:
            Optional :class:`~quick_convert.components.decoders.flow_matching.base.BASECFM`
            used to reconstruct a target from the quantised latent.
        lr:
            Peak learning rate for AdamW.
        weight_decay:
            AdamW weight decay.
        commitment_loss_weight:
            Scale applied to the VQ commitment loss.
        codebook_loss_weight:
            Scale applied to the VQ codebook loss.
        cfm_loss_weight:
            Scale applied to the flow-matching reconstruction loss.
        distillation_loss_weight:
            Scale applied to the distillation loss from SSL teacher models.
        freeze_feature_extractor:
            When ``True`` (default), gradients are blocked for
            ``model.feature_extractor`` (the frozen SSL backbone).
        cosine_t_max:
            ``T_max`` (number of steps per cosine half-cycle) for
            :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.
            Set to 0 to disable the scheduler.
    """

    def __init__(
        self,
        model: RVQDisentangler,
        decoder: Optional[BASECFM] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        commitment_loss_weight: float = 1.0,
        codebook_loss_weight: float = 1.0,
        cfm_loss_weight: float = 1.0,
        distillation_loss_weight: list[float] = [1.0, 1.0, 1.0],
        freeze_feature_extractor: bool = True,
        cosine_t_max: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "decoder"])

        self.model = model
        self.decoder = decoder

        if freeze_feature_extractor:
            self._freeze_feature_extractor()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _freeze_feature_extractor(self) -> None:
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        self.model.feature_extractor.eval()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor,
    ):
        """Thin wrapper around :meth:`RVQDisentangler.forward`."""
        return self.model(waveform, lengths)

    # ------------------------------------------------------------------
    # Shared step logic
    # ------------------------------------------------------------------

    def forward(self, batch: AudioBatch, stage: str) -> torch.Tensor:
        waveform = batch.waveforms # (B, T)
        lengths = batch.lengths    # (B,)

        z_q, spk_q, pros_q, text_q, commitment_loss, codebook_loss = self.model(
            waveform, lengths
        )

        loss = (
            self.hparams.commitment_loss_weight * commitment_loss
            + self.hparams.codebook_loss_weight * codebook_loss
        )

        cfm_loss = torch.tensor(0.0, device=self.device)
        if self.decoder is not None:
            # z_q: (B, T, F) — use as both target x1 and conditioning
            cfm_loss = self.decoder.compute_loss(
                x1=z_q.detach().transpose(1, 2),  # (B, F, T)
                mu=z_q.transpose(1, 2),
            )
            loss = loss + self.hparams.cfm_loss_weight * cfm_loss

        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/commitment_loss": commitment_loss,
                f"{stage}/codebook_loss": codebook_loss,
                f"{stage}/cfm_loss": cfm_loss,
            },
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage == "train"),
            sync_dist=True,
        )

        return loss

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: AudioBatch, batch_idx: int) -> torch.Tensor:
        # Keep the SSL backbone in eval mode even during training
        if self.hparams.freeze_feature_extractor:
            self.model.feature_extractor.eval()

        return self._shared_step(batch, "train")

    def validation_step(self, batch: AudioBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    # ------------------------------------------------------------------
    # Optimizers / schedulers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        # Only optimise parameters that require grad
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.cosine_t_max <= 0:
            return {"optimizer": optimizer}

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.cosine_t_max,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
