from __future__ import annotations

from typing import Optional

import torch

from quick_convert.components.encoders import RVQDisentangler
from quick_convert.components.decoders.flow_matching.base import BASECFM
from quick_convert.data.types import AudioBatch
from quick_convert.trainers.abs_anonymizer_trainer import AbsAnonymizerTrainer


class ControllableRVQTrainer(AbsAnonymizerTrainer):
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
        encoder:
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
            ``encoder.feature_extractor`` (the frozen SSL backbone).
        cosine_t_max:
            ``T_max`` (number of steps per cosine half-cycle) for
            :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.
            Set to 0 to disable the scheduler.
    """

    def __init__(
        self,
        encoder: RVQDisentangler,
        decoder: Optional[BASECFM] = None,
        spk_encoder: Optional[torch.nn.Module] = None,
        emo_encoder: Optional[torch.nn.Module] = None,
        pros_encoder: Optional[torch.nn.Module] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        cosine_t_max: int = 0,
        commitment_loss_weight: float = 1.0,
        codebook_loss_weight: float = 1.0,
        cfm_loss_weight: float = 1.0,
        distillation_loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0],
        freeze_feature_extractor: bool = True,
        tokenizer: Optional[object] = None,  # Placeholder for future use if needed
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            lr=lr,
            weight_decay=weight_decay,
            cosine_t_max=cosine_t_max,
        )
        self.tokenizer = tokenizer  

        self.spk_encoder = spk_encoder.eval() if spk_encoder is not None else None
        self.emo_encoder = emo_encoder.eval() if emo_encoder is not None else None
        self.pros_encoder = pros_encoder.eval() if pros_encoder is not None else None

        self.save_hyperparameters(ignore=["encoder", "decoder", "tokenizer", 
                                          "spk_encoder", "emo_encoder", "pros_encoder"])

        if freeze_feature_extractor:
            self._freeze_feature_extractor()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _freeze_feature_extractor(self) -> None:
        for param in self.encoder.feature_extractor.parameters():
            param.requires_grad = False
        self.encoder.feature_extractor.eval()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor,
    ):
        """Thin wrapper around :meth:`RVQDisentangler.forward`."""
        return self.decoder(self.encoder(waveform, lengths))

    # ------------------------------------------------------------------
    # Shared step logic
    # ------------------------------------------------------------------

    def _shared_step(self, batch: AudioBatch, stage: str) -> torch.Tensor:
        waveform = batch.waveforms  # (B, T)
        lengths = batch.lengths     # (B,)
        targets = batch.targets     # (B, T_text) or None

        with torch.no_grad():
            spk_targets = self.spk_encoder(targets) if self.spk_encoder is not None else None
            emo_targets = self.emo_encoder(targets) if self.emo_encoder is not None else None
            pros_targets = self.pros_encoder(targets) if self.pros_encoder is not None else None

        targets = self.tokenizer(targets)

        z_q, text_q, spk_q, pros_q, emo_q,  \
        commitment_loss, codebook_loss, \
        ctc_loss, spk_loss, pros_loss, emo_loss = self.encoder.compute_loss(
            waveform, lengths, targets, spk_targets, pros_targets, emo_targets
        )

        loss = (
            self.hparams.commitment_loss_weight * commitment_loss
            + self.hparams.codebook_loss_weight * codebook_loss
            + self.hparams.distillation_loss_weights[0] * ctc_loss
            + self.hparams.distillation_loss_weights[1] * spk_loss
            + self.hparams.distillation_loss_weights[2] * emo_loss
            + self.hparams.distillation_loss_weights[3] * pros_loss
        )

        # z_q: (B, T, F) — use as both target x1 and conditioning
        cfm_loss = self.decoder.compute_loss(
            x1=z_q.detach().transpose(1, 2),  # (B, F, T)
            mu=z_q.transpose(1, 2),
        )

        loss = loss + self.hparams.cfm_loss_weight * cfm_loss

        # External supervision loss
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
            self.encoder.feature_extractor.eval()

        return self._shared_step(batch, "train")