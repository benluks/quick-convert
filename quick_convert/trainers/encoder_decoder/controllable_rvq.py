from __future__ import annotations

from typing import Any, Optional

import torch
import sentencepiece as spm

from quick_convert.components.encoders import RVQDisentangler
from quick_convert.components.speaker import SpeakerEncoder, SpeakerEmbedding
from quick_convert.components.ssl.base import ContentEncoder, ContentFeatures
from quick_convert.components.spectrogram_generator import ChatterboxSpectrogramGenerator as CSG

from quick_convert.data.types import AudioBatch
from quick_convert.trainers.encoder_decoder.base import BaseEncoderDecoderTrainingModule


class ControllableRVQTrainingModule(BaseEncoderDecoderTrainingModule):
    """
    PyTorch Lightning trainer for the controllable RVQ disentanglement model.

    Training objective
    ------------------
    The model is trained end-to-end with four groups of loss terms:

    * **RVQ losses** — ``commitment_loss`` (encoder commits to codebook entries)
      and ``codebook_loss`` (codebook entries move toward encoder outputs).
    * **Distillation losses** — ``ctc_loss``, ``spk_loss``, ``emo_loss``, and
      ``pros_loss``, computed against frozen teacher encoders for linguistics,
      speaker identity, emotion, and prosody respectively.
    * **Adversarial losses** — ``adv_spk_loss_ling``, ``adv_spk_loss_pros``,
      ``adv_ling_loss_spk``, ``adv_ling_loss_pros``, used to enforce
      disentanglement across attribute subspaces.
    * **Decoder loss** — reconstruction loss from the spectrogram decoder,
      operating on the quantised latent.

    All teacher encoders (``spk_encoder``, ``emo_encoder``, ``pros_encoder``) and
    the SSL backbone inside ``encoder`` are frozen at construction time.

    Args:
        encoder:
            Instantiated :class:`~quick_convert.components.encoders.RVQDisentangler`
            whose ``feature_extractor`` will be frozen.
        decoder:
            :class:`~quick_convert.components.spectrogram_generator.ChatterboxSpectrogramGenerator`
            used to reconstruct spectrograms from quantised latents.
        spk_encoder:
            Frozen speaker encoder providing speaker-identity distillation targets.
        emo_encoder:
            Frozen content/emotion encoder providing emotion distillation targets.
        tokenizer:
            SentencePiece processor used to tokenise transcript targets for CTC
            distillation.
        decoder_loss_weight:
            Scale applied to the decoder reconstruction loss.
        rvq_loss_weights:
            Per-term weights for the RVQ losses.  Expected keys:
            ``'commitment_loss'`` and ``'codebook_loss'``.
        distillation_loss_weights:
            Per-term weights for distillation losses.  Expected keys:
            ``'ling'``, ``'spk'``, ``'emo'``.  ``'pros'`` is only required
            when ``pros_encoder`` is not ``None``.
        adv_loss_weights:
            Per-term weights for adversarial disentanglement losses.  Expected
            keys: ``'spk_ling'``, ``'spk_pros'``, ``'ling_spk'``, ``'ling_pros'``.
        pros_encoder:
            Optional frozen prosody encoder.  When ``None`` the prosody
            distillation loss is skipped.
        lr_scheduler_cls:
            Optional LR-scheduler class to instantiate after the optimiser.
        lr_scheduler_kwargs:
            Keyword arguments forwarded to ``lr_scheduler_cls``.
        optimizer_cls:
            Optimiser class (default: :class:`~torch.optim.AdamW`).
        optimizer_kwargs:
            Keyword arguments forwarded to ``optimizer_cls``.
    """

    def __init__(
        self,
        encoder: RVQDisentangler,
        decoder: CSG,
        spk_encoder: SpeakerEncoder,
        emo_encoder: ContentEncoder,
        tokenizer: spm.SentencePieceProcessor,
        decoder_loss_weight: float,
        rvq_loss_weights: dict[str, float],
        distillation_loss_weights: dict[str, float],
        adv_loss_weights: dict[str, float],
        pros_encoder: ContentEncoder = None,
        lr_scheduler_cls: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lr_scheduler_kwargs: Optional[dict[str, Any]] = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: dict[str, Any] = {"weight_decay": 1e-2},
        *kwargs: Any,
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.spk_encoder = spk_encoder
        self.emo_encoder = emo_encoder
        self.pros_encoder = pros_encoder
        self.tokenizer = tokenizer

        self.save_hyperparameters(
            ignore=["encoder", "decoder", "tokenizer", "spk_encoder", "emo_encoder", "pros_encoder"]
        )
        self._validate_inputs(
            rvq_loss_weights=rvq_loss_weights,
            distillation_loss_weights=distillation_loss_weights,
            adv_loss_weights=adv_loss_weights,
            pros_encoder=pros_encoder,
        )
        self._freeze_feature_extractors()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        rvq_loss_weights: dict[str, float],
        distillation_loss_weights: dict[str, float],
        adv_loss_weights: dict[str, float],
        pros_encoder: Optional[object],
    ) -> None:
        """Raise ``ValueError`` if any loss-weight dict is missing required keys."""
        distil_keys = {"ling", "spk", "emo"} | ({"pros"} if pros_encoder is not None else set())
        checks = [
            ("rvq_loss_weights", rvq_loss_weights, {"commitment_loss", "codebook_loss"}),
            ("distillation_loss_weights", distillation_loss_weights, distil_keys),
            ("adv_loss_weights", adv_loss_weights, {"spk_ling", "spk_pros", "ling_spk", "ling_pros"}),
        ]
        for name, weights, required in checks:
            if missing := required - weights.keys():
                raise ValueError(f"{name} is missing keys: {missing}")

    def _freeze_feature_extractors(self) -> None:
        """Freeze all teacher encoders and the SSL backbone inside the encoder.

        All frozen modules are also switched to eval mode so that batch-norm
        and dropout layers behave deterministically during training.
        """
        for param in self.encoder.feature_extractor.parameters():
            param.requires_grad = False

        for param in self.spk_encoder.parameters():
            param.requires_grad = False

        for param in self.emo_encoder.parameters():
            param.requires_grad = False

        if self.pros_encoder is not None:
            for param in self.pros_encoder.parameters():
                param.requires_grad = False

        self.encoder.feature_extractor.eval()
        self.spk_encoder = self.spk_encoder.eval()
        self.emo_encoder = self.emo_encoder.eval()

        self.pros_encoder = self.pros_encoder.eval() if self.pros_encoder is not None else None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor,
    ):
        """Encode ``waveform`` with the RVQ disentangler and decode the result.

        Args:
            waveform: Raw audio tensor of shape ``(B, T)``.
            lengths: Valid sample lengths of shape ``(B,)``.

        Returns:
            Decoder output produced from the quantised latent.
        """
        return self.decoder(self.encoder(waveform, lengths))

    # ------------------------------------------------------------------
    # Shared step logic
    # ------------------------------------------------------------------

    def _shared_step(self, batch: AudioBatch, stage: str) -> torch.Tensor:
        """Compute all losses, log them, and return the total loss.

        Called by both :meth:`training_step` and :meth:`validation_step` with
        the appropriate ``stage`` prefix (``'train'`` / ``'val'``).

        Args:
            batch: Batch of waveforms, lengths, and optional transcripts.
            stage: Logging prefix; controls ``on_step`` logging behaviour.

        Returns:
            Scalar total loss tensor passed to the Lightning optimiser.
        """
        waveform = batch.waveforms  # (B, T)
        lengths = batch.lengths  # (B,)
        targets = batch.targets  # transcript token ids, shape (B, T_text) or None
        target_lengths = batch.target_lengths  # (B,) or None

        with torch.no_grad():
            spk_targets = self.spk_encoder(waveform).values
            emo_targets = self.emo_encoder(waveform).values
            pros_targets = self.pros_encoder(waveform).values if self.pros_encoder is not None else None

        ling_targets = self.tokenizer(targets)

        z_q, spk_q, text_q, pros_emo_q, loss_dict = self.encoder.compute_loss(
            waveform,
            lengths,
            ling_targets,
            target_lengths,
            spk_targets,
            emo_targets,
            pros_targets,
        )

        # Weighted sum of VQ commitment and codebook losses
        rvq_loss = (
            self.hparams.rvq_loss_weights["commitment_loss"] * loss_dict["rvq_losses"]["commitment_loss"]
            + self.hparams.rvq_loss_weights["codebook_loss"] * loss_dict["rvq_losses"]["codebook_loss"]
        )

        # Weighted sum of per-attribute distillation losses from frozen teacher encoders
        distil_loss = (
            self.hparams.distillation_loss_weights["ling"] * loss_dict["distill_losses"]["ctc_loss"]
            + self.hparams.distillation_loss_weights["spk"] * loss_dict["distill_losses"]["spk_loss"]
            + self.hparams.distillation_loss_weights["emo"] * loss_dict["distill_losses"]["emo_loss"]
            + self.hparams.distillation_loss_weights["pros"] * loss_dict["distill_losses"]["pros_loss"]
        )

        # Weighted sum of adversarial losses that enforce cross-attribute disentanglement
        adv_loss = (
            self.hparams.adv_loss_weights["spk_ling"] * loss_dict["adv_losses"]["adv_spk_loss_ling"]
            + self.hparams.adv_loss_weights["spk_pros"] * loss_dict["adv_losses"]["adv_spk_loss_pros"]
            + self.hparams.adv_loss_weights["ling_spk"] * loss_dict["adv_losses"]["adv_ling_loss_spk"]
            + self.hparams.adv_loss_weights["ling_pros"] * loss_dict["adv_losses"]["adv_ling_loss_pros"]
        )

        # Decoder reconstruction loss: z_q is used as both the flow target (x1)
        # and the conditioning signal; detach x1 so gradients flow only through mu
        # TODO
        decoder_loss = self.decoder.compute_loss(waveform, lengths, text_q, pros_emo_q, spk_q)

        loss = rvq_loss + distil_loss + adv_loss + self.hparams.decoder_loss_weight * decoder_loss
        log_dict = {
            f"{stage}/loss": loss,
            # RVQ losses
            f"{stage}/rvq_loss": rvq_loss,
            f"{stage}/commitment_loss": loss_dict["rvq_losses"]["commitment_loss"],
            f"{stage}/codebook_loss": loss_dict["rvq_losses"]["codebook_loss"],
            # Decoder loss
            f"{stage}/decoder_loss": decoder_loss,
            # Distillation losses
            f"{stage}/distil_loss": distil_loss,
            f"{stage}/ctc_loss": loss_dict["distill_losses"]["ctc_loss"],
            f"{stage}/spk_loss": loss_dict["distill_losses"]["spk_loss"],
            f"{stage}/emo_loss": loss_dict["distill_losses"]["emo_loss"],
            f"{stage}/pros_loss": loss_dict["distill_losses"]["pros_loss"],
            # Adversarial losses
            f"{stage}/adv_loss": adv_loss,
            f"{stage}/adv_spk_loss_ling": loss_dict["adv_losses"]["adv_spk_loss_ling"],
            f"{stage}/adv_spk_loss_pros": loss_dict["adv_losses"]["adv_spk_loss_pros"],
            f"{stage}/adv_ling_loss_spk": loss_dict["adv_losses"]["adv_ling_loss_spk"],
            f"{stage}/adv_ling_loss_pros": loss_dict["adv_losses"]["adv_ling_loss_pros"],
        }

        self.log_dict(
            log_dict,
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
        return self._shared_step(batch, "train")
