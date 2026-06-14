from __future__ import annotations

from typing import Any, Optional

import torch
import sentencepiece as spm

from quick_convert.components.encoders import RVQDisentangler
from quick_convert.components.decoders import ChatterboxSpectrogramGenerator as CSG

from quick_convert.data.index.base import Indexer
from quick_convert.data.types import AudioBatch
from quick_convert.pipelines.training.modules.encoder_decoder.base import BaseEncoderDecoderTrainingModule


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
            :class:`~quick_convert.components.decoders.ChatterboxSpectrogramGenerator`
            used to reconstruct spectrograms from quantised latents.
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
        # either pass `tokenizer` or `tokenizer_pad_id`
        rvq_loss_weights: dict[str, float],
        distillation_loss_weights: dict[str, float],
        adv_loss_weights: dict[str, float],
        tokenizer: Optional[spm.SentencePieceProcessor] = None,
        tokenizer_pad_id: Optional[int] = None,
        decoder_loss_weight: Optional[float] = None,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        adv_loss_hold_off: int = 1000,
        *kwargs: Any,
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        self.tokenizer = tokenizer
        self.tokenizer_pad_id = tokenizer_pad_id if tokenizer_pad_id is not None else tokenizer.pad_id()
        self.adv_loss_hold_off = adv_loss_hold_off

        self.save_hyperparameters(ignore=["encoder", "decoder", "tokenizer"])
        self._validate_inputs(
            rvq_loss_weights=rvq_loss_weights,
            distillation_loss_weights=distillation_loss_weights,
            adv_loss_weights=adv_loss_weights,
        )

        # has to be declared like this, since the parent lightning trainer has
        # access to the datasets, which are used to fit the indexer
        self.indexers = {"speaker": Indexer("{sample.resources.spkid.value}")}

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        rvq_loss_weights: dict[str, float],
        distillation_loss_weights: dict[str, float],
        adv_loss_weights: dict[str, float],
    ) -> None:
        """Raise ``ValueError`` if any loss-weight dict is missing required keys."""
        distil_keys = {"ling", "spk", "emo"}
        checks = [
            (
                "rvq_loss_weights",
                rvq_loss_weights,
                {"commitment_loss", "codebook_loss", "load_balancing_loss", "mse_loss"},
            ),
            ("distillation_loss_weights", distillation_loss_weights, distil_keys),
            ("adv_loss_weights", adv_loss_weights, {"spk_ling", "spk_pros", "ling_spk", "ling_pros"}),
        ]
        for name, weights, required in checks:
            if missing := required - weights.keys():
                raise ValueError(f"{name} is missing keys: {missing}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: AudioBatch):
        """Encode ``waveform`` with the RVQ disentangler and decode the result.

        Args:
            waveform: Raw audio tensor of shape ``(B, T)``.
            lengths: Valid sample lengths of shape ``(B,)``.

        Returns:
            Decoder output produced from the quantised latent.
        """
        return self.decoder(self.encoder(batch.waveforms, batch.lengths))

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
        # lengths = batch.lengths  # (B,)

        features = batch.resources["content"].values
        lengths = batch.resources["content"].lengths

        emo_targets = batch.resources["emo2vec"].values
        emo_lengths = batch.resources["emo2vec"].lengths

        spk_targets = torch.LongTensor(self.indexers["speaker"].encode_many(batch.resources["spkid"])).to(self.device)

        pros_targets = (
            batch.resources["pros"].values
            if "pros" in batch.resources and batch.resources["pros"] is not None
            else None
        )

        # tokenized_transcripts = self.tokenizer.encode(targets)
        token_ids = batch.resources["token_ids"]  # transcript token ids, shape (B, T_text) or None

        z_q, spk_q, spk_output, text_q, pros_emo_q, loss_dict, spk_acc_dict = self.encoder.compute_loss(
            features,
            lengths,
            linguistic_targets=token_ids.values,
            target_lengths=token_ids.lengths,
            speaker_seq=spk_targets,
            emotion_seq=emo_targets,
            emotion_lengths=emo_lengths,
            prosody_seq=pros_targets,
            run_adv=self.global_step > self.hparams.adv_loss_hold_off,
        )

        # Weighted sum of VQ commitment and codebook losses
        rvq_loss = (
            self.hparams.rvq_loss_weights["commitment_loss"] * loss_dict["rvq_losses"]["commitment_loss"]
            + self.hparams.rvq_loss_weights["codebook_loss"] * loss_dict["rvq_losses"]["codebook_loss"]
            + self.hparams.rvq_loss_weights["mse_loss"] * loss_dict["rvq_losses"]["mse_loss"]
            + self.hparams.rvq_loss_weights["load_balancing_loss"] * loss_dict["rvq_losses"]["load_balancing_loss"]
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

        # DECODING

        # Decoder reconstruction loss: z_q is used as both the flow target (x1)
        # and the conditioning signal; detach x1 so gradients flow only through mu
        # TODO
        decoder_features = torch.cat([text_q, pros_emo_q], dim=-1)

        decoder_loss, decoder_output, decoder_mae = self.decoder.compute_loss(
            features=decoder_features,
            # feature lengths
            lengths=lengths,
            target_wav=batch.waveforms,
            # audio waveform lengths
            wav_lens=batch.lengths,
            sampling_rate=batch.sample_rates[0],
            speaker_embedding=spk_output,
        )

        loss = rvq_loss + distil_loss + adv_loss + self.hparams.decoder_loss_weight * decoder_loss
        log_dict = {
            f"{stage}/loss": loss,
            # RVQ losses
            f"{stage}/rvq/loss": rvq_loss,
            f"{stage}/rvq/commitment_loss": loss_dict["rvq_losses"]["commitment_loss"],
            f"{stage}/rvq/codebook_loss": loss_dict["rvq_losses"]["codebook_loss"],
            f"{stage}/mse_loss": loss_dict["rvq_losses"]["mse_loss"],
            f"{stage}/load_balancing_loss": loss_dict["rvq_losses"]["load_balancing_loss"],
            # Decoder loss
            f"{stage}/decoder/loss": decoder_loss,
            f"{stage}/decoder/mae": decoder_mae,
            # Distillation losses
            f"{stage}/distil/loss": distil_loss,
            f"{stage}/distil/ctc_loss": loss_dict["distill_losses"]["ctc_loss"],
            f"{stage}/distil/spk_loss": loss_dict["distill_losses"]["spk_loss"],
            f"{stage}/distil/emo_loss": loss_dict["distill_losses"]["emo_loss"],
            f"{stage}/distil/pros_loss": loss_dict["distill_losses"]["pros_loss"],
            # Adversarial losses
            f"{stage}/adv/loss": adv_loss,
            f"{stage}/adv/spk_loss_ling": loss_dict["adv_losses"]["adv_spk_loss_ling"],
            f"{stage}/adv/spk_loss_pros": loss_dict["adv_losses"]["adv_spk_loss_pros"],
            f"{stage}/adv/ling_loss_spk": loss_dict["adv_losses"]["adv_ling_loss_spk"],
            f"{stage}/adv/ling_loss_pros": loss_dict["adv_losses"]["adv_ling_loss_pros"],
            # spk accuracy
            f"{stage}/adv_spk_acc_ling": spk_acc_dict["adv_spk_acc_ling"],
            f"{stage}/adv_spk_acc_pros": spk_acc_dict["adv_spk_acc_pros"],
        }

        self.log_dict(
            log_dict,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage == "train"),
            sync_dist=True,
            batch_size=len(batch.paths),
        )

        return (
            loss,
            decoder_output,
            spk_output,
            text_q,
            pros_emo_q,
        )

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: AudioBatch, batch_idx: int) -> torch.Tensor:
        loss, decoder_output, spk_output, text_q, pros_emo_q = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: AudioBatch, batch_idx: int) -> torch.Tensor:
        (
            loss,
            decoder_output,
            spk_output,
            text_q,
            pros_emo_q,
        ) = self._shared_step(batch, "val")

        with torch.no_grad():
            if self.trainer.is_global_zero and batch_idx == 0:
                # Log the first sample in the batch for qualitative monitoring
                decoder_features = torch.cat([text_q, pros_emo_q], dim=-1)
                max_len = batch.resources["content"].values.shape[1]
                mel = torch.cat(
                    [
                        self.decoder(feat.unsqueeze(0), length.unsqueeze(0), spk.unsqueeze(0), max_len=max_len)
                        for feat, length, spk in zip(decoder_features, batch.resources["content"].lengths, spk_output)
                    ]
                )

                gen_audio = self.decoder.mel2wav(mel)

                # log spectrograms
                for i, sample in enumerate(batch):
                    # only log up to 16 samples. Anything more is overkills
                    if i >= 16:
                        break

                    gen_sample_rate = int(self.decoder.vocoder.sampling_rate)
                    orig_sample_rate = int(sample.sample_rate)
                    tag_name = sample.utt_id

                    # TODO: remove padding from the target mel spectrogram and waveform before logging

                    # self.logger.experiment.add_image(f"{tag_prefix}/target_spectrogram")
                    self.logger.experiment.add_image(
                        f"generated/{tag_name}", mel[i].unsqueeze(0).detach().cpu(), self.global_step
                    )
                    # compute vocoder output, log audio
                    self.logger.experiment.add_audio(
                        f"generated/{tag_name}",
                        gen_audio[i].unsqueeze(-1).detach().cpu()[..., : batch.lengths[i]],
                        self.global_step,
                        sample_rate=gen_sample_rate,
                    )
                    if self.global_step == 0:
                        self.logger.experiment.add_audio(
                            f"original/{tag_name}",
                            sample.waveform.detach().cpu()[..., : batch.lengths[i]],
                            self.global_step,
                            sample_rate=orig_sample_rate,
                        )

        return loss

    @torch.inference_mode()
    def inference(
        self,
        batch: AudioBatch,
    ) -> torch.Tensor:
        """Run the decoder in inference mode on the provided features and speaker embedding."""
        features = batch.resources["content"].values
        lengths = batch.resources["content"].lengths
        z_q, z_quantized, text_q, spk_q, spk_output, emo_pros_q = self.encoder(features, lengths)
        decoder_features = torch.cat([text_q, emo_pros_q], dim=-1)
        mel = torch.stack(
            [self.decoder(feat.unsqueeze(0), length, spk_output) for feat, length in zip(decoder_features, lengths)]
        )
        return mel
