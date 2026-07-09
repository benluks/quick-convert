from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import sentencepiece as spm

from quick_convert.components.encoders import RVQDisentangler
from quick_convert.components.decoders import ChatterboxSpectrogramGenerator as CSG

from quick_convert.components.encoders.rvq_disentangler import RVQDisentanglerOutput
from quick_convert.components.ssl import ContentEncoder, ContentFeatures
from quick_convert.data.index.base import Indexer
from quick_convert.data.types import AudioBatch
from quick_convert.pipelines.training.modules.encoder_decoder.base import BaseEncoderDecoderTrainingModule


@dataclass
class ControllableRVQOutput:
    # mel spectrogram
    decoder: torch.Tensor
    encoder: RVQDisentanglerOutput
    loss: Optional[torch.tensor]


# @dataclass
# class OnTheFlyContentEncoder:
#     model: ContentEncoder
#     max_length: Optional[int] = None
#     requires_grad: Optional[bool] = False


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
        rvq_loss_weights: dict[str, float] = {
            "commitment_loss": 1,
            "codebook_loss": 1,
            "load_balancing_loss": 1,
            "mse_loss": 1,
        },
        distillation_loss_weights: dict[str, float] = {"ling": 1, "spk": 1, "emo": 1, "pros": 1},
        adv_loss_weights: dict[str, float] = {"spk_ling": 1, "spk_pros": 1, "ling_spk": 1, "ling_pros": 1},
        tokenizer: Optional[spm.SentencePieceProcessor] = None,
        tokenizer_pad_id: Optional[int] = None,
        decoder_loss_weight: Optional[float] = None,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        adv_loss_hold_off: int = 1000,
        # encoders that'll run online, as opposed to using pre-computed featurs
        online_encoders: dict[str, ContentEncoder] = {},
        **kwargs: Any,
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
        # render + log qualitative audio/mel every N train steps (0 disables);
        # logged to wandb (or tensorboard) and saved under <out_dir>/samples/.
        self.sample_log_every_n_steps = 2000

        self.save_hyperparameters(ignore=["encoder", "decoder", "tokenizer", "content_encoder"])
        self._validate_inputs(
            rvq_loss_weights=rvq_loss_weights,
            distillation_loss_weights=distillation_loss_weights,
            adv_loss_weights=adv_loss_weights,
        )

        # has to be declared like this, since the parent lightning trainer has
        # access to the datasets, which are used to fit the indexer
        self.indexers = {"speaker": Indexer("{sample.resources.spkid.value}")}

        # set encoders (frozen, instead of precomputed features)
        object.__setattr__(self, "online_encoders", {})

        for enc_name, online_encoder in online_encoders.items() or []:
            online_encoder.requires_grad_(False)
            online_encoder.model.eval()
            self.online_encoders[enc_name] = online_encoder

            # store as attribute to it's accessible, but won't be added to state dict

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

    def _get_resource(self, batch: AudioBatch, name, feature_encoder_name=None):
        """
        Method to optionally use precomputed features, otherwise rely on an encoder (must be passed to this class' __init__,
        and referenced by name). If `feautre_encoder_name`
        """
        if batch.resources.get(name, None) is not None:
            resource = batch.resources[name]
            return resource.values, resource.lengths
        elif feature_encoder_name is not None and (feature_encoder := self.online_encoders.get(name, None)) is not None:
            # feature_encoder.model.eval()
            with torch.inference_mode():
                content: ContentFeatures = feature_encoder(batch)
            return content.values.detach(), content.lengths
        else:
            raise RuntimeError(f"""Ensure batch has resource named {name} or this value can be computed using the appropriate encoder.
                               name='{name}', feature_encoder_name='{feature_encoder_name}'""")

    def _shared_step(self, batch: AudioBatch, stage: str) -> ControllableRVQOutput:
        """Compute all losses, log them, and return the total loss.

        Called by both :meth:`training_step` and :meth:`validation_step` with
        the appropriate ``stage`` prefix (``'train'`` / ``'val'``).

        Args:
            batch: Batch of waveforms, lengths, and optional transcripts.
            stage: Logging prefix; controls ``on_step`` logging behaviour.

        Returns:
            Scalar total loss tensor passed to the Lightning optimiser.
        """
        features, lengths = self._get_resource(batch, "content", "content_encoder")
        # for now, only precomputed emo2vec
        emo_targets, emo_lengths = self._get_resource(batch, "emo2vec")

        spk_targets = torch.LongTensor(self.indexers["speaker"].encode_many(batch.resources["spkid"])).to(self.device)

        pros_targets = (
            batch.resources["pros"].values
            if "pros" in batch.resources and batch.resources["pros"] is not None
            else None
        )

        # tokenized_transcripts = self.tokenizer.encode(targets)
        token_ids = batch.resources["token_ids"]  # transcript token ids, shape (B, T_text) or None

        encoder_output: RVQDisentanglerOutput = self.encoder.compute_loss(
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
            self.hparams.rvq_loss_weights["commitment_loss"] * encoder_output.loss.rvq["commitment_loss"]
            + self.hparams.rvq_loss_weights["codebook_loss"] * encoder_output.loss.rvq["codebook_loss"]
            + self.hparams.rvq_loss_weights["mse_loss"] * encoder_output.loss.rvq["mse_loss"]
            + self.hparams.rvq_loss_weights["load_balancing_loss"] * encoder_output.loss.rvq["load_balancing_loss"]
        )

        # Weighted sum of per-attribute distillation losses from frozen teacher encoders
        distil_loss = (
            self.hparams.distillation_loss_weights["ling"] * encoder_output.loss.distill["ctc_loss"]
            + self.hparams.distillation_loss_weights["spk"] * encoder_output.loss.distill["spk_loss"]
            + self.hparams.distillation_loss_weights["emo"] * encoder_output.loss.distill["emo_loss"]
            + self.hparams.distillation_loss_weights["pros"] * encoder_output.loss.distill["pros_loss"]
        )

        # Weighted sum of adversarial losses that enforce cross-attribute disentanglement
        adv_loss = (
            self.hparams.adv_loss_weights["spk_ling"] * encoder_output.loss.adv["spk_loss_ling"]
            + self.hparams.adv_loss_weights["spk_pros"] * encoder_output.loss.adv["spk_loss_pros"]
            + self.hparams.adv_loss_weights["ling_spk"] * encoder_output.loss.adv["ling_loss_spk"]
            + self.hparams.adv_loss_weights["ling_pros"] * encoder_output.loss.adv["ling_loss_pros"]
        )

        # DECODING

        # Decoder reconstruction loss: z_q is used as both the flow target (x1)
        # and the conditioning signal; detach x1 so gradients flow only through mu
        # TODO
        decoder_features = torch.cat([encoder_output.router.zs[1], encoder_output.router.zs[2]], dim=-1)

        decoder_loss, decoder_output, decoder_mae = self.decoder.compute_loss(
            features=decoder_features,
            # feature lengths
            lengths=lengths,
            target_wav=batch.waveforms,
            # audio waveform lengths
            wav_lens=batch.lengths,
            sampling_rate=batch.sample_rates[0],
            speaker_embedding=encoder_output.head_outputs["spk"].speaker_features,
        )

        loss = rvq_loss + distil_loss + adv_loss + self.hparams.decoder_loss_weight * decoder_loss
        log_dict = {
            f"{stage}/loss": loss,
            # RVQ losses
            f"{stage}/rvq/loss": rvq_loss,
            f"{stage}/rvq/commitment_loss": encoder_output.loss.rvq["commitment_loss"],
            f"{stage}/rvq/codebook_loss": encoder_output.loss.rvq["codebook_loss"],
            f"{stage}/rvq/mse_loss": encoder_output.loss.rvq["mse_loss"],
            f"{stage}/rvq/load_balancing_loss": encoder_output.loss.rvq["load_balancing_loss"],
            # Distillation losses
            f"{stage}/distill/loss": distil_loss,
            f"{stage}/distill/ctc_loss": encoder_output.loss.distill["ctc_loss"],
            f"{stage}/distill/spk_loss": encoder_output.loss.distill["spk_loss"],
            f"{stage}/distill/emo_loss": encoder_output.loss.distill["emo_loss"],
            f"{stage}/distill/pros_loss": encoder_output.loss.distill["pros_loss"],
            # Adversarial losses
            f"{stage}/adv/loss": adv_loss,
            f"{stage}/adv/spk_loss_ling": encoder_output.loss.adv["spk_loss_ling"],
            f"{stage}/adv/spk_loss_pros": encoder_output.loss.adv["spk_loss_pros"],
            f"{stage}/adv/ling_loss_spk": encoder_output.loss.adv["ling_loss_spk"],
            f"{stage}/adv/ling_loss_pros": encoder_output.loss.adv["ling_loss_pros"],
            # spk accuracy
            f"{stage}/adv/spk_acc_ling": encoder_output.loss.metrics["spk_acc_ling"],
            f"{stage}/adv/spk_acc_pros": encoder_output.loss.metrics["spk_acc_pros"],
            f"{stage}/spk_acc": encoder_output.loss.metrics["spk_acc"],
            # Decoder loss
            f"{stage}/decoder/loss": decoder_loss,
            f"{stage}/decoder/mae": decoder_mae,
        }

        self.log_dict(
            log_dict,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage == "train"),
            sync_dist=True,
            batch_size=len(batch.paths),
        )

        self.logger.experiment.add_image(
            "router/probabilities",
            encoder_output.loss.states["router_probabilities"].detach().unsqueeze(0),  # (1, 8, 3)
            self.global_step,
            dataformats="CHW",
        )
        self.logger.experiment.add_image(
            "router/logits",
            encoder_output.loss.states["router_logits"].detach().unsqueeze(0),  # (1, 8, 3)
            self.global_step,
            dataformats="CHW",
        )

        return ControllableRVQOutput(decoder=decoder_output, encoder=encoder_output, loss=loss)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: AudioBatch, batch_idx: int) -> torch.Tensor:
        output = self._shared_step(batch, "train")
        return output.loss

    def validation_step(self, batch: AudioBatch, batch_idx: int) -> torch.Tensor:
        output = self._shared_step(batch, "val")

        if self.trainer.is_global_zero and batch_idx == 0:
            mel, gen_audio = self._generate_media(output.decoder)
            self._log_media(batch, mel, output.encoder.lengths, gen_audio)

        return output.loss

    @torch.no_grad()
    def _generate_media(self, mel):
        """Render mel-spectrograms + waveforms for the first samples of a batch,
        for qualitative monitoring. Returns ``(mel, gen_audio)``."""
        gen_audio = self.decoder.mel2wav(mel)
        return mel, gen_audio

    def _log_media(self, batch, mel, feat_lengths, gen_audio, max_samples: int = 16) -> None:
        """Persist + log generated spectrograms and audio.

        Always writes wavs to ``<trainer.default_root_dir>/samples/step_<N>/`` so the
        results exist regardless of logger. Also logs to the active logger:
        ``wandb.Audio`` / ``wandb.Image`` for a WandbLogger, or ``add_audio`` /
        ``add_image`` for a TensorBoardLogger (so TensorBoard still works). Source audio
        is logged once, at global_step 0. Audio is not trimmed, so trailing padding may
        be silent.
        """
        import torchaudio
        from pathlib import Path

        step = int(self.global_step)
        gen_sr = int(self.decoder.vocoder.sampling_rate)

        out_dir = getattr(self.trainer, "default_root_dir", None) or "."
        sample_dir = Path(out_dir) / "samples" / f"step_{step:08d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        logger_name = type(self.logger).__name__ if self.logger is not None else ""
        experiment = getattr(self.logger, "experiment", None)
        wandb_payload: dict = {}

        for i, (sample, wav_length) in enumerate(zip(batch, batch.lengths)):
            if i >= max_samples:
                break
            utt = sample.utt_id
            wav = gen_audio[i].detach().cpu().float().reshape(-1)  # (T,)

            # always persist to disk
            torchaudio.save(str(sample_dir / f"{utt}.wav"), wav.unsqueeze(0), gen_sr)

            mel_img = mel[i].detach().cpu().float()
            mel_img = (mel_img - mel_img.min()) / (mel_img.max() - mel_img.min() + 1e-8)

            # todo: abstract loggers
            if self.global_rank == 0:
                if logger_name == "WandbLogger":
                    import wandb

                    wandb_payload[f"generated/{utt}"] = wandb.Audio(wav.numpy(), sample_rate=gen_sr)
                    wandb_payload[f"mel/{utt}"] = wandb.Image(mel_img.numpy())
                    if step == 0:
                        orig = sample.waveform.detach().cpu().float().reshape(-1)
                        wandb_payload[f"original/{utt}"] = wandb.Audio(
                            orig.numpy(), sample_rate=int(sample.sample_rate)
                        )
                elif logger_name == "TensorBoardLogger" and experiment is not None:
                    experiment.add_image(f"generated/{utt}", mel_img.unsqueeze(0)[..., : feat_lengths[i]], step)
                    experiment.add_audio(f"generated/{utt}", wav.unsqueeze(0)[:, :wav_length], step, sample_rate=gen_sr)
                    if step == 0:
                        orig = sample.waveform.detach().cpu().float().reshape(-1)
                        experiment.add_audio(
                            f"original/{utt}",
                            orig.unsqueeze(0)[:, :wav_length],
                            step,
                            sample_rate=int(sample.sample_rate),
                        )

        if wandb_payload and experiment is not None:
            experiment.log(wandb_payload)

    @torch.inference_mode()
    def inference(
        self,
        batch: AudioBatch,
    ) -> torch.Tensor:
        """Run the decoder in inference mode on the provided features and speaker embedding."""
        features, lengths = self._get_resource(batch, "content", "content_encoder")
        emo_targets, emo_lengths = self._get_resource(batch, "emo2vec")

        encoder_output = self.encoder.inference(features, lengths, emo_targets, emo_lengths)
        decoder_features = torch.cat([encoder_output.router.zs[1], encoder_output.router.zs[2]], dim=-1)
        mel = self.decoder.inference(decoder_features, lengths, encoder_output.head_outputs["spk"].speaker_features)

        wav = self.decoder.mel2wav(mel)

        return mel, wav
