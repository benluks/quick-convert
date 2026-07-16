from dataclasses import dataclass
from typing import Callable, Optional

from pathlib import Path

import torch
from torch import nn
from sentencepiece import SentencePieceProcessor


from quick_convert.components.encoders import LinguisticCTCHead
from quick_convert.components.layers import VectorQuantize, LayerWeightedSum
from quick_convert.components.layers.rvq import VQOutput
from quick_convert.components.losses.asr_losses import CTCOutput
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.components.ssl import ContentEncoder

from torch.optim.lr_scheduler import LRScheduler

from quick_convert.data.types import AudioBatch
from quick_convert.pipelines.training.modules.base import BaseTrainingModule
from quick_convert.pipelines.training.optim.base import Optimization
from quick_convert.utils.masking import make_padding_mask
from quick_convert.systems.asr.utils import greedy_ctc_decode
from quick_convert.pipelines.evaluation.metrics.wer.jiwer_wer import JiwerWER


@dataclass
class VQASROutput:
    vq: VQOutput
    ctc: CTCOutput
    loss: torch.Tensor


class VQASRTrainingModule(OnlineResourceMixin, BaseTrainingModule):
    def __init__(
        self,
        quantizer: VectorQuantize,
        ctc_head: LinguisticCTCHead,
        optimization: Optimization,
        tokenizer_model_path: Path = None,
        layer_fusion: Optional[LayerWeightedSum] = None,
        # contextual model to be able to break the independence sampling,
        # something but cpaable of modelling context
        post_quantization_network: Optional[nn.Module] = None,
        online_encoders: Optional[dict[str, ContentEncoder]] = None,
        ctc_loss_weight: float = 1.0,
        commitment_loss_weight: float = 1.0,
        codebook_loss_weight: float = 1.0,
    ):
        super().__init__(
            optimization=optimization,
        )

        self.quantizer = quantizer
        self.ctc_head = ctc_head
        self.online_encoders = nn.ModuleDict(online_encoders or {})
        self.layer_fusion = layer_fusion or nn.Identity()
        self.ctc_loss_weight = ctc_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.post_quantization_network = post_quantization_network

        # decoding for eval
        self.tokenizer = SentencePieceProcessor(model_file=tokenizer_model_path)
        self.val_hyps = []
        self.val_refs = []

        self.save_hyperparameters(
            ignore=[
                "quantizer",
                "ctc_head",
                "online_encoders",
                "optimizer",
                "lr_scheduler",
            ]
        )
        for encoder in self.online_encoders.values():
            encoder.requires_grad_(False)
            encoder.eval()

        self.online_encoders.eval()
        self.online_encoders.requires_grad_(False)

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> VQASROutput:
        features, feature_lengths = self.get_resource(batch, "content")
        token_ids = self.get_resource(batch, "token_ids")

        padding_mask = make_padding_mask(
            feature_lengths,
            max_length=features.shape[1],
        )

        features = self.layer_fusion(features)
        quantizer_output = self.quantizer(features.transpose(1, 2), padding_mask, loss_reduction="batch_by_sample")

        # back to [B, T, D]
        quantized = quantizer_output.z_q.transpose(1, 2)
        contextual_output = self.post_quantization_network(quantized, padding_mask)

        ctc_output: CTCOutput = self.ctc_head.compute_loss(
            contextual_output,
            token_ids.values,
            input_lengths=feature_lengths,
            target_lengths=token_ids.lengths,
        )

        loss = (
            self.ctc_loss_weight * ctc_output.loss
            + self.commitment_loss_weight * quantizer_output.loss.commitment_loss
            + self.codebook_loss_weight * quantizer_output.loss.codebook_loss
        )

        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/ctc_loss": ctc_output.loss,
                f"{stage}/vq/commitment_loss": quantizer_output.loss.commitment_loss,
                f"{stage}/vq/codebook_loss": quantizer_output.loss.codebook_loss,
            },
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch.paths),
        )

        return VQASROutput(
            vq=quantizer_output,
            ctc=ctc_output,
            loss=loss,
        )

    def on_validation_epoch_start(self):
        self.hypothesis_fp = (Path(self.trainer.log_dir) / "hypothesis.txt").open("w")
        self.references_fp = (Path(self.trainer.log_dir) / "ground_truth.txt").open("w")

    def log_validation_output(self, batch: AudioBatch, output: VQASROutput, batch_idx: int):
        writer: torch.utils.tensorboard.SummaryWriter = self.logger.experiment

        transcripts = self.get_resource(batch, "transcript")
        self.references_fp.write("\n".join(transcripts))
        # reshape logits to [B T V]
        for i, (item, logits) in enumerate(zip(batch, output.ctc.logits.transpose(0, 1))):
            if self.tokenizer is not None:
                hypothesis_ids = greedy_ctc_decode(logits=logits)
                hypothesis = self.tokenizer.decode_ids(hypothesis_ids.tolist())
                self.hypothesis_fp.write(hypothesis + "\n")

                if batch_idx == 0:
                    writer.add_text(f"transcript/{item.utt_id}/hypothesis", hypothesis, self.global_step)
                    writer.add_text(f"transcript/{item.utt_id}/ground_truth", transcripts[i], self.global_step)

    def on_validation_epoch_end(self):
        self.hypothesis_fp.close()
        self.references_fp.close()

        wer = JiwerWER().compute(self.references_fp, self.hypothesis_fp)["wer"]
        self.log(
            "wer",
            wer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
