from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from quick_convert.components.encoders.conformer_encoder import ConformerEncoderSSL
from quick_convert.components.layers.grl import GradientReversalLayer
from quick_convert.components.layers.heads import HeadOutput, HeadTarget, SupervisedHead
from quick_convert.components.layers.routers import DeterministicRVQLayerRouter, LearnedRVQLayerRouter, RouterOutput
from quick_convert.components.layers.rvq import RVQOutput
from quick_convert.utils.masking import make_padding_mask, masked_loss

from ..layers import ResidualVectorQuantizer


@dataclass(frozen=True)
class HeadSpec:
    """
    Wiring for a supervised head.

    Args:
        route:
            Routed RVQ representation consumed by the head.
        target:
            Resource name used as supervision.
        loss_weight:
            Weight applied to the supervised loss.
        indexer:
            Optional indexer used to convert categorical resources into
            integer class labels.
        optional:
            Whether the head may be skipped when its target resource is absent.
    """

    route: str
    target: str
    loss_weight: float = 1.0
    indexer: str | None = None
    optional: bool = False


@dataclass(frozen=True)
class AdversarialSpec:
    """
    Describes an adversarial prediction task.

    `predictor` identifies the task whose target and head type should be used.
    `source_route` identifies the representation from which that task should
    not be recoverable.
    """

    predictor: str
    source_route: str
    loss_weight: float = 1.0


@dataclass
class RVQDisentanglerLoss:
    rvq: dict[str, torch.Tensor]
    supervised: dict[str, torch.Tensor]
    adv: dict[str, torch.Tensor] | None = None
    metrics: dict[str, torch.Tensor] | None = None
    states: dict[str, Any] | None = None


@dataclass
class RVQDisentanglerOutput:
    content: torch.Tensor

    # having both lengths and padding mask is techincally redundant, but we will need different ones for different
    # purposes and it makes more sense to store both than to constantly be re-computing one from the other.
    lengths: torch.Tensor
    padding_mask: torch.Tensor

    rvq: RVQOutput
    router: RouterOutput

    # the head output exists on the disentangler (encoder) level. It's akin to an x-vector
    head_outputs: dict[str, HeadOutput] | None = None
    loss: RVQDisentanglerLoss | None = None


class RVQDisentangler(nn.Module):
    def __init__(
        self,
        content_encoder: ConformerEncoderSSL,
        rvq: ResidualVectorQuantizer,
        router: LearnedRVQLayerRouter | DeterministicRVQLayerRouter,
        heads: dict[str, SupervisedHead],
        head_specs: dict[str, HeadSpec],
        adversarial_specs: dict[str, AdversarialSpec] | None = None,
        **kwargs,
    ):
        super().__init__()

        self.content_encoder = content_encoder
        self.rvq = rvq
        self.router = router

        self.heads = nn.ModuleDict(heads)
        self.head_specs = dict(head_specs)

        self._validate_head_configuration()

        self.adversarial_specs = dict(adversarial_specs or {})
        self._validate_adversarial_configuration()

        self.adversarial_heads = nn.ModuleDict(
            {name: deepcopy(self.heads[spec.predictor]) for name, spec in self.adversarial_specs.items()}
        )

        self.grl = GradientReversalLayer()

    def _validate_head_configuration(self) -> None:
        head_names = set(self.heads)
        spec_names = set(self.head_specs)

        if missing := head_names - spec_names:
            raise ValueError(f"Heads missing HeadSpec entries: {sorted(missing)}")

        if unknown := spec_names - head_names:
            raise ValueError(f"HeadSpec entries without heads: {sorted(unknown)}")

        for name, spec in self.head_specs.items():
            if not spec.route:
                raise ValueError(f"Head {name!r} has an empty route.")

            if not spec.target:
                raise ValueError(f"Head {name!r} has an empty target resource.")

    def _validate_adversarial_configuration(self) -> None:
        for adversary_name, spec in self.adversarial_specs.items():
            if spec.predictor not in self.heads:
                raise ValueError(
                    f"Adversary {adversary_name!r} references unknown "
                    f"predictor head {spec.predictor!r}. "
                    f"Available heads: {sorted(self.heads)}"
                )

            if not spec.source_route:
                raise ValueError(f"Adversary {adversary_name!r} has an empty source route.")

            predictor_route = self.head_specs[spec.predictor].route

            if spec.source_route == predictor_route:
                raise ValueError(
                    f"Adversary {adversary_name!r} predicts "
                    f"{spec.predictor!r} from its own supervised route "
                    f"{spec.source_route!r}. This would directly oppose the "
                    "normal supervised objective."
                )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> RVQDisentanglerOutput:
        padding_mask = make_padding_mask(
            lengths,
            max_length=features.shape[1],
        )
        return self.encode(
            features,
            padding_mask,
            lengths=lengths,
        )

    def encode(
        self,
        features: int["b t d"],
        padding_mask: int["b t [1]"],
        lengths: int["b"] | None = None,
    ) -> RVQDisentanglerOutput:
        content = self.content_encoder(features, padding_mask=padding_mask)

        # RVQ expects B, D, T.
        content = content.transpose(1, 2)

        rvq_output: RVQOutput = self.rvq(content, padding_mask)

        rvq_output = replace(rvq_output, z_q=rvq_output.z_q.transpose(1, 2))
        router_output = self._route(rvq_output.layer_z_qs)

        return RVQDisentanglerOutput(
            content=content,
            # having lengths and padding
            lengths=lengths if lengths is not None else padding_mask.sum(dim=1),
            padding_mask=padding_mask,
            rvq=rvq_output,
            router=router_output,
        )

    def _route(self, layer_z_qs) -> RouterOutput:
        return self.router(
            self.rvq.quantizers,
            layer_z_qs,
            compute_loss=True,
        )

    def compute_loss(
        self, features, lengths, head_targets: dict[str, HeadTarget], run_adv: bool = True
    ) -> RVQDisentanglerOutput:
        output = self.forward(features, lengths)

        rvq_mse_loss = masked_loss(
            F.mse_loss,
            preds=output.rvq.z_q,
            targets=output.content.detach().transpose(1, 2),
            mask=output.padding_mask,
        )

        rvq_losses = {
            "commitment_loss": output.rvq.loss.commitment_loss,
            "codebook_loss": output.rvq.loss.codebook_loss,
            "mse_loss": rvq_mse_loss,
            "load_balancing_loss": output.router.loss,
        }

        head_outputs: dict[str, HeadOutput] = {}
        supervised_losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        unknown_targets = set(head_targets) - set(self.heads)
        if unknown_targets:
            raise ValueError(f"Targets were provided for unknown heads: {sorted(unknown_targets)}")

        for name, head in self.heads.items():
            target = head_targets.get(name)

            # This permits optional heads whose target is not present in a
            # particular dataset or batch.
            if target is None:
                continue

            spec = self.head_specs[name]

            try:
                routed_features = output.router.zs[spec.route]
            except KeyError as error:
                available = sorted(output.router.zs.keys())
                raise KeyError(
                    f"Head {name!r} requires route {spec.route!r}, but the router produced: {available}"
                ) from error

            head_output = head.compute_loss(
                routed_features,
                targets=target,
                lengths=output.lengths,
                padding_mask=output.padding_mask,
            )

            head_outputs[name] = head_output

            if head_output.loss is not None:
                supervised_losses[name] = head_output.loss

            for metric_name, metric in head_output.metrics.items():
                metrics[f"{name}/{metric_name}"] = metric

        adversarial_outputs: dict[str, HeadOutput] = {}
        adversarial_losses: dict[str, torch.Tensor] = {}

        if run_adv:
            for adversary_name, adversarial_head in self.adversarial_heads.items():
                spec = self.adversarial_specs[adversary_name]

                target = head_targets.get(spec.predictor)

                # This also handles an optional predictor target that was absent
                # from this batch.
                if target is None:
                    continue

                try:
                    source_features = output.router.zs[spec.source_route]
                except KeyError as error:
                    available = sorted(output.router.zs)

                    raise KeyError(
                        f"Adversary {adversary_name!r} requires route "
                        f"{spec.source_route!r}, but the router produced "
                        f"{available}."
                    ) from error

                adversarial_output = adversarial_head.compute_loss(
                    self.grl(source_features),
                    targets=target,
                    lengths=output.lengths,
                    padding_mask=output.padding_mask,
                )

                adversarial_outputs[adversary_name] = adversarial_output

                if adversarial_output.loss is not None:
                    adversarial_losses[adversary_name] = adversarial_output.loss

                for metric_name, metric in adversarial_output.metrics.items():
                    metrics[f"adversarial/{adversary_name}/{metric_name}"] = metric

        loss = RVQDisentanglerLoss(
            rvq=rvq_losses,
            supervised=supervised_losses,
            adv=adversarial_losses,
            metrics=metrics,
            states={
                "router_probabilities": output.router.layer_probabilities,
                "router_logits": output.router.layer_logits,
                "adversarial_outputs": adversarial_outputs,
            },
        )

        return replace(
            output,
            head_outputs=head_outputs,
            loss=loss,
        )

    @torch.inference_mode()
    def inference(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        head_names: list[str] | tuple[str, ...] | None = None,
    ) -> RVQDisentanglerOutput:
        output = self.forward(features, lengths)

        names = tuple(head_names) if head_names is not None else tuple(self.heads.keys())

        head_outputs: dict[str, HeadOutput] = {}

        for name in names:
            if name not in self.heads:
                raise KeyError(f"Unknown head: {name!r}")

            head = self.heads[name]
            spec = self.head_specs[name]
            routed_features = output.router.zs[spec.route]

            head_outputs[name] = head.predict(
                routed_features,
                lengths=output.lengths,
                padding_mask=output.padding_mask,
            )

        return replace(
            output,
            head_outputs=head_outputs,
        )
