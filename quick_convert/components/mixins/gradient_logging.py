from __future__ import annotations

from collections.abc import Mapping

import torch


class ObjectiveGradientLoggingMixin:
    @staticmethod
    def _gradient_squared_norm(
        grads: tuple[torch.Tensor | None, ...],
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        value = reference.new_zeros((), dtype=torch.float32)

        for grad in grads:
            if grad is not None:
                value = value + grad.float().square().sum()

        return value

    @staticmethod
    def _gradient_dot_product(
        first: tuple[torch.Tensor | None, ...],
        second: tuple[torch.Tensor | None, ...],
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        value = reference.new_zeros((), dtype=torch.float32)

        for first_grad, second_grad in zip(first, second):
            if first_grad is None or second_grad is None:
                continue

            value = value + (first_grad.float() * second_grad.float()).sum()

        return value

    def log_objective_gradients(
        self,
        *,
        losses: Mapping[str, torch.Tensor],
        parameters: tuple[torch.nn.Parameter, ...],
        prefix: str,
    ) -> None:
        if not parameters or not losses:
            return

        reference = parameters[0]

        objective_grads: dict[
            str,
            tuple[torch.Tensor | None, ...],
        ] = {}

        objective_norms: dict[str, torch.Tensor] = {}

        for name, loss in losses.items():
            grads = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            grads = tuple(grad.detach() if grad is not None else None for grad in grads)

            objective_grads[name] = grads

            squared_norm = self._gradient_squared_norm(
                grads,
                reference=reference,
            )

            objective_norms[name] = squared_norm.sqrt()

        metrics: dict[str, torch.Tensor] = {}

        for name, norm in objective_norms.items():
            metrics[f"{prefix}/{name}/norm"] = norm

        names = sorted(objective_grads)

        for i, first_name in enumerate(names):
            for second_name in names[i + 1 :]:
                dot = self._gradient_dot_product(
                    objective_grads[first_name],
                    objective_grads[second_name],
                    reference=reference,
                )

                denominator = (objective_norms[first_name] * objective_norms[second_name]).clamp_min(1e-12)

                metrics[f"{prefix}/{first_name}_vs_{second_name}/cosine"] = dot / denominator

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
