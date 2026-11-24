import torch
from torch import nn

from quick_convert.training.hifigan import HiFiGANObjective


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 3, padding=1)

    def forward(self, real, fake):
        r, g = self.conv(real), self.conv(fake)
        return [r], [g], [[r]], [[g]]


def test_discriminator_detaches_generator():
    objective = HiFiGANObjective(discriminators=[TinyDiscriminator()])
    real = torch.randn(2, 1280)
    fake = torch.randn_like(real, requires_grad=True)
    objective(real, fake, "discriminator").backward()
    assert fake.grad is None
    assert objective.discriminators[0].conv.weight.grad is not None


def test_generator_gradient_through_frozen_discriminator():
    objective = HiFiGANObjective(discriminators=[TinyDiscriminator()])
    objective.requires_grad_(False)
    real = torch.randn(2, 1280)
    fake = torch.randn_like(real, requires_grad=True)
    loss = objective(real, fake, "generator")
    loss.backward()
    assert torch.isfinite(loss)
    assert fake.grad is not None
    assert fake.grad.abs().sum() > 0
    assert objective.discriminators[0].conv.weight.grad is None
