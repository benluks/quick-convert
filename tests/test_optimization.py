import copy

import pytest
import torch

from quick_convert.training.lightning.optim import LinearWarmup, Optimization


def build(total_steps=100, warmup=None, **kwargs):
    optimization = Optimization(
        optimizer_kwargs={"lr": 1e-4},
        lr_scheduler_kwargs={"T_max": "auto", "eta_min": 1e-6},
        warmup=warmup,
        **kwargs,
    )
    result = optimization.configure([torch.nn.Parameter(torch.ones(1))], total_steps=total_steps)
    return optimization, result["optimizer"], result["lr_scheduler"]["scheduler"]


@pytest.mark.parametrize("warmup", [None, LinearWarmup(5), LinearWarmup(0.05)])
def test_auto_cosine_decays_and_stays_at_minimum(warmup):
    optimization, optimizer, scheduler = build(warmup=warmup)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(120):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    start = 0 if warmup is None else 5
    assert rates[start] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(rates[start:], rates[start + 1 :], strict=False))
    assert rates[100:] == pytest.approx([1e-6] * 21)
    assert optimization.lr_scheduler_kwargs["T_max"] == "auto"


def test_auto_cosine_resume_preserves_schedule():
    _, optimizer, scheduler = build(warmup=LinearWarmup(0.05))
    for _ in range(60):
        optimizer.step()
        scheduler.step()
    _, restored_optimizer, restored_scheduler = build(warmup=LinearWarmup(0.05))
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    for _ in range(60):
        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()
        assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("total_steps", [None, float("inf"), 0, -1, True, 1.5])
def test_auto_requires_finite_positive_integer_budget(total_steps):
    with pytest.raises(ValueError):
        build(total_steps=total_steps)


@pytest.mark.parametrize("warmup", [LinearWarmup(100), LinearWarmup(1.0), LinearWarmup(101)])
def test_auto_requires_remaining_decay_steps(warmup):
    with pytest.raises(ValueError, match="leave at least one"):
        build(warmup=warmup)


@pytest.mark.parametrize("kwargs", [{"interval": "epoch"}, {"frequency": 2}])
def test_auto_requires_step_cadence(kwargs):
    with pytest.raises(ValueError, match="interval='step'"):
        build(**kwargs)


def test_explicit_cosine_duration_is_unchanged():
    optimization = Optimization(optimizer_kwargs={"lr": 1e-4}, lr_scheduler_kwargs={"T_max": 10})
    result = optimization.configure([torch.nn.Parameter(torch.ones(1))])
    assert type(result["lr_scheduler"]["scheduler"]) is torch.optim.lr_scheduler.CosineAnnealingLR
