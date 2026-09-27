import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from optimizer_resurrection.models.pure_resnet import ProductUnitConv2d
from optimizer_resurrection.pure_landscape import (
    LandscapeDiagnostics,
    _tensor_stats,
    fixed_probe_indices,
    probe_epochs,
)


class ScalarBinaryLogits(nn.Module):
    def __init__(self, value=0.2):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([value], dtype=torch.float32))

    def forward(self, x):
        positive = x[:, 0] * self.scale[0]
        return torch.stack((positive, torch.zeros_like(positive)), dim=1)


class BatchNormBinaryLogits(ScalarBinaryLogits):
    def __init__(self):
        super().__init__(0.2)
        self.bn = nn.BatchNorm1d(1, affine=False)

    def forward(self, x):
        positive = self.bn(x)[:, 0] * self.scale[0]
        return torch.stack((positive, torch.zeros_like(positive)), dim=1)


def args(**updates):
    values = dict(weight_decay=0.1, seed=3, stage="test", epochs=3)
    values.update(updates)
    return SimpleNamespace(**values)


def scalar_probe():
    return torch.ones(4, 1), torch.zeros(4, dtype=torch.long)


def test_fixed_probe_and_short_schedule_are_deterministic():
    assert fixed_probe_indices(100) == fixed_probe_indices(100)
    assert len(fixed_probe_indices(100)) == 32
    assert len(set(fixed_probe_indices(100))) == 32
    assert probe_epochs(args()) == (0, 1, 3)
    assert probe_epochs(args(stage="exploratory", epochs=160)) == (0, 1, 40, 80, 81, 120, 121, 160)
    with pytest.raises(ValueError):
        fixed_probe_indices(3, 4)


def test_finite_large_gradient_norm_uses_stable_diagnostic_reduction():
    stats = _tensor_stats(torch.tensor([1e30, 1e30], dtype=torch.float32))
    assert stats["l2"] == pytest.approx(2.0**0.5 * 1e30, rel=1e-6)
    assert stats["rms"] == pytest.approx(1e30, rel=1e-6)
    assert stats["max_abs"] == pytest.approx(1e30, rel=1e-6)


def test_probe_hessian_trace_and_loss_slices_match_scalar_logistic_geometry():
    model = ScalarBinaryLogits(0.2)
    model.train()
    diagnostics = LandscapeDiagnostics(model, args(), scalar_probe())
    result = diagnostics.probe(1)
    probability = torch.sigmoid(torch.tensor(0.2)).item()
    expected_data_hessian = probability * (1.0 - probability)
    assert result["landscape/probe/valid"]
    assert result["landscape/probe/hessian_valid"]
    assert result["landscape/probe/hessian_signed_rayleigh"] == pytest.approx(expected_data_hessian + 0.1, rel=1e-5)
    assert result["landscape/probe/hessian_rayleigh_data"] == pytest.approx(expected_data_hessian, rel=1e-5)
    assert result["landscape/probe/hessian_trace_estimate"] == pytest.approx(expected_data_hessian + 0.1, rel=1e-5)
    assert result["landscape/probe/hessian_trace_data_estimate"] == pytest.approx(expected_data_hessian, rel=1e-5)
    assert result["landscape/probe/hessian_trace_standard_error"] == pytest.approx(0.0, abs=1e-8)
    assert result["landscape/probe/hessian_relative_residual"] < 1e-5
    assert result["landscape/probe/slice_valid"]
    assert result["landscape/probe/slice1/objective_alpha_+0.00"] == pytest.approx(
        result["landscape/probe/objective"]
    )
    center = "landscape/probe/slice2d/objective_a_+0.00_b_+0.00"
    assert result[center] == pytest.approx(result["landscape/probe/objective"])
    assert sum(key.startswith("landscape/probe/slice2d/objective_") for key in result) == 25
    assert all(key in result for key in (
        "landscape/probe/slice2d/valid_a_-0.10_b_+0.10",
        "landscape/probe/slice2d/ce_a_+0.05_b_-0.05",
    ))


def test_loss_slices_survive_an_unavailable_hessian(monkeypatch):
    model = ScalarBinaryLogits()
    diagnostic = LandscapeDiagnostics(model, args(), scalar_probe())

    def unavailable(*_args):
        raise NotImplementedError("second derivatives unsupported")

    monkeypatch.setattr(diagnostic, "_local_geometry", unavailable)
    result = diagnostic.probe(1)
    assert result["landscape/probe/valid"]
    assert not result["landscape/probe/hessian_valid"]
    assert result["landscape/probe/hessian_status"] == "unsupported"
    assert result["landscape/probe/slice_valid"]
    assert result["landscape/probe/slice2d/objective_a_+0.00_b_+0.00"] is not None


def test_probe_restores_parameters_buffers_gradients_mode_and_rng():
    model = BatchNormBinaryLogits()
    model.train()
    model.scale.grad = torch.tensor([0.75])
    parameters = [p.detach().clone() for p in model.parameters()]
    buffers = [b.detach().clone() for b in model.buffers()]
    gradient = model.scale.grad.clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    diagnostics = LandscapeDiagnostics(
        model, args(weight_decay=0.0),
        (torch.arange(4, dtype=torch.float32).reshape(-1, 1), torch.zeros(4, dtype=torch.long)),
    )
    result = diagnostics.probe(1)
    assert result["landscape/probe/bn_comparison_valid"]
    assert result["landscape/probe/bn_eval_vs_batch_logit_max_abs"] > 0
    assert result["landscape/probe/bn_misclassification_count_delta"] is not None
    assert model.training
    assert all(torch.equal(p, saved) for p, saved in zip(model.parameters(), parameters))
    assert all(torch.equal(b, saved) for b, saved in zip(model.buffers(), buffers))
    assert torch.equal(model.scale.grad, gradient)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and np.array_equal(numpy_rng[1], after_numpy[1])
    assert numpy_rng[2:] == after_numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


class TinyProductModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.product = ProductUnitConv2d(1, 2, kernel_size=1)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.product(x).flatten(1))


def test_minibatch_metrics_capture_product_activations_and_actual_updates():
    model = TinyProductModel()
    x = torch.ones(3, 1, 2, 2)
    y = torch.tensor([0, 1, 0])
    diagnostic = LandscapeDiagnostics(model, args(weight_decay=0.001), (x, y))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    diagnostic.begin_epoch(1)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    diagnostic.before_step()
    optimizer.step()
    diagnostic.after_step()
    metrics = diagnostic.epoch_metrics()
    assert metrics["landscape/train/minibatch_count"] == 1
    assert metrics["landscape/train/gradient_valid_batches"] == 1
    assert metrics["landscape/train/global/data_l2_mean"] > 0
    assert metrics["landscape/train/global/actual_update_l2_mean"] > 0
    assert metrics["landscape/train/global/actual_update_l2_max_batch"] == 0
    assert metrics["landscape/activation/product/input_floor_fraction"] is not None
    assert metrics["landscape/activation/product/pre_exp_max"] is not None
    assert any(key.startswith("landscape/train/layer/product/") for key in metrics)
    probe_metrics = diagnostic.probe(1)
    assert probe_metrics["landscape/probe/activation/product/input_floor_fraction"] is not None


def test_diagnostic_call_does_not_change_a_training_step():
    baseline = ScalarBinaryLogits(0.2)
    measured = copy.deepcopy(baseline)
    x, y = scalar_probe()
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.03, weight_decay=0.1)
    measured_optimizer = torch.optim.SGD(measured.parameters(), lr=0.03, weight_decay=0.1)
    baseline_optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(baseline(x), y).backward()
    baseline_optimizer.step()

    diagnostic = LandscapeDiagnostics(measured, args(), (x, y))
    diagnostic.begin_epoch(1)
    measured_optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(measured(x), y).backward()
    diagnostic.before_step()
    measured_optimizer.step()
    diagnostic.after_step()
    measured_state = [p.detach().clone() for p in measured.parameters()]
    diagnostic.probe(1)
    assert all(torch.equal(p, after) for p, after in zip(measured.parameters(), measured_state))
    assert torch.equal(baseline.scale, measured.scale)


def test_bn_probe_keeps_finite_batch_statistics_when_eval_is_invalid():
    class EvalInvalid(ScalarBinaryLogits):
        def forward(self, x):
            output = super().forward(x)
            return output if self.training else output * float("nan")
    model = EvalInvalid()
    model.train()
    result = LandscapeDiagnostics(model, args(), scalar_probe()).probe(0)
    assert not result["landscape/probe/bn_eval_finite"]
    assert result["landscape/probe/bn_batch_stats_finite"]
    assert result["landscape/probe/bn_batch_stats_ce_loss"] is not None
    assert result["landscape/probe/bn_eval_vs_batch_logit_max_abs"] is None
    assert model.training


def test_slices_remain_valid_when_higher_derivatives_are_unsupported(monkeypatch):
    model = ScalarBinaryLogits()
    diagnostic = LandscapeDiagnostics(model, args(), scalar_probe())
    def unsupported(*unused):
        raise NotImplementedError("test HVP unavailable")
    monkeypatch.setattr(diagnostic, "_hvp", unsupported)
    result = diagnostic.probe(1)
    assert result["landscape/probe/hessian_status"] == "unsupported"
    assert result["landscape/probe/slice_status"] == "valid"
    assert result["landscape/probe/slice2d/objective_a_+0.00_b_+0.00"] == pytest.approx(
        result["landscape/probe/objective"])
