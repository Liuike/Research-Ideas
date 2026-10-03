from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from optimizer_resurrection import punn_architecture_stability as stability
from optimizer_resurrection import punn_manifold_recorded as recorded
from optimizer_resurrection.optim import manifold_muon
from optimizer_resurrection.punn_architecture_data import TASK_SPECS
from optimizer_resurrection.punn_landscape import Dataset


ROOT = Path(__file__).resolve().parents[1]
CPU_RECORDED_CONFIG = ROOT / "configs/product_unit/manifold_landscape_cpu_continuation.yaml"


def _training_args(
    method: str,
    task: str = "f1",
    architecture: str = "small",
    seed: int = 7,
    recipe_override: dict | None = None,
):
    recipe = stability.MANIFOLD_RECIPES[method] if recipe_override is None else recipe_override
    args = [
        "--protocol", stability.MANIFOLD_PROTOCOL,
        "--task", task,
        "--architecture", architecture,
        "--method", method,
        "--seed", str(seed),
        "--data-seed", str(seed + 100000),
        "--epochs", "2",
        "--regularization-lambda", "0.0001" if architecture == "regularized" else "0",
        "--run-group", "test-no-momentum",
        "--stage", "test",
    ]
    for key, value in recipe.items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        args.extend(["--" + key.replace("_", "-"), rendered])
    return stability.parse_args(args)


def _dataset_pair(task: str, data_seed: int):
    input_dim, _, _, output_dim = TASK_SPECS[task]
    generator = torch.Generator(device="cpu").manual_seed(data_seed)
    tensors = (
        torch.rand((12, input_dim), generator=generator) + 0.25,
        torch.randn((12, output_dim), generator=generator) * 0.1,
        torch.rand((5, input_dim), generator=generator) + 0.25,
        torch.randn((5, output_dim), generator=generator) * 0.1,
    )
    digest = hashlib.sha256(
        b"".join(tensor.contiguous().numpy().tobytes() for tensor in tensors)
    ).hexdigest()
    return Dataset(*tensors, digest), {"data_seed": data_seed}


def test_zero_momentum_da10_feeds_each_current_gradient_without_buffer(monkeypatch):
    """The recorded run must pass each sample gradient directly into DA-10."""
    method = stability.MANIFOLD_NO_MOMENTUM_METHOD
    original_direction = manifold_muon.manifold_muon_direction
    original_optimizer = manifold_muon.ManifoldMuon
    observed_gradients = []
    observed_current_gradients = []
    constructor_settings = []
    step_count = 0

    class CapturedManifoldMuon(original_optimizer):
        def __init__(self, params, *args, **kwargs):
            constructor_settings.append({
                "momentum": kwargs.get("momentum"),
                "nesterov": kwargs.get("nesterov"),
            })
            super().__init__(params, *args, **kwargs)

        def step(self, closure=None):
            nonlocal step_count
            result = super().step(closure)
            step_count += 1
            assert all("momentum_buffer" not in state for state in self.state.values())
            return result

    def capture_direction(weight, gradient, *args, **kwargs):
        observed_gradients.append(gradient.detach().clone())
        assert weight.grad is not None
        observed_current_gradients.append(weight.grad.detach().clone())
        return original_direction(weight, gradient, *args, **kwargs)

    monkeypatch.setattr(manifold_muon, "ManifoldMuon", CapturedManifoldMuon)
    monkeypatch.setattr(manifold_muon, "manifold_muon_direction", capture_direction)
    args = _training_args(method)
    result = stability.run_trial(args, dataset_pair=_dataset_pair(args.task, args.data_seed))

    assert result["terminal_outcome"] == "completed"
    assert result["method"] == method
    assert result["momentum"] == 0.0
    assert result["manifold_nesterov"] is False
    assert constructor_settings == [{"momentum": 0.0, "nesterov": False}]
    assert step_count >= 2
    assert len(observed_gradients) == step_count
    assert all(torch.equal(actual, current)
               for actual, current in zip(observed_gradients, observed_current_gradients))
    assert any(not torch.equal(observed_gradients[0], later)
               for later in observed_gradients[1:])


def test_no_momentum_and_existing_da10_expand_to_distinct_paired_recorded_plans():
    config = yaml.safe_load(CPU_RECORDED_CONFIG.read_text(encoding="utf-8"))
    existing_commands = recorded.expand_config(config)
    existing = [recorded.parse_args(command[3:]) for command in existing_commands]

    method = stability.MANIFOLD_NO_MOMENTUM_METHOD
    assert method in stability.MANIFOLD_METHODS
    recipe = copy.deepcopy(stability.MANIFOLD_RECIPES[method])
    assert recipe["momentum"] == 0.0
    assert {key: value for key, value in recipe.items() if key != "momentum"} == {
        key: value for key, value in stability.MANIFOLD_RECIPE.items() if key != "momentum"
    }

    no_momentum_config = {
        **config,
        "methods": [method],
        "optimizer_settings": {method: recipe},
        "run_group": "test-no-momentum-recorded-v1",
    }
    no_momentum_commands = recorded.expand_config(no_momentum_config)
    no_momentum = [recorded.parse_args(command[3:]) for command in no_momentum_commands]

    assert len(existing) == len(no_momentum) == 480
    assert len({args.condition_id for args in existing}) == 480
    assert len({args.condition_id for args in no_momentum}) == 480
    assert {args.task for args in no_momentum} == set(TASK_SPECS)
    assert {args.seed for args in no_momentum} == set(range(30))
    assert all(args.method == method and args.momentum == 0.0 for args in no_momentum)
    assert {args.condition_id for args in existing}.isdisjoint(
        args.condition_id for args in no_momentum
    )

    old_by_cell = {(args.task, args.architecture, args.seed): args for args in existing}
    new_by_cell = {(args.task, args.architecture, args.seed): args for args in no_momentum}
    assert old_by_cell.keys() == new_by_cell.keys()
    for cell, old in old_by_cell.items():
        new = new_by_cell[cell]
        for field in (
            "data_seed", "epochs", "learning_rate", "weight_decay", "aux_learning_rate",
            "secondary", "regularization_lambda", "input_dim", "hidden_units", "output_dim",
            "device", "landscape_device", "landscape_seed_offset", "slice_points",
            "slice_radius", "slice_every", "landscape_batch_size", "landscape_views",
        ):
            assert getattr(old, field) == getattr(new, field), (cell, field)
        assert old.method == stability.MANIFOLD_METHOD
        assert old.momentum == stability.MANIFOLD_RECIPE["momentum"]
        assert new.momentum == 0.0

    # Pin one existing condition identity so registering the new method cannot
    # silently change the already completed Nesterov DA-10 experiment.
    old_f1_small_seed0 = next(args for args in existing
                              if (args.task, args.architecture, args.seed) == ("f1", "small", 0))
    assert old_f1_small_seed0.condition_id == (
        "849b8ac9808d03a24fb6b996b2f6c7d958ec867b3fd695fb11c0f7cec2fa4c86"
    )


def test_each_method_rejects_the_other_methods_momentum_recipe():
    old_method = stability.MANIFOLD_METHOD
    no_momentum_method = stability.MANIFOLD_NO_MOMENTUM_METHOD
    no_momentum_args = _training_args(no_momentum_method)
    assert no_momentum_args.momentum == 0.0
    with pytest.raises(SystemExit):
        _training_args(
            no_momentum_method,
            recipe_override=copy.deepcopy(stability.MANIFOLD_RECIPES[old_method]),
        )
    with pytest.raises(SystemExit):
        _training_args(
            old_method,
            recipe_override=copy.deepcopy(stability.MANIFOLD_RECIPES[no_momentum_method]),
        )

