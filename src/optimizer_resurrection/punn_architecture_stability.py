"""PUNN numerical-stability runs for registered architecture and optimizer recipes.

The 2024 study's oversized networks use the corresponding summation-unit
network (SUNN) widths. Its selected L2 setting (lambda=1e-4) was used only on classification
problems. The original protocol keeps the paper's scalar FP32 update order and
uses plain SGD (learning rate 0.1, momentum 0). The adaptive extension uses the
frozen AdamW and Moonlight Muon recipes from ``gradient_defaults.yaml``. Both
protocols record finite-error outcomes separately from numerical failures.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from .models.product_unit import ProductUnitNetwork
from .punn_architecture_data import TASK_SPECS, make_architecture_dataset
from .train import (
    _determinism_metadata,
    _environment_metadata,
    _git_metadata,
    _require_reproducible_source,
    _sanitized_invocation,
    _source_tree_digest,
    seed_everything,
)
from .tracking import load_wandb_credentials, require_online_wandb


PROTOCOL = "punn-architecture-stability-v1"
ADAPTIVE_PROTOCOL = "punn-adaptive-architecture-stability-v1"
MANIFOLD_PROTOCOL = "punn-manifold-architecture-stability-v1"
METHOD = "sgd"
ADAPTIVE_METHODS = ("adamw", "muon_moonlight")
MANIFOLD_METHOD = "manifold_muon_da10"
ARCHITECTURES = ("small", "oversized", "regularized")
CLASSIFICATION_TASKS = {"xor", "iris", "wine", "diabetes"}
RUNTIME_FIELDS = {"condition_id", "dry_run", "run_group", "run_name", "stage"}
CONFIG_FIELDS = {
    "protocol",
    "conditions",
    "seeds",
    "data_seed_offset",
    "epochs",
    "learning_rate",
    "momentum",
    "regularization_lambda",
    "log_every",
    "device",
    "stage",
    "run_group",
}
ADAPTIVE_CONFIG_FIELDS = {
    "protocol",
    "conditions",
    "methods",
    "seeds",
    "data_seed_offset",
    "epochs",
    "regularization_lambda",
    "log_every",
    "device",
    "stage",
    "run_group",
    "optimizer_settings",
}
ADAPTIVE_RECIPE_FIELDS = {
    "learning_rate", "momentum", "weight_decay", "aux_learning_rate", "secondary"
}
MANIFOLD_RECIPE_FIELDS = ADAPTIVE_RECIPE_FIELDS | {
    "manifold_scale", "dual_learning_rate", "dual_iterations"
}
ADAPTIVE_RECIPES = {
    "adamw": {
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "aux_learning_rate": 0.001,
        "secondary": True,
    },
    "muon_moonlight": {
        "learning_rate": 0.02,
        "momentum": 0.95,
        "weight_decay": 0.0,
        "aux_learning_rate": 0.001,
        "secondary": False,
    },
}
MANIFOLD_RECIPE = {
    "learning_rate": 0.01,
    "momentum": 0.95,
    "weight_decay": 0.0,
    "aux_learning_rate": 0.001,
    "secondary": False,
    "manifold_scale": 1.0,
    "dual_learning_rate": 0.01,
    "dual_iterations": 10,
}


def _dimensions(task: str, architecture: str) -> tuple[int, int, int]:
    try:
        input_dim, small_hidden, oversized_hidden, output_dim = TASK_SPECS[task]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown architecture-stability task: {task}") from exc
    if architecture == "small":
        hidden_units = small_hidden
    elif architecture in {"oversized", "regularized"}:
        hidden_units = oversized_hidden
    else:
        raise ValueError(f"unknown architecture variant: {architecture}")
    return int(input_dim), int(hidden_units), int(output_dim)


def _scientific_values(args: argparse.Namespace) -> dict[str, Any]:
    """Fields that define one immutable condition, including its resolved shape."""
    if args.protocol in {ADAPTIVE_PROTOCOL, MANIFOLD_PROTOCOL}:
        values = {
            "protocol": args.protocol,
            "task": args.task,
            "architecture": args.architecture,
            "method": args.method,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "regularization_lambda": args.regularization_lambda,
            "weight_decay": args.weight_decay,
            "aux_learning_rate": args.aux_learning_rate,
            "secondary": args.secondary,
            "input_dim": args.input_dim,
            "hidden_units": args.hidden_units,
            "output_dim": args.output_dim,
        }
        if args.protocol == MANIFOLD_PROTOCOL:
            # Keep the registered CPU condition IDs stable for the partial
            # CPU sweep; CUDA gets a distinct identity for different arithmetic.
            if args.device != "cpu":
                values["device"] = args.device
            values.update(manifold_scale=args.manifold_scale,
                          dual_learning_rate=args.dual_learning_rate,
                          dual_iterations=args.dual_iterations)
        return values
    return {
        "protocol": args.protocol,
        "task": args.task,
        "architecture": args.architecture,
        "method": args.method,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "regularization_lambda": args.regularization_lambda,
        "input_dim": args.input_dim,
        "hidden_units": args.hidden_units,
        "output_dim": args.output_dim,
    }


def _condition_id(values: dict[str, Any]) -> str:
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=[PROTOCOL, ADAPTIVE_PROTOCOL, MANIFOLD_PROTOCOL], default=PROTOCOL)
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--method", choices=[METHOD, *ADAPTIVE_METHODS, MANIFOLD_METHOD], default=METHOD)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--regularization-lambda", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--aux-learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--secondary", choices=["true", "false"], default=argparse.SUPPRESS)
    parser.add_argument("--manifold-scale", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dual-learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dual-iterations", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--stage", default="exploratory")
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--condition-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.device == "cuda" and args.protocol != MANIFOLD_PROTOCOL:
        parser.error("CUDA is registered only for the Manifold DA-10 stability protocol")

    if args.epochs < 1 or args.log_every < 1:
        parser.error("epochs and log-every must be positive")
    if args.seed < 0 or args.data_seed < 0:
        parser.error("seeds must be nonnegative")
    if args.protocol == PROTOCOL:
        if args.method != METHOD:
            parser.error("the frozen plain-SGD stability protocol requires method=sgd")
        if not math.isfinite(args.learning_rate) or args.learning_rate != 0.1:
            parser.error("this frozen plain-SGD stability protocol requires learning-rate=0.1")
        if not math.isfinite(args.momentum) or args.momentum != 0.0:
            parser.error("this frozen plain-SGD stability protocol requires momentum=0")
        if any(hasattr(args, key) for key in MANIFOLD_RECIPE_FIELDS - {"learning_rate", "momentum"}):
            parser.error("adaptive optimizer settings require the adaptive stability protocol")
    elif args.protocol == ADAPTIVE_PROTOCOL:
        if args.method not in ADAPTIVE_METHODS:
            parser.error("the adaptive stability protocol requires AdamW or Muon Moonlight")
        required = ("weight_decay", "aux_learning_rate", "secondary")
        if any(not hasattr(args, key) for key in required):
            parser.error("adaptive stability runs require explicit weight-decay, aux-learning-rate, and secondary")
        args.secondary = args.secondary == "true"
        expected_recipe = ADAPTIVE_RECIPES[args.method]
        actual_recipe = {key: getattr(args, key) for key in ADAPTIVE_RECIPE_FIELDS}
        if actual_recipe != expected_recipe:
            parser.error(f"adaptive {args.method} settings must match the frozen gradient-defaults recipe")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("learning-rate must be finite and positive")
        if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
            parser.error("momentum must be finite and in [0,1)")
        if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
            parser.error("weight-decay must be finite and nonnegative")
        if not math.isfinite(args.aux_learning_rate) or args.aux_learning_rate <= 0:
            parser.error("aux-learning-rate must be finite and positive")
        if any(hasattr(args, key) for key in ("manifold_scale", "dual_learning_rate", "dual_iterations")):
            parser.error("manifold settings require the manifold stability protocol")
    else:
        if args.method != MANIFOLD_METHOD:
            parser.error("the manifold stability protocol requires manifold_muon_da10")
        if any(not hasattr(args, key) for key in MANIFOLD_RECIPE_FIELDS):
            parser.error("manifold stability runs require every frozen DA-10 setting")
        args.secondary = args.secondary == "true"
        actual_recipe = {key: getattr(args, key) for key in MANIFOLD_RECIPE_FIELDS}
        if actual_recipe != MANIFOLD_RECIPE:
            parser.error("manifold settings must match the registered plain DA-10 recipe")
    if not math.isfinite(args.regularization_lambda) or args.regularization_lambda < 0:
        parser.error("regularization-lambda must be finite and nonnegative")
    if args.architecture == "regularized":
        if args.task not in CLASSIFICATION_TASKS:
            parser.error("the regularized architecture is only registered for classification tasks")
        if args.regularization_lambda != 0.0001:
            parser.error("regularized conditions require the selected lambda=0.0001")
    elif args.regularization_lambda != 0.0:
        parser.error("small and oversized architectures require regularization-lambda=0")

    args.input_dim, args.hidden_units, args.output_dim = _dimensions(args.task, args.architecture)
    actual = _condition_id(_scientific_values(args))
    if args.condition_id is not None and args.condition_id != actual:
        parser.error("condition-id does not match resolved scientific configuration")
    args.condition_id = actual
    if args.run_name is None:
        method_suffix = f"-{args.method}" if args.protocol in {ADAPTIVE_PROTOCOL, MANIFOLD_PROTOCOL} else ""
        args.run_name = f"punn-stability-{args.task}-{args.architecture}{method_suffix}-seed{args.seed}"
    return args


def expand_config(config: dict[str, Any]) -> list[list[str]]:
    if config.get("protocol") == MANIFOLD_PROTOCOL:
        return _expand_manifold_config(config)
    if config.get("protocol") == ADAPTIVE_PROTOCOL:
        return _expand_adaptive_config(config)
    if set(config) != CONFIG_FIELDS or config.get("protocol") != PROTOCOL:
        raise ValueError(
            f"invalid architecture-stability config fields: missing={CONFIG_FIELDS-set(config)} "
            f"extra={set(config)-CONFIG_FIELDS}"
        )
    conditions = config["conditions"]
    seeds = config["seeds"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("conditions must be a nonempty list")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if type(config["data_seed_offset"]) is not int or config["data_seed_offset"] < 0:
        raise ValueError("data-seed-offset must be a nonnegative integer")
    keys_seen: set[tuple[str, str]] = set()
    for condition in conditions:
        if not isinstance(condition, dict) or set(condition) != {"task", "architecture"}:
            raise ValueError("each condition must contain exactly task and architecture")
        task, architecture = condition["task"], condition["architecture"]
        if task not in TASK_SPECS or architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported task/architecture condition: {condition}")
        if architecture == "regularized" and task not in CLASSIFICATION_TASKS:
            raise ValueError(f"regularization is not registered for regression task {task}")
        key = (task, architecture)
        if key in keys_seen:
            raise ValueError(f"duplicate task/architecture condition: {key}")
        keys_seen.add(key)
    if config["epochs"] < 1 or config["log_every"] < 1:
        raise ValueError("epochs and log-every must be positive")
    if config["learning_rate"] != 0.1 or config["momentum"] != 0.0:
        raise ValueError("the frozen plain-SGD recipe is learning-rate=0.1, momentum=0")
    reg_lambda = config["regularization_lambda"]
    if not math.isfinite(reg_lambda) or reg_lambda != 0.0001:
        raise ValueError("regularization-lambda must be the selected 0.0001")
    if config["device"] != "cpu":
        raise ValueError("architecture-stability runs use CPU FP32 scalar updates")

    commands: list[list[str]] = []
    for condition, seed in itertools.product(conditions, seeds):
        task = condition["task"]
        architecture = condition["architecture"]
        values = {
            "protocol": PROTOCOL,
            "task": task,
            "architecture": architecture,
            "method": METHOD,
            "seed": seed,
            "data_seed": seed + config["data_seed_offset"],
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "momentum": config["momentum"],
            "regularization_lambda": reg_lambda if architecture == "regularized" else 0.0,
            "log_every": config["log_every"],
            "device": config["device"],
            "stage": config["stage"],
            "run_group": config["run_group"],
            "run_name": f"punn-stability-{task}-{architecture}-seed{seed}",
        }
        command = ["python", "-m", "optimizer_resurrection.punn_architecture_stability"]
        for key, value in values.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        parse_args(command[3:])
        commands.append(command)
    return commands


def _expand_adaptive_config(config: dict[str, Any]) -> list[list[str]]:
    """Expand the frozen AdamW/Muon architecture stability matrix."""
    if set(config) != ADAPTIVE_CONFIG_FIELDS:
        raise ValueError(
            f"invalid adaptive architecture-stability config fields: "
            f"missing={ADAPTIVE_CONFIG_FIELDS-set(config)} extra={set(config)-ADAPTIVE_CONFIG_FIELDS}"
        )
    conditions = config["conditions"]
    methods = config["methods"]
    seeds = config["seeds"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("conditions must be a nonempty list")
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a nonempty unique list")
    if any(method not in ADAPTIVE_METHODS for method in methods):
        raise ValueError(f"adaptive methods must be selected from {ADAPTIVE_METHODS}")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if type(config["data_seed_offset"]) is not int or config["data_seed_offset"] < 0:
        raise ValueError("data-seed-offset must be a nonnegative integer")
    if config["epochs"] < 1 or config["log_every"] < 1:
        raise ValueError("epochs and log-every must be positive")
    if config["device"] != "cpu":
        raise ValueError("adaptive architecture-stability runs use CPU FP32 scalar updates")
    if set(config["optimizer_settings"]) != set(methods):
        raise ValueError("optimizer_settings must cover exactly the requested methods")
    for method in methods:
        recipe = config["optimizer_settings"][method]
        if not isinstance(recipe, dict) or set(recipe) != ADAPTIVE_RECIPE_FIELDS:
            raise ValueError(f"invalid optimizer setting fields for {method}")
        if recipe != ADAPTIVE_RECIPES[method]:
            raise ValueError(f"optimizer settings for {method} differ from the frozen gradient-defaults recipe")

    reg_lambda = config["regularization_lambda"]
    if not math.isfinite(reg_lambda) or reg_lambda != 0.0001:
        raise ValueError("regularization-lambda must be the selected 0.0001")
    keys_seen: set[tuple[str, str]] = set()
    for condition in conditions:
        if not isinstance(condition, dict) or set(condition) != {"task", "architecture"}:
            raise ValueError("each condition must contain exactly task and architecture")
        task, architecture = condition["task"], condition["architecture"]
        if task not in TASK_SPECS or architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported task/architecture condition: {condition}")
        if architecture == "regularized" and task not in CLASSIFICATION_TASKS:
            raise ValueError(f"regularization is not registered for regression task {task}")
        key = (task, architecture)
        if key in keys_seen:
            raise ValueError(f"duplicate task/architecture condition: {key}")
        keys_seen.add(key)

    commands: list[list[str]] = []
    for condition, method, seed in itertools.product(conditions, methods, seeds):
        task = condition["task"]
        architecture = condition["architecture"]
        recipe = config["optimizer_settings"][method]
        values = {
            "protocol": ADAPTIVE_PROTOCOL,
            "task": task,
            "architecture": architecture,
            "method": method,
            "seed": seed,
            "data_seed": seed + config["data_seed_offset"],
            "epochs": config["epochs"],
            "learning_rate": recipe["learning_rate"],
            "momentum": recipe["momentum"],
            "regularization_lambda": reg_lambda if architecture == "regularized" else 0.0,
            "weight_decay": recipe["weight_decay"],
            "aux_learning_rate": recipe["aux_learning_rate"],
            "secondary": recipe["secondary"],
            "log_every": config["log_every"],
            "device": config["device"],
            "stage": config["stage"],
            "run_group": config["run_group"],
            "run_name": f"punn-stability-{task}-{architecture}-{method}-seed{seed}",
        }
        command = ["python", "-m", "optimizer_resurrection.punn_architecture_stability"]
        for key, value in values.items():
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            command.extend(["--" + key.replace("_", "-"), rendered])
        parse_args(command[3:])
        commands.append(command)
    return commands


def _expand_manifold_config(config: dict[str, Any]) -> list[list[str]]:
    """Register only the plain DA-10 condition over the existing 16-cell matrix."""
    if set(config) != ADAPTIVE_CONFIG_FIELDS or config.get("protocol") != MANIFOLD_PROTOCOL:
        raise ValueError("invalid manifold architecture-stability config fields")
    if config.get("methods") != [MANIFOLD_METHOD]:
        raise ValueError("the plain DA-10 plan requires only manifold_muon_da10")
    if config.get("optimizer_settings") != {MANIFOLD_METHOD: MANIFOLD_RECIPE}:
        raise ValueError("the plain DA-10 plan must use the fixed registered recipe")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("Manifold stability device must be cpu or cuda")
    # Reuse the original plan's structural checks for tasks, architecture
    # variants, seeds, L2 and duplicate-condition rejection. The adaptive
    # plan remains CPU-only; device is validated above for Manifold.
    validation_config = {
        **config,
        "device": "cpu",
        "protocol": ADAPTIVE_PROTOCOL,
        "methods": ["muon_moonlight"],
        "optimizer_settings": {"muon_moonlight": ADAPTIVE_RECIPES["muon_moonlight"]},
    }
    _expand_adaptive_config(validation_config)
    commands: list[list[str]] = []
    for condition, seed in itertools.product(config["conditions"], config["seeds"]):
        task, architecture = condition["task"], condition["architecture"]
        values = {
            "protocol": MANIFOLD_PROTOCOL,
            "task": task,
            "architecture": architecture,
            "method": MANIFOLD_METHOD,
            "seed": seed,
            "data_seed": seed + config["data_seed_offset"],
            "epochs": config["epochs"],
            "regularization_lambda": (config["regularization_lambda"]
                                      if architecture == "regularized" else 0.0),
            **MANIFOLD_RECIPE,
            "log_every": config["log_every"],
            "device": config["device"],
            "stage": config["stage"],
            "run_group": config["run_group"],
            "run_name": f"punn-stability-{task}-{architecture}-{MANIFOLD_METHOD}-seed{seed}",
        }
        command = ["python", "-m", "optimizer_resurrection.punn_architecture_stability"]
        for key, value in values.items():
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            command.extend(["--" + key.replace("_", "-"), rendered])
        parse_args(command[3:])
        commands.append(command)
    return commands


def _tensor_pattern(named_values: list[tuple[str, torch.Tensor]]) -> dict[str, Any] | None:
    """Return a bounded, serializable location report for the first bad tensor."""
    bad_names = []
    bad_count = 0
    examples: list[dict[str, Any]] = []
    for name, tensor in named_values:
        mask = ~torch.isfinite(tensor)
        count = int(mask.sum())
        if not count:
            continue
        bad_names.append(name)
        bad_count += count
        if len(examples) < 8:
            for index in mask.nonzero(as_tuple=False)[: max(0, 8 - len(examples))]:
                examples.append({"tensor": name, "index": [int(part) for part in index]})
    if not bad_count:
        return None
    return {"tensor_names": bad_names[:16], "nonfinite_values": bad_count, "examples": examples}


def _first_failure(
    *,
    reason: str,
    phase: str,
    epoch: int,
    examples_processed: int,
    pattern: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "failure_reason": reason,
        "failure_phase": phase,
        "failed_epoch": int(epoch),
        "examples_processed": int(examples_processed),
        "failure_pattern": pattern or {"metric": reason},
    }


def _mse(model: ProductUnitNetwork, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    prediction = model(x)
    return (prediction - y).square().mean()


def _regularization_term(model: ProductUnitNetwork, strength: float) -> torch.Tensor:
    # The selected penalty is lambda * sum(p**2), including the output bias.
    # Avoid evaluating 0 * sum(p**2): a very large but finite parameter can
    # overflow its square and turn an otherwise unregularized run into NaN.
    zero = next(model.parameters()).new_zeros(())
    if strength == 0.0:
        return zero
    return strength * sum((parameter.square().sum() for parameter in model.parameters()),
                          start=zero)


def _dataset_digest(dataset: Any) -> str:
    digest = hashlib.sha256()
    for value in (dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_dataset(dataset: Any, args: argparse.Namespace) -> None:
    for name in ("train_x", "train_y", "test_x", "test_y"):
        value = getattr(dataset, name)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise TypeError(f"dataset {name} must be a float32 tensor")
        if value.device.type != "cpu" or value.ndim != 2 or value.shape[0] == 0:
            raise ValueError(f"dataset {name} must be a nonempty rank-2 CPU tensor")
    for part in ("train", "test"):
        x, y = getattr(dataset, part + "_x"), getattr(dataset, part + "_y")
        if x.shape[1] != args.input_dim or y.shape[1] != args.output_dim or len(x) != len(y):
            raise ValueError(f"dataset {part} shapes do not match the registered task dimensions")


def _active_optimizers(optimizer: Any) -> list[tuple[str, torch.optim.Optimizer]]:
    """Expose every optimizer in a method bundle for state finiteness checks."""
    if hasattr(optimizer, "primary") and hasattr(optimizer, "auxiliary"):
        active = [("primary", optimizer.primary)]
        if optimizer.auxiliary is not None:
            active.append(("auxiliary", optimizer.auxiliary))
        return active
    return [("primary", optimizer)]


def run_trial(
    args: argparse.Namespace,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    *,
    dataset_pair: tuple[Any, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one FP32 scalar-update condition and stop at its first nonfinite."""
    started = time.perf_counter()
    seed_everything(args.seed)
    if dataset_pair is None:
        dataset, data_metadata = make_architecture_dataset(args.task, args.data_seed)
    else:
        dataset, data_metadata = dataset_pair
    _validate_dataset(dataset, args)
    dataset_digest = _dataset_digest(dataset)
    declared_digest = getattr(dataset, "digest", dataset_digest)
    if declared_digest != dataset_digest:
        raise ValueError("dataset digest does not match the four stored train/test tensors")

    # Decouple initialization from data-loader RNG use so oversized and
    # regularized conditions with the same seed have bitwise-identical starts.
    seed_everything(args.seed)
    model = ProductUnitNetwork(args.input_dim, args.hidden_units, args.output_dim,
                               input_domain="real_complex", init_bound=1.0).cpu()
    def model_digest() -> str:
        return hashlib.sha256(
            b"".join(parameter.detach().cpu().contiguous().numpy().tobytes()
                     for parameter in model.parameters())
        ).hexdigest()

    preprojection_initialization_digest = model_digest()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA device is unavailable")
    model = model.to(device)
    if device.type == "cuda":
        from dataclasses import replace

        dataset = replace(
            dataset,
            train_x=dataset.train_x.to(device),
            train_y=dataset.train_y.to(device),
            test_x=dataset.test_x.to(device),
            test_y=dataset.test_y.to(device),
        )
    if args.protocol == MANIFOLD_PROTOCOL:
        from .optim.manifold_muon import retract_stiefel
        from .diagnostics.spectra import stiefel_residual

        with torch.no_grad():
            model.exponents.copy_(retract_stiefel(model.exponents, args.manifold_scale))
        initial_stiefel_residual = stiefel_residual(model.exponents, args.manifold_scale)
    initialization_digest = model_digest()
    order_seed = args.seed + 1_000_003
    generator = torch.Generator(device="cpu").manual_seed(order_seed)
    if args.protocol == MANIFOLD_PROTOCOL:
        from .optim.manifold_muon import ManifoldMuon
        from .optim.param_groups import OptimizerBundle

        optimizer = OptimizerBundle(
            ManifoldMuon([model.exponents], lr=args.learning_rate,
                         momentum=args.momentum, nesterov=True,
                         dual_lr=args.dual_learning_rate,
                         max_iterations=args.dual_iterations,
                         scale=args.manifold_scale),
            torch.optim.AdamW(model.output.parameters(), lr=args.aux_learning_rate,
                              betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        )
    elif args.protocol == ADAPTIVE_PROTOCOL:
        # Reuse the frozen comparison builder so AdamW/Muon assignment and
        # optimizer defaults match the existing gradient-defaults experiments.
        from .punn_gradients import build_optimizer as build_product_optimizer

        optimizer = build_product_optimizer(model, args)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=0.0, weight_decay=0.0)
    active_optimizers = _active_optimizers(optimizer)

    epochs_completed = 0
    examples_processed = 0
    max_gradient_norm = 0.0
    max_gradient_abs = 0.0
    max_parameter_abs = max(float(parameter.detach().abs().max())
                            for parameter in model.parameters())
    failure: dict[str, Any] | None = None
    current_sample_index: int | None = None
    current_within_epoch_position: int | None = None
    initial_train_mse: float | None = None
    train_mse: float | None = None
    test_mse: float | None = None
    regularization_term: float | None = None
    objective: float | None = None
    train_x, train_y = dataset.train_x, dataset.train_y

    def note_failure(**values: Any) -> None:
        nonlocal failure
        failure = _first_failure(**values)
        if current_sample_index is not None:
            failure["sample_index"] = current_sample_index
            failure["within_epoch_position"] = current_within_epoch_position

    def measure(epoch: int, phase: str) -> dict[str, float] | dict[str, Any]:
        nonlocal failure
        with torch.no_grad():
            predictions = model(train_x)
            if not bool(torch.isfinite(predictions).all()):
                note_failure(
                    reason="nonfinite_prediction", phase=phase, epoch=epoch,
                    examples_processed=examples_processed,
                    pattern=_tensor_pattern([("prediction", predictions)]),
                )
                return {}
            data_mse_tensor = (predictions - train_y).square().mean()
            reg_tensor = _regularization_term(model, args.regularization_lambda)
            objective_tensor = data_mse_tensor + reg_tensor
        for name, tensor, reason in (
            ("train_mse", data_mse_tensor, "nonfinite_train_mse"),
            ("regularization_term", reg_tensor, "nonfinite_regularization_term"),
            ("objective", objective_tensor, "nonfinite_objective"),
        ):
            if not bool(torch.isfinite(tensor)):
                note_failure(
                    reason=reason, phase=phase, epoch=epoch,
                    examples_processed=examples_processed,
                    pattern={"metric": name},
                )
                return {}
        return {
            "train_mse": float(data_mse_tensor),
            "regularization_term": float(reg_tensor),
            "objective": float(objective_tensor),
        }

    initial = measure(0, "initialization")
    if failure is None:
        initial_train_mse = initial["train_mse"]
        train_mse = initial["train_mse"]
        regularization_term = initial["regularization_term"]
        objective = initial["objective"]

    for epoch in range(1, args.epochs + 1):
        if failure is not None:
            break
        for within_epoch_position, sample_index in enumerate(torch.randperm(len(train_x), generator=generator), start=1):
            current_sample_index = int(sample_index)
            current_within_epoch_position = within_epoch_position
            optimizer.zero_grad(set_to_none=True)
            sample_x = train_x[current_sample_index:current_sample_index + 1]
            sample_y = train_y[current_sample_index:current_sample_index + 1]
            prediction = model(sample_x)
            examples_processed += 1
            if not bool(torch.isfinite(prediction).all()):
                note_failure(
                    reason="nonfinite_prediction", phase="training_forward", epoch=epoch,
                    examples_processed=examples_processed,
                    pattern=_tensor_pattern([("prediction", prediction)]),
                )
                break
            data_loss = (prediction - sample_y).square().mean()
            if not bool(torch.isfinite(data_loss)):
                note_failure(
                    reason="nonfinite_loss", phase="training_forward", epoch=epoch,
                    examples_processed=examples_processed,
                    pattern={"metric": "per_example_data_mse"},
                )
                break
            penalty = _regularization_term(model, args.regularization_lambda)
            if not bool(torch.isfinite(penalty)):
                note_failure(
                    reason="nonfinite_regularization_term", phase="training_forward", epoch=epoch,
                    examples_processed=examples_processed, pattern={"metric": "regularization_term"},
                )
                break
            loss = data_loss + penalty
            if not bool(torch.isfinite(loss)):
                note_failure(
                    reason="nonfinite_loss", phase="training_forward", epoch=epoch,
                    examples_processed=examples_processed, pattern={"metric": "objective"},
                )
                break
            loss.backward()
            bad_gradient = _tensor_pattern([
                (name, parameter.grad) for name, parameter in model.named_parameters()
                if parameter.grad is not None
            ])
            if bad_gradient is not None:
                note_failure(
                    reason="nonfinite_gradient", phase="backward", epoch=epoch,
                    examples_processed=examples_processed, pattern=bad_gradient,
                )
                break
            gradient_norm = math.sqrt(sum(
                float(parameter.grad.detach().double().square().sum())
                for parameter in model.parameters() if parameter.grad is not None
            ))
            if not math.isfinite(gradient_norm):
                note_failure(
                    reason="nonfinite_gradient_norm", phase="backward", epoch=epoch,
                    examples_processed=examples_processed, pattern={"metric": "gradient_norm"},
                )
                break
            max_gradient_norm = max(max_gradient_norm, gradient_norm)
            max_gradient_abs = max(max_gradient_abs, max(
                (float(parameter.grad.detach().abs().max())
                 for parameter in model.parameters() if parameter.grad is not None),
                default=0.0,
            ))
            try:
                optimizer.step()
            except torch.linalg.LinAlgError as exc:
                if args.protocol != MANIFOLD_PROTOCOL:
                    raise
                # Finite external gradients can overflow inside DA-10's
                # dual-ascent matrix products before SVD. A nonconvergent SVD
                # is therefore a scientific numerical outcome, not an
                # infrastructure failure that should abort the sweep.
                note_failure(
                    reason="manifold_svd_failure", phase="optimizer_update", epoch=epoch,
                    examples_processed=examples_processed,
                    pattern={"operation": "manifold_muon_svd",
                             "exception_type": type(exc).__name__},
                )
                break
            bad_parameters = _tensor_pattern(list(model.named_parameters()))
            if bad_parameters is not None:
                note_failure(
                    reason="nonfinite_parameter", phase="optimizer_update", epoch=epoch,
                    examples_processed=examples_processed, pattern=bad_parameters,
                )
                break
            max_parameter_abs = max(max_parameter_abs, max(
                float(parameter.detach().abs().max()) for parameter in model.parameters()
            ))
            bad_optimizer_state = _tensor_pattern([
                (f"optimizer_state.{optimizer_name}.{name}.{key}", value)
                for optimizer_name, active_optimizer in active_optimizers
                for name, parameter in model.named_parameters()
                for key, value in active_optimizer.state.get(parameter, {}).items()
                if isinstance(value, torch.Tensor)
            ])
            if bad_optimizer_state is not None:
                note_failure(
                    reason="nonfinite_optimizer_state", phase="optimizer_update", epoch=epoch,
                    examples_processed=examples_processed, pattern=bad_optimizer_state,
                )
                break
            if args.protocol == MANIFOLD_PROTOCOL:
                scalar_state = [
                    (f"optimizer_state.{optimizer_name}.{name}.{key}", value)
                    for optimizer_name, active_optimizer in active_optimizers
                    for name, parameter in model.named_parameters()
                    for key, value in active_optimizer.state.get(parameter, {}).items()
                    if isinstance(value, float) and not math.isfinite(value)
                ]
                if scalar_state:
                    note_failure(
                        reason="nonfinite_optimizer_state", phase="optimizer_update", epoch=epoch,
                        examples_processed=examples_processed,
                        pattern={"scalar_state_names": [name for name, _ in scalar_state]},
                    )
                    break
        current_sample_index = None
        current_within_epoch_position = None
        if failure is not None:
            break

        # Evaluate the full training objective every epoch to expose any bad
        # result promptly; reporting cadence never changes numerical checks.
        epoch_result = measure(epoch, "epoch_evaluation")
        if failure is not None:
            break
        epochs_completed = epoch
        train_mse = epoch_result["train_mse"]
        regularization_term = epoch_result["regularization_term"]
        objective = epoch_result["objective"]
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            if on_epoch is not None:
                on_epoch({
                    "epoch": epoch,
                    **epoch_result,
                    "examples_processed": examples_processed,
                    "max_gradient_norm": max_gradient_norm,
                    "max_gradient_abs": max_gradient_abs,
                    "max_parameter_abs": max_parameter_abs,
                })

    if failure is None:
        with torch.no_grad():
            test_predictions = model(dataset.test_x)
            if not bool(torch.isfinite(test_predictions).all()):
                note_failure(
                    reason="nonfinite_prediction", phase="final_test", epoch=epochs_completed,
                    examples_processed=examples_processed,
                    pattern=_tensor_pattern([("test_prediction", test_predictions)]),
                )
            else:
                test_mse_tensor = (test_predictions - dataset.test_y).square().mean()
                if not bool(torch.isfinite(test_mse_tensor)):
                    note_failure(
                        reason="nonfinite_test_mse", phase="final_test", epoch=epochs_completed,
                        examples_processed=examples_processed, pattern={"metric": "test_mse"},
                    )
                else:
                    test_mse = float(test_mse_tensor)

    numerical_failure = failure is not None
    result: dict[str, Any] = {
        "condition_id": args.condition_id,
        "protocol": args.protocol,
        "task": args.task,
        "architecture": args.architecture,
        "method": args.method,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "epochs_completed": epochs_completed,
        "failed_epoch": failure["failed_epoch"] if failure else None,
        "failure_phase": failure["failure_phase"] if failure else None,
        "failure_reason": failure["failure_reason"] if failure else None,
        "failure_pattern": failure["failure_pattern"] if failure else None,
        "failure_sample_index": failure.get("sample_index") if failure else None,
        "failure_within_epoch_position": failure.get("within_epoch_position") if failure else None,
        "examples_processed": examples_processed,
        "initial_train_mse": initial_train_mse,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "regularization_term": regularization_term,
        "objective": objective,
        "max_gradient_norm": max_gradient_norm,
        "max_gradient_abs": max_gradient_abs,
        "max_parameter_abs": max_parameter_abs,
        "wall_seconds": time.perf_counter() - started,
        "numerical_failure": numerical_failure,
        "dataset_digest": dataset_digest,
        "initialization_digest": initialization_digest,
        "order_seed": order_seed,
        "terminal_outcome": "numerical_failure" if numerical_failure else "completed",
    }
    if args.protocol in {ADAPTIVE_PROTOCOL, MANIFOLD_PROTOCOL}:
        if args.method == "adamw":
            optimizer_name = "torch.optim.AdamW"
            optimizer_assignment = "AdamW over all model parameters"
            optimizer_betas = [0.9, 0.999]
        elif args.method == "muon_moonlight":
            optimizer_name = "MoonlightMuon plus AdamW auxiliary"
            optimizer_assignment = "Moonlight Muon on exponents; AdamW on output weight and bias"
            optimizer_betas = {"auxiliary_adamw": [0.9, 0.95]}
        else:
            optimizer_name = "ManifoldMuon DA-10 plus AdamW auxiliary"
            optimizer_assignment = "DA-10 Manifold Muon on projected exponents; AdamW on output weight and bias"
            optimizer_betas = {"auxiliary_adamw": [0.9, 0.95]}
        result.update({
            "epochs": args.epochs,
            "input_dim": args.input_dim,
            "hidden_units": args.hidden_units,
            "output_dim": args.output_dim,
            "regularization_lambda": args.regularization_lambda,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "aux_learning_rate": args.aux_learning_rate,
            "secondary": args.secondary,
            "optimizer_name": optimizer_name,
            "optimizer_assignment": optimizer_assignment,
            "optimizer_betas": optimizer_betas,
            "optimizer_epsilon": 1e-8,
        })
        if args.method == "muon_moonlight":
            result["muon_nesterov"] = True
            result["muon_ns_steps"] = 5
        if args.method == MANIFOLD_METHOD:
            result.update({
                "preprojection_initialization_digest": preprojection_initialization_digest,
                "manifold_scale": args.manifold_scale,
                "dual_learning_rate": args.dual_learning_rate,
                "dual_iterations": args.dual_iterations,
                "manifold_nesterov": True,
                "initial_stiefel_residual": initial_stiefel_residual,
                "final_stiefel_residual": stiefel_residual(model.exponents, args.manifold_scale)
                if bool(torch.isfinite(model.exponents).all()) else None,
            })
    if failure:
        result["failed_epoch"] = failure["failed_epoch"]
        result["failure_examples_processed"] = failure["examples_processed"]
    return result


def _log_dataset_artifact(run: Any, args: argparse.Namespace, dataset: Any,
                          metadata: dict[str, Any], digest: str) -> str:
    import wandb

    artifact = wandb.Artifact(
        name=f"punn-architecture-data-{args.task}-{args.data_seed}",
        type="dataset",
        description="Exact train/test tensors and source metadata for one PUNN stability data seed",
        metadata={"task": args.task, "data_seed": args.data_seed, "dataset_digest": digest},
    )
    with artifact.new_file("dataset.npz", mode="wb") as handle:
        np.savez_compressed(
            handle,
            train_x=dataset.train_x.numpy(),
            train_y=dataset.train_y.numpy(),
            test_x=dataset.test_x.numpy(),
            test_y=dataset.test_y.numpy(),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, default=str)),
        )
    logged = run.log_artifact(artifact)
    logged.wait()
    return str(logged.name)


def _log_training_epoch(run: Any, values: dict[str, Any]) -> None:
    """Keep live epoch history separate from terminal summary metric names."""
    epoch = values["epoch"]
    record = {"epoch": epoch}
    record.update({f"training/{key}": value for key, value in values.items() if key != "epoch"})
    run.log(record, step=epoch)


def _record_terminal_result(run: Any, result: dict[str, Any]) -> None:
    """Publish both the nested authoritative snapshot and legacy flat fields."""
    snapshot = dict(result)
    run.summary.update(snapshot)
    run.summary["terminal_result"] = snapshot


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), sort_keys=True))
        return None

    root = Path(__file__).resolve().parents[2]
    git = _git_metadata(root)
    _require_reproducible_source(args.stage, git)
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    seed_everything(args.seed)
    torch.set_num_threads(1)
    invocation = [
        sys.executable,
        "-m",
        "optimizer_resurrection.punn_architecture_stability",
        *(sys.argv[1:] if argv is None else argv),
    ]
    if args.protocol in {ADAPTIVE_PROTOCOL, MANIFOLD_PROTOCOL}:
        if args.method == "adamw":
            optimizer_name = "torch.optim.AdamW"
            optimizer_assignment = "AdamW over all model parameters"
            optimizer_betas: Any = [0.9, 0.999]
            optimizer_note = "AdamW recipe from configs/product_unit/gradient_defaults.yaml"
        elif args.method == "muon_moonlight":
            from .optim.moonlight_muon import SOURCE as MOONLIGHT_SOURCE

            optimizer_name = "MoonlightMuon + torch.optim.AdamW"
            optimizer_assignment = "Moonlight Muon on exponents; AdamW on output weight and bias"
            optimizer_betas = {"auxiliary_adamw": [0.9, 0.95]}
            optimizer_note = "Muon recipe from configs/product_unit/gradient_defaults.yaml"
        else:
            optimizer_name = "ManifoldMuon DA-10 + torch.optim.AdamW"
            optimizer_assignment = "DA-10 on projected exponents; AdamW on output weight and bias"
            optimizer_betas = {"auxiliary_adamw": [0.9, 0.95]}
            optimizer_note = "Plain fixed-scale DA-10 recipe; no tuning or extra controls"
        optimizer_provenance = {
            "optimizer": optimizer_name,
            "optimizer_assignment": optimizer_assignment,
            "optimizer_recipe_source": ("configs/product_unit/manifold_architecture_stability.yaml"
                                        if args.protocol == MANIFOLD_PROTOCOL else
                                        "configs/product_unit/gradient_defaults.yaml"),
            "optimizer_recipe": {
                "learning_rate": args.learning_rate,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "aux_learning_rate": args.aux_learning_rate,
                "secondary": args.secondary,
            },
            "optimizer_betas": optimizer_betas,
            "optimizer_epsilon": 1e-8,
            "optimizer_note": optimizer_note,
            "secondary": args.secondary,
        }
        if args.method == "muon_moonlight":
            optimizer_provenance["moonlight_source"] = MOONLIGHT_SOURCE
            optimizer_provenance["muon_nesterov"] = True
            optimizer_provenance["muon_ns_steps"] = 5
        if args.method == MANIFOLD_METHOD:
            optimizer_provenance.update({
                "manifold_scale": args.manifold_scale,
                "dual_learning_rate": args.dual_learning_rate,
                "dual_iterations": args.dual_iterations,
                "manifold_nesterov": True,
                "exponent_projection": "SVD polar projection before training and after every DA-10 update",
            })
    else:
        optimizer_provenance = {
            "optimizer": "torch.optim.SGD over all parameters; lr=0.1; momentum=0; weight_decay=0"
        }
    provenance = {
        "git": git,
        "source_tree": _source_tree_digest(root),
        "invocation": _sanitized_invocation(invocation, credentials),
        "environment": _environment_metadata(torch.device(args.device)),
        "determinism": _determinism_metadata(args.seed, args.data_seed),
        "precision": f"{args.device.upper()} FP32; scalar batch-one updates; no AMP",
        "batch_size": 1,
        "gradient_clipping": False,
        "parameter_projection": args.protocol == MANIFOLD_PROTOCOL,
        "regularization": "lambda * sum(parameter ** 2), including output bias",
        "regularization_lambda": args.regularization_lambda,
        "order_seed": args.seed + 1_000_003,
        "tuning_budget": 0,
        "task_spec": {
            "input_dim": args.input_dim,
            "hidden_units": args.hidden_units,
            "output_dim": args.output_dim,
        },
        **optimizer_provenance,
    }
    import wandb

    config = {**vars(args), "provenance": provenance}
    run = wandb.init(
        project=credentials["WANDB_PROJECT"],
        entity=credentials["WANDB_ENTITY"],
        group=args.run_group,
        name=args.run_name,
        mode="online",
        config=config,
        save_code=False,
    )
    if run is None:
        raise RuntimeError("W&B online initialization returned no run")
    run.config.update({"wandb_run_id": str(run.id), "wandb_run_url": str(run.url)})
    try:
        dataset, data_metadata = make_architecture_dataset(args.task, args.data_seed)
        _validate_dataset(dataset, args)
        digest = _dataset_digest(dataset)
        if getattr(dataset, "digest", digest) != digest:
            raise ValueError("dataset digest does not match stored tensors")
        data_metadata = {
            **data_metadata,
            "task": args.task,
            "data_seed": args.data_seed,
            "dataset_digest": digest,
        }
        dataset_artifact = _log_dataset_artifact(run, args, dataset, data_metadata, digest)
        run.config.update({"data_metadata": data_metadata,
                           "dataset_artifact": dataset_artifact}, allow_val_change=True)
        run.summary["dataset_artifact"] = dataset_artifact
        run.summary["dataset_digest"] = digest
        result = run_trial(args, lambda values: _log_training_epoch(run, values),
                           dataset_pair=(dataset, data_metadata))
        if args.protocol in {ADAPTIVE_PROTOCOL, MANIFOLD_PROTOCOL}:
            result["dataset_artifact"] = dataset_artifact
        _record_terminal_result(run, result)
    except BaseException:
        run.summary["terminal_outcome"] = "failed"
        run.finish(exit_code=1)
        raise
    # A detected numerical failure is a completed scientific observation; the
    # plan runner must continue through every registered cell. Infrastructure
    # exceptions still finish nonzero through the exception path above.
    run.finish(exit_code=0)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
