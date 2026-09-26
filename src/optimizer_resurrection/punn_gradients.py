"""Untuned gradient-optimizer experiments on the reconstructed f1/f4 PUNNs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from .models import ProductUnitNetwork
from .optim.param_groups import OptimizerBundle
from .punn_landscape import TASKS, TrialResult, _finite_mse, make_dataset
from .tracking import load_wandb_credentials, require_online_wandb
from .train import (_determinism_metadata, _environment_metadata, _git_metadata,
                    _require_reproducible_source, _sanitized_invocation,
                    _source_tree_digest, seed_everything)

PROTOCOL = "punn-gradient-defaults-v1"
METHODS = ("sgd", "adam", "adamw", "muon_moonlight")
RUNTIME = {"condition_id", "dry_run", "run_group", "run_name", "stage", "log_every"}


def build_optimizer(model: ProductUnitNetwork, args: argparse.Namespace) -> OptimizerBundle:
    if args.method == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.method in {"adam", "adamw"}:
        cls = torch.optim.Adam if args.method == "adam" else torch.optim.AdamW
        optimizer = cls(model.parameters(), lr=args.learning_rate,
                        betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    else:
        from .optim.moonlight_muon import MoonlightMuon
        return OptimizerBundle(
            MoonlightMuon([model.exponents], lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay),
            torch.optim.AdamW(model.output.parameters(), lr=args.aux_learning_rate,
                              betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        )
    return OptimizerBundle(optimizer, None)


def _call_observer(callback: Callable[..., None] | None, *values: Any) -> None:
    """Call an observer without letting its random draws affect the run."""
    if callback is None:
        return
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        callback(*values)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def run_trial(
    args: argparse.Namespace,
    on_epoch: Callable[[int, float | None], None] | None = None,
    on_state: Callable[[int, str, torch.Tensor, torch.Tensor | None, torch.Tensor | None], None] | None = None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    torch.set_num_threads(1)
    dataset = make_dataset(args.task, args.data_seed)
    inputs, hidden, _ = TASKS[args.task]
    model = ProductUnitNetwork(inputs, hidden, input_domain="real_complex", init_bound=1.0)
    init_digest = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    initial_mse = _finite_mse(model, dataset.train_x, dataset.train_y)

    def record_state(epoch: int, kind: str) -> None:
        if on_state is not None:
            vector = parameters_to_vector(model.parameters()).detach().clone()
            _call_observer(on_state, epoch, kind, vector, None, None)

    record_state(0, "initial")
    optimizer = build_optimizer(model, args)
    order_seed = args.seed + 1_000_003
    generator = torch.Generator().manual_seed(order_seed)
    evaluations, completed, max_grad = 0, 0, 0.0
    failed_epoch, failure_reason = None, None
    for epoch in range(1, args.epochs + 1):
        for index in torch.randperm(len(dataset.train_x), generator=generator):
            optimizer.zero_grad()
            loss = (model(dataset.train_x[index:index + 1]) - dataset.train_y[index:index + 1]).square().mean()
            evaluations += 1
            if not torch.isfinite(loss):
                failure_reason = "nonfinite_loss"
                break
            loss.backward()
            norm = math.sqrt(sum(float(p.grad.double().square().sum())
                                 for p in model.parameters() if p.grad is not None))
            if not math.isfinite(norm):
                failure_reason = "nonfinite_gradient"
                break
            max_grad = max(max_grad, norm)
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                failure_reason = "nonfinite_parameters"
                break
            active = [optimizer.primary] + ([optimizer.auxiliary] if optimizer.auxiliary else [])
            if any(not bool(torch.isfinite(value).all()) for opt in active
                   for state in opt.state.values() for value in state.values()
                   if isinstance(value, torch.Tensor)):
                failure_reason = "nonfinite_optimizer_state"
                break
        if failure_reason:
            failed_epoch = epoch
            break
        completed = epoch
        if on_epoch and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            _call_observer(on_epoch, epoch, _finite_mse(model, dataset.train_x, dataset.train_y))
        record_state(epoch, "epoch")
    train_mse = _finite_mse(model, dataset.train_x, dataset.train_y)
    test_mse = _finite_mse(model, dataset.test_x, dataset.test_y)
    if train_mse is None or test_mse is None:
        failure_reason = failure_reason or "nonfinite_final_mse"
    failed = failure_reason is not None
    record_state(failed_epoch or completed, "numerical_failure" if failed else "final")
    result = TrialResult(args.condition_id, args.task, args.method, args.seed,
                         args.data_seed, completed, failed_epoch, evaluations, 0,
                         train_mse, test_mse, initial_mse, max_grad, failed,
                         dataset.digest, "numerical_failure" if failed else "completed")
    return {**asdict(result), "failure_reason": failure_reason,
            "initialization_digest": init_digest, "order_seed": order_seed}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=[PROTOCOL], default=PROTOCOL)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--momentum", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--aux-learning-rate", type=float, default=0.001)
    parser.add_argument("--secondary", choices=["true", "false"], default="false")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--stage", default="exploratory")
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--condition-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.secondary = args.secondary == "true"
    if args.epochs < 1 or args.log_every < 1:
        parser.error("epochs and log-every must be positive")
    for key in ("learning_rate", "aux_learning_rate"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        parser.error("momentum must be in [0,1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if args.weight_decay and not args.secondary:
        parser.error("nonzero weight decay requires secondary=true")
    scientific = {k: v for k, v in vars(args).items() if k not in RUNTIME}
    actual = hashlib.sha256(json.dumps(scientific, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if args.condition_id and args.condition_id != actual:
        parser.error("condition-id does not match the resolved config")
    args.condition_id = actual
    return args


def expand_config(config: dict[str, Any]) -> list[list[str]]:
    fields = {"protocol", "tasks", "methods", "seeds", "data_seed_offset", "epochs",
              "log_every", "device", "stage", "run_group", "recipes"}
    if set(config) != fields or config["protocol"] != PROTOCOL:
        raise ValueError("invalid PUNN gradient config fields or protocol")
    for axis in ("tasks", "methods", "seeds"):
        if not isinstance(config[axis], list) or not config[axis] or len(set(config[axis])) != len(config[axis]):
            raise ValueError(f"{axis} must be a nonempty unique list")
    if set(config["recipes"]) != set(config["methods"]):
        raise ValueError("recipes must cover exactly the requested methods")
    commands = []
    for task, method, seed in itertools.product(config["tasks"], config["methods"], config["seeds"]):
        recipe = config["recipes"][method]
        if set(recipe) != {"learning_rate", "momentum", "weight_decay", "aux_learning_rate", "secondary"}:
            raise ValueError("invalid optimizer recipe fields")
        values = {k: v for k, v in config.items() if k not in {"tasks", "methods", "seeds", "data_seed_offset", "recipes"}}
        values.update(recipe, task=task, method=method, seed=seed,
                      data_seed=seed + config["data_seed_offset"],
                      run_name=f"punn-{task}-{method}-seed{seed}")
        command = ["python", "-m", "optimizer_resurrection.punn_gradients"]
        for key, value in values.items():
            command.extend(["--" + key.replace("_", "-"), str(value).lower() if isinstance(value, bool) else str(value)])
        parse_args(command[3:])
        commands.append(command)
    return commands


def main(argv=None):
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
    invocation = [sys.executable, "-m", "optimizer_resurrection.punn_gradients",
                  *(sys.argv[1:] if argv is None else argv)]
    provenance = {"git": git, "source_tree": _source_tree_digest(root),
                  "invocation": _sanitized_invocation(invocation, credentials),
                  "environment": _environment_metadata(torch.device("cpu")),
                  "determinism": _determinism_metadata(args.seed, args.data_seed),
                  "optimizer_assignment": "Muon on exponent matrix; AdamW on output head" if args.method == "muon_moonlight" else "all parameters",
                  "precision": "FP32 including Moonlight Newton-Schulz; no AMP",
                  "tuning_budget": 0, "batch_size": 1, "gradient_clipping": False,
                  "torch_num_threads": torch.get_num_threads(),
                  "exponent_projection": False, "order_seed": args.seed + 1_000_003}
    provenance["adam_betas"] = [0.9, 0.95] if args.method == "muon_moonlight" else [0.9, 0.999]
    provenance["adam_epsilon"] = 1e-8
    if args.method == "muon_moonlight":
        from .optim.moonlight_muon import SOURCE
        provenance["moonlight_source"] = SOURCE
    import wandb
    run = wandb.init(project=credentials["WANDB_PROJECT"], entity=credentials["WANDB_ENTITY"],
                     group=args.run_group, name=args.run_name, mode="online",
                     config={**vars(args), "provenance": provenance}, save_code=False)
    if run is None:
        raise RuntimeError("W&B initialization returned no run")
    run.config.update({"wandb_run_id": str(run.id), "wandb_run_url": str(run.url)})
    try:
        result = run_trial(args, lambda epoch, mse: run.log({"epoch": epoch, "train_mse": mse}, step=epoch))
        run.summary.update(result)
    except BaseException:
        run.summary["terminal_outcome"] = "failed"
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=1 if result["numerical_failure"] else 0)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
