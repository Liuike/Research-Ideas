"""Focused reconstruction of Engelbrecht and Gouldie's 2024 PUNN benchmarks.

Source: https://www.mdpi.com/1999-4893/17/6/241, Sections 2, 6, and 8.
The paper specifies the model, synthetic functions, 75/25 split, 500 epochs,
30 runs, uniform [-1,1] initialization, and SGD/PSO/DE control parameters.
It does not publish the sampled datasets, seeds, minibatch order, population
size, or all boundary behavior. These are explicit reconstruction choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .models import ProductUnitNetwork
from .tracking import load_wandb_credentials, require_online_wandb
from .train import (
    _determinism_metadata,
    _environment_metadata,
    _git_metadata,
    _require_reproducible_source,
    _sanitized_invocation,
    _source_tree_digest,
    seed_everything,
)


PROTOCOL = "engelbrecht-gouldie-2024-basic-v1"
SOURCE = "https://www.mdpi.com/1999-4893/17/6/241"
TASKS = {"f1": (1, 1, 50), "f4": (2, 3, 300)}
_RUNTIME_KEYS = {"condition_id", "device", "dry_run", "run_group", "run_name", "stage", "log_every"}


@dataclass(frozen=True)
class Dataset:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    digest: str


@dataclass(frozen=True)
class TrialResult:
    condition_id: str
    task: str
    method: str
    seed: int
    data_seed: int
    epochs_completed: int
    failed_epoch: int | None
    pattern_evaluations: int
    full_objective_evaluations: int
    train_mse: float | None
    test_mse: float | None
    initial_train_mse: float | None
    max_gradient_norm: float | None
    numerical_failure: bool
    dataset_digest: str
    terminal_outcome: str


def make_dataset(task: str, data_seed: int) -> Dataset:
    """Generate the paper's f1 or f4 inputs and take a reproducible 75/25 split."""
    if task not in TASKS:
        raise ValueError(f"unsupported 2024 basic task: {task}")
    input_dim, _, count = TASKS[task]
    generator = torch.Generator(device="cpu").manual_seed(data_seed)
    x = torch.empty(count, input_dim, dtype=torch.float32).uniform_(-1, 1, generator=generator)
    # The source does not specify its policy for a rare exact zero. Resample it
    # because the published logarithmic forward formula is undefined there.
    while (x == 0).any():
        mask = x == 0
        x[mask] = torch.empty(int(mask.sum()), dtype=x.dtype).uniform_(-1, 1, generator=generator)
    if task == "f1":
        y = x[:, :1].square()
    else:
        y = x[:, 1:2].pow(7) * x[:, :1].pow(3) - 0.5 * x[:, :1].pow(6)
    order = torch.randperm(count, generator=generator)
    train_count = int(0.75 * count)
    train_x, test_x = x[order[:train_count]], x[order[train_count:]]
    train_y, test_y = y[order[:train_count]], y[order[train_count:]]
    digest = hashlib.sha256()
    for tensor in (train_x, train_y, test_x, test_y):
        digest.update(tensor.contiguous().numpy().tobytes())
    return Dataset(train_x, train_y, test_x, test_y, digest.hexdigest())


def _finite_mse(model: ProductUnitNetwork, x: torch.Tensor, y: torch.Tensor) -> float | None:
    with torch.no_grad():
        value = torch.mean((model(x) - y).square())
    result = float(value)
    return result if math.isfinite(result) else None


def _run_sgd(
    model: ProductUnitNetwork,
    dataset: Dataset,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    momentum: float,
    on_epoch: Callable[[int, float | None], None] | None,
) -> tuple[int, int, float | None, bool, int | None]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    max_grad = 0.0
    evaluations = 0
    for epoch in range(1, epochs + 1):
        for index in torch.randperm(len(dataset.train_x), generator=generator):
            optimizer.zero_grad()
            prediction = model(dataset.train_x[index : index + 1])
            loss = (prediction - dataset.train_y[index : index + 1]).square().mean()
            evaluations += 1
            if not torch.isfinite(loss):
                return epoch - 1, evaluations, max_grad, True, epoch
            loss.backward()
            squared = sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None)
            grad_norm = math.sqrt(squared)
            if not math.isfinite(grad_norm):
                return epoch - 1, evaluations, max_grad, True, epoch
            max_grad = max(max_grad, grad_norm)
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                return epoch - 1, evaluations, max_grad, True, epoch
        if on_epoch is not None:
            on_epoch(epoch, _finite_mse(model, dataset.train_x, dataset.train_y))
    return epochs, evaluations, max_grad, False, None


def run_trial(
    *,
    task: str,
    method: str,
    seed: int,
    data_seed: int,
    epochs: int,
    population_size: int = 20,
    search_bound: float = 1.0,
    boundary: str = "clip",
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    on_epoch: Callable[[int, float | None], None] | None = None,
    condition_id: str = "",
) -> TrialResult:
    if task not in TASKS or method not in {"sgd", "pso", "de"}:
        raise ValueError("unsupported task or method")
    if epochs < 1 or population_size < 4 or not math.isfinite(search_bound) or search_bound <= 0:
        raise ValueError("invalid experiment budget or search bound")
    if learning_rate <= 0 or not 0 <= momentum < 1:
        raise ValueError("invalid SGD parameters")
    seed_everything(seed)
    dataset = make_dataset(task, data_seed)
    input_dim, hidden_units, _ = TASKS[task]
    model = ProductUnitNetwork(input_dim, hidden_units, input_domain="real_complex", init_bound=1.0)
    initial_mse = _finite_mse(model, dataset.train_x, dataset.train_y)
    max_gradient_norm: float | None = None
    numerical_failure = False
    failed_epoch: int | None = None
    if method == "sgd":
        completed, pattern_evaluations, max_gradient_norm, numerical_failure, failed_epoch = _run_sgd(
            model, dataset, seed=seed, epochs=epochs,
            learning_rate=learning_rate, momentum=momentum, on_epoch=on_epoch,
        )
        full_objective_evaluations = 0
    else:
        from .punn_population import differential_evolution, particle_swarm

        initial_vector = parameters_to_vector(model.parameters()).detach().clone()

        def objective(vector: torch.Tensor) -> float:
            if not torch.isfinite(vector).all():
                return float("inf")
            vector_to_parameters(vector, model.parameters())
            value = _finite_mse(model, dataset.train_x, dataset.train_y)
            return value if value is not None else float("inf")

        algorithm = particle_swarm if method == "pso" else differential_evolution
        population_result = algorithm(
            objective, dimension=initial_vector.numel(),
            bounds=(-search_bound, search_bound), seed=seed, epochs=epochs,
            population_size=population_size, initial_vector=initial_vector,
            boundary=boundary,
        )
        vector_to_parameters(population_result.best_parameters, model.parameters())
        completed = epochs
        full_objective_evaluations = population_result.evaluations
        pattern_evaluations = full_objective_evaluations * len(dataset.train_x)
        numerical_failure = not math.isfinite(population_result.best_loss)
    train_mse = _finite_mse(model, dataset.train_x, dataset.train_y)
    test_mse = _finite_mse(model, dataset.test_x, dataset.test_y)
    numerical_failure = numerical_failure or train_mse is None or test_mse is None
    return TrialResult(condition_id, task, method, seed, data_seed, completed, failed_epoch,
                       pattern_evaluations, full_objective_evaluations,
                       train_mse, test_mse, initial_mse, max_gradient_norm,
                       numerical_failure, dataset.digest,
                       "numerical_failure" if numerical_failure else "completed")


def _condition_id(config: dict[str, Any]) -> str:
    scientific = {key: value for key, value in config.items() if key not in _RUNTIME_KEYS}
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=[PROTOCOL], default=PROTOCOL)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--method", choices=["sgd", "pso", "de"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--search-bound", type=float, default=1.0)
    parser.add_argument("--boundary", choices=["clip", "reflect"], default="clip")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--stage", default="reproduction")
    parser.add_argument("--run-group", default="punn-landscape-basic-v1")
    parser.add_argument("--run-name")
    parser.add_argument("--condition-id")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.population_size < 4 or args.log_every < 1:
        parser.error("epochs, population-size, and log-every must be positive")
    if not math.isfinite(args.search_bound) or args.search_bound <= 0:
        parser.error("search-bound must be finite and positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        parser.error("momentum must be in [0, 1)")
    actual_id = _condition_id(vars(args))
    if args.condition_id is not None and args.condition_id != actual_id:
        parser.error("condition-id does not match resolved scientific configuration")
    args.condition_id = actual_id
    return args


def main(argv: list[str] | None = None) -> TrialResult | None:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), sort_keys=True, indent=2))
        return None
    project_root = Path(__file__).resolve().parents[2]
    git = _git_metadata(project_root)
    _require_reproducible_source(args.stage, git)
    credentials = load_wandb_credentials(project_root)
    require_online_wandb()
    invocation = [sys.executable, "-m", "optimizer_resurrection.punn_landscape",
                  *(sys.argv[1:] if argv is None else argv)]
    seed_everything(args.seed)
    provenance = {
        "invocation": _sanitized_invocation(invocation, credentials),
        "git": git,
        "source_tree": _source_tree_digest(project_root),
        "environment": _environment_metadata(torch.device("cpu")),
        "determinism": _determinism_metadata(args.seed, args.data_seed),
        "source": {"url": SOURCE, "sections": [2, 6, 8]},
        "reconstruction_choices": {
            "sampled_data": "new deterministic samples; original samples unavailable",
            "zero_input": "resample exact zero",
            "sgd_order": "seeded random permutation each epoch, batch size one",
            "population_size": args.population_size,
            "population_boundary": args.boundary,
            "search_bounds": [-args.search_bound, args.search_bound],
            "dtype": "float32",
        },
    }
    import wandb

    run = wandb.init(
        project=credentials["WANDB_PROJECT"], entity=credentials["WANDB_ENTITY"],
        name=args.run_name or f"punn-{args.task}-{args.method}-seed{args.seed}",
        group=args.run_group, config={**vars(args), "provenance": provenance},
        mode="online", settings=wandb.Settings(mode="online"), save_code=False,
    )
    if run is None:
        raise RuntimeError("W&B online initialization returned no run")
    run.config.update({"provenance": {**provenance, "wandb": {
        "id": str(run.id), "url": str(run.url), "group": args.run_group,
    }}}, allow_val_change=True)
    def log_epoch(epoch: int, mse: float | None) -> None:
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            run.log({"epoch": epoch, "train_mse": mse}, step=epoch)
    try:
        result = run_trial(
            task=args.task, method=args.method, seed=args.seed, data_seed=args.data_seed,
            epochs=args.epochs, population_size=args.population_size,
            search_bound=args.search_bound, boundary=args.boundary,
            learning_rate=args.learning_rate, momentum=args.momentum,
            on_epoch=log_epoch, condition_id=args.condition_id,
        )
        payload = asdict(result)
        run.summary.update({**payload, "run_result": payload})
    except BaseException:
        run.summary.update({"terminal_outcome": "failed"})
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0 if not result.numerical_failure else 1)
    print(json.dumps(asdict(result), sort_keys=True))
    return result


if __name__ == "__main__":
    main()
