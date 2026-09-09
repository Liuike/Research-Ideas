from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .diagnostics import (
    activation_statistics,
    input_output_jacobian_spectrum,
    matrix_statistics,
    stiefel_residual,
    temporal_jacobian_statistics,
)
from .models import SigmoidMLP, VanillaRNN
from .optim import build_optimizer
from .tasks import OnlineShapeSet, ShapeSetDataset, make_sequence_batch
from .tracking import load_wandb_credentials, require_online_wandb


@dataclass(frozen=True)
class RunResult:
    train_loss: float
    best_train_loss: float
    validation_loss: float
    validation_accuracy: float
    success: bool
    final_activation_saturation_mean: float
    max_stiefel_residual: float | None


def seed_everything(seed: int) -> None:
    # CUBLAS_WORKSPACE_CONFIG must be present before deterministic CUDA kernels
    # execute. Setting it here is early enough: model/data construction follows.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _image_loaders(args: argparse.Namespace):
    if args.dataset == "shapeset":
        train = OnlineShapeSet(args.data_seed)
        validation = ShapeSetDataset(args.validation_examples, args.data_seed + 10_000_000)
        return (
            DataLoader(train, batch_size=args.batch_size),
            DataLoader(validation, batch_size=args.batch_size),
            32 * 32,
            9,
        )

    from torchvision import datasets, transforms

    root = Path(args.data_dir)
    transform = transforms.ToTensor()
    dataset_cls = datasets.MNIST if args.dataset == "mnist" else datasets.CIFAR10
    full = dataset_cls(root, train=True, download=args.download, transform=transform)
    val_size = min(args.validation_examples, len(full) // 5)
    train_size = min(args.train_examples, len(full) - val_size)
    remainder = len(full) - train_size - val_size
    train, validation, _ = random_split(
        full,
        [train_size, val_size, remainder],
        generator=torch.Generator().manual_seed(args.data_seed),
    )
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.data_seed)),
        DataLoader(validation, batch_size=args.batch_size),
        32 * 32 * 3 if args.dataset == "cifar10" else 28 * 28,
        10,
    )


def _log_record(wandb_run, record: dict[str, Any]) -> None:
    wandb_run.log(record, step=record.get("step"))


def _parameter_diagnostics(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2:
            continue
        stats = matrix_statistics(parameter)
        stats["gradient_rms"] = (
            float(parameter.grad.square().mean().sqrt()) if parameter.grad is not None else 0.0
        )
        stats["update_rms"] = float((parameter.detach() - before[name]).square().mean().sqrt())
        result[name] = stats
    return result


def train_mlp(args: argparse.Namespace, device: torch.device, wandb_run) -> RunResult:
    train_loader, validation_loader, input_dim, classes = _image_loaders(args)
    init = "orthogonal" if args.optimizer.endswith("_o") or args.optimizer in {"rsgd", "mm", "scaled_mm"} else "historical"
    init_scale = args.manifold_scale if args.optimizer == "scaled_mm" else 1.0
    model = SigmoidMLP(input_dim, args.width, args.depth, classes, args.activation, init, init_scale).to(device)
    active_scale = args.manifold_scale if args.optimizer == "scaled_mm" else 1.0
    optimizer = build_optimizer(model, args.optimizer, args.learning_rate, args.aux_learning_rate, args.weight_decay, active_scale, args.muon_backend, args.assignment)
    criterion = nn.CrossEntropyLoss()
    iterator = iter(train_loader)
    loss_value = float("nan")
    best_loss = float("inf")
    final_saturation = float("nan")
    max_residual: float | None = None
    for step in range(1, args.steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, activations = model(x, return_activations=True)
        loss = criterion(logits, y)
        loss.backward()
        before = {name: p.detach().clone() for name, p in model.named_parameters() if p.ndim == 2}
        optimizer.step()
        loss_value = float(loss.detach())
        best_loss = min(best_loss, loss_value)
        if step == 1 or step % args.diagnostic_interval == 0 or step == args.steps:
            record = {
                "step": step,
                "train_loss": loss_value,
                "parameters": _parameter_diagnostics(model, before),
                "activations": [activation_statistics(a, args.activation) for a in activations],
            }
            final_saturation = float(
                np.mean([value["saturation_fraction"] for value in record["activations"]])
            )
            if args.optimizer in {"rsgd", "mm", "scaled_mm"}:
                record["compared_stiefel_residuals"] = [
                    stiefel_residual(weight, active_scale) for weight in model.compared_weights
                ]
                observed = max(record["compared_stiefel_residuals"], default=0.0)
                max_residual = observed if max_residual is None else max(max_residual, observed)
            if step == args.steps or (args.jacobian_interval > 0 and step % args.jacobian_interval == 0):
                record["input_output_jacobian_singular_values"] = input_output_jacobian_spectrum(model, x)
            _log_record(wandb_run, record)
    val_loss, val_accuracy = evaluate_mlp(model, validation_loader, criterion, device)
    success = loss_value <= args.success_loss and val_accuracy >= args.success_accuracy
    return RunResult(
        loss_value,
        best_loss,
        val_loss,
        val_accuracy,
        success,
        final_saturation,
        max_residual,
    )


@torch.no_grad()
def evaluate_mlp(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(criterion(logits, y)) * y.numel()
        total_correct += int((logits.argmax(1) == y).sum())
        total += y.numel()
    model.train()
    return total_loss / total, total_correct / total


def train_rnn(args: argparse.Namespace, device: torch.device, wandb_run) -> RunResult:
    input_size = 2 if args.task == "latch" else 1
    init = "orthogonal" if args.optimizer.endswith("_o") or args.optimizer in {"rsgd", "mm", "scaled_mm"} else "historical"
    init_scale = args.manifold_scale if args.optimizer == "scaled_mm" else 1.0
    model = VanillaRNN(input_size, args.hidden_size, init=init, orthogonal_scale=init_scale).to(device)
    active_scale = args.manifold_scale if args.optimizer == "scaled_mm" else 1.0
    optimizer = build_optimizer(model, args.optimizer, args.learning_rate, args.aux_learning_rate, args.weight_decay, active_scale, args.muon_backend, args.assignment)
    criterion = nn.CrossEntropyLoss()
    loss_value = float("nan")
    best_loss = float("inf")
    final_saturation = float("nan")
    max_residual: float | None = None
    for step in range(1, args.steps + 1):
        generator = torch.Generator().manual_seed(args.data_seed + step)
        batch = make_sequence_batch(args.task, args.batch_size, args.max_length, generator)
        x, y, lengths = batch.inputs.to(device), batch.targets.to(device), batch.lengths.to(device)
        optimizer.zero_grad()
        logits, states = model(x, lengths, return_states=True)
        loss = criterion(logits, y)
        loss.backward()
        before = {name: p.detach().clone() for name, p in model.named_parameters() if p.ndim == 2}
        optimizer.step()
        loss_value = float(loss.detach())
        best_loss = min(best_loss, loss_value)
        if step == 1 or step % args.diagnostic_interval == 0 or step == args.steps:
            valid = torch.arange(states.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
            record = {
                "step": step,
                "train_loss": loss_value,
                "parameters": _parameter_diagnostics(model, before),
                "state_saturation": float((states.abs() > 0.95)[valid].float().mean()),
                "recurrent_stiefel_residual": stiefel_residual(model.W_hh, active_scale),
                "temporal_jacobians": temporal_jacobian_statistics(model.W_hh, states, lengths),
            }
            final_saturation = record["state_saturation"]
            observed = record["recurrent_stiefel_residual"]
            max_residual = observed if max_residual is None else max(max_residual, observed)
            _log_record(wandb_run, record)
    val_loss, val_accuracy = evaluate_rnn(model, args, criterion, device)
    success = loss_value <= args.success_loss and val_accuracy >= args.success_accuracy
    return RunResult(
        loss_value,
        best_loss,
        val_loss,
        val_accuracy,
        success,
        final_saturation,
        max_residual,
    )


@torch.no_grad()
def evaluate_rnn(model, args, criterion, device) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(args.data_seed + 999_999)
    batch = make_sequence_batch(args.task, args.validation_examples, args.max_length, generator)
    logits = model(batch.inputs.to(device), batch.lengths.to(device))
    targets = batch.targets.to(device)
    return float(criterion(logits, targets)), float((logits.argmax(1) == targets).float().mean())


_RUNTIME_CONFIG_KEYS = {
    "data_dir",
    "device",
    "download",
    "dry_run",
    "run_group",
    "run_name",
    "stage",
}


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key not in _RUNTIME_CONFIG_KEYS
    }


def _condition_id(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_tree_digest(project_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files: list[Path] = []
    for relative in (Path("src"), Path("configs"), Path("scripts")):
        root = project_root / relative
        if root.exists():
            files.extend(
                path for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    for name in ("pyproject.toml",):
        path = project_root / name
        if path.is_file():
            files.append(path)
    unique = sorted(set(files), key=lambda path: path.relative_to(project_root).as_posix())
    for path in unique:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"algorithm": "sha256", "digest": digest.hexdigest(), "file_count": len(unique)}


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*command: str) -> str | None:
        try:
            return subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    revision = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain", "--untracked-files=all")
    return {"revision": revision, "dirty": status is None or bool(status)}


def _require_reproducible_source(stage: str, git: dict[str, Any]) -> None:
    if stage in {"engineering-smoke", "test"}:
        return
    if not git.get("revision") or git.get("dirty"):
        raise RuntimeError(
            "scientific experiments require a clean committed git revision; "
            "use --stage engineering-smoke only for non-claimable validation"
        )


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "torch", "torchvision", "PyYAML", "wandb"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            cuda_devices.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            })
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_PROCID",
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "dependencies": _dependency_versions(),
        "selected_device": str(device),
        "cuda": {
            "available": cuda_available,
            "runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": cuda_devices,
        },
        "slurm": {key: os.environ[key] for key in slurm_keys if os.environ.get(key)},
    }


def _determinism_metadata(seed: int, data_seed: int) -> dict[str, Any]:
    return {
        "model_seed": seed,
        "data_seed": data_seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "dataloader_workers": 0,
    }


def _sanitized_invocation(argv: list[str], credentials: dict[str, str]) -> dict[str, Any]:
    redacted: list[str] = []
    redact_next = False
    secret_values = {value for value in credentials.values() if value}
    for token in argv:
        lowered = token.lower()
        if redact_next or token in secret_values:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if any(marker in lowered for marker in ("api-key", "password", "token", "secret")):
            if "=" in token:
                redacted.append(token.split("=", 1)[0] + "=<redacted>")
            else:
                redacted.append(token)
                redact_next = True
            continue
        sanitized = token
        for value in secret_values:
            sanitized = sanitized.replace(value, "<redacted>")
        redacted.append(sanitized)
    return {"argv": redacted, "shell": shlex.join(redacted)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["mlp", "rnn"], required=True)
    parser.add_argument("--optimizer", choices=["bptt_sgd", "sgd_x", "sgd_o", "adam_x", "adam_o", "muon_x", "muon_o", "rsgd", "mm", "scaled_mm"], default="adam_o")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--diagnostic-interval", type=int, default=100)
    parser.add_argument("--jacobian-interval", type=int, default=1000)
    parser.add_argument("--success-loss", type=float, default=0.2)
    parser.add_argument("--success-accuracy", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name")
    parser.add_argument("--run-group")
    parser.add_argument("--stage", default="experiment")
    parser.add_argument("--recipe-version", default="pilot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--muon-backend", choices=["ns", "svd"], default="ns")
    parser.add_argument("--assignment", choices=["primary", "standard"], default="primary")
    parser.add_argument("--manifold-scale", type=float, default=1.0)
    parser.add_argument("--activation", choices=["logistic", "tanh"], default="logistic")
    parser.add_argument("--dataset", choices=["shapeset", "mnist", "cifar10"], default="shapeset")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-examples", type=int, default=10_000)
    parser.add_argument("--validation-examples", type=int, default=1_000)
    parser.add_argument("--task", choices=["latch", "two_sequence", "parity"], default="latch")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=100)
    args = parser.parse_args(argv)
    if args.data_seed is None:
        args.data_seed = args.seed
    return args


def main(argv: list[str] | None = None) -> RunResult | None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    scientific_config = _scientific_config(args)
    condition_id = _condition_id(scientific_config)
    if args.run_group is None:
        args.run_group = f"{args.stage}/{args.recipe_version}"
    if args.dry_run:
        print(json.dumps({**vars(args), "condition_id": condition_id}, indent=2, sort_keys=True))
        return None

    credentials = load_wandb_credentials(project_root)
    require_online_wandb()
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    invocation_argv = (
        list(sys.argv)
        if argv is None
        else [sys.executable, "-m", "optimizer_resurrection.train", *argv]
    )
    git_metadata = _git_metadata(project_root)
    _require_reproducible_source(args.stage, git_metadata)
    provenance = {
        "invocation": _sanitized_invocation(invocation_argv, credentials),
        "git": git_metadata,
        "source_tree": _source_tree_digest(project_root),
        "environment": _environment_metadata(device),
        "determinism": _determinism_metadata(args.seed, args.data_seed),
    }
    run_config = {
        **scientific_config,
        "condition_id": condition_id,
        "run_group": args.run_group,
        "stage": args.stage,
        "provenance": provenance,
    }
    run_name = args.run_name or f"{args.track}-{args.optimizer}-s{args.seed}-{condition_id[:8]}"

    import wandb

    wandb_run = wandb.init(
        project=credentials["WANDB_PROJECT"],
        entity=credentials["WANDB_ENTITY"],
        name=run_name,
        group=args.run_group,
        config=run_config,
        mode="online",
        settings=wandb.Settings(mode="online"),
        save_code=False,
    )
    if wandb_run is None:
        raise RuntimeError("W&B online initialization returned no run")

    identity = {
        "id": str(wandb_run.id),
        "name": str(wandb_run.name),
        "entity": str(getattr(wandb_run, "entity", credentials["WANDB_ENTITY"])),
        "project": str(getattr(wandb_run, "project", credentials["WANDB_PROJECT"])),
        "group": args.run_group,
        "url": str(wandb_run.url),
    }
    wandb_run.config.update({"provenance": {**provenance, "wandb": identity}}, allow_val_change=True)
    try:
        result = (
            train_mlp(args, device, wandb_run)
            if args.track == "mlp"
            else train_rnn(args, device, wandb_run)
        )
        result_payload = asdict(result)
        wandb_run.summary.update({**result_payload, "run_result": result_payload})
    except BaseException:
        wandb_run.finish(exit_code=1)
        raise
    else:
        wandb_run.finish(exit_code=0)
    print(json.dumps(asdict(result), sort_keys=True))
    return result


if __name__ == "__main__":
    main()
