"""Frozen PURe ResNet-18 adaptation of arXiv:2505.04397v3 on CIFAR-10."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import datasets, transforms

from .tracking import load_wandb_credentials, require_online_wandb
from .train import (_determinism_metadata, _environment_metadata, _git_metadata,
                    _require_reproducible_source, _sanitized_invocation,
                    _source_tree_digest, seed_everything)

PROTOCOL = "pure-cifar10-resnet18-v1"
PARAMETERS = 11_173_970
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
RUNTIME = {"condition_id", "dry_run", "run_group", "run_name", "stage",
           "data_dir", "download", "device", "num_workers"}
FIELDS = {"protocol", "seeds", "data_seed_offset", "epochs", "batch_size",
          "learning_rate", "momentum", "weight_decay", "milestones", "gamma",
          "device", "num_workers", "data_dir", "download", "train_examples",
          "test_examples", "secondary", "stage", "run_group"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", choices=[PROTOCOL], default=PROTOCOL)
    for name, default in (("seed", 0), ("data-seed", 100000), ("epochs", 160),
                          ("batch-size", 128), ("num-workers", 4),
                          ("train-examples", 50000), ("test-examples", 10000)):
        p.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", .01), ("momentum", .9),
                          ("weight-decay", .001), ("gamma", .1)):
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--milestones", default="80,120")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--data-dir", default="data/cifar10")
    p.add_argument("--download", choices=["true", "false"], default="false")
    p.add_argument("--secondary", choices=["true", "false"], default="true")
    p.add_argument("--stage", default="reproduction")
    p.add_argument("--run-group", required=True)
    p.add_argument("--run-name")
    p.add_argument("--condition-id")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    try:
        a.milestones = [] if a.milestones == "none" else [int(v) for v in a.milestones.split(",")]
    except ValueError:
        p.error("milestones must be comma-separated integers or 'none'")
    a.download, a.secondary = a.download == "true", a.secondary == "true"
    if min(a.epochs, a.batch_size, a.train_examples, a.test_examples) < 1 or a.num_workers < 0:
        p.error("budgets must be positive and workers nonnegative")
    if a.train_examples > 50000 or a.test_examples > 10000 or min(a.seed, a.data_seed) < 0:
        p.error("invalid CIFAR budgets or seeds")
    if a.milestones != sorted(set(a.milestones)) or any(v < 1 or v >= a.epochs for v in a.milestones):
        p.error("milestones must be unique, increasing, and within the epoch budget")
    if not all(math.isfinite(v) for v in (a.learning_rate, a.momentum, a.weight_decay, a.gamma)):
        p.error("optimizer values must be finite")
    if a.learning_rate <= 0 or not 0 <= a.momentum < 1 or a.weight_decay < 0 or not 0 < a.gamma < 1:
        p.error("invalid optimizer recipe")
    if not a.secondary:
        p.error("paper's scheduled, decayed recipe must be marked secondary")
    if a.stage not in {"engineering-smoke", "test"}:
        frozen = (a.epochs, a.batch_size, a.learning_rate, a.momentum, a.weight_decay,
                  a.milestones, a.gamma, a.train_examples, a.test_examples)
        if frozen != (160, 128, .01, .9, .001, [80, 120], .1, 50000, 10000):
            p.error("scientific runs must use the frozen paper reconstruction recipe")
    scientific = {k: v for k, v in vars(a).items() if k not in RUNTIME}
    actual = hashlib.sha256(json.dumps(scientific, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if a.condition_id and a.condition_id != actual:
        p.error("condition-id mismatch")
    a.condition_id = actual
    return a


def expand_config(config):
    if set(config) != FIELDS or config["protocol"] != PROTOCOL:
        raise ValueError("invalid PURe config fields or protocol")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique list")
    commands = []
    for seed in seeds:
        values = {k: v for k, v in config.items() if k not in {"seeds", "data_seed_offset"}}
        values.update(seed=seed, data_seed=seed + config["data_seed_offset"],
                      run_name=f"pure-resnet18-cifar10-sgd-seed{seed}")
        command = ["python", "-m", "optimizer_resurrection.pure_cifar"]
        for key, value in values.items():
            if key == "milestones":
                value = ",".join(map(str, value)) or "none"
            elif isinstance(value, bool):
                value = str(value).lower()
            command.extend(["--" + key.replace("_", "-"), str(value)])
        parse_args(command[3:])
        commands.append(command)
    return commands


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(.5),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = datasets.CIFAR10(args.data_dir, train=True, download=args.download, transform=train_transform)
    test = datasets.CIFAR10(args.data_dir, train=False, download=args.download, transform=test_transform)
    digest = hashlib.sha256()
    for dataset, count in ((train, args.train_examples), (test, args.test_examples)):
        digest.update(np.ascontiguousarray(dataset.data[:count]).tobytes())
        digest.update(np.asarray(dataset.targets[:count], dtype="<i8").tobytes())
    train_subset, test_subset = Subset(train, range(args.train_examples)), Subset(test, range(args.test_examples))
    order_seed, worker_seed = args.data_seed + 1_000_003, args.data_seed + 2_000_003
    sampler = RandomSampler(train_subset, generator=torch.Generator().manual_seed(order_seed))
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  worker_init_fn=seed_worker, pin_memory=args.device == "cuda", drop_last=False)
    train_loader = DataLoader(train_subset, sampler=sampler,
                             generator=torch.Generator().manual_seed(worker_seed), **common)
    test_loader = DataLoader(test_subset, shuffle=False,
                            generator=torch.Generator().manual_seed(worker_seed + 1), **common)
    return train_loader, test_loader, {"dataset_digest": digest.hexdigest(),
        "order_seed": order_seed, "worker_seed": worker_seed,
        "augmentation_seed": args.data_seed + 3_000_003,
        "train_examples": len(train_subset), "test_examples": len(test_subset)}


def all_finite(tensors):
    values = [torch.isfinite(t.detach()).all() for t in tensors]
    return not values or bool(torch.stack(values).all())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        if not all_finite([logits, loss]):
            raise FloatingPointError("nonfinite_test_output")
        total += y.numel()
        correct += int((logits.argmax(1) == y).sum())
        loss_sum += float(loss)
    return {"test_loss": loss_sum / total, "test_accuracy": correct / total,
            "test_examples_evaluated": total}


def run_trial(args, on_epoch=None, loaders=None, model_factory=None):
    from .models.pure_resnet import ProductUnitConv2d, pure_resnet18_cifar10
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(1)
    if loaders is None:
        loaders = make_loaders(args)
    train_loader, test_loader, metadata = loaders
    # Dataset preparation must not change the model initialization stream.
    seed_everything(args.seed)
    model = (model_factory or pure_resnet18_cifar10)()
    initialization = hashlib.sha256()
    for name, parameter in model.named_parameters():
        initialization.update(name.encode())
        initialization.update(parameter.detach().numpy().tobytes())
    count = sum(p.numel() for p in model.parameters())
    if model_factory is None and count != PARAMETERS:
        raise RuntimeError(f"unexpected parameter count: {count}")
    model = model.to(device)
    torch.manual_seed(metadata["augmentation_seed"])
    random.seed(metadata["augmentation_seed"])
    np.random.seed(metadata["augmentation_seed"] % 2**32)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                               momentum=args.momentum, weight_decay=args.weight_decay,
                               dampening=0, nesterov=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.gamma)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    completed, failure, failed_epoch = 0, None, None
    result = {**metadata, "parameter_count": count,
              "initialization_digest": initialization.hexdigest()}
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        lr = optimizer.param_groups[0]["lr"]
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if not all_finite([logits, loss]):
                failure = "nonfinite_loss"
                break
            loss.backward()
            if not all_finite(p.grad for p in model.parameters() if p.grad is not None):
                failure = "nonfinite_gradient"
                break
            optimizer.step()
            state = [v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
            if not all_finite([*model.parameters(), *model.buffers(), *state]):
                failure = "nonfinite_parameters_or_state"
                break
            total += y.numel()
            correct += int((logits.detach().argmax(1) == y).sum())
            loss_sum += float(loss.detach()) * y.numel()
        if failure:
            failed_epoch = epoch
            break
        if total != metadata["train_examples"]:
            raise RuntimeError("incomplete training dataset coverage")
        completed = epoch
        thresholds = {name: float(module.threshold.detach()) for name, module in model.named_modules()
                      if isinstance(module, ProductUnitConv2d)}
        record = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
                  "learning_rate": lr, "train_examples_seen": total, "thresholds": thresholds,
                  "epoch_seconds": time.perf_counter() - epoch_start}
        result.update({k: v for k, v in record.items() if k != "thresholds"})
        if on_epoch:
            on_epoch(record)
        scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result["training_seconds"] = time.perf_counter() - start
    if not failure:
        try:
            result.update(evaluate(model, test_loader, device))
        except FloatingPointError as exc:
            failure, failed_epoch = str(exc), args.epochs
    result.update(epochs_completed=completed, numerical_failure=failure is not None,
                  failure_reason=failure, failed_epoch=failed_epoch,
                  terminal_outcome="numerical_failure" if failure else "completed",
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), sort_keys=True))
        return
    root = Path(__file__).resolve().parents[2]
    git = _git_metadata(root)
    _require_reproducible_source(args.stage, git)
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    seed_everything(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)
    invocation = [sys.executable, "-m", "optimizer_resurrection.pure_cifar",
                  *(sys.argv[1:] if argv is None else argv)]
    provenance = {"git": git, "source_tree": _source_tree_digest(root),
        "environment": _environment_metadata(device),
        "invocation": _sanitized_invocation(invocation, credentials),
        "determinism": {**_determinism_metadata(args.seed, args.data_seed),
                        "dataloader_workers": args.num_workers,
                        "augmentation_rng": "worker_seed/iterator-generated worker torch seeds when workers>0; augmentation_seed when workers=0"},
        "paper": "https://arxiv.org/html/2505.04397v3",
        "architecture_adaptation": "four-stage PURe ResNet-18; no published CIFAR-18 target",
        "normalization_mean": MEAN, "normalization_std": STD,
        "augmentation": "RandomCrop(32,padding=4); RandomHorizontalFlip(0.5)",
        "initialization": "Kaiming normal fan_out kernels; theta=0; BN=1/0; torchvision head",
        "evaluation": "final model only; official test set once; no validation split",
        "precision": "FP32; no AMP/TF32", "gradient_clipping": False,
        "tuning_budget": 0, "optimizer": "SGD all parameters; dampening=0; nesterov=false"}
    import wandb
    run = wandb.init(project=credentials["WANDB_PROJECT"], entity=credentials["WANDB_ENTITY"],
                     name=args.run_name, group=args.run_group, mode="online", save_code=False,
                     config={**vars(args), "provenance": provenance})
    if run is None:
        raise RuntimeError("W&B initialization returned no run")
    run.config.update({"wandb_run_id": run.id, "wandb_run_url": run.url})
    try:
        result = run_trial(args, lambda record: run.log(record, step=record["epoch"]))
        run.summary.update(result)
    except BaseException:
        run.summary["terminal_outcome"] = "failed"
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=1 if result["numerical_failure"] else 0)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
