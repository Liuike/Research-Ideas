"""Registered 2024 basic replication with W&B-owned loss-landscape records.

Training retains the original CPU FP32 scalar objective and RNG order. Separate
batched landscape evaluation may use CUDA. Diagnostic RNGs never train models.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .models import ProductUnitNetwork
from .punn_landscape import TASKS, make_dataset, run_trial
from .tracking import load_wandb_credentials, require_online_wandb
from .train import (_determinism_metadata, _environment_metadata, _git_metadata,
                    _require_reproducible_source, _sanitized_invocation,
                    _source_tree_digest, seed_everything)

PROTOCOL = "engelbrecht-gouldie-2024-recorded-v1"
RUNTIME = {"condition_id", "dry_run", "run_group", "run_name", "stage"}
FIELDS = {"protocol", "tasks", "methods", "seeds", "data_seed_offset", "epochs",
          "population_size", "search_bound", "boundary", "learning_rate", "momentum",
          "log_every", "device", "stage", "run_group", "landscape_device",
          "landscape_seed_offset", "slice_points", "slice_radius", "prw_steps",
          "mrw_steps", "uniform_per_dim", "dispersion_samples", "landscape_batch_size",
          "landscape_bounds"}
OPTIONAL_FIELDS = {"landscape_cpu_reference"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=[PROTOCOL], default=PROTOCOL)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--method", choices=["sgd", "pso", "de"], required=True)
    for name in ("seed", "data-seed"):
        parser.add_argument("--" + name, type=int, required=True)
    for name, default in (("epochs", 500), ("population-size", 20), ("log-every", 50),
                          ("landscape-seed-offset", 2_000_000), ("slice-points", 31),
                          ("prw-steps", 1000), ("mrw-steps", 2000),
                          ("uniform-per-dim", 500), ("dispersion-samples", 2000),
                          ("landscape-batch-size", 512)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("search-bound", 1.0), ("learning-rate", 0.1),
                          ("momentum", 0.9), ("slice-radius", 1.0)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--boundary", choices=["clip", "reflect"], default="clip")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--landscape-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--landscape-cpu-reference", choices=["true", "false"], default="false")
    parser.add_argument("--landscape-bounds", choices=["training", "training-and-paper-wide"],
                        default="training-and-paper-wide")
    parser.add_argument("--stage", default="reproduction")
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--condition-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for key in ("epochs", "log_every", "prw_steps", "mrw_steps", "uniform_per_dim",
                "dispersion_samples", "landscape_batch_size"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.population_size < 4 or args.slice_points < 3 or args.slice_points % 2 != 1:
        parser.error("population_size must be >=4 and slice_points must be odd and >=3")
    for key in ("search_bound", "learning_rate", "slice_radius"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        parser.error("momentum must be in [0,1)")
    if min(args.seed, args.data_seed, args.landscape_seed_offset) < 0:
        parser.error("seeds and seed offset must be nonnegative")
    values = {k: v for k, v in vars(args).items() if k not in RUNTIME
              and not (k == "landscape_cpu_reference" and v == "false")}
    condition = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    if args.condition_id is not None and args.condition_id != condition:
        parser.error("condition-id does not match resolved scientific configuration")
    args.condition_id = condition
    return args


def expand_config(config: dict[str, Any]) -> list[list[str]]:
    if not FIELDS <= config.keys() or config.keys() - FIELDS - OPTIONAL_FIELDS or config.get("protocol") != PROTOCOL:
        raise ValueError(f"invalid recorded PUNN config fields: missing={FIELDS-config.keys()} "
                         f"extra={config.keys()-FIELDS-OPTIONAL_FIELDS}")
    for axis in ("tasks", "methods", "seeds"):
        values = config[axis]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{axis} must be a nonempty unique list")
    commands = []
    for task, method, seed in itertools.product(config["tasks"], config["methods"], config["seeds"]):
        values = {k: v for k, v in config.items()
                  if k not in {"tasks", "methods", "seeds", "data_seed_offset"}}
        values.update(task=task, method=method, seed=seed,
                      data_seed=seed + config["data_seed_offset"],
                      run_name=f"punn-recorded-{task}-{method}-seed{seed}")
        command = ["python", "-m", "optimizer_resurrection.punn_recorded"]
        for key, value in values.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        parse_args(command[3:])
        commands.append(command)
    return commands


def _write_payload(artifact, name: str, payload: dict[str, Any]) -> None:
    """Serialize arrays into a W&B-managed artifact file, never project outputs."""
    arrays = {}
    metadata = {}

    def flatten(prefix, value):
        if isinstance(value, torch.Tensor):
            arrays[prefix] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arrays[prefix] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                flatten(f"{prefix}/{key}" if prefix else str(key), child)
        else:
            metadata[prefix] = value

    flatten("", payload)
    # JSON strings are stored without object dtype; archives load with allow_pickle=False.
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    with artifact.new_file(name, mode="wb") as handle:
        np.savez_compressed(handle, **arrays)


def _update_array_digest(digest, payload, prefix=""):
    if isinstance(payload, dict):
        for key in sorted(payload):
            _update_array_digest(digest, payload[key], f"{prefix}/{key}")
    elif isinstance(payload, (torch.Tensor, np.ndarray)):
        value = payload.detach().cpu().numpy() if isinstance(payload, torch.Tensor) else payload
        digest.update(prefix.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())


def _add_cpu_reference(record, model, dataset, args):
    """Retain platform cancellation/overflow differences instead of hiding them."""
    vectors = record.get("vectors", record.get("grid_coordinates"))
    original = record["mse"]
    # Candidate batching itself changes FP32 GEMM cancellation. Use the exact
    # original one-model/full-dataset forward, even for CPU diagnostic arrays.
    reference_model = copy.deepcopy(model).cpu()
    flat = vectors.reshape(-1, vectors.shape[-1])
    reference = torch.empty(len(flat), dtype=torch.float32)
    with torch.no_grad():
        for index, vector in enumerate(flat):
            vector_to_parameters(vector, reference_model.parameters())
            reference[index] = (reference_model(dataset.train_x) - dataset.train_y).square().mean()
    reference = reference.reshape(original.shape)
    record["cpu_reference_mse"] = reference
    record["cpu_reference_nonfinite"] = ~torch.isfinite(reference)
    finite = torch.isfinite(original) & torch.isfinite(reference)
    relative = ((original[finite].double() - reference[finite].double()).abs()
                / reference[finite].double().abs().clamp_min(1e-12))
    record["precision_comparison"] = {
        "reference": "original CPU FP32 ProductUnitNetwork.forward, one candidate and the full training dataset",
        "finite_mask_disagreements": int((torch.isfinite(original) != torch.isfinite(reference)).sum()),
        "common_finite_count": int(finite.sum()),
        "maximum_common_finite_relative_difference": float(relative.max()) if relative.numel() else None,
        "common_finite_relative_difference_gt_1e_minus4": int((relative > 1e-4).sum()),
    }


class LandscapeRecorder:
    def __init__(self, args, model, dataset, run, artifact):
        from .punn_sampling import orthonormal_directions
        self.args, self.model, self.dataset = args, model, dataset
        self.run, self.artifact = run, artifact
        self.directions = orthonormal_directions(
            sum(p.numel() for p in model.parameters()), args.data_seed + args.landscape_seed_offset)
        self.states, self.slices, self.populations = [], [], []
        self.seen_slices = set()

    def __call__(self, epoch, kind, vector, population, losses):
        from .punn_sampling import sample_slice
        self.states.append({"epoch": epoch, "kind": kind, "parameters": vector.clone()})
        selected = kind != "epoch" or epoch == 1 or epoch % self.args.log_every == 0
        if not selected:
            return
        if population is not None:
            self.populations.append({"epoch": epoch, "parameters": population, "train_mse": losses})
        # Keep nonfinite terminal states, but do not invent a valid slice around them.
        digest = hashlib.sha256(vector.cpu().numpy().tobytes()).hexdigest()
        key = (epoch, digest)
        if key in self.seen_slices or not bool(torch.isfinite(vector).all()):
            return
        self.seen_slices.add(key)
        sampled = sample_slice(self.model, self.dataset.train_x, self.dataset.train_y, vector,
                               directions=self.directions, radius=self.args.slice_radius,
                               points=self.args.slice_points, device=self.args.landscape_device,
                               batch_size=self.args.landscape_batch_size)
        if self.args.landscape_cpu_reference == "true":
            _add_cpu_reference(sampled, self.model, self.dataset, self.args)
        self.slices.append({"epoch": epoch, "kind": kind, "center": vector.clone(), "sample": sampled})

    def finish(self):
        import wandb

        payload = {"directions": self.directions}
        for category in ("states", "slices", "populations"):
            for index, value in enumerate(getattr(self, category)):
                payload[f"{category}/{index:04d}"] = value
        _write_payload(self.artifact, "training_landscapes.npz", payload)
        table = wandb.Table(columns=["epoch", "kind", "u", "v", "train_mse", "nonfinite",
                                    "cpu_reference_mse", "cpu_reference_nonfinite"])
        for entry in self.slices:
            sample = entry["sample"]
            u, v = torch.meshgrid(sample["coordinates_1d"], sample["coordinates_1d"], indexing="ij")
            coordinates = torch.stack((u, v), dim=-1).reshape(-1, 2)
            losses = sample["mse"].reshape(-1)
            reference_losses = (sample["cpu_reference_mse"].reshape(-1)
                                if "cpu_reference_mse" in sample else [None] * len(losses))
            for point, loss, reference in zip(coordinates, losses, reference_losses):
                finite = math.isfinite(float(loss))
                reference_finite = reference is not None and math.isfinite(float(reference))
                table.add_data(entry["epoch"], entry["kind"], float(point[0]), float(point[1]),
                               float(loss) if finite else None, not finite,
                               float(reference) if reference_finite else None,
                               not reference_finite if reference is not None else None)
        self.artifact.add(table, "loss_slices")
        return {"landscape_states": len(self.states), "landscape_slices": len(self.slices),
                "landscape_population_snapshots": len(self.populations)}


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
    torch.set_num_threads(1)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.landscape_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA landscape device is unavailable")
    from .punn_sampling import sample_global
    dataset = make_dataset(args.task, args.data_seed)
    inputs, hidden, _ = TASKS[args.task]
    model = ProductUnitNetwork(inputs, hidden)
    initial = parameters_to_vector(model.parameters()).detach().clone()
    initialization_digest = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
    sampling_seed = args.data_seed + args.landscape_seed_offset
    bounds = [args.search_bound]
    if args.landscape_bounds == "training-and-paper-wide":
        wider = 3.0 if args.task == "f1" else 7.0
        if wider not in bounds:
            bounds.append(wider)
    provenance = {
        "invocation": _sanitized_invocation([sys.executable, "-m", "optimizer_resurrection.punn_recorded",
                                             *(sys.argv[1:] if argv is None else argv)], credentials),
        "git": git, "source_tree": _source_tree_digest(root),
        "environment": _environment_metadata(torch.device(args.landscape_device)),
        "determinism": _determinism_metadata(args.seed, args.data_seed),
        "source": {"url": "https://www.mdpi.com/1999-4893/17/6/241", "sections": [2, 6, 8]},
        "training": "unchanged original CPU FP32 scalar objective, batch-one seeded SGD",
        "training_device": "cpu", "landscape_device": args.landscape_device,
        "cpu_reference": args.landscape_cpu_reference == "true",
        "numerical_precision_note": "Extreme product terms may cancel differently in CPU and CUDA FP32 matrix products; both raw evaluations are retained when CPU reference is enabled.",
        "sampling_seed": sampling_seed, "global_bounds": bounds,
        "parameter_order": [{"name": n, "shape": list(p.shape)} for n, p in model.named_parameters()],
        "diagnostic_note": "Raw walks support later FLA; local slices and finite-difference summaries are not exact paper metric replications.",
    }
    import wandb
    run = wandb.init(project=credentials["WANDB_PROJECT"], entity=credentials["WANDB_ENTITY"],
                     name=args.run_name, group=args.run_group, job_type="punn-landscape-recorded",
                     config={**vars(args), "provenance": provenance}, mode="online", save_code=False)
    if run is None:
        raise RuntimeError("W&B online initialization returned no run")
    run.config.update({"provenance": {**provenance, "wandb": {
        "id": str(run.id), "url": str(run.url), "group": args.run_group,
    }}}, allow_val_change=True)
    artifact = wandb.Artifact(f"punn-landscape-{run.id}", type="loss-landscape",
                              metadata={"condition_id": args.condition_id, "dataset_digest": dataset.digest,
                                        "sampling_seed": sampling_seed, "protocol": PROTOCOL})
    started = time.monotonic()
    try:
        _write_payload(artifact, "dataset.npz", {"train_x": dataset.train_x, "train_y": dataset.train_y,
                                                "test_x": dataset.test_x, "test_y": dataset.test_y,
                                                "dataset_digest": dataset.digest})
        global_summaries = {}
        global_digest = hashlib.sha256()
        for bound in bounds:
            sampled = sample_global(model, dataset.train_x, dataset.train_y, seed=sampling_seed,
                                    bound=bound, prw_steps=args.prw_steps, mrw_steps=args.mrw_steps,
                                    uniform_per_dim=args.uniform_per_dim,
                                    dispersion_samples=args.dispersion_samples,
                                    device=args.landscape_device, batch_size=args.landscape_batch_size)
            if args.landscape_cpu_reference == "true":
                for record in sampled["samples"].values():
                    _add_cpu_reference(record, model, dataset, args)
                sampled["summary"]["precision_comparisons"] = {
                    name: record["precision_comparison"] for name, record in sampled["samples"].items()}
            _write_payload(artifact, f"global_bound_{bound:g}.npz", sampled)
            _update_array_digest(global_digest, sampled, f"bound_{bound:g}")
            global_summaries[f"bound_{bound:g}"] = sampled.get("summary", {})
        recorder = LandscapeRecorder(args, model, dataset, run, artifact)

        def log_epoch(epoch, mse):
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                run.log({"epoch": epoch, "train_mse": mse}, step=epoch)

        result = run_trial(task=args.task, method=args.method, seed=args.seed, data_seed=args.data_seed,
                           epochs=args.epochs, population_size=args.population_size,
                           search_bound=args.search_bound, boundary=args.boundary,
                           learning_rate=args.learning_rate, momentum=args.momentum,
                           on_epoch=log_epoch, on_state=recorder, condition_id=args.condition_id)
        counts = recorder.finish()
        with artifact.new_file("provenance.json", mode="w", encoding="utf-8") as handle:
            json.dump({"config": vars(args), "provenance": provenance, "result": asdict(result)},
                      handle, sort_keys=True, allow_nan=False)
        run.log_artifact(artifact)
        artifact.wait()
        # Verify the immutable manifest before marking the landscape record complete.
        required = {"dataset.npz", "training_landscapes.npz", "provenance.json",
                    *(f"global_bound_{bound:g}.npz" for bound in bounds)}
        if not required.issubset(artifact.manifest.entries):
            raise RuntimeError("uploaded landscape artifact is incomplete")
        payload = {**asdict(result), **counts, "initialization_digest": initialization_digest,
                   "order_seed": args.seed + 1_000_003, "landscape_seed": sampling_seed,
                   "landscape_samples_digest": global_digest.hexdigest(),
                   "landscape_cpu_reference": args.landscape_cpu_reference == "true",
                   "landscape_artifact": artifact.qualified_name, "landscape_status": "completed",
                   "landscape_global_summaries": global_summaries,
                   "wall_seconds": time.monotonic()-started}
        run.summary.update(payload)
        run.finish(exit_code=1 if result.numerical_failure else 0)
    except BaseException:
        run.summary.update({"terminal_outcome": "failed", "landscape_status": "failed"})
        run.finish(exit_code=1)
        raise
    print(json.dumps({"run_id": run.id, "url": run.url,
                      **{key: payload[key] for key in ("task", "method", "seed", "terminal_outcome",
                         "train_mse", "test_mse", "landscape_status", "landscape_artifact",
                         "landscape_states", "landscape_slices", "wall_seconds")}},
                     sort_keys=True, allow_nan=False), flush=True)
    return payload


if __name__ == "__main__":
    main()
