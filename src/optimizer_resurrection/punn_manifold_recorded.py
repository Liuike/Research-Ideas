"""GPU DA-10 rerun with Euclidean and manifold-respecting loss slices.

Training is delegated to the frozen architecture-stability runner. This module
records its exact parameter trajectory and evaluates full-training-data MSE on
two local two-dimensional planes: ordinary Euclidean perturbations and
tangent perturbations retracted onto the exponent Stiefel constraint.
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
from typing import Any

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from . import punn_architecture_stability as stability
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

PROTOCOL = "punn-manifold-recorded-v1"
TRAINING_PROTOCOL = stability.MANIFOLD_PROTOCOL
VIEWS = ("ambient", "manifold")
RUNTIME_FIELDS = {"condition_id", "dry_run", "run_group", "run_name", "stage"}
RECORDING_FIELDS = {
    "landscape_device",
    "landscape_seed_offset",
    "slice_points",
    "slice_radius",
    "slice_every",
    "landscape_batch_size",
    "landscape_views",
}


def _condition_values(training: argparse.Namespace, recording: argparse.Namespace) -> dict[str, Any]:
    training_values = argparse.Namespace(**vars(training))
    training_values.protocol = TRAINING_PROTOCOL
    values = stability._scientific_values(training_values)
    values.update(
        protocol=PROTOCOL,
        training_protocol=TRAINING_PROTOCOL,
        landscape_device=recording.landscape_device,
        landscape_seed_offset=recording.landscape_seed_offset,
        slice_points=recording.slice_points,
        slice_radius=recording.slice_radius,
        slice_every=recording.slice_every,
        landscape_batch_size=recording.landscape_batch_size,
        landscape_views=list(recording.landscape_views),
        slice_schedule="initialization, epoch 1, every slice_every epochs, terminal",
        landscape_objective="full training-set prediction MSE; L2 objective retained separately",
    )
    return values


def _condition_id(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=[PROTOCOL], required=True)
    parser.add_argument("--landscape-device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--landscape-seed-offset", type=int, required=True)
    parser.add_argument("--slice-points", type=int, required=True)
    parser.add_argument("--slice-radius", type=float, required=True)
    parser.add_argument("--slice-every", type=int, required=True)
    parser.add_argument("--landscape-batch-size", type=int, required=True)
    parser.add_argument("--landscape-views", nargs="+", choices=VIEWS, required=True)
    recording, training_argv = parser.parse_known_args(argv)
    if recording.landscape_seed_offset < 0:
        parser.error("landscape-seed-offset must be nonnegative")
    if recording.slice_points < 3 or recording.slice_points % 2 != 1:
        parser.error("slice-points must be odd and at least 3")
    if not math.isfinite(recording.slice_radius) or recording.slice_radius <= 0:
        parser.error("slice-radius must be finite and positive")
    if recording.slice_every < 1 or recording.landscape_batch_size < 1:
        parser.error("slice-every and landscape-batch-size must be positive")
    if not recording.landscape_views or len(set(recording.landscape_views)) != len(recording.landscape_views):
        parser.error("landscape-views must be a nonempty unique list")

    # The base parser remains the authority for every frozen training setting.
    training = stability.parse_args(["--protocol", TRAINING_PROTOCOL, *training_argv])
    if training.method != stability.MANIFOLD_METHOD:
        parser.error("recorded DA-10 runs require the plain manifold_muon_da10 recipe")
    if training.device != recording.landscape_device:
        parser.error("training and landscape devices must match in the registered run")
    if training.device != "cuda" and training.stage not in {"engineering-smoke", "test"}:
        parser.error("scientific recorded DA-10 runs require CUDA for training and landscape evaluation")
    args = argparse.Namespace(**vars(training))
    for name in RECORDING_FIELDS:
        setattr(args, name, getattr(recording, name))
    args.protocol = PROTOCOL
    args.training_protocol = TRAINING_PROTOCOL
    args.condition_id = _condition_id(_condition_values(training, recording))
    if args.run_name == f"punn-stability-{args.task}-{args.architecture}-{stability.MANIFOLD_METHOD}-seed{args.seed}":
        args.run_name = f"punn-manifold-recorded-{args.task}-{args.architecture}-seed{args.seed}"
    return args


def expand_config(config: dict[str, Any]) -> list[list[str]]:
    """Expand the registered 16-cell by 30-seed recorded DA-10 sweep."""
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"recorded manifold config protocol must be {PROTOCOL}")
    required_recording = RECORDING_FIELDS
    missing = required_recording - config.keys()
    extra = config.keys() - stability.ADAPTIVE_CONFIG_FIELDS - required_recording
    if missing or extra:
        raise ValueError(f"invalid recorded manifold config fields: missing={missing} extra={extra}")
    if config["landscape_views"] != list(VIEWS):
        raise ValueError("the registered comparison requires landscape_views [ambient, manifold]")
    if config["device"] != config["landscape_device"]:
        raise ValueError("training and landscape devices must match")
    if config["device"] != "cuda" and config["stage"] not in {"engineering-smoke", "test"}:
        raise ValueError("scientific recorded DA-10 runs require CUDA")
    if config["stage"] not in {"engineering-smoke", "test"} and len(config["seeds"]) != 30:
        raise ValueError("scientific recorded DA-10 sweep requires all 30 registered seeds")
    if config["stage"] not in {"engineering-smoke", "test"} and len(config["conditions"]) != 16:
        raise ValueError("scientific recorded DA-10 sweep requires all 16 registered cells")

    training_config = {
        key: value for key, value in config.items()
        if key not in required_recording and key != "protocol"
    }
    training_config["protocol"] = TRAINING_PROTOCOL
    base_commands = stability._expand_manifold_config(training_config)
    commands: list[list[str]] = []
    for base in base_commands:
        # Keep the frozen command arguments intact and change only the module
        # entrypoint/protocol plus the registered recorder fields.
        tail = list(base[3:])
        protocol_index = tail.index("--protocol") + 1
        tail[protocol_index] = PROTOCOL
        command = [sys.executable, "-m", "optimizer_resurrection.punn_manifold_recorded", *tail]
        for field in sorted(RECORDING_FIELDS):
            value = config[field]
            if field == "landscape_views":
                command.extend(["--landscape-views", *[str(item) for item in value]])
            else:
                command.extend(["--" + field.replace("_", "-"), str(value)])
        parsed = parse_args(command[3:])
        # Existing plan expansion validates the old recipe; this second pass
        # proves the recorded protocol and all recorder values are represented
        # in each emitted command.
        if parsed.protocol != PROTOCOL:
            raise AssertionError("expanded recorded run lost its protocol")
        commands.append(command)
    return commands


def _flat_parameter_layout(model: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
         "dtype": str(parameter.dtype)}
        for name, parameter in model.named_parameters()
    ]


def _parameter_digest(vector: torch.Tensor) -> str:
    array = vector.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _stiefel_residual_batch(exponents: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Match diagnostics.spectra.stiefel_residual for one or many matrices."""
    q = exponents.float() / scale
    if q.shape[-2] < q.shape[-1]:
        q = q.transpose(-1, -2)
    gram = q.transpose(-1, -2) @ q
    dimension = gram.shape[-1]
    identity = torch.eye(dimension, device=gram.device, dtype=gram.dtype)
    return torch.linalg.matrix_norm(gram - identity, ord="fro", dim=(-2, -1)) / math.sqrt(dimension)


def _float_or_none(value: torch.Tensor | float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _quantile_summary(values: torch.Tensor, center_value: float | None) -> dict[str, Any]:
    finite = values.detach().cpu().double()
    mask = torch.isfinite(finite)
    selected = finite[mask]
    if not selected.numel() or center_value is None or not math.isfinite(center_value):
        return {
            "finite_count": int(mask.sum()),
            "nonfinite_count": int((~mask).sum()),
            "center_value": center_value,
            "signed_delta_quantiles": None,
            "absolute_delta_quantiles": None,
        }
    delta = selected - center_value
    absolute_delta = delta.abs()
    probabilities = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0], dtype=torch.float64)
    labels = ("q0", "q25", "q50", "q75", "q90", "q95", "q100")
    return {
        "finite_count": int(mask.sum()),
        "nonfinite_count": int((~mask).sum()),
        "center_value": center_value,
        "signed_delta_quantiles": {
            label: float(value) for label, value in zip(labels, torch.quantile(delta, probabilities))
        },
        "absolute_delta_quantiles": {
            label: float(value) for label, value in zip(labels, torch.quantile(absolute_delta, probabilities))
        },
    }


def _orthonormalize_pair(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Stable two-vector Gram-Schmidt, retaining whether projection collapsed."""
    work = values.detach().cpu().double().clone()
    norms = work.norm(dim=1)
    degenerate = bool((norms <= 1e-12).any())
    if degenerate:
        return torch.full_like(values.detach().cpu(), float("nan")), True
    first = work[0] / norms[0]
    second = work[1] - torch.dot(work[1], first) * first
    second_norm = second.norm()
    if not torch.isfinite(second_norm) or second_norm <= 1e-12:
        return torch.full_like(values.detach().cpu(), float("nan")), True
    result = torch.stack((first, second / second_norm)).to(dtype=torch.float32)
    return result.contiguous(), False


def _directions(
    model: torch.nn.Module,
    seed: int,
    center: torch.Tensor,
    *,
    ambient: torch.Tensor | None = None,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Create comparable ambient directions and their Stiefel-tangent projection."""
    from .punn_sampling import orthonormal_directions

    ambient = orthonormal_directions(center.numel(), seed) if ambient is None else ambient
    exponent_count = model.exponents.numel()
    input_dim = model.exponents.shape[1]
    hidden = model.exponents.shape[0]
    exponent_center = center[:exponent_count].reshape(hidden, input_dim).double()
    projected_rows: list[torch.Tensor] = []
    preprojection_norms: list[float] = []
    tangent_norms: list[float] = []
    tangent_dimension_zero = hidden == input_dim == 1
    center_residual = None
    center_residual_nonfinite = False
    if bool(torch.isfinite(exponent_center).all()):
        raw_residual = float(_stiefel_residual_batch(exponent_center, scale))
        center_residual_nonfinite = not math.isfinite(raw_residual)
        center_residual = raw_residual if math.isfinite(raw_residual) else None
    center_feasible = center_residual is not None and center_residual <= 1e-4
    if bool(torch.isfinite(exponent_center).all()) and center_feasible:
        q = exponent_center / scale
        transposed = hidden < input_dim
        if transposed:
            q = q.T
        for direction in ambient.double():
            exponent_direction = direction[:exponent_count].reshape(hidden, input_dim)
            transformed = exponent_direction.T if transposed else exponent_direction
            sym = (q.T @ transformed + transformed.T @ q) * 0.5
            tangent = transformed - q @ sym
            if transposed:
                tangent = tangent.T
            tangent_flat = tangent.reshape(-1)
            preprojection_norms.append(float(exponent_direction.norm()))
            tangent_norms.append(float(tangent_flat.norm()))
            projected = direction.clone()
            projected[:exponent_count] = tangent_flat
            projected_rows.append(projected)
        manifold, degenerate = _orthonormalize_pair(torch.stack(projected_rows))
    else:
        preprojection_norms = [float(direction[:exponent_count].norm()) for direction in ambient]
        tangent_norms = [None, None]
        manifold = torch.full_like(ambient, float("nan"))
        degenerate = True
    return {
        "ambient": ambient,
        "manifold": manifold,
        "metadata": {
            "seed": seed,
            "ambient_source": "punn_sampling.orthonormal_directions: seeded CPU float64 QR, same parameter dimension and seed as earlier PUNN slices",
            "ambient_digest": hashlib.sha256(ambient.contiguous().numpy().tobytes()).hexdigest(),
            "manifold_source": "same ambient draws; exponent block projected with G - Q sym(Q^T G), head block unchanged, then global two-vector Gram-Schmidt",
            "parameter_order": ["exponents", "output.weight", "output.bias"],
            "exponent_tangent_dimensions": {
                "ambient_projection_norms": preprojection_norms,
                "tangent_projection_norms": tangent_norms,
                "zero_dimensional_exponent_tangent": tangent_dimension_zero,
                "direction_pair_degenerate": degenerate,
                "degeneracy_threshold": 1e-12,
                "center_stiefel_residual": center_residual,
                "center_feasibility_threshold": 1e-4,
                "center_feasible": center_feasible,
                "center_stiefel_residual_nonfinite": center_residual_nonfinite,
                "unavailable_reason": (None if center_feasible
                                       else "nonfinite or off-manifold center"),
            },
        },
    }


def _candidate_vectors(
    center: torch.Tensor,
    directions: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    view: str,
    hidden: int,
    input_dim: int,
    scale: float,
    device: torch.device,
) -> torch.Tensor:
    first, second = torch.meshgrid(coordinates, coordinates, indexing="ij")
    centers = center.to(device=device, dtype=torch.float32)
    axes = directions.to(device=device, dtype=torch.float32)
    candidates = (
        centers[None, None, :]
        + first.to(device=device)[..., None] * axes[0]
        + second.to(device=device)[..., None] * axes[1]
    ).reshape(-1, center.numel())
    if view == "ambient":
        return candidates
    if view != "manifold":
        raise ValueError(f"unknown landscape view: {view}")

    exponent_count = hidden * input_dim
    exponent_values = candidates[:, :exponent_count].reshape(-1, hidden, input_dim)
    finite_rows = torch.isfinite(exponent_values).all(dim=(1, 2))
    retracted = torch.full_like(exponent_values, float("nan"))
    if bool(finite_rows.any()):
        values = exponent_values[finite_rows]
        if hidden < input_dim:
            tall = values.transpose(1, 2) / scale
            u, _, vh = torch.linalg.svd(tall.float(), full_matrices=False)
            result = (u @ vh).transpose(1, 2).mul(scale)
        else:
            tall = values / scale
            u, _, vh = torch.linalg.svd(tall.float(), full_matrices=False)
            result = (u @ vh).mul(scale)
        # Preserve the exact center rather than allowing roundoff from an
        # otherwise identity retraction to move the shared plane origin.
        center_index = (coordinates.numel() // 2) * coordinates.numel() + coordinates.numel() // 2
        positions = finite_rows.nonzero(as_tuple=False).flatten()
        center_rows = positions == center_index
        if bool(center_rows.any()):
            result[center_rows] = exponent_values[finite_rows][center_rows]
        retracted[finite_rows] = result
    candidates[:, :exponent_count] = retracted.reshape(-1, exponent_count)
    return candidates


def evaluate_slice(
    model: torch.nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    view: str,
    points: int,
    radius: float,
    batch_size: int,
    device: str,
    regularization_lambda: float,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate raw full-data MSE plus the registered objective on one 2D plane."""
    from .punn_sampling import evaluate_vectors

    coordinates = torch.linspace(-radius, radius, points, dtype=torch.float32)
    grid_shape = (points, points)
    center_cpu = center.detach().cpu().float()
    center_finite = bool(torch.isfinite(center_cpu).all())
    center_stiefel_residual = None
    center_stiefel_residual_nonfinite = False
    if center_finite:
        raw_residual = float(
            _stiefel_residual_batch(center_cpu[:model.exponents.numel()].reshape_as(model.exponents), scale)
        )
        center_stiefel_residual_nonfinite = not math.isfinite(raw_residual)
        center_stiefel_residual = raw_residual if math.isfinite(raw_residual) else None
    center_feasible = center_stiefel_residual is not None and center_stiefel_residual <= 1e-4
    if not center_finite or (view == "manifold" and not center_feasible):
        nan = torch.full(grid_shape, float("nan"), dtype=torch.float32)
        false_mask = torch.zeros(grid_shape, dtype=torch.bool)
        invalid_status = "invalid_center_nonfinite" if not center_finite else "invalid_center_off_manifold"
        return {
            "coordinates_1d": coordinates,
            "mse": nan,
            "mse_nonfinite": torch.ones(grid_shape, dtype=torch.bool),
            "regularization": nan.clone(),
            "regularization_nonfinite": torch.ones(grid_shape, dtype=torch.bool),
            "objective": nan.clone(),
            "objective_nonfinite": torch.ones(grid_shape, dtype=torch.bool),
            "realized_displacement": nan.clone(),
            "stiefel_residual": nan.clone(),
            "finite_parameter_mask": false_mask,
            "candidate_evaluated_mask": false_mask.clone(),
            "status": invalid_status,
            "center_stiefel_residual": center_stiefel_residual,
            "center_stiefel_residual_nonfinite": center_stiefel_residual_nonfinite,
            "surface_evaluated": False,
            "sensitivity": {
                "analysis_scope": "two-dimensional plane only",
                "unavailable_surface_policy": "retain NaN loss arrays and explicit masks; do not call SVD or evaluate unless the center is finite and feasible",
                "train_mse": _quantile_summary(nan, None),
                "objective": _quantile_summary(nan, None),
            },
        }
    candidate_vectors = _candidate_vectors(
        center_cpu, directions, coordinates, view=view, hidden=model.exponents.shape[0],
        input_dim=model.exponents.shape[1], scale=scale, device=torch.device(device),
    )
    finite_parameter_mask = torch.isfinite(candidate_vectors).all(dim=1)
    mse = torch.full((candidate_vectors.shape[0],), float("nan"), dtype=torch.float32)
    finite_indices = finite_parameter_mask.nonzero(as_tuple=False).flatten()
    if finite_indices.numel():
        finite_indices_cpu = finite_indices.cpu()
        mse[finite_indices_cpu] = evaluate_vectors(
            model, train_x, train_y, candidate_vectors[finite_indices],
            device=device, batch_size=batch_size,
        )
    if regularization_lambda == 0.0:
        regularization = torch.zeros_like(mse)
    else:
        parameter_square_sum = candidate_vectors.square().sum(dim=1)
        regularization = parameter_square_sum.mul(regularization_lambda)
    regularization = regularization.cpu()
    objective = mse + regularization
    realized_displacement = torch.linalg.vector_norm(
        candidate_vectors - center.to(device=candidate_vectors.device, dtype=torch.float32), dim=1
    )
    if view == "manifold":
        exponent_count = model.exponents.numel()
        exponents = candidate_vectors[:, :exponent_count].reshape(
            -1, model.exponents.shape[0], model.exponents.shape[1]
        )
        stiefel_residual = _stiefel_residual_batch(exponents, scale)
        stiefel_residual = torch.where(
            torch.isfinite(exponents).all(dim=(1, 2)), stiefel_residual,
            torch.full_like(stiefel_residual, float("nan")),
        )
    else:
        stiefel_residual = torch.full_like(realized_displacement, float("nan"))
    mse = mse.reshape(grid_shape)
    regularization = regularization.reshape(grid_shape)
    objective = objective.reshape(grid_shape)
    finite_parameter_mask = finite_parameter_mask.reshape(grid_shape).cpu()
    candidate_evaluated_mask = finite_parameter_mask.clone()
    realized_displacement = realized_displacement.reshape(grid_shape).cpu()
    stiefel_residual = stiefel_residual.reshape(grid_shape).cpu()
    center_index = points // 2
    center_mse = _float_or_none(mse[center_index, center_index])
    center_objective = _float_or_none(objective[center_index, center_index])
    return {
        "coordinates_1d": coordinates,
        "mse": mse,
        "mse_nonfinite": ~torch.isfinite(mse),
        "regularization": regularization,
        "regularization_nonfinite": ~torch.isfinite(regularization),
        "objective": objective,
        "objective_nonfinite": ~torch.isfinite(objective),
        "finite_parameter_mask": finite_parameter_mask,
        "candidate_evaluated_mask": candidate_evaluated_mask,
        "realized_displacement": realized_displacement,
        "stiefel_residual": stiefel_residual,
        "status": "completed" if bool(torch.isfinite(mse).all()) else "completed_with_nonfinite_values",
        "center_stiefel_residual": center_stiefel_residual,
        "center_stiefel_residual_nonfinite": center_stiefel_residual_nonfinite,
        "surface_evaluated": True,
        "sensitivity": {
            "analysis_scope": "two-dimensional plane only; these quantiles do not estimate the full-dimensional landscape",
            "reference": "evaluated center of this same plane",
            "nonfinite_policy": "raw nonfinite values are retained and excluded only from finite quantile summaries",
            "train_mse": _quantile_summary(mse, center_mse),
            "objective": _quantile_summary(objective, center_objective),
        },
    }


class ManifoldLandscapeRecorder:
    """Read-only model-state observer collecting trajectories and 2D slices."""

    def __init__(self, args: argparse.Namespace, model: torch.nn.Module):
        self.args = args
        self.model = model
        self.seed = int(args.data_seed + args.landscape_seed_offset)
        self.states: list[dict[str, Any]] = []
        self.slices: list[dict[str, Any]] = []
        self.parameter_layout = _flat_parameter_layout(model)
        self.manifold_scale = float(args.manifold_scale)
        # The callback only observes the model. Diagnostics use their own
        # deterministic CPU generator and evaluate on cloned vectors.
        from .punn_sampling import orthonormal_directions

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.ambient_directions = orthonormal_directions(parameter_count, self.seed)
        self.direction_metadata: dict[str, Any] | None = None

    def __call__(self, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
        phase = str(metadata["phase"])
        epoch = int(metadata["epoch"])
        vector = parameters_to_vector(model.parameters()).detach().cpu().clone().float()
        state_meta = {key: value for key, value in metadata.items()}
        state_meta.update(
            phase=phase,
            epoch=epoch,
            state_index=len(self.states),
            parameter_digest=_parameter_digest(vector),
            finite_parameter_count=int(torch.isfinite(vector).sum()),
            nonfinite_parameter_count=int((~torch.isfinite(vector)).sum()),
        )
        self.states.append({"metadata": state_meta, "parameters": vector,
                            "finite_mask": torch.isfinite(vector)})
        directions = _directions(model, self.seed, vector, ambient=self.ambient_directions,
                                 scale=self.manifold_scale)
        if self.direction_metadata is None and phase == "initialization":
            self.direction_metadata = directions["metadata"]
        selected = (
            phase in {"initialization", "terminal"}
            or phase == "epoch" and (epoch == 1 or epoch % self.args.slice_every == 0)
        )
        if not selected:
            return
        for view in self.args.landscape_views:
            direction = directions[view]
            direction_finite = bool(torch.isfinite(direction).all())
            scalar_check: dict[str, Any] = {
                "status": "not_comparable_nonfinite_or_missing_reference",
                "passed": None,
                "reported_mse": None,
                "grid_center_mse": None,
            }
            if direction_finite:
                sample = evaluate_slice(
                    model, self._train_x, self._train_y, vector, direction,
                    view=view, points=self.args.slice_points, radius=self.args.slice_radius,
                    batch_size=self.args.landscape_batch_size, device=self.args.landscape_device,
                    regularization_lambda=self.args.regularization_lambda, scale=self.manifold_scale,
                )
                # Recompute the scalar reference at this exact callback state.
                # In particular, on a failure terminal callback the runner's
                # last finite train_mse may describe the previous completed
                # epoch, before a partially applied optimizer update.
                with torch.no_grad():
                    x = self._train_x.to(device=self.args.device)
                    y = self._train_y.to(device=self.args.device)
                    prediction = model(x)
                    reported_mse = None
                    if bool(torch.isfinite(prediction).all()):
                        reported_mse = _float_or_none((prediction - y).square().mean())
                grid_center = _float_or_none(
                    sample["mse"][self.args.slice_points // 2, self.args.slice_points // 2]
                )
                if reported_mse is None or grid_center is None:
                    scalar_check = {
                        "status": "not_comparable_nonfinite_or_missing_reference",
                        "passed": None,
                        "reported_mse": reported_mse,
                        "grid_center_mse": grid_center,
                    }
                else:
                    reported_mse = float(reported_mse)
                    difference = abs(grid_center - reported_mse)
                    scalar_check = {
                        "status": "passed" if math.isclose(grid_center, reported_mse, rel_tol=1e-4, abs_tol=1e-7) else "failed",
                        "passed": math.isclose(grid_center, reported_mse, rel_tol=1e-4, abs_tol=1e-7),
                        "reported_mse": reported_mse,
                        "grid_center_mse": grid_center,
                        "absolute_difference": difference,
                        "relative_difference": difference / max(abs(reported_mse), 1e-12),
                        "runner_reported_mse": metadata.get("train_mse"),
                        "tolerance": {"relative": 1e-4, "absolute": 1e-7},
                    }
            else:
                nan = torch.full((self.args.slice_points, self.args.slice_points), float("nan"))
                sample = {
                    "coordinates_1d": torch.linspace(-self.args.slice_radius, self.args.slice_radius,
                                                      self.args.slice_points),
                    "mse": nan, "mse_nonfinite": torch.ones_like(nan, dtype=torch.bool),
                    "regularization": nan.clone(), "regularization_nonfinite": torch.ones_like(nan, dtype=torch.bool),
                    "objective": nan.clone(), "objective_nonfinite": torch.ones_like(nan, dtype=torch.bool),
                    "realized_displacement": nan.clone(), "stiefel_residual": nan.clone(),
                    "finite_parameter_mask": torch.zeros_like(nan, dtype=torch.bool),
                    "candidate_evaluated_mask": torch.zeros_like(nan, dtype=torch.bool),
                    "status": (
                        ("invalid_center_nonfinite" if state_meta["nonfinite_parameter_count"] else
                         "invalid_center_off_manifold")
                        if view == "manifold" and not directions["metadata"]["exponent_tangent_dimensions"]["center_feasible"]
                        else "degenerate_direction_pair"
                    ),
                    "center_stiefel_residual": directions["metadata"]["exponent_tangent_dimensions"]["center_stiefel_residual"],
                    "surface_evaluated": False,
                    "scalar_center_check": {
                        "status": "not_comparable_degenerate_directions", "passed": None,
                    },
                    "sensitivity": {"analysis_scope": "two-dimensional plane only",
                                     "train_mse": _quantile_summary(nan, None),
                                     "objective": _quantile_summary(nan, None)},
                }
            self.slices.append({
                "metadata": {
                    **state_meta,
                    "direction_metadata": directions["metadata"],
                    "scalar_center_check": scalar_check,
                    "view": view,
                    "center_digest": state_meta["parameter_digest"],
                    "direction_pair_degenerate": not direction_finite,
                    "grid_order": "first direction varies along axis 0; second direction along axis 1",
                    "retraction": "none" if view == "ambient" else "polar SVD retraction of exponent block for every off-center grid point; output head unchanged",
                },
                "directions": direction.clone(),
                "center": vector.clone(),
                "sample": sample,
            })

    def bind_dataset(self, dataset: Any) -> None:
        self._train_x = dataset.train_x
        self._train_y = dataset.train_y

    def payload(self) -> dict[str, Any]:
        dimension = sum(int(item["numel"]) for item in self.parameter_layout)
        parameters = (torch.stack([state["parameters"] for state in self.states])
                      if self.states else torch.empty((0, dimension), dtype=torch.float32))
        finite_mask = (torch.stack([state["finite_mask"] for state in self.states])
                       if self.states else torch.empty((0, dimension), dtype=torch.bool))
        values: dict[str, Any] = {
            "trajectory/parameters": parameters,
            "trajectory/finite_mask": finite_mask,
            "trajectory/metadata": [state["metadata"] for state in self.states],
            "parameter_layout": self.parameter_layout,
            "direction_metadata": self.direction_metadata or {},
            "directions/ambient": self.ambient_directions,
        }
        for index, entry in enumerate(self.slices):
            prefix = f"slices/{index:04d}"
            values[f"{prefix}/metadata"] = entry["metadata"]
            values[f"{prefix}/center"] = entry["center"]
            values[f"{prefix}/directions"] = entry["directions"]
            for key, value in entry["sample"].items():
                values[f"{prefix}/{key}"] = value
        return values

    def summary(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        slice_summaries = []
        finite_counts: dict[str, int] = {}
        slice_count_by_view: dict[str, int] = {}
        for entry in self.slices:
            sample = entry["sample"]
            status = sample["status"]
            statuses[status] = statuses.get(status, 0) + 1
            view = entry["metadata"]["view"]
            slice_count_by_view[view] = slice_count_by_view.get(view, 0) + 1
            for metric in ("mse", "objective"):
                finite = int(torch.isfinite(sample[metric]).sum())
                finite_counts[f"{view}/{metric}"] = (
                    finite_counts.get(f"{view}/{metric}", 0) + finite
                )
            slice_summaries.append({
                "state_index": entry["metadata"]["state_index"],
                "phase": entry["metadata"]["phase"],
                "epoch": entry["metadata"]["epoch"],
                "view": entry["metadata"]["view"],
                "status": status,
                "sensitivity": sample["sensitivity"],
            })
        return {
            "recorded_state_count": len(self.states),
            "parameter_states_count": len(self.states),
            "recorded_slice_count": len(self.slices),
            "slice_count_by_view": slice_count_by_view,
            "slice_status_counts": statuses,
            "finite_landscape_value_counts": finite_counts,
            "slice_summaries": slice_summaries,
            "plane_sensitivity_warning": "Quantiles summarize only the sampled two-dimensional planes.",
        }

    def validate_complete(self, result: dict[str, Any]) -> None:
        """Require every model callback and scheduled center check before upload."""
        epochs = int(result["epochs_completed"])
        expected_states = epochs + 2  # initialization, each completed epoch, terminal
        if len(self.states) != expected_states:
            raise RuntimeError(
                f"parameter trajectory is incomplete: {len(self.states)} of {expected_states} states"
            )
        expected_states_meta = [("initialization", 0)] + [("epoch", epoch) for epoch in range(1, epochs + 1)] + [("terminal", epochs)]
        observed_states_meta = [(state["metadata"]["phase"], state["metadata"]["epoch"])
                                for state in self.states]
        if observed_states_meta != expected_states_meta:
            raise RuntimeError("parameter trajectory has a missing, duplicated, or out-of-order phase/epoch")
        terminal = self.states[-1]["metadata"]
        if (terminal["phase"] != "terminal"
                or terminal.get("terminal_outcome") != result["terminal_outcome"]):
            raise RuntimeError("parameter trajectory lacks the matching terminal outcome")
        selected_epochs = [epoch for epoch in range(1, epochs + 1)
                           if epoch == 1 or epoch % self.args.slice_every == 0]
        expected_slice_meta = ([('initialization', 0)]
                               + [('epoch', epoch) for epoch in selected_epochs]
                               + [('terminal', epochs)])
        counts: dict[str, int] = {}
        observed_by_view: dict[str, list[tuple[str, int]]] = {}
        for entry in self.slices:
            view = entry["metadata"]["view"]
            counts[view] = counts.get(view, 0) + 1
            observed_by_view.setdefault(view, []).append(
                (entry["metadata"]["phase"], entry["metadata"]["epoch"])
            )
            check = entry["metadata"].get("scalar_center_check", {})
            if check.get("passed") is False:
                raise RuntimeError(
                    f"{view} slice center failed its scalar MSE check at "
                    f"{entry['metadata']['phase']} epoch {entry['metadata']['epoch']}"
                )
            sample = entry["sample"]
            if view == "manifold" and sample.get("surface_evaluated"):
                candidate_mask = sample["candidate_evaluated_mask"]
                residuals = sample["stiefel_residual"][candidate_mask]
                if residuals.numel() and (
                    not bool(torch.isfinite(residuals).all()) or float(residuals.max()) > 1e-4
                ):
                    raise RuntimeError("feasible-view grid contains an invalid Stiefel residual")
        if any(counts.get(view, 0) != len(expected_slice_meta)
               for view in self.args.landscape_views):
            raise RuntimeError(
                f"landscape slice schedule is incomplete: {counts}; "
                f"expected {len(expected_slice_meta)} per view"
            )
        if any(observed_by_view.get(view) != expected_slice_meta for view in self.args.landscape_views):
            raise RuntimeError("landscape slice phases are missing, duplicated, or out of order")


def _write_payload(artifact: Any, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            arrays[prefix] = value.detach().cpu().contiguous().numpy()
        elif isinstance(value, np.ndarray):
            arrays[prefix] = value
        elif isinstance(value, dict):
            for key in sorted(value):
                flatten(f"{prefix}/{key}" if prefix else str(key), value[key])
        else:
            metadata[prefix] = value

    flatten("", payload)
    with artifact.new_file(name, mode="wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        )
    return {
        "file": name,
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for key, value in sorted(arrays.items())},
    }


def _dataset_artifact(run: Any, args: argparse.Namespace, dataset: Any,
                      metadata: dict[str, Any], digest: str) -> str:
    import wandb

    artifact = wandb.Artifact(
        name=f"punn-manifold-recorded-data-{args.task}-{args.data_seed}-{digest[:12]}",
        type="dataset",
        description="Exact PUNN train/test tensors and transformation metadata for recorded DA-10 landscapes",
        metadata={"protocol": PROTOCOL, "task": args.task, "data_seed": args.data_seed,
                  "dataset_digest": digest},
    )
    with artifact.new_file("dataset.npz", mode="wb") as handle:
        np.savez_compressed(
            handle,
            train_x=dataset.train_x.detach().cpu().contiguous().numpy(),
            train_y=dataset.train_y.detach().cpu().contiguous().numpy(),
            test_x=dataset.test_x.detach().cpu().contiguous().numpy(),
            test_y=dataset.test_y.detach().cpu().contiguous().numpy(),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, default=str, allow_nan=False)),
            dataset_digest=np.asarray(digest),
        )
    logged = run.log_artifact(artifact)
    logged.wait()
    return str(logged.name)


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), sort_keys=True, default=str))
        return None

    root = Path(__file__).resolve().parents[2]
    git = _git_metadata(root)
    _require_reproducible_source(args.stage, git)
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA device is unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(args.seed)
    invocation = [
        sys.executable,
        "-m",
        "optimizer_resurrection.punn_manifold_recorded",
        *(sys.argv[1:] if argv is None else argv),
    ]
    provenance = {
        "protocol": PROTOCOL,
        "training_protocol": TRAINING_PROTOCOL,
        "git": git,
        "source_tree": _source_tree_digest(root),
        "invocation": _sanitized_invocation(invocation, credentials),
        "environment": _environment_metadata(torch.device(args.device)),
        "determinism": _determinism_metadata(args.seed, args.data_seed),
        "precision": f"{args.device.upper()} FP32 scalar batch-one DA-10 training; no AMP, clipping, or TF32",
        "training_device": args.device,
        "landscape_device": args.landscape_device,
        "batch_size": 1,
        "optimizer": "ManifoldMuon DA-10 plus AdamW auxiliary",
        "optimizer_assignment": "DA-10 on projected exponents; AdamW on output weight and bias",
        "optimizer_recipe": {key: getattr(args, key) for key in stability.MANIFOLD_RECIPE},
        "training_recipe_source": "configs/product_unit/manifold_architecture_stability.yaml",
        "recording": {
            "views": list(args.landscape_views),
            "slice_points": args.slice_points,
            "slice_radius": args.slice_radius,
            "slice_every": args.slice_every,
            "slice_schedule": "initialization, epoch 1, every slice_every epochs, terminal",
            "landscape_seed": args.data_seed + args.landscape_seed_offset,
            "landscape_batch_size": args.landscape_batch_size,
            "parameter_trajectory": "initialization, every completed epoch, terminal (including failures)",
            "objective": "full training-set unregularized prediction MSE; regularization and training objective retained separately",
        },
        "condition_id": args.condition_id,
        "condition_values": _condition_values(args, args),
    }
    import wandb

    run = wandb.init(
        project=credentials["WANDB_PROJECT"],
        entity=credentials["WANDB_ENTITY"],
        group=args.run_group,
        name=args.run_name,
        job_type="punn-manifold-recorded-landscape",
        mode="online",
        config={**vars(args), "provenance": provenance},
        save_code=False,
    )
    if run is None:
        raise RuntimeError("W&B online initialization returned no run")
    run.config.update({"wandb_run_id": str(run.id), "wandb_run_url": str(run.url)})
    artifact = wandb.Artifact(
        name=f"punn-manifold-landscape-{run.id}",
        type="loss-landscape",
        description="DA-10 training trajectory and ambient/manifold-retracted two-dimensional loss slices",
        metadata={"protocol": PROTOCOL, "condition_id": args.condition_id,
                  "task": args.task, "architecture": args.architecture, "seed": args.seed},
    )
    recorder: ManifoldLandscapeRecorder | None = None
    started = time.monotonic()
    try:
        dataset, data_metadata = make_architecture_dataset(args.task, args.data_seed)
        stability._validate_dataset(dataset, args)
        digest = stability._dataset_digest(dataset)
        if getattr(dataset, "digest", digest) != digest:
            raise ValueError("dataset digest does not match stored train/test tensors")
        data_metadata = {**data_metadata, "task": args.task, "data_seed": args.data_seed,
                         "dataset_digest": digest, "tensor_order": ["train_x", "train_y", "test_x", "test_y"],
                         "tensor_dtype": "float32", "tensor_shapes": {
                             key: list(getattr(dataset, key).shape)
                             for key in ("train_x", "train_y", "test_x", "test_y")}}
        dataset_artifact = _dataset_artifact(run, args, dataset, data_metadata, digest)
        run.config.update({"data_metadata": data_metadata,
                           "dataset_artifact": dataset_artifact}, allow_val_change=True)
        run.summary.update({"dataset_artifact": dataset_artifact, "dataset_digest": digest})

        # The training runner moves a copied Dataset to the requested device.
        # For observation, the original exact CPU tensors are used by the
        # evaluator, which moves them onto the same registered device.
        from .models.product_unit import ProductUnitNetwork

        input_dim, _, _, output_dim = TASK_SPECS[args.task]
        hidden_units = (TASK_SPECS[args.task][1] if args.architecture == "small"
                        else TASK_SPECS[args.task][2])
        seed_everything(args.seed)
        model = ProductUnitNetwork(input_dim, hidden_units, output_dim,
                                   input_domain="real_complex", init_bound=1.0).cpu()
        if args.device == "cuda":
            model = model.cuda()
        recorder = ManifoldLandscapeRecorder(args, model)
        recorder.bind_dataset(dataset)

        # The model instance used by the callback belongs to run_trial. The
        # recorder updates its reference on the first callback before use.
        def observe(model_state: torch.nn.Module, metadata: dict[str, Any]) -> None:
            recorder.model = model_state
            recorder(model_state, metadata)

        def log_epoch(values: dict[str, Any]) -> None:
            stability._log_training_epoch(run, values)

        training_args = argparse.Namespace(**vars(args))
        training_args.protocol = TRAINING_PROTOCOL
        result = stability.run_trial(
            training_args,
            on_epoch=log_epoch,
            dataset_pair=(dataset, data_metadata),
            on_model_state=observe,
        )
        if result["dataset_digest"] != digest:
            raise RuntimeError("recording and training datasets differ")
        initial_state = recorder.states[0]
        if result["initialization_digest"] != initial_state["metadata"]["parameter_digest"]:
            raise RuntimeError("recorded and training initial parameter states differ")
        result.update({"protocol": PROTOCOL, "condition_id": args.condition_id,
                       "recorded_dataset_artifact": dataset_artifact})
        recording_summary = recorder.summary()
        result.update(recording_summary)
        result["landscape_status"] = "completed" if result["terminal_outcome"] == "completed" else "completed_with_numerical_failure"
        recorder.validate_complete(result)

        dataset_payload_meta = {
            "task": args.task,
            "data_seed": args.data_seed,
            "dataset_digest": digest,
            "dataset_artifact": dataset_artifact,
            "metadata": data_metadata,
            "tensor_order": ["train_x", "train_y", "test_x", "test_y"],
        }
        dataset_index = _write_payload(artifact, "dataset_metadata.npz", dataset_payload_meta)
        trajectory_index = _write_payload(artifact, "recorded_landscapes.npz", recorder.payload())
        with artifact.new_file("provenance.json", mode="w", encoding="utf-8") as handle:
            json.dump({"config": vars(args), "provenance": provenance,
                       "dataset": dataset_payload_meta, "terminal_result": result},
                      handle, sort_keys=True, allow_nan=False, default=str)
        manifest = {
            "schema": "punn_manifold_recorded_v1",
            "protocol": PROTOCOL,
            "condition_id": args.condition_id,
            "dataset_digest": digest,
            "dataset_artifact": dataset_artifact,
            "training_device": args.device,
            "landscape_device": args.landscape_device,
            "trajectory": {
                "states": recording_summary["recorded_state_count"],
                "state_schedule": "initialization, each completed epoch, terminal",
                "parameter_layout": recorder.parameter_layout,
                "array_file": "recorded_landscapes.npz",
            },
            "slices": {
                "count": recording_summary["recorded_slice_count"],
                "count_by_view": recording_summary["slice_count_by_view"],
                "views": list(args.landscape_views),
                "schedule": "initialization, epoch 1, every slice_every epochs, terminal",
                "grid": [args.slice_points, args.slice_points],
                "radius": args.slice_radius,
                "slice_every": args.slice_every,
                "array_file": "recorded_landscapes.npz",
                "finite_masks": "stored separately for MSE, objective, regularization, and candidate parameter vectors",
                "sensitivity_scope": "sampled two-dimensional planes only",
                "summary": recording_summary["slice_summaries"],
            },
            "files": ["dataset_metadata.npz", "recorded_landscapes.npz", "provenance.json"],
            "array_indexes": [dataset_index, trajectory_index],
            "terminal_result": result,
        }
        with artifact.new_file("manifest.json", mode="w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, allow_nan=False, default=str)
        logged_artifact = run.log_artifact(artifact)
        logged_artifact.wait()
        if not {"dataset_metadata.npz", "recorded_landscapes.npz", "provenance.json", "manifest.json"}.issubset(logged_artifact.manifest.entries):
            raise RuntimeError("uploaded recorded landscape artifact is incomplete")
        result.update({"landscape_artifact": logged_artifact.qualified_name,
                       "wall_seconds": time.monotonic() - started})
        result["landscape_recording_complete"] = True
        stability._record_terminal_result(run, result)
        run.summary.update({
            "landscape_artifact": logged_artifact.qualified_name,
            "landscape_status": result["landscape_status"],
            "condition_id": args.condition_id,
            "recorded_state_count": recording_summary["recorded_state_count"],
            "recorded_slice_count": recording_summary["recorded_slice_count"],
            "parameter_states_count": recording_summary["parameter_states_count"],
            "slice_count_by_view": recording_summary["slice_count_by_view"],
            "landscape_recording_complete": True,
            "slice_status_counts": recording_summary["slice_status_counts"],
            "wall_seconds": result["wall_seconds"],
        })
        # A numerical optimizer failure is an accepted finished scientific
        # observation. Infrastructure exceptions take the separate failure
        # path below and exit nonzero.
        run.finish(exit_code=0)
        print(json.dumps({"run_id": run.id, "url": run.url, **result},
                         sort_keys=True, allow_nan=False, default=str), flush=True)
        return result
    except BaseException as error:
        failure = {
            "protocol": PROTOCOL,
            "condition_id": args.condition_id,
            "terminal_outcome": "infrastructure_failure",
            "landscape_status": "partial_recording" if recorder is not None else "failed_before_recording",
            "recording_failure_type": type(error).__name__,
            "recording_failure_message": str(error),
            "landscape_recording_complete": False,
        }
        try:
            if recorder is not None:
                partial_summary = recorder.summary()
                partial_index = _write_payload(artifact, "recorded_landscapes.npz", recorder.payload())
                manifest = {
                    "schema": "punn_manifold_recorded_v1",
                    "protocol": PROTOCOL,
                    "condition_id": args.condition_id,
                    "terminal_outcome": "infrastructure_failure",
                    "recording_failure_type": type(error).__name__,
                    "recorded_state_count": len(recorder.states),
                    "recorded_slice_count": len(recorder.slices),
                    "parameter_states_count": len(recorder.states),
                    "slice_count_by_view": partial_summary["slice_count_by_view"],
                    "array_index": partial_index,
                }
                with artifact.new_file("manifest.json", mode="w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, sort_keys=True, allow_nan=False, default=str)
                run.log_artifact(artifact).wait()
                failure["landscape_artifact"] = artifact.qualified_name
                failure["parameter_states_count"] = len(recorder.states)
                failure["slice_count_by_view"] = partial_summary["slice_count_by_view"]
            run.summary.update(failure)
        except BaseException:
            run.summary.update(failure)
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
