"""Read-only validation and summaries for recorded DA-10 landscape artifacts.

This module consumes W&B-downloaded payloads but never contacts W&B, launches
training, writes project outputs, or changes the supplied arrays. Its primary
unit of comparison is one run's two planes around the same parameter center.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


_DATASET_TENSORS = ("train_x", "train_y", "test_x", "test_y")
_VIEWS = ("ambient", "manifold")
_CENTER_RTOL = 1e-3
_CENTER_ATOL = 2e-7
_IDENTITY_RTOL = 2e-5
_IDENTITY_ATOL = 5e-7


def _fail(message: str) -> None:
    raise ValueError(message)


def _unpack_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _payload_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("metadata_json")
    if raw is None:
        _fail("recorded_landscapes.npz is missing metadata_json")
    try:
        metadata = json.loads(str(_unpack_scalar(raw)))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("recorded_landscapes.npz has invalid metadata_json") from error
    if not isinstance(metadata, dict):
        _fail("recorded_landscapes metadata_json must contain a mapping")
    return metadata


def _metadata_subtree(metadata: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Rebuild one nested dict from the recorder's flattened slash keys."""
    direct = metadata.get(prefix)
    if isinstance(direct, Mapping):
        return dict(direct)
    subtree: dict[str, Any] = {}
    marker = prefix + "/"
    for key, value in metadata.items():
        if not key.startswith(marker):
            continue
        parts = key[len(marker):].split("/")
        current = subtree
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                _fail(f"metadata keys collide at {prefix}/{part}")
            current = child
        current[parts[-1]] = value
    return subtree


def _array(payload: Mapping[str, Any], key: str, *, dtype: Any | None = None) -> np.ndarray:
    if key not in payload:
        _fail(f"recorded_landscapes.npz is missing array {key!r}")
    value = np.asarray(payload[key])
    if dtype is not None and value.dtype != np.dtype(dtype):
        _fail(f"array {key!r} has dtype {value.dtype}, expected {np.dtype(dtype)}")
    return value


def _digest_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _dataset_digest(dataset: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in _DATASET_TENSORS:
        if name not in dataset:
            _fail(f"dataset artifact is missing {name}")
        value = np.ascontiguousarray(np.asarray(dataset[name]))
        if value.dtype != np.float32:
            _fail(f"dataset tensor {name} must be float32, got {value.dtype}")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _optional_embedded_json(mapping: Mapping[str, Any]) -> dict[str, Any]:
    raw = mapping.get("metadata_json")
    if raw is None:
        return {}
    try:
        value = json.loads(str(_unpack_scalar(raw)))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("dataset artifact has invalid metadata_json") from error
    if not isinstance(value, dict):
        _fail("dataset metadata_json must contain a mapping")
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _equal_array(left: np.ndarray, right: np.ndarray, *, atol: float = 0.0,
                 rtol: float = 0.0, equal_nan: bool = False) -> bool:
    if left.shape != right.shape:
        return False
    if atol == 0.0 and rtol == 0.0:
        if equal_nan and (np.issubdtype(left.dtype, np.floating)
                          or np.issubdtype(right.dtype, np.floating)):
            return bool(np.array_equal(left, right, equal_nan=True))
        return bool(np.array_equal(left, right))
    return bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=equal_nan))


def _check_close(actual: float, expected: float, *, what: str,
                 rtol: float = _CENTER_RTOL, atol: float = _CENTER_ATOL) -> tuple[float, float]:
    absolute = abs(actual - expected)
    relative = absolute / max(abs(expected), 1e-12)
    if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
        _fail(f"{what} mismatch: stored={actual:.9g} recomputed={expected:.9g}")
    return absolute, relative


def _make_model(config: Mapping[str, Any], layout: Sequence[Mapping[str, Any]]) -> torch.nn.Module:
    from .models.product_unit import ProductUnitNetwork

    by_name = {str(item["name"]): item for item in layout}
    if set(by_name) != {"exponents", "output.weight", "output.bias"}:
        _fail("parameter_layout must contain exponents, output.weight, and output.bias")
    exponent_shape = tuple(int(value) for value in by_name["exponents"]["shape"])
    weight_shape = tuple(int(value) for value in by_name["output.weight"]["shape"])
    bias_shape = tuple(int(value) for value in by_name["output.bias"]["shape"])
    if len(exponent_shape) != 2 or len(weight_shape) != 2 or len(bias_shape) != 1:
        _fail("parameter_layout contains invalid PUNN parameter dimensions")
    hidden, input_dim = exponent_shape
    output_dim = weight_shape[0]
    if weight_shape[1] != hidden or bias_shape != (output_dim,):
        _fail("parameter_layout output-head dimensions are inconsistent")
    for key, actual in (("input_dim", input_dim), ("hidden_units", hidden),
                        ("output_dim", output_dim)):
        configured = config.get(key)
        if configured is not None and int(configured) != actual:
            _fail(f"config {key}={configured} disagrees with parameter_layout ({actual})")
    model = ProductUnitNetwork(input_dim, hidden, output_dim,
                               input_domain="real_complex", init_bound=1.0).cpu()
    expected_layout = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
    ]
    for expected, observed in zip(expected_layout, layout):
        for key in ("name", "shape", "numel"):
            if key in observed and observed[key] != expected[key]:
                _fail(f"parameter_layout {key} mismatch for {expected['name']}")
    return model


def _set_model_vector(model: torch.nn.Module, vector: np.ndarray,
                      layout: Sequence[Mapping[str, Any]]) -> None:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    offset = 0
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for item in layout:
            name = str(item["name"])
            shape = tuple(int(value) for value in item["shape"])
            count = int(item.get("numel", math.prod(shape)))
            if count != math.prod(shape) or name not in parameters:
                _fail(f"invalid parameter layout entry {name}")
            parameter = parameters[name]
            if tuple(parameter.shape) != shape:
                _fail(f"model parameter shape mismatch for {name}")
            if offset + count > flat.size:
                _fail("parameter vector is shorter than parameter_layout")
            parameter.copy_(torch.from_numpy(flat[offset:offset + count].reshape(shape)))
            offset += count
    if offset != flat.size:
        _fail("parameter vector is longer than parameter_layout")


def _mse(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> float | None:
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        _fail("dataset tensors contain nonfinite values")
    with torch.no_grad():
        prediction = model(torch.from_numpy(np.ascontiguousarray(x)))
        value = (prediction - torch.from_numpy(np.ascontiguousarray(y))).square().mean()
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_row(values: np.ndarray, center: float | None) -> dict[str, float | None]:
    finite = np.asarray(values)[np.isfinite(values)].astype(np.float64, copy=False)
    if not finite.size or center is None:
        return {"signed_p95_mse_delta": None, "absolute_p95_mse_delta": None}
    delta = finite - center
    return {
        "signed_p95_mse_delta": float(np.percentile(delta, 95)),
        "absolute_p95_mse_delta": float(np.percentile(np.abs(delta), 95)),
    }


def _slice_key(index: int) -> str:
    return f"slices/{index:04d}"


def audit_payload(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    terminal_result: Mapping[str, Any],
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    check_terminal_corners: bool = False,
) -> dict[str, Any]:
    """Validate a downloaded DA-10 artifact and return compact JSON-ready rows.

    `payload` is the key/value view of ``recorded_landscapes.npz`` and
    `dataset` is the corresponding ``dataset.npz``. Nonfinite outcomes remain
    explicit in the returned per-slice rows; broken invariants raise ValueError.
    CUDA-produced scalars are checked against the independent CPU FP32 model
    with relative tolerance 1e-3 and absolute tolerance 2e-7.
    """
    if manifest.get("schema") != "punn_manifold_recorded_v1":
        _fail("unexpected recorded landscape manifest schema")
    if manifest.get("protocol") != "punn-manifold-recorded-v1":
        _fail("unexpected recorded landscape protocol")
    metadata = _payload_metadata(payload)
    trajectory_meta = metadata.get("trajectory/metadata")
    layout = metadata.get("parameter_layout")
    if not isinstance(trajectory_meta, list) or not isinstance(layout, list):
        _fail("recorded payload lacks trajectory metadata or parameter layout")
    parameters = _array(payload, "trajectory/parameters", dtype=np.float32)
    finite_mask = _array(payload, "trajectory/finite_mask", dtype=np.bool_)
    if parameters.ndim != 2 or finite_mask.shape != parameters.shape:
        _fail("trajectory parameters and finite mask must be equally shaped matrices")
    if len(trajectory_meta) != parameters.shape[0]:
        _fail("trajectory metadata count differs from parameter state count")
    if not _equal_array(finite_mask, np.isfinite(parameters)):
        _fail("trajectory finite mask does not match parameter values")

    task = str(config.get("task", terminal_result.get("task", "")))
    architecture = str(config.get("architecture", terminal_result.get("architecture", "")))
    method = str(config.get("method", terminal_result.get("method", "")))
    seed = int(config.get("seed", terminal_result.get("seed", -1)))
    data_seed = int(config.get("data_seed", terminal_result.get("data_seed", -1)))
    for key, resolved in (("task", task), ("architecture", architecture), ("method", method),
                          ("seed", seed), ("data_seed", data_seed)):
        result_value = terminal_result.get(key)
        if result_value is not None and str(result_value) != str(resolved):
            _fail(f"terminal result {key} disagrees with resolved config")
    condition_id = config.get("condition_id")
    if condition_id and manifest.get("condition_id") != condition_id:
        _fail("manifest condition_id differs from resolved config")
    if terminal_result.get("condition_id") and condition_id:
        if terminal_result["condition_id"] != condition_id:
            _fail("terminal result condition_id differs from resolved config")
    training_device = str(config.get("device", manifest.get("training_device", "unknown")))
    landscape_device = str(config.get("landscape_device", manifest.get("landscape_device", "unknown")))
    if manifest.get("training_device") != training_device:
        _fail("manifest training device differs from resolved config")
    if manifest.get("landscape_device") != landscape_device:
        _fail("manifest landscape device differs from resolved config")
    if training_device not in {"cpu", "cuda"} or landscape_device != training_device:
        _fail("recorded DA-10 run has an unsupported device/fallback combination")
    epochs = int(terminal_result.get("epochs_completed", config.get("epochs", -1)))
    terminal_outcome = str(terminal_result.get("terminal_outcome", ""))
    if terminal_outcome not in {"completed", "numerical_failure"}:
        _fail(f"unsupported scientific terminal outcome {terminal_outcome!r}")
    if terminal_outcome == "completed" and epochs != int(config.get("epochs", epochs)):
        _fail("completed terminal epoch count differs from the registered budget")
    expected_state_count = epochs + 2
    if parameters.shape != (expected_state_count, sum(int(item["numel"]) for item in layout)):
        _fail("trajectory shape is inconsistent with completed epochs or parameter layout")
    expected_state_meta = ([ ("initialization", 0) ]
                           + [("epoch", epoch) for epoch in range(1, epochs + 1)]
                           + [("terminal", epochs)])
    for index, (state, (phase, epoch)) in enumerate(zip(trajectory_meta, expected_state_meta)):
        if not isinstance(state, Mapping):
            _fail(f"trajectory metadata entry {index} is not a mapping")
        if state.get("phase") != phase or int(state.get("epoch", -1)) != epoch:
            _fail(f"trajectory state {index} is out of order or has the wrong phase")
        if int(state.get("state_index", -1)) != index:
            _fail(f"trajectory state index mismatch at {index}")
        if ((phase == "initialization" and index != 0)
                or (phase == "epoch" and index != epoch)
                or (phase == "terminal" and index != epochs + 1)):
            _fail(f"trajectory phase {phase} epoch {epoch} is stored at the wrong state index")
        row = parameters[index]
        digest = _digest_array(row)
        if state.get("parameter_digest") != digest:
            _fail(f"trajectory parameter digest mismatch at state {index}")
        if int(state.get("finite_parameter_count", -1)) != int(np.isfinite(row).sum()):
            _fail(f"trajectory finite parameter count mismatch at state {index}")
        if int(state.get("nonfinite_parameter_count", -1)) != int((~np.isfinite(row)).sum()):
            _fail(f"trajectory nonfinite parameter count mismatch at state {index}")

    if manifest.get("trajectory", {}).get("states") != expected_state_count:
        _fail("manifest trajectory state count does not match payload")
    if terminal_result.get("recorded_state_count", expected_state_count) != expected_state_count:
        _fail("terminal result state count does not match payload")
    manifest_layout = manifest.get("trajectory", {}).get("parameter_layout")
    if manifest_layout != layout:
        _fail("manifest parameter layout does not match payload")
    views = list(config.get("landscape_views", ()))
    if views != list(_VIEWS):
        _fail("registered landscape views must be [ambient, manifold]")
    slices_manifest = manifest.get("slices", {})
    if slices_manifest.get("views") != views:
        _fail("manifest landscape views differ from resolved config")
    points = int(config.get("slice_points", 0))
    radius = float(config.get("slice_radius", 0.0))
    slice_every = int(config.get("slice_every", 0))
    if points < 3 or points % 2 != 1 or radius <= 0 or slice_every < 1:
        _fail("invalid slice grid in resolved config")
    if slices_manifest.get("grid") != [points, points] or not math.isclose(float(slices_manifest.get("radius", -1)), radius):
        _fail("manifest grid dimensions/radius differ from config")
    expected_slice_schedule = ([ ("initialization", 0) ]
                               + [("epoch", epoch) for epoch in range(1, epochs + 1)
                                  if epoch == 1 or epoch % slice_every == 0]
                               + [("terminal", epochs)])
    expected_slice_count = len(expected_slice_schedule) * len(views)
    if slices_manifest.get("count") != expected_slice_count:
        _fail("manifest slice count does not match the expected schedule")
    if slices_manifest.get("count_by_view") != {view: len(expected_slice_schedule) for view in views}:
        _fail("manifest per-view slice counts are incomplete")
    if terminal_result.get("recorded_slice_count", expected_slice_count) != expected_slice_count:
        _fail("terminal result slice count does not match payload")
    if terminal_result.get("slice_count_by_view", {view: len(expected_slice_schedule) for view in views}) != {
            view: len(expected_slice_schedule) for view in views}:
        _fail("terminal result per-view slice counts are incomplete")
    manifest_slice_summary = slices_manifest.get("summary")
    if isinstance(manifest_slice_summary, list) and len(manifest_slice_summary) != expected_slice_count:
        _fail("manifest slice summary count does not match payload")

    computed_dataset_digest = _dataset_digest(dataset)
    manifest_digest = manifest.get("dataset_digest")
    result_digest = terminal_result.get("dataset_digest")
    dataset_digest_value = _unpack_scalar(dataset.get("dataset_digest", ""))
    dataset_metadata = _optional_embedded_json(dataset)
    if not isinstance(manifest_digest, str) or computed_dataset_digest != manifest_digest:
        _fail("downloaded dataset tensors do not match the manifest dataset digest")
    if result_digest and result_digest != manifest_digest:
        _fail("terminal result dataset digest differs from manifest")
    if dataset_digest_value and str(dataset_digest_value) != manifest_digest:
        _fail("dataset artifact's embedded digest differs from manifest")
    for key, expected_value in (("task", task), ("data_seed", data_seed),
                                ("dataset_digest", manifest_digest)):
        observed = dataset_metadata.get(key)
        if observed is not None and observed != expected_value:
            _fail(f"dataset artifact metadata {key} differs from run provenance")
    if int(config.get("data_seed", data_seed)) != data_seed:
        _fail("resolved config data_seed mismatch")
    if manifest.get("dataset_artifact") and terminal_result.get("recorded_dataset_artifact"):
        if manifest["dataset_artifact"].split(":")[0] != terminal_result["recorded_dataset_artifact"].split(":")[0]:
            _fail("manifest and terminal result refer to different dataset artifacts")

    train_x, train_y, test_x, test_y = (np.asarray(dataset[name]) for name in _DATASET_TENSORS)
    if train_x.ndim != 2 or train_y.ndim != 2 or test_x.ndim != 2 or test_y.ndim != 2:
        _fail("dataset tensors must be rank two")
    if train_x.shape[0] != train_y.shape[0] or test_x.shape[0] != test_y.shape[0]:
        _fail("dataset input/target row counts differ")
    model = _make_model(config, layout)
    if train_x.shape[1] != model.input_dim or test_x.shape[1] != model.input_dim:
        _fail("dataset input dimensions disagree with the model")
    if train_y.shape[1] != model.output.out_features or test_y.shape[1] != model.output.out_features:
        _fail("dataset target dimensions disagree with the model")

    # The ambient RNG is shared across both views and every recorded center.
    from .punn_manifold_recorded import _candidate_vectors, _directions
    from .punn_sampling import orthonormal_directions

    direction_seed = data_seed + int(config.get("landscape_seed_offset", 0))
    global_ambient = _array(payload, "directions/ambient", dtype=np.float32)
    parameter_dim = parameters.shape[1]
    if global_ambient.shape != (2, parameter_dim) or not np.isfinite(global_ambient).all():
        _fail("global ambient directions have invalid shape or nonfinite values")
    expected_ambient = orthonormal_directions(parameter_dim, direction_seed).numpy()
    if not _equal_array(global_ambient, expected_ambient, atol=1e-6, rtol=1e-5):
        _fail("global ambient directions do not match the registered seed")
    if not np.allclose(global_ambient @ global_ambient.T, np.eye(2), atol=1e-5, rtol=1e-5):
        _fail("global ambient direction vectors are not orthonormal")

    slice_rows: list[dict[str, Any]] = []
    paired_centers: dict[tuple[int, str, int], np.ndarray] = {}
    expected_order = [(phase, epoch, view)
                      for phase, epoch in expected_slice_schedule for view in views]
    for index, (expected_phase, expected_epoch, expected_view) in enumerate(expected_order):
        prefix = _slice_key(index)
        slice_payload_meta = _metadata_subtree(metadata, prefix)
        slice_meta = slice_payload_meta.get("metadata")
        if not isinstance(slice_meta, Mapping):
            _fail(f"slice {index} lacks metadata")
        sample_meta = {key: value for key, value in slice_payload_meta.items() if key != "metadata"}
        phase = str(slice_meta.get("phase", ""))
        epoch = int(slice_meta.get("epoch", -1))
        view = str(slice_meta.get("view", ""))
        state_index = int(slice_meta.get("state_index", -1))
        if (phase, epoch, view) != (expected_phase, expected_epoch, expected_view):
            _fail(f"slice {index} has an unexpected view or schedule position")
        if isinstance(manifest_slice_summary, list):
            listed = manifest_slice_summary[index]
            if (listed.get("phase"), int(listed.get("epoch", -1)), listed.get("view")) != (
                    phase, epoch, view):
                _fail(f"manifest slice summary differs from payload at slice {index}")
        if state_index < 0 or state_index >= expected_state_count:
            _fail(f"slice {index} references an invalid trajectory state")
        expected_state_index = (0 if phase == "initialization" else
                                epochs + 1 if phase == "terminal" else epoch)
        if state_index != expected_state_index:
            _fail(f"slice {index} phase/epoch maps to trajectory state {state_index}, "
                  f"expected {expected_state_index}")
        center = _array(payload, f"{prefix}/center", dtype=np.float32)
        directions = _array(payload, f"{prefix}/directions", dtype=np.float32)
        if center.shape != (parameter_dim,) or directions.shape != (2, parameter_dim):
            _fail(f"slice {index} center or directions have invalid dimensions")
        state_vector = parameters[state_index]
        if not _equal_array(center, state_vector, equal_nan=True):
            _fail(f"slice {index} center differs from its recorded trajectory state")
        center_digest = _digest_array(center)
        if slice_meta.get("center_digest") != center_digest:
            _fail(f"slice {index} center digest mismatch")
        if not _equal_array(global_ambient, directions, atol=1e-6, rtol=1e-5) and view == "ambient":
            _fail(f"ambient directions changed at slice {index}")
        paired_centers[(state_index, phase, epoch)] = center
        expected_direction_pair = _directions(
            model, direction_seed, torch.from_numpy(center.copy()),
            ambient=torch.from_numpy(global_ambient.copy()),
            scale=float(config.get("manifold_scale", 1.0)),
        )[view].numpy()
        if not _equal_array(directions, expected_direction_pair, atol=1e-5, rtol=1e-5,
                            equal_nan=True):
            _fail(f"slice {index} directions do not match the center-projected registered pair")
        if not bool(slice_meta.get("direction_pair_degenerate", False)):
            if not np.isfinite(directions).all() or not np.allclose(
                    directions @ directions.T, np.eye(2), atol=1e-5, rtol=1e-5):
                _fail(f"slice {index} direction pair is not finite and orthonormal")

        arrays = {
            key: _array(payload, f"{prefix}/{key}")
            for key in ("coordinates_1d", "mse", "regularization", "objective",
                        "realized_displacement", "stiefel_residual")
        }
        for key in ("mse_nonfinite", "regularization_nonfinite", "objective_nonfinite",
                    "finite_parameter_mask", "candidate_evaluated_mask"):
            arrays[key] = _array(payload, f"{prefix}/{key}", dtype=np.bool_)
        coordinates = arrays["coordinates_1d"]
        if coordinates.shape != (points,) or not np.allclose(
                coordinates, np.linspace(-radius, radius, points, dtype=np.float32),
                atol=1e-7, rtol=1e-7):
            _fail(f"slice {index} coordinates do not match the registered grid")
        for key in ("mse", "mse_nonfinite", "regularization", "regularization_nonfinite",
                    "objective", "objective_nonfinite", "finite_parameter_mask",
                    "candidate_evaluated_mask", "realized_displacement", "stiefel_residual"):
            if arrays[key].shape != (points, points):
                _fail(f"slice {index} array {key} does not match the registered grid")
        for key in ("mse", "regularization", "objective"):
            mask_key = f"{key}_nonfinite"
            if not _equal_array(arrays[mask_key], ~np.isfinite(arrays[key])):
                _fail(f"slice {index} {mask_key} does not match raw values")
        if not _equal_array(arrays["objective"], arrays["mse"] + arrays["regularization"],
                            atol=_IDENTITY_ATOL, rtol=_IDENTITY_RTOL, equal_nan=True):
            _fail(f"slice {index} objective is not MSE plus regularization")

        surface_evaluated = bool(sample_meta.get("surface_evaluated", False))
        center_index = points // 2
        grid_center = _float_or_none(arrays["mse"][center_index, center_index])
        verified_center = None
        center_abs_error = None
        center_relative_error = None
        if surface_evaluated:
            if not bool(slice_meta.get("scalar_center_check", {}).get("passed")):
                _fail(f"slice {index} evaluated surface has no passing scalar center check")
            if not np.isfinite(center).all():
                _fail(f"slice {index} evaluates a nonfinite center")
            _set_model_vector(model, center, layout)
            verified_center = _mse(model, train_x, train_y)
            if verified_center is None or grid_center is None:
                _fail(f"slice {index} center MSE is nonfinite")
            center_abs_error, center_relative_error = _check_close(
                grid_center, verified_center, what=f"slice {index} grid center")
            check = slice_meta["scalar_center_check"]
            reported_mse = _float_or_none(check.get("reported_mse"))
            if reported_mse is None:
                _fail(f"slice {index} scalar center reference is missing")
            _check_close(reported_mse, verified_center, what=f"slice {index} stored scalar reference")
            if check.get("passed") is False:
                _fail(f"slice {index} scalar_center_check failed")

            candidate_vectors = _candidate_vectors(
                torch.from_numpy(center.copy()), torch.from_numpy(directions.copy()),
                torch.from_numpy(coordinates.copy()), view=view, hidden=model.exponents.shape[0],
                input_dim=model.exponents.shape[1], scale=float(config.get("manifold_scale", 1.0)),
                device=torch.device("cpu"),
            ).reshape(points, points, parameter_dim).numpy()
            candidate_finite = np.isfinite(candidate_vectors).all(axis=-1)
            if not _equal_array(arrays["finite_parameter_mask"], candidate_finite):
                _fail(f"slice {index} candidate-parameter finite mask is incorrect")
            if not _equal_array(arrays["candidate_evaluated_mask"], candidate_finite):
                _fail(f"slice {index} candidate evaluation mask is incorrect")
            regularization_lambda = float(config.get("regularization_lambda", 0.0))
            if regularization_lambda < 0 or not math.isfinite(regularization_lambda):
                _fail("regularization_lambda must be finite and nonnegative")
            if regularization_lambda == 0.0:
                expected_regularization = np.zeros_like(arrays["regularization"])
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    squared_norm = np.square(candidate_vectors).sum(axis=-1, dtype=np.float32)
                    expected_regularization = squared_norm * np.float32(regularization_lambda)
            if not _equal_array(arrays["regularization"], expected_regularization,
                                atol=_IDENTITY_ATOL, rtol=_IDENTITY_RTOL, equal_nan=True):
                _fail(f"slice {index} regularization is not lambda times squared parameter norm")
            displacement = np.linalg.norm(candidate_vectors - center[None, None, :], axis=-1)
            if not _equal_array(arrays["realized_displacement"], displacement,
                                atol=1e-5, rtol=1e-5, equal_nan=True):
                _fail(f"slice {index} realized displacement is incorrect")
            if view == "manifold":
                from .punn_manifold_recorded import _stiefel_residual_batch

                exponent_count = int(model.exponents.numel())
                exponent_candidates = torch.from_numpy(
                    candidate_vectors[..., :exponent_count].copy()
                ).reshape(-1, model.exponents.shape[0], model.exponents.shape[1])
                reconstructed_residual = _stiefel_residual_batch(
                    exponent_candidates, float(config.get("manifold_scale", 1.0))
                ).numpy().reshape(points, points)
                finite_candidates = arrays["candidate_evaluated_mask"]
                stored_residual = arrays["stiefel_residual"]
                if not np.isfinite(stored_residual[finite_candidates]).all():
                    _fail(f"slice {index} has nonfinite feasible residuals for evaluated candidates")
                if (finite_candidates.any() and float(stored_residual[finite_candidates].max()) > 1e-4):
                    _fail(f"slice {index} has a feasible-grid Stiefel residual above 1e-4")
                if not _equal_array(stored_residual, reconstructed_residual,
                                    atol=1e-4, rtol=1e-3, equal_nan=True):
                    _fail(f"slice {index} stored feasibility residuals differ from reconstructed grid")
            elif np.isfinite(arrays["stiefel_residual"]).any():
                _fail(f"ambient slice {index} unexpectedly records Stiefel feasibility values")
        else:
            if bool(np.isfinite(arrays["mse"]).any()) or not bool(arrays["mse_nonfinite"].all()):
                _fail(f"unevaluated slice {index} must retain only nonfinite MSE values")
            for key in ("regularization", "objective"):
                if bool(np.isfinite(arrays[key]).any()) or not bool(arrays[f"{key}_nonfinite"].all()):
                    _fail(f"unevaluated slice {index} must retain only nonfinite {key} values")
            if bool(arrays["candidate_evaluated_mask"].any()):
                _fail(f"unevaluated slice {index} marks parameter candidates as evaluated")
            if slice_meta.get("scalar_center_check", {}).get("passed") is True:
                _fail(f"unevaluated slice {index} claims a passed scalar center check")

        mse_values = arrays["mse"]
        finite_mse = mse_values[np.isfinite(mse_values)]
        metric_summary = _metric_row(finite_mse, grid_center)
        finite_displacements = arrays["realized_displacement"][
            np.isfinite(arrays["realized_displacement"])
        ]
        slice_rows.append({
            "view": view,
            "phase": phase,
            "epoch": epoch,
            "state_index": state_index,
            "status": str(sample_meta.get("status", "unknown")),
            "surface_evaluated": surface_evaluated,
            "center_mse": grid_center,
            **metric_summary,
            "actual_displacement_p95": (float(np.percentile(finite_displacements, 95))
                                         if finite_displacements.size else None),
            "mse_finite_count": int(np.isfinite(arrays["mse"]).sum()),
            "mse_nonfinite_count": int((~np.isfinite(arrays["mse"])).sum()),
            "objective_finite_count": int(np.isfinite(arrays["objective"]).sum()),
            "objective_nonfinite_count": int((~np.isfinite(arrays["objective"])).sum()),
            "candidate_evaluated_count": int(arrays["candidate_evaluated_mask"].sum()),
            "candidate_not_evaluated_count": int((~arrays["candidate_evaluated_mask"]).sum()),
            "max_stiefel_residual": (
                float(np.nanmax(arrays["stiefel_residual"]))
                if view == "manifold" and np.isfinite(arrays["stiefel_residual"]).any() else None
            ),
            "center_verified_mse": verified_center,
            "center_abs_error": center_abs_error,
            "center_relative_error": center_relative_error,
            "directions_digest": _digest_array(directions),
            "center_check": dict(slice_meta.get("scalar_center_check", {})),
        })

        if check_terminal_corners and phase == "terminal" and surface_evaluated:
            _set_model_vector(model, center, layout)
            candidate_flat = _candidate_vectors(
                torch.from_numpy(center.copy()), torch.from_numpy(directions.copy()),
                torch.from_numpy(coordinates.copy()), view=view, hidden=model.exponents.shape[0],
                input_dim=model.exponents.shape[1], scale=float(config.get("manifold_scale", 1.0)),
                device=torch.device("cpu"),
            ).reshape(points, points, parameter_dim).numpy()
            corner_results = []
            for row_index, column_index in ((0, 0), (points - 1, points - 1)):
                if not arrays["candidate_evaluated_mask"][row_index, column_index]:
                    continue
                _set_model_vector(model, candidate_flat[row_index, column_index], layout)
                recomputed = _mse(model, train_x, train_y)
                recorded = _float_or_none(arrays["mse"][row_index, column_index])
                if recomputed is None or recorded is None:
                    _fail(f"terminal corner check is nonfinite at slice {index}")
                absolute, relative = _check_close(
                    recorded, recomputed, what=f"terminal corner {row_index},{column_index}")
                corner_results.append({"row": row_index, "column": column_index,
                                       "recorded_mse": recorded, "verified_mse": recomputed,
                                       "absolute_error": absolute, "relative_error": relative})
            slice_rows[-1]["terminal_corner_checks"] = corner_results

    # Every initialization/epoch/terminal center must have both views and the
    # exact same trajectory vector. The vector digests are also carried in rows.
    if len(paired_centers) != len(expected_slice_schedule):
        _fail("ambient/manifold slice centers are missing or duplicated")
    for key, center in paired_centers.items():
        del center
        pair = [row for row in slice_rows
                if (row["state_index"], row["phase"], row["epoch"]) == key]
        if {row["view"] for row in pair} != set(views):
            _fail(f"slice center {key} does not have both registered views")
        # The exact center vector identity was checked against the trajectory
        # for each member above; this catches any view-pair schedule mismatch.

    final_state = parameters[-1]
    verified_train_mse = None
    verified_test_mse = None
    if np.isfinite(final_state).all():
        _set_model_vector(model, final_state, layout)
        verified_train_mse = _mse(model, train_x, train_y)
        verified_test_mse = _mse(model, test_x, test_y)
    stored_train_mse = _float_or_none(terminal_result.get("train_mse"))
    stored_test_mse = _float_or_none(terminal_result.get("test_mse"))
    if terminal_outcome == "completed":
        if verified_train_mse is None or verified_test_mse is None:
            _fail("completed terminal state has nonfinite independent MSE")
        if stored_train_mse is None or stored_test_mse is None:
            _fail("completed terminal result lacks finite train/test MSE")
        _check_close(stored_train_mse, verified_train_mse, what="terminal train MSE")
        _check_close(stored_test_mse, verified_test_mse, what="terminal test MSE")

    return {
        "condition": {
            "task": task,
            "architecture": architecture,
            "method": method,
            "seed": seed,
            "data_seed": data_seed,
            "input_dim": model.input_dim,
            "hidden_units": model.exponents.shape[0],
            "output_dim": model.output.out_features,
            "training_device": training_device,
            "landscape_device": landscape_device,
        },
        "dataset_digest": manifest_digest,
        "terminal_outcome": terminal_outcome,
        "epochs_completed": epochs,
        "train_mse": stored_train_mse,
        "test_mse": stored_test_mse,
        "verified_train_mse": verified_train_mse,
        "verified_test_mse": verified_test_mse,
        "state_count": int(parameters.shape[0]),
        "slice_count": len(slice_rows),
        "validation": {
            "dataset_digest": True,
            "trajectory_schedule_and_digests": True,
            "parameter_finite_masks": True,
            "slice_schedule_and_center_identity": True,
            "scalar_center_checks": True,
            "independent_cpu_center_mse": True,
            "objective_equals_mse_plus_regularization": True,
            "candidate_masks_and_displacement": True,
            "manifold_feasibility": True,
            "landscape_scope": "sampled two-dimensional slices only",
        },
        "slices": slice_rows,
    }


def aggregate_audits(audits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize per-seed slice metrics without discarding invalid surfaces."""
    grouped: dict[tuple[Any, ...], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for audit in audits:
        condition = audit["condition"]
        for row in audit["slices"]:
            grouped[(condition["task"], condition["architecture"], row["view"],
                     row["phase"], int(row["epoch"]))].append((audit, row))
    result: list[dict[str, Any]] = []
    for key, entries in sorted(grouped.items()):
        columns = {
            name: np.asarray([row[name] for _, row in entries if row[name] is not None], dtype=np.float64)
            for name in ("signed_p95_mse_delta", "absolute_p95_mse_delta", "actual_displacement_p95")
        }
        summary: dict[str, Any] = {
            "task": key[0], "architecture": key[1], "view": key[2],
            "phase": key[3], "epoch": key[4], "run_count": len(entries),
            "surface_run_count": sum(bool(row["surface_evaluated"]) for _, row in entries),
            "unevaluated_run_count": sum(not bool(row["surface_evaluated"]) for _, row in entries),
            "mse_finite_count": sum(int(row["mse_finite_count"]) for _, row in entries),
            "mse_nonfinite_count": sum(int(row["mse_nonfinite_count"]) for _, row in entries),
            "objective_finite_count": sum(int(row["objective_finite_count"]) for _, row in entries),
            "objective_nonfinite_count": sum(int(row["objective_nonfinite_count"]) for _, row in entries),
            "candidate_evaluated_count": sum(int(row["candidate_evaluated_count"]) for _, row in entries),
            "candidate_not_evaluated_count": sum(int(row["candidate_not_evaluated_count"]) for _, row in entries),
        }
        for name, values in columns.items():
            summary[f"{name}_run_count"] = int(values.size)
            summary[f"{name}_q25"] = float(np.percentile(values, 25)) if values.size else None
            summary[f"{name}_median"] = float(np.percentile(values, 50)) if values.size else None
            summary[f"{name}_q75"] = float(np.percentile(values, 75)) if values.size else None
        result.append(summary)
    return result

