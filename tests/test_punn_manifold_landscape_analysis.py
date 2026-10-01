from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from optimizer_resurrection.models.product_unit import ProductUnitNetwork
from optimizer_resurrection.punn_manifold_landscape_analysis import (
    aggregate_audits,
    audit_payload,
)
from optimizer_resurrection.punn_manifold_recorded import (
    _directions,
    _parameter_digest,
    evaluate_slice,
)
from optimizer_resurrection.punn_sampling import orthonormal_directions


def _fixture(regularization_lambda=0.0):
    torch.set_num_threads(1)
    config = {
        "task": "f1", "architecture": "small", "method": "manifold_muon_da10",
        "seed": 3, "data_seed": 100003, "epochs": 1, "input_dim": 1,
        "hidden_units": 1, "output_dim": 1, "device": "cpu",
        "landscape_device": "cpu", "landscape_views": ["ambient", "manifold"],
        "landscape_seed_offset": 2_000_000, "slice_points": 3,
        "slice_radius": 0.1, "slice_every": 1,
        "regularization_lambda": regularization_lambda,
        "manifold_scale": 1.0,
    }
    train_x = np.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
    train_y = np.square(train_x).astype(np.float32)
    test_x = np.asarray([[-0.75], [0.75]], dtype=np.float32)
    test_y = np.square(test_x).astype(np.float32)
    digest = hashlib.sha256()
    for tensor in (train_x, train_y, test_x, test_y):
        digest.update(np.ascontiguousarray(tensor).tobytes())
    dataset_digest = digest.hexdigest()
    dataset = {
        "train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y,
        "dataset_digest": np.asarray(dataset_digest),
        "metadata_json": np.asarray(json.dumps({
            "task": "f1", "data_seed": 100003, "dataset_digest": dataset_digest,
        })),
    }

    model = ProductUnitNetwork(1, 1, 1, input_domain="real_complex", init_bound=1.0)
    with torch.no_grad():
        model.exponents.fill_(1.0)
        model.output.weight.fill_(0.4)
        model.output.bias.fill_(0.1)
    center = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().float()
    vector = center.numpy().copy()
    layout = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
         "dtype": str(parameter.dtype)}
        for name, parameter in model.named_parameters()
    ]
    states = []
    for index, (phase, epoch) in enumerate((("initialization", 0), ("epoch", 1), ("terminal", 1))):
        states.append({
            "phase": phase, "epoch": epoch, "state_index": index,
            "parameter_digest": _parameter_digest(center),
            "finite_parameter_count": int(torch.isfinite(center).sum()),
            "nonfinite_parameter_count": int((~torch.isfinite(center)).sum()),
        })
    direction_seed = config["data_seed"] + config["landscape_seed_offset"]
    ambient = orthonormal_directions(center.numel(), direction_seed)
    payload_arrays = {
        "trajectory/parameters": np.stack([vector, vector, vector]).astype(np.float32),
        "trajectory/finite_mask": np.ones((3, vector.size), dtype=np.bool_),
        "directions/ambient": ambient.numpy().astype(np.float32),
    }
    metadata = {"trajectory/metadata": states, "parameter_layout": layout}
    with torch.no_grad():
        terminal_train = float(((model(torch.from_numpy(train_x)) - torch.from_numpy(train_y)) ** 2).mean())
        terminal_test = float(((model(torch.from_numpy(test_x)) - torch.from_numpy(test_y)) ** 2).mean())
    coordinates = torch.linspace(-config["slice_radius"], config["slice_radius"], config["slice_points"])
    slice_phases = (("initialization", 0, 0), ("epoch", 1, 1), ("terminal", 1, 2))
    slice_summaries = []
    per_view_count = {"ambient": 0, "manifold": 0}
    slice_index = 0
    for phase, epoch, state_index in slice_phases:
        for view in config["landscape_views"]:
            direction_bundle = _directions(model, direction_seed, center, ambient=ambient, scale=1.0)
            directions = direction_bundle[view]
            sample = evaluate_slice(
                model, torch.from_numpy(train_x), torch.from_numpy(train_y), center, directions,
                view=view, points=config["slice_points"], radius=config["slice_radius"],
                    batch_size=16, device="cpu", regularization_lambda=regularization_lambda, scale=1.0,
            )
            center_mse = float(sample["mse"][1, 1])
            sample_meta = {
                "phase": phase, "epoch": epoch, "view": view, "state_index": state_index,
                "center_digest": _parameter_digest(center), "surface_evaluated": sample["surface_evaluated"],
                    "direction_pair_degenerate": not bool(torch.isfinite(directions).all()),
                "scalar_center_check": {
                    "passed": True, "reported_mse": center_mse,
                    "grid_center_mse": center_mse,
                },
                "status": sample["status"],
            }
            prefix = f"slices/{slice_index:04d}"
            metadata[f"{prefix}/metadata"] = sample_meta
            payload_arrays[f"{prefix}/center"] = center.numpy().astype(np.float32)
            payload_arrays[f"{prefix}/directions"] = directions.numpy().astype(np.float32)
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    payload_arrays[f"{prefix}/{key}"] = value.detach().cpu().numpy()
                else:
                    metadata[f"{prefix}/{key}"] = value
            slice_summaries.append({"phase": phase, "epoch": epoch, "view": view})
            per_view_count[view] += 1
            slice_index += 1
    flattened_metadata = {}

    def flatten(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(f"{prefix}/{key}" if prefix else str(key), child)
        else:
            flattened_metadata[prefix] = value

    for key, value in metadata.items():
        flatten(key, value)
    payload = {**payload_arrays,
               "metadata_json": np.asarray(json.dumps(flattened_metadata, sort_keys=True))}
    terminal_result = {
        "task": "f1", "architecture": "small", "method": "manifold_muon_da10",
        "seed": 3, "data_seed": 100003, "epochs_completed": 1,
        "terminal_outcome": "completed", "dataset_digest": dataset_digest,
        "train_mse": terminal_train, "test_mse": terminal_test,
    }
    manifest = {
        "schema": "punn_manifold_recorded_v1", "protocol": "punn-manifold-recorded-v1",
        "dataset_digest": dataset_digest,
        "dataset_artifact": "punn-manifold-recorded-data-f1-100003-abcd:v0",
        "training_device": "cpu", "landscape_device": "cpu",
        "trajectory": {"states": 3, "parameter_layout": layout},
        "slices": {"count": 6, "count_by_view": per_view_count,
                    "views": config["landscape_views"], "grid": [3, 3], "radius": 0.1,
                    "slice_every": 1, "summary": slice_summaries},
    }
    return payload, manifest, terminal_result, config, dataset


def test_audit_payload_checks_recording_and_returns_center_relative_metrics():
    payload, manifest, terminal_result, config, dataset = _fixture()

    audit = audit_payload(payload, manifest, terminal_result, config, dataset)

    assert audit["validation"]["dataset_digest"] is True
    assert audit["state_count"] == 3
    assert audit["slice_count"] == 6
    assert audit["verified_train_mse"] == pytest.approx(terminal_result["train_mse"])
    assert audit["verified_test_mse"] == pytest.approx(terminal_result["test_mse"])
    assert {row["view"] for row in audit["slices"]} == {"ambient", "manifold"}
    assert all(row["center_abs_error"] is not None for row in audit["slices"])
    assert all(row["absolute_p95_mse_delta"] is not None for row in audit["slices"])
    summaries = aggregate_audits([audit])
    assert len(summaries) == 6
    assert all(row["run_count"] == 1 for row in summaries)


def test_audit_payload_rejects_corrupt_trajectory_finite_mask():
    payload, manifest, terminal_result, config, dataset = _fixture()
    payload = dict(payload)
    mask = payload["trajectory/finite_mask"].copy()
    mask[0, 0] = False
    payload["trajectory/finite_mask"] = mask

    with pytest.raises(ValueError, match="trajectory finite mask"):
        audit_payload(payload, manifest, terminal_result, config, dataset)


def test_audit_payload_rejects_slice_center_that_does_not_match_state():
    payload, manifest, terminal_result, config, dataset = _fixture()
    payload = dict(payload)
    payload["slices/0000/center"] = payload["slices/0000/center"].copy()
    payload["slices/0000/center"][0] += 0.01

    with pytest.raises(ValueError, match="center differs from its recorded trajectory state"):
        audit_payload(payload, manifest, terminal_result, config, dataset)


def test_audit_payload_recomputes_center_even_when_vector_and_digests_match():
    payload, manifest, terminal_result, config, dataset = _fixture()
    payload = dict(payload)
    states = payload["trajectory/parameters"].copy()
    states[0, -1] += 0.2
    payload["trajectory/parameters"] = states
    metadata = json.loads(str(payload["metadata_json"].item()))
    changed_center = states[0]
    digest = hashlib.sha256(np.ascontiguousarray(changed_center).tobytes()).hexdigest()
    metadata["trajectory/metadata"][0]["parameter_digest"] = digest
    for index in (0, 1):
        center = payload[f"slices/{index:04d}/center"].copy()
        center[-1] += 0.2
        payload[f"slices/{index:04d}/center"] = center
        metadata[f"slices/{index:04d}/metadata/center_digest"] = digest
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    with pytest.raises(ValueError, match="grid center mismatch"):
        audit_payload(payload, manifest, terminal_result, config, dataset)


def test_audit_payload_rejects_slice_phase_mapped_to_wrong_trajectory_index():
    payload, manifest, terminal_result, config, dataset = _fixture()
    payload = dict(payload)
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata["slices/0000/metadata/state_index"] = 1
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    with pytest.raises(ValueError, match="phase/epoch maps to trajectory state"):
        audit_payload(payload, manifest, terminal_result, config, dataset)


def test_audit_payload_rejects_wrong_objective_grid():
    payload, manifest, terminal_result, config, dataset = _fixture()
    payload = dict(payload)
    objective = payload["slices/0000/objective"].copy()
    objective[0, 0] += 1.0
    payload["slices/0000/objective"] = objective

    with pytest.raises(ValueError, match="objective is not MSE plus regularization"):
        audit_payload(payload, manifest, terminal_result, config, dataset)


def test_audit_payload_rejects_joint_regularization_and_objective_shift_at_zero_lambda():
    payload, manifest, terminal_result, config, dataset = _fixture(regularization_lambda=0.0)
    payload = dict(payload)
    regularization = payload["slices/0000/regularization"].copy()
    objective = payload["slices/0000/objective"].copy()
    regularization[0, 0] = -0.25
    objective[0, 0] -= 0.25
    payload["slices/0000/regularization"] = regularization
    payload["slices/0000/objective"] = objective

    with pytest.raises(ValueError, match="regularization is not lambda times"):
        audit_payload(payload, manifest, terminal_result, config, dataset)


def test_audit_payload_accepts_registered_nonzero_regularization():
    payload, manifest, terminal_result, config, dataset = _fixture(regularization_lambda=1e-4)

    audit = audit_payload(payload, manifest, terminal_result, config, dataset)

    assert audit["validation"]["objective_equals_mse_plus_regularization"] is True


def test_audit_payload_rejects_incomplete_manifest_schedule():
    payload, manifest, terminal_result, config, dataset = _fixture()
    manifest = {**manifest, "slices": {**manifest["slices"], "count": 5}}

    with pytest.raises(ValueError, match="manifest slice count"):
        audit_payload(payload, manifest, terminal_result, config, dataset)
