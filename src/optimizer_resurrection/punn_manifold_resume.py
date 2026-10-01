"""Reconcile online DA-10 landscape attempts before a recorded-plan resume."""

from __future__ import annotations

from typing import Any

from .punn_architecture_analysis import _terminal_snapshot


_RUNTIME_FIELDS = {"dry_run", "stage", "run_name", "run_group"}
_CONTINUATION_ONLY_FIELDS = {"condition_id", "device", "landscape_device"}
_LOGICAL_FIELDS = ("task", "architecture", "method", "seed", "data_seed")


def _logical_cell(values: dict[str, Any]) -> tuple[Any, ...]:
    try:
        return tuple(values[field] for field in _LOGICAL_FIELDS)
    except KeyError as error:
        raise ValueError(f"continuation recipe is missing logical field: {error.args[0]}") from error


def _expected_by_logical_cell(
    expected: dict[str, dict[str, Any]], label: str,
) -> dict[tuple[Any, ...], tuple[str, dict[str, Any]]]:
    cells: dict[tuple[Any, ...], tuple[str, dict[str, Any]]] = {}
    for condition, values in expected.items():
        if values.get("condition_id") != condition:
            raise ValueError(f"{label} expected condition ID mismatch: {condition}")
        cell = _logical_cell(values)
        if cell in cells:
            raise ValueError(f"duplicate {label} expected logical cell: {cell}")
        cells[cell] = (condition, values)
    return cells


def _validate_continuation_recipes(
    source_expected: dict[str, dict[str, Any]],
    current_expected: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[Any, ...], tuple[str, dict[str, Any]]],
           dict[tuple[Any, ...], tuple[str, dict[str, Any]]]]:
    source_cells = _expected_by_logical_cell(source_expected, "source")
    current_cells = _expected_by_logical_cell(current_expected, "continuation")
    if source_cells.keys() != current_cells.keys():
        missing = sorted(source_cells.keys() - current_cells.keys(), key=repr)
        extra = sorted(current_cells.keys() - source_cells.keys(), key=repr)
        raise ValueError(f"continuation logical-cell mismatch: missing={missing} extra={extra}")

    allowed = _RUNTIME_FIELDS | _CONTINUATION_ONLY_FIELDS
    for cell in source_cells:
        _source_id, source_values = source_cells[cell]
        _current_id, current_values = current_cells[cell]
        source_recipe = {key: value for key, value in source_values.items() if key not in allowed}
        current_recipe = {key: value for key, value in current_values.items() if key not in allowed}
        if source_recipe != current_recipe:
            differing = sorted(
                key for key in source_recipe.keys() | current_recipe.keys()
                if source_recipe.get(key) != current_recipe.get(key)
            )
            raise ValueError(f"continuation recipe mismatch for {cell}: {differing}")
    return source_cells, current_cells


def _validate_continuation_records(
    records: list[Any], expected: dict[str, dict[str, Any]], *, group: str,
    device: str, revision: str, source_tree_digest: str,
) -> None:
    for record in records:
        condition = record.config.get("condition_id")
        if condition not in expected:
            raise ValueError(f"unregistered landscape condition: {record.run_id}")
        if record.config.get("stage") != "exploratory":
            raise ValueError(f"continuation requires scientific exploratory stage: {record.run_id}")
        if record.config.get("run_group") != group:
            raise ValueError(f"unknown landscape run group: {record.run_id}")
        if (record.config.get("device") != device
                or record.config.get("landscape_device") != device):
            raise ValueError(f"landscape device mismatch: {record.run_id}")
        if record.state in {"running", "pending", "preempting"}:
            raise ValueError(f"active landscape run prevents a duplicate resume: {record.run_id}")
        for key, value in expected[condition].items():
            if record.config.get(key) != value:
                raise ValueError(f"landscape recipe mismatch: {record.run_id}: {key}")
        provenance = record.config.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"landscape resume requires source provenance: {record.run_id}")
        git = provenance.get("git", {})
        if not isinstance(git, dict):
            git = {}
        if git.get("revision") != revision or git.get("dirty") is not False:
            raise ValueError(f"landscape resume requires the original clean source: {record.run_id}")
        source_tree = provenance.get("source_tree", {})
        observed_digest = source_tree.get("digest") if isinstance(source_tree, dict) else None
        if observed_digest != source_tree_digest:
            raise ValueError(f"landscape resume requires the original source bytes: {record.run_id}")


def pending_continuation_condition_ids(
    source_expected: dict[str, dict[str, Any]],
    current_expected: dict[str, dict[str, Any]],
    source_records: list[Any],
    current_records: list[Any],
    *,
    source_group: str,
    current_group: str,
    source_revision: str,
    source_tree_digest: str,
    current_revision: str,
    current_tree_digest: str,
) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconcile GPU source outcomes against a CPU continuation plan.

    Source and current condition IDs differ because the device is part of the
    registered condition. Pairing is instead by task, architecture, method,
    model seed, and data seed; all remaining scientific fields must match.
    """
    source_cells, current_cells = _validate_continuation_recipes(
        source_expected, current_expected
    )
    _validate_continuation_records(
        source_records, source_expected, group=source_group, device="cuda",
        revision=source_revision, source_tree_digest=source_tree_digest,
    )
    _validate_continuation_records(
        current_records, current_expected, group=current_group, device="cpu",
        revision=current_revision, source_tree_digest=current_tree_digest,
    )

    source_pending, source_excluded = pending_condition_ids(
        source_expected, source_records, source_revision, source_tree_digest
    )
    current_pending, current_excluded = pending_condition_ids(
        current_expected, current_records, current_revision, current_tree_digest
    )
    source_accepted = set(source_expected) - source_pending
    current_accepted = set(current_expected) - current_pending
    accepted_source_cells = {
        cell for cell, (condition, _values) in source_cells.items()
        if condition in source_accepted
    }
    accepted_current_cells = {
        cell for cell, (condition, _values) in current_cells.items()
        if condition in current_accepted
    }
    duplicate_cells = accepted_source_cells & accepted_current_cells
    if duplicate_cells:
        raise ValueError(
            f"duplicate accepted GPU and CPU continuation outcomes: {sorted(duplicate_cells, key=repr)}"
        )
    completed_cells = accepted_source_cells | accepted_current_cells
    pending_current = {
        condition for cell, (condition, _values) in current_cells.items()
        if cell not in completed_cells
    }

    def decorate_excluded(
        excluded: list[dict[str, Any]], records: list[Any], group: str,
    ) -> list[dict[str, Any]]:
        by_id = {record.run_id: record for record in records}
        return [
            {**item, "device": by_id[item["run_id"]].config.get("device"),
             "group": group}
            for item in excluded
        ]

    excluded = decorate_excluded(source_excluded, source_records, source_group)
    excluded.extend(decorate_excluded(current_excluded, current_records, current_group))
    accepted_sources = []
    source_records_by_condition = {
        record.config["condition_id"]: record for record in source_records
        if record.config["condition_id"] in source_accepted
    }
    for condition in sorted(source_accepted):
        record = source_records_by_condition[condition]
        result = _terminal_snapshot(record.summary, record.run_id)
        accepted_sources.append({
            "run_id": record.run_id,
            "condition_id": condition,
            "device": record.config.get("device"),
            "landscape_artifact": result["landscape_artifact"],
        })
    return pending_current, excluded, accepted_sources


def pending_condition_ids(expected: dict[str, dict[str, Any]], records: list[Any],
                          source_revision: str, source_tree_digest: str | None = None,
                          ) -> tuple[set[str], list[dict[str, Any]]]:
    """Accept unique uploaded scientific outcomes; reject overlapping active jobs."""
    accepted: set[str] = set()
    excluded: list[dict[str, Any]] = []
    for record in records:
        condition = record.config.get("condition_id")
        if condition not in expected:
            raise ValueError(f"unregistered landscape condition: {record.run_id}")
        if record.state in {"running", "pending", "preempting"}:
            raise ValueError(f"active landscape run prevents a duplicate resume: {record.run_id}")
        try:
            result = _terminal_snapshot(record.summary, record.run_id)
        except ValueError:
            result = {}
        if (record.state != "finished"
                or result.get("terminal_outcome") not in {"completed", "numerical_failure"}
                or result.get("landscape_recording_complete") is not True):
            excluded.append({"run_id": record.run_id, "condition_id": condition,
                             "state": record.state, "reason": "incomplete_execution_or_recording"})
            continue
        for key, value in expected[condition].items():
            if record.config.get(key) != value:
                raise ValueError(f"landscape recipe mismatch: {record.run_id}: {key}")
        git = record.config.get("provenance", {}).get("git", {})
        if git.get("revision") != source_revision or git.get("dirty") is not False:
            raise ValueError(f"landscape resume requires the original clean source: {record.run_id}")
        if source_tree_digest is not None:
            observed_digest = record.config.get("provenance", {}).get("source_tree", {}).get("digest")
            if observed_digest != source_tree_digest:
                raise ValueError(f"landscape resume requires the original source bytes: {record.run_id}")
        if result.get("terminal_outcome") == "completed":
            if result.get("epochs_completed") != expected[condition]["epochs"]:
                raise ValueError(f"incomplete final epoch: {record.run_id}")
        if not isinstance(result.get("landscape_artifact"), str) or ":v" not in result["landscape_artifact"]:
            raise ValueError(f"missing immutable landscape artifact: {record.run_id}")
        if condition in accepted:
            raise ValueError(f"duplicate accepted landscape condition: {condition}")
        accepted.add(condition)
    return set(expected) - accepted, excluded


def select_pending_commands(commands: list[list[str]], config: dict[str, Any]) -> list[list[str]]:
    import wandb
    from pathlib import Path
    import subprocess
    import yaml
    from .punn_manifold_recorded import parse_args
    from .tracking import RunRecord, load_wandb_credentials, require_online_wandb
    from .train import _git_metadata, _require_reproducible_source
    from .train import _source_tree_digest

    root = Path(__file__).resolve().parents[2]
    git = _git_metadata(root)
    if "continuation" in config and config.get("stage") != "exploratory":
        raise ValueError("registered continuation requires scientific exploratory stage")
    _require_reproducible_source(config["stage"], git)
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    parsed = [(command, vars(parse_args(command[3:]))) for command in commands]
    expected = {
        values["condition_id"]: {key: value for key, value in values.items()
                                 if key not in _RUNTIME_FIELDS}
        for _command, values in parsed
    }
    current_digest = _source_tree_digest(root)["digest"]

    continuation = config.get("continuation")
    if continuation is None:
        query = {"group": config["run_group"], "config.protocol": config["protocol"]}
        records = [RunRecord(str(r.id), str(r.state), dict(r.config), dict(r.summary), str(r.url))
                   for r in wandb.Api(timeout=90).runs(
                       f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}", filters=query)]
        pending, excluded = pending_condition_ids(
            expected, records, git["revision"], current_digest)
        # Keep ordinary --resume's output and same-source reconciliation stable.
        import json
        print(json.dumps({"resume_pending": len(pending),
                          "resume_accepted": len(expected) - len(pending),
                          "excluded_incomplete_attempts": excluded}), flush=True)
        return [command for command, values in parsed
                if values["condition_id"] in pending]

    frozen_continuation = {
        "source_config": "configs/product_unit/manifold_landscape_gpu.yaml",
        "source_group": "punn-manifold-landscape-gpu-v1",
        "source_revision": "c55600e4755adf687753e6dee01b63462bbc6f65",
        "source_tree_digest": "853afa71ba5f0e4df8f1a78e4967d573c5f476e6ef35522c176e5639768946c6",
        "source_device": "cuda",
        "reason": "user_requested_cpu_completion",
    }
    if continuation != frozen_continuation:
        raise ValueError("invalid frozen GPU-to-CPU continuation metadata")
    if (config.get("run_group") != "punn-manifold-landscape-cpu-continuation-v1"
            or config.get("device") != "cpu"
            or config.get("landscape_device") != "cpu"):
        raise ValueError("GPU-to-CPU continuation requires the registered CPU group and devices")

    source_config_path = root / frozen_continuation["source_config"]
    try:
        frozen_source_bytes = subprocess.run(
            ["git", "show", f"{frozen_continuation['source_revision']}:{frozen_continuation['source_config']}"],
            cwd=root, check=True, capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError("frozen continuation source config is unavailable at its recorded revision") from error
    current_source_bytes = source_config_path.read_bytes()
    if (current_source_bytes.replace(b"\r\n", b"\n")
            != frozen_source_bytes.replace(b"\r\n", b"\n")):
        raise ValueError("frozen continuation source config differs from its recorded revision")
    source_config = yaml.safe_load(current_source_bytes.decode("utf-8"))
    if (not isinstance(source_config, dict)
            or source_config.get("run_group") != frozen_continuation["source_group"]
            or source_config.get("stage") != "exploratory"
            or source_config.get("device") != "cuda"
            or source_config.get("landscape_device") != "cuda"):
        raise ValueError("frozen continuation source config has changed identity or device")
    from .punn_manifold_recorded import expand_config
    source_commands = expand_config(source_config)
    source_parsed = [(command, vars(parse_args(command[3:])))
                     for command in source_commands]
    source_expected = {
        values["condition_id"]: {key: value for key, value in values.items()
                                 if key not in _RUNTIME_FIELDS}
        for _command, values in source_parsed
    }

    api = wandb.Api(timeout=90)

    def query_records(group: str) -> list[RunRecord]:
        query = {"group": group, "config.protocol": config["protocol"]}
        return [RunRecord(str(record.id), str(record.state), dict(record.config),
                          dict(record.summary), str(record.url))
                for record in api.runs(
                    f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}",
                    filters=query)]

    source_records = query_records(frozen_continuation["source_group"])
    current_records = query_records(config["run_group"])
    pending, excluded, accepted_sources = pending_continuation_condition_ids(
        source_expected, expected, source_records, current_records,
        source_group=frozen_continuation["source_group"],
        current_group=config["run_group"],
        source_revision=frozen_continuation["source_revision"],
        source_tree_digest=frozen_continuation["source_tree_digest"],
        current_revision=git["revision"], current_tree_digest=current_digest,
    )
    import json
    print(json.dumps({
        "continuation_reconciliation": {
            "source_group": frozen_continuation["source_group"],
            "source_expected": len(source_expected),
            "current_expected": len(expected),
            "accepted_gpu_sources": accepted_sources,
            "resume_accepted": len(expected) - len(pending),
            "resume_pending": len(pending),
            "excluded_incomplete_attempts": excluded,
        }
    }, sort_keys=True), flush=True)
    return [command for command, values in parsed
            if values["condition_id"] in pending]
