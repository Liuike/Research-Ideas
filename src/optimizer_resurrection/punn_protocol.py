"""Expand registered basic PUNN landscape experiments into one-run commands."""

from __future__ import annotations

import itertools
from typing import Any


FIELDS = {
    "protocol", "tasks", "methods", "seeds", "data_seed_offset", "epochs",
    "population_size", "search_bound", "boundary", "learning_rate", "momentum",
    "log_every", "device", "stage", "run_group",
}


def expand_config(config: dict[str, Any]) -> list[list[str]]:
    from .punn_landscape import PROTOCOL, parse_args

    missing, extra = FIELDS - config.keys(), config.keys() - FIELDS
    if missing or extra:
        raise ValueError(f"PUNN config missing={sorted(missing)} unknown={sorted(extra)}")
    if config["protocol"] != PROTOCOL:
        raise ValueError("unsupported PUNN protocol")
    for axis in ("tasks", "methods", "seeds"):
        values = config[axis]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{axis} must be a nonempty unique list")
    commands = []
    for task, method, seed in itertools.product(config["tasks"], config["methods"], config["seeds"]):
        values = {key: value for key, value in config.items() if key not in {"tasks", "methods", "seeds", "data_seed_offset"}}
        values.update(task=task, method=method, seed=seed,
                      data_seed=seed + config["data_seed_offset"],
                      run_name=f"punn-{task}-{method}-seed{seed}")
        command = ["python", "-m", "optimizer_resurrection.punn_landscape"]
        for key, value in values.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        parse_args(command[3:])
        commands.append(command)
    return commands
