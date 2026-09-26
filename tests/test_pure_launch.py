import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "pure_cifar"
SCRIPT = ROOT / "scripts" / "oscar_pure_cifar.sbatch"
CONFIG_FIELDS = {
    "protocol",
    "seeds",
    "data_seed_offset",
    "epochs",
    "batch_size",
    "learning_rate",
    "momentum",
    "weight_decay",
    "milestones",
    "gamma",
    "device",
    "num_workers",
    "data_dir",
    "download",
    "train_examples",
    "test_examples",
    "secondary",
    "stage",
    "run_group",
}


def _load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_pure_cifar_configs_have_the_frozen_protocol_shape_and_values():
    configs = {
        name: _load_config(name)
        for name in (
            "defaults.yaml",
            "cpu_smoke.yaml",
            "local_gpu_smoke.yaml",
            "oscar_smoke.yaml",
        )
    }
    assert all(set(config) == CONFIG_FIELDS for config in configs.values())

    defaults = configs["defaults.yaml"]
    assert defaults == {
        "protocol": "pure-cifar10-resnet18-v1",
        "seeds": [0, 1, 2, 3, 4],
        "data_seed_offset": 100000,
        "epochs": 160,
        "batch_size": 128,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.001,
        "milestones": [80, 120],
        "gamma": 0.1,
        "device": "cuda",
        "num_workers": 4,
        "data_dir": "data/cifar10",
        "download": False,
        "train_examples": 50000,
        "test_examples": 10000,
        "secondary": True,
        "stage": "reproduction",
        "run_group": "pure-cifar10-resnet18-v1",
    }
    assert configs["cpu_smoke.yaml"]["device"] == "cpu"
    assert configs["cpu_smoke.yaml"]["stage"] == "engineering-smoke"
    assert configs["cpu_smoke.yaml"]["train_examples"] == 16
    assert configs["local_gpu_smoke.yaml"]["device"] == "cuda"
    assert configs["local_gpu_smoke.yaml"]["train_examples"] == 1024
    assert configs["local_gpu_smoke.yaml"]["test_examples"] == 256
    assert configs["oscar_smoke.yaml"]["num_workers"] == 4
    assert configs["oscar_smoke.yaml"]["run_group"] == "pure-cifar-oscar-smoke-v1"


def _bash() -> str:
    candidates = []
    if os.name == "nt":
        candidates.append(r"C:\Program Files\Git\bin\bash.exe")
    candidates.append(shutil.which("bash"))
    return next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), "")


def _shell_path(bash: str, path: Path) -> str:
    if os.name != "nt":
        return str(path)
    converted = subprocess.run(
        [bash, "-lc", 'cygpath -u "$1"', "bash", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return converted.stdout.strip()


def test_oscar_script_syntax_and_array_resources():
    bash = _bash()
    if not bash:
        pytest.skip("bash is unavailable")
    subprocess.run([bash, "-n", str(SCRIPT)], check=True, cwd=ROOT)
    source = SCRIPT.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --constraint=l40s",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32G",
        "#SBATCH --time=12:00:00",
        "#SBATCH --array=0-4%2",
        "#SBATCH --output=logs/slurm/%x-%A_%a.out",
        "#SBATCH --error=logs/slurm/%x-%A_%a.err",
    ):
        assert directive in source


def test_oscar_dry_run_maps_array_task_to_jsonl_row_and_sets_cache_first(tmp_path):
    bash = _bash()
    python = shutil.which("python") or shutil.which("python3")
    if not bash or not python:
        pytest.skip("bash and python are required for launcher dry-run")

    plan = tmp_path / "plan.jsonl"
    rows = []
    for seed in range(5):
        rows.append(
            [
                "python",
                "-m",
                "optimizer_resurrection.pure_cifar",
                "--protocol",
                "pure-cifar10-resnet18-v1",
                "--seed",
                str(seed),
                "--milestones",
                "80,120",
                "--download",
                "false",
                "--secondary",
                "true",
                "--stage",
                "reproduction",
                "--run-group",
                "pure-cifar10-resnet18-v1",
                "--run-name",
                f"pure-cifar10-resnet18-v1__seed{seed}",
            ]
        )
    plan.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    fake_uv = tmp_path / "fake-uv"
    capture = tmp_path / "observed-env.txt"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"${UV_CACHE_DIR:-unset}\" \"${WANDB_DIR:-unset}\" > \"${CAPTURE_ENV}\"\n"
        "shift 3\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    for key in (
        "UV_CACHE_DIR",
        "WANDB_DIR",
        "WANDB_DATA_DIR",
        "WANDB_CONFIG_DIR",
        "WANDB_CACHE_DIR",
    ):
        env.pop(key, None)
    env.update(
        {
            "UV_BIN": _shell_path(bash, fake_uv),
            "CAPTURE_ENV": _shell_path(bash, capture),
            "PLAN_TASK_ID": "4",
            "DRY_RUN": "1",
            "PROJECT_ROOT": _shell_path(bash, ROOT),
        }
    )
    result = subprocess.run(
        [bash, _shell_path(bash, SCRIPT), _shell_path(bash, plan)],
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert "Plan line: 5" in result.stdout
    assert "Seed: 4" in result.stdout
    assert "Run name: pure-cifar10-resnet18-v1__seed4" in result.stdout
    assert "DRY_RUN=1" in result.stdout
    assert "Final command:" in result.stdout
    assert "--seed 4" in result.stdout
    observed_cache, observed_wandb = capture.read_text(encoding="utf-8").splitlines()
    assert observed_cache == _shell_path(bash, ROOT / ".cache" / "uv")
    assert observed_wandb == _shell_path(bash, ROOT / "wandb")


def test_oscar_dry_run_rejects_array_task_outside_plan(tmp_path):
    bash = _bash()
    python = shutil.which("python") or shutil.which("python3")
    if not bash or not python:
        pytest.skip("bash and python are required for launcher dry-run")

    plan = tmp_path / "plan.jsonl"
    plan.write_text(json.dumps(["python", "-m", "optimizer_resurrection.pure_cifar"]) + "\n", encoding="utf-8")
    fake_uv = tmp_path / "fake-uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nshift 3\nexec \"$@\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "UV_BIN": _shell_path(bash, fake_uv),
            "PLAN_TASK_ID": "1",
            "DRY_RUN": "1",
            "PROJECT_ROOT": _shell_path(bash, ROOT),
        }
    )
    result = subprocess.run(
        [bash, _shell_path(bash, SCRIPT), _shell_path(bash, plan)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "outside the 1-command plan" in result.stderr
