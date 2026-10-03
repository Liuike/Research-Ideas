import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "oscar_pure_manifold.sbatch"


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


def test_manifold_wrapper_uses_submit_project_root_from_slurm_spool(tmp_path):
    bash = _bash()
    if not bash:
        import pytest
        pytest.skip("bash is unavailable")

    project = tmp_path / "project"
    scripts = project / "scripts"
    spool = tmp_path / "slurm-spool"
    plan = project / "plans" / "paired.jsonl"
    scripts.mkdir(parents=True)
    spool.mkdir()
    plan.parent.mkdir()
    plan.write_text("paired plan placeholder\n", encoding="utf-8")
    wrapper = spool / "oscar_pure_manifold.sbatch"
    shutil.copy2(WRAPPER, wrapper)
    stub = scripts / "oscar_pure_cifar.sbatch"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'PROJECT_ROOT=<%s>\\n' \"$PROJECT_ROOT\"\n"
        "printf 'ARGC=<%s>\\n' \"$#\"\n"
        "printf 'PLAN=<%s>\\n' \"$1\"\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.pop("PROJECT_ROOT", None)
    env["SLURM_SUBMIT_DIR"] = _shell_path(bash, project)
    result = subprocess.run(
        [bash, _shell_path(bash, wrapper), _shell_path(bash, plan)],
        check=True,
        cwd=spool,
        env=env,
        text=True,
        capture_output=True,
    )

    assert f"PROJECT_ROOT=<{_shell_path(bash, project)}>" in result.stdout
    assert "ARGC=<1>" in result.stdout
    assert f"PLAN=<{_shell_path(bash, plan)}>" in result.stdout

    source = WRAPPER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --constraint=l40s",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32G",
        "#SBATCH --time=36:00:00",
        "#SBATCH --array=0-1%2",
    ):
        assert directive in source
