import tomllib
from pathlib import Path

import torch

from optimizer_resurrection.models import SigmoidMLP
from optimizer_resurrection.train import seed_everything


def test_model_initialization_repeats_for_same_seed():
    seed_everything(19)
    first = SigmoidMLP(12, 8, 3, 2)
    state = {name: value.clone() for name, value in first.state_dict().items()}
    seed_everything(19)
    second = SigmoidMLP(12, 8, 3, 2)
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in state.items())


def test_uv_runtime_is_pinned_for_local_and_oscar():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(project["project"]["dependencies"])

    assert Path(".python-version").read_text(encoding="utf-8").strip() == "3.11"
    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    assert project["tool"]["uv"]["required-version"] == "==0.12.11"
    assert "torch==2.11.0" in dependencies
    assert "torchvision==0.26.0" in dependencies
    assert Path("uv.lock").is_file()

    for script_path in (
        "scripts/oscar_plan.sbatch",
        "scripts/oscar_select_exploratory.sbatch",
        "scripts/oscar_submit_exploratory.sh",
    ):
        script = Path(script_path).read_text(encoding="utf-8")
        assert "uv" in script
        assert "--frozen" in script
        assert "--no-sync" in script
        assert "source \"${VENV_ACTIVATE" not in script
