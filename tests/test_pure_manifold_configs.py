from pathlib import Path

import pytest
import yaml

from optimizer_resurrection import pure_cifar as pc


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "pure_cifar"


def load_config(name):
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_manifold_seed_config_expands_to_paired_da10_conditions():
    config = load_config("manifold_seed0.yaml")
    commands = pc.expand_config(config)

    assert len(commands) == 2
    args = [pc.parse_args(command[3:]) for command in commands]
    assert [item.momentum for item in args] == [.95, 0.0]
    assert [item.seed for item in args] == [0, 0]
    assert [item.data_seed for item in args] == [100000, 100000]
    assert len({item.condition_id for item in args}) == 2
    assert all(item.protocol == pc.MANIFOLD_PROTOCOL for item in args)
    assert all(item.landscape for item in args)
    assert all(item.weight_decay == 0 for item in args)
    assert all((item.aux_learning_rate, item.manifold_scale,
                item.dual_learning_rate, item.dual_iterations) == (.001, 1, .01, 10)
               for item in args)
    assert all((item.epochs, item.batch_size, item.learning_rate,
                item.milestones, item.gamma) == (160, 128, .01, [80, 120], .1)
               for item in args)


@pytest.mark.parametrize(("field", "value"), [
    ("momentums", [0.0, .95]),
    ("aux_learning_rate", .01),
    ("manifold_scale", 2.0),
    ("dual_learning_rate", .02),
    ("dual_iterations", 9),
    ("weight_decay", .001),
])
def test_manifold_scientific_plan_rejects_nonfrozen_da10_values(field, value):
    config = load_config("manifold_seed0.yaml")
    config[field] = value
    with pytest.raises((ValueError, SystemExit)):
        pc.expand_config(config)


@pytest.mark.parametrize(("name", "device", "batch_size", "train", "test", "workers", "group"), [
    ("manifold_cpu_smoke.yaml", "cpu", 4, 4, 4, 0, "pure-cifar-manifold-cpu-smoke-v1"),
    ("manifold_local_gpu_smoke.yaml", "cuda", 128, 256, 64, 0,
     "pure-cifar-manifold-local-smoke-v2"),
    ("manifold_oscar_smoke.yaml", "cuda", 128, 256, 64, 4,
     "pure-cifar-manifold-oscar-smoke-v1"),
])
def test_manifold_smoke_configs_are_small_paired_engineering_runs(
        name, device, batch_size, train, test, workers, group):
    config = load_config(name)
    commands = pc.expand_config(config)
    args = [pc.parse_args(command[3:]) for command in commands]

    assert len(args) == 2
    assert [item.momentum for item in args] == [.95, 0.0]
    assert all(item.seed == 999 and item.epochs == 1 for item in args)
    assert all(item.device == device and item.batch_size == batch_size for item in args)
    assert all(item.train_examples == train and item.test_examples == test for item in args)
    assert all(item.num_workers == workers and item.run_group == group for item in args)
    assert all(item.weight_decay == 0 and item.landscape for item in args)
