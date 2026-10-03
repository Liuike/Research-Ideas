import hashlib
import pytest
import torch


from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optimizer_resurrection import pure_cifar as pc
from optimizer_resurrection.models.pure_resnet import ProductUnitConv2d
from optimizer_resurrection.optim import manifold_muon
from optimizer_resurrection.pure_manifold import (
    build_manifold_optimizer,
    parameter_digest,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for mixed-device AdamW state")
def test_finite_check_accepts_cpu_step_counter_and_cuda_parameter():
    assert pc.all_finite([torch.tensor(1.0), torch.ones(2, device="cuda")])
    assert not pc.all_finite([torch.tensor(float("nan")), torch.ones(2, device="cuda")])


class TinyPURe(nn.Module):
    """Small image model with product, ordinary convolution, and classifier weights."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1)
        self.product = ProductUnitConv2d(2, 2, kernel_size=1)
        self.head = nn.Linear(2, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x)) + 0.5
        x = self.product(x).mean(dim=(2, 3))
        return self.head(x)


def manifold_args(momentum=0.95):
    return pc.parse_args([
        "--protocol", pc.MANIFOLD_PROTOCOL,
        "--landscape", "true",
        "--run-group", "test-manifold",
        "--stage", "test",
        "--device", "cpu",
        "--seed", "7",
        "--data-seed", "123",
        "--epochs", "3",
        "--batch-size", "2",
        "--num-workers", "0",
        "--train-examples", "4",
        "--test-examples", "2",
        "--learning-rate", "0.01",
        "--momentum", str(momentum),
        "--weight-decay", "0",
        "--milestones", "1,2",
        "--gamma", "0.1",
        "--aux-learning-rate", "0.001",
        "--manifold-scale", "1.0",
        "--dual-learning-rate", "0.01",
        "--dual-iterations", "10",
    ])


def paired_config():
    return {
        "protocol": pc.MANIFOLD_PROTOCOL,
        "seeds": [7],
        "data_seed_offset": 100000,
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 0.01,
        "momentums": [0.95, 0.0],
        "weight_decay": 0,
        "milestones": [1, 2],
        "gamma": 0.1,
        "device": "cpu",
        "num_workers": 0,
        "data_dir": "unused",
        "download": False,
        "train_examples": 4,
        "test_examples": 2,
        "secondary": True,
        "stage": "test",
        "run_group": "test-manifold",
        "landscape": True,
        "aux_learning_rate": 0.001,
        "manifold_scale": 1.0,
        "dual_learning_rate": 0.01,
        "dual_iterations": 10,
    }


def tiny_loaders():
    generator = torch.Generator().manual_seed(512)
    train_x = torch.rand(4, 1, 3, 3, generator=generator)
    train_y = torch.tensor([0, 1, 0, 1])
    test_x = torch.rand(2, 1, 3, 3, generator=generator)
    test_y = torch.tensor([1, 0])
    loaders = (
        DataLoader(TensorDataset(train_x, train_y), batch_size=2),
        DataLoader(TensorDataset(test_x, test_y), batch_size=2),
        {
            "train_examples": 4,
            "test_examples": 2,
            "dataset_digest": hashlib.sha256(train_x.numpy().tobytes()).hexdigest(),
            "order_seed": 1100010,
            "worker_seed": 2100010,
            "augmentation_seed": 3100010,
        },
    )
    probe = (train_x[:2].clone(), train_y[:2].clone())
    return loaders, probe


def test_pair_config_and_projected_initialization_pair_by_momentum():
    commands = pc.expand_config(paired_config())
    assert len(commands) == 2
    args_pair = [pc.parse_args(command[3:]) for command in commands]
    assert [args.momentum for args in args_pair] == [0.95, 0.0]

    metadata_pair = []
    bundles = []
    for args in args_pair:
        torch.manual_seed(1907)
        model = TinyPURe()
        unprojected_digest = parameter_digest(model)
        bundle, metadata = build_manifold_optimizer(model, args)
        metadata_pair.append(metadata)
        bundles.append((bundle, model))

        product_weight = model.product.weight
        primary_ids = {
            id(parameter)
            for group in bundle.primary.param_groups
            for parameter in group["params"]
        }
        auxiliary_ids = {
            id(parameter)
            for group in bundle.auxiliary.param_groups
            for parameter in group["params"]
        }
        assert primary_ids == {id(product_weight)}
        assert id(model.conv.weight) in auxiliary_ids
        assert id(model.head.weight) in auxiliary_ids
        assert id(model.product.theta) in auxiliary_ids
        assert primary_ids.isdisjoint(auxiliary_ids)
        assert metadata["manifold_parameter_names"] == ["product.weight"]
        assert metadata["manifold_matrix_shapes"] == {"product.weight": [2, 2]}
        assert metadata["unprojected_initialization_digest"] == unprojected_digest
        assert metadata["unprojected_initialization_digest"] != metadata["initialization_digest"]
        assert metadata["initialization_digest"] == parameter_digest(model)
        constraints = bundle.constraint_metrics()
        assert constraints["manifold/constraint/product.weight/rms"] < 1e-5
        assert constraints["manifold/constraint/product.weight/max_abs"] < 1e-5

    for key in ("unprojected_initialization_digest", "initialization_digest"):
        assert metadata_pair[0][key] == metadata_pair[1][key]
    assert metadata_pair[0]["unprojected_initialization_digest"] != metadata_pair[0]["initialization_digest"]

    zero_momentum_bundle, zero_momentum_model = bundles[1]
    zero_momentum_bundle.zero_grad(set_to_none=True)
    zero_momentum_model.product.weight.grad = torch.ones_like(zero_momentum_model.product.weight)
    zero_momentum_bundle.step()
    assert "momentum_buffer" not in zero_momentum_bundle.primary.state[zero_momentum_model.product.weight]


def test_runner_logs_both_manifold_learning_rate_schedules_at_milestones():
    args = manifold_args(momentum=0.95)
    loaders, probe = tiny_loaders()
    records = []
    unprojected_digests = []

    def model_factory():
        model = TinyPURe()
        unprojected_digests.append(parameter_digest(model))
        return model

    result = pc.run_trial(
        args,
        on_epoch=records.append,
        loaders=loaders,
        model_factory=model_factory,
        landscape_probe=probe,
    )

    assert result["terminal_outcome"] == "completed"
    assert result["epochs_completed"] == 3
    assert result["test_examples_evaluated"] == 2
    epochs = [record for record in records if record["epoch"] > 0]
    assert [record["learning_rate"] for record in epochs] == pytest.approx([0.01, 0.001, 0.0001])
    assert [record["auxiliary_learning_rate"] for record in epochs] == pytest.approx(
        [0.001, 0.0001, 0.00001]
    )
    assert all(record["manifold/minibatch_count"] == 2 for record in epochs)
    assert all(record["manifold/dual_measurement_count"] == 2 for record in epochs)
    assert result["unprojected_initialization_digest"] == unprojected_digests[0]
    assert result["initialization_digest"] != unprojected_digests[0]


def test_svd_failure_is_terminal_and_skips_final_test_evaluation(monkeypatch):
    args = manifold_args(momentum=0.0)
    loaders, probe = tiny_loaders()
    original_svd = manifold_muon.matrix_sign_svd
    call_count = 0

    def fail_during_optimizer_step(matrix):
        nonlocal call_count
        call_count += 1
        # Initial projection succeeds; the first DA-10 iterate then fails.
        if call_count == 2:
            raise torch.linalg.LinAlgError("injected SVD breakdown")
        return original_svd(matrix)

    monkeypatch.setattr(manifold_muon, "matrix_sign_svd", fail_during_optimizer_step)
    monkeypatch.setattr(pc, "evaluate", lambda *args, **kwargs: pytest.fail("test evaluation ran after SVD failure"))
    records = []
    result = pc.run_trial(
        args,
        on_epoch=records.append,
        loaders=loaders,
        model_factory=TinyPURe,
        landscape_probe=probe,
    )

    assert result["terminal_outcome"] == "numerical_failure"
    assert result["numerical_failure"]
    assert result["failure_reason"] == "manifold_optimizer_numerical_failure"
    assert result["optimizer_failure_detail"] == "injected SVD breakdown"
    assert result["failed_epoch"] == 1
    assert result["epochs_completed"] == 0
    assert "test_examples_evaluated" not in result
    assert records[-1]["landscape/failure/reason"] == "manifold_optimizer_numerical_failure"

