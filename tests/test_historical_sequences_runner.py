from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
import yaml
from optimizer_resurrection.experiment import expand_config
from optimizer_resurrection.historical_sequences import (
    parse_args, resolved_config, train, losses, temporal_diagnostics,
)
from optimizer_resurrection.historical_sequence_protocol import select_recipe
from optimizer_resurrection.historical_sequence_tasks import TwoSequence, Parity, generate_fixed_dataset
from optimizer_resurrection.tracking import RunRecord


def config(task, mode="smoke"):
    return yaml.safe_load(Path(f"configs/historical/{task}_{mode}.yaml").read_text())


@pytest.mark.parametrize("task", ["two_sequence", "parity"])
def test_config_mapping_and_strict_identity(task):
    c = config(task, "scout")
    commands = expand_config(c)
    assert len(commands) == 6
    resolved = [resolved_config(parse_args(command[3:])) for command in commands]
    assert len({r["condition_id"] for r in resolved}) == 6
    assert [(r["noise_amplitude"], r["learning_rate"]) for r in resolved] == [
        (noise, lr) for noise in (0.0, 0.2) for lr in (0.001, 0.01, 0.1)]
    args = parse_args(commands[0][3:])
    args.presentations += 1
    with pytest.raises(ValueError, match="condition_id"):
        resolved_config(args)
    for mutate in (lambda x: x.update(momentum=0.1),
                   lambda x: x.update(seeds=[1000,1000])):
        bad = deepcopy(c)
        mutate(bad)
        with pytest.raises(ValueError):
            expand_config(bad)


@pytest.mark.parametrize("task", ["two_sequence", "parity"])
def test_cpu_budget_and_finite_diagnostics(task):
    args = SimpleNamespace(**resolved_config(parse_args(expand_config(config(task))[0][3:])))
    history = []
    run = SimpleNamespace(log=lambda record, step: history.append((step, record)))
    result = train(args, device=torch.device("cpu"), wandb_run=run,
                   resolved_condition_id=args.condition_id)
    assert result.presentations == 10
    assert result.terminal_outcome == "completed"
    assert [step for step,_ in history] == [0,1,5,10]
    assert result.diagnostic_evaluation_sequences == 4 * (10+20+4)
    assert len(result.train_dataset_digest) == 64
    for _,record in history:
        for probe in record["temporal_gradients"].values():
            for dtype in ("fp32", "float64_reference"):
                values = probe[dtype]["input_gradient"]
                assert torch.isfinite(torch.tensor(values)).all()
                assert len(values) >= 5


@pytest.mark.parametrize("task,model_type", [("two_sequence",TwoSequence),("parity",Parity)])
def test_diagnostics_do_not_modify_model_or_rng(task, model_type):
    model = model_type(3)
    saved = deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    protocol = config(task)["protocol"]
    temporal_diagnostics(model, torch.tensor([0.2,-0.3,0.4]), 1, protocol)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v, saved[k]) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_objective_targets_differ_explicitly():
    zero = torch.tensor([0.0])
    label = torch.tensor([1])
    assert losses(zero,label,"bsf1994-parity-v1").item() == 0.5
    assert losses(zero,label,"bsf1994-two-sequence-v1").item() == pytest.approx(0.32)


@pytest.mark.parametrize("task,model_type", [("two_sequence",TwoSequence),("parity",Parity)])
def test_runner_performs_exact_plain_sgd_update(task, model_type):
    raw = parse_args(expand_config(config(task))[0][3:])
    raw.condition_id = None
    raw.presentations = 1
    args = SimpleNamespace(**resolved_config(raw))
    data = generate_fixed_dataset(args.protocol, args.train_size, args.max_length,
                                  args.noise_amplitude, args.data_seed, "train")
    model = model_type(args.seed)
    loss = losses(model(data.inputs[:1], data.lengths[:1]), data.labels[:1], args.protocol).mean()
    loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(parameter.grad, alpha=-args.learning_rate)
    history = []
    train(args, device=torch.device("cpu"),
          wandb_run=SimpleNamespace(log=lambda record,step: history.append(record)),
          resolved_condition_id=args.condition_id, train_data=data)
    for name, expected in model.parameter_values().items():
        torch.testing.assert_close(torch.tensor(history[-1]["parameters"][name]),
                                   torch.tensor(expected), rtol=0, atol=0)


@pytest.mark.parametrize("task", ["two_sequence", "parity"])
def test_selector_rejects_missing_duplicate_or_failed_cells(task):
    c = config(task, "scout")
    rows = []
    for index, command in enumerate(expand_config(c)):
        resolved = resolved_config(parse_args(command[3:]))
        rows.append(RunRecord(str(index), "finished", resolved, dict(
            terminal_outcome="completed", presentations=5000,
            train_loss=0.01, validation_loss=abs(resolved["learning_rate"]-0.01),
            train_accuracy=1.0, validation_accuracy=1.0), "https://example.invalid"))
    assert select_recipe(rows,c)["learning_rate"] == 0.01
    for bad in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError):
            select_recipe(bad,c)
    bad = deepcopy(rows)
    bad[0].summary["terminal_outcome"] = "failed"
    with pytest.raises(ValueError, match="incomplete"):
        select_recipe(bad,c)


def test_initialized_run_is_closed_if_provenance_update_fails(monkeypatch):
    import wandb
    import optimizer_resurrection.historical_sequences as runner
    finished = []
    def fail(*args, **kwargs):
        raise RuntimeError("simulated config service failure")
    run = SimpleNamespace(id="test", name="test", url="https://example.invalid",
        config=SimpleNamespace(update=fail), summary={},
        finish=lambda exit_code: finished.append(exit_code))
    monkeypatch.setattr(wandb,"init",lambda **kwargs: run)
    monkeypatch.setattr(runner,"load_wandb_credentials",
                        lambda *args: {"WANDB_PROJECT":"test","WANDB_ENTITY":"test"})
    monkeypatch.setattr(runner,"require_online_wandb",lambda: None)
    with pytest.raises(RuntimeError,match="simulated config"):
        runner.main(expand_config(config("parity"))[0][3:])
    assert finished == [1]
    assert run.summary["terminal_outcome"] == "failed"
