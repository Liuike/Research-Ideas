"""Independent checks for the reconstructed Bengio--Simard--Frasconi Latch.

These tests intentionally exercise the historical module directly instead of
the generic RNN path.  That keeps the protocol checks sensitive to accidental
changes in recurrence, data pairing, and the presentation budget.
"""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from optimizer_resurrection.historical_latch import (
    HistoricalLatch,
    generate_fixed_dataset,
    parse_args,
    temporal_gradient_diagnostics,
    train,
)


def _loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = labels.to(dtype=outputs.dtype).mul(2).sub(1).mul(0.8)
    return 0.5 * (outputs - targets).square().mean()


def test_historical_latch_has_three_parameters_and_matches_hand_recurrence():
    model = HistoricalLatch(seed=4)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3
    assert {name for name, _ in model.named_parameters()} == {
        "w",
        "u_negative",
        "u_positive",
    }

    with torch.no_grad():
        model.w.fill_(0.7)
        model.u_negative.fill_(-0.3)
        model.u_positive.fill_(0.4)
    labels = torch.tensor([0, 1])
    lengths = torch.tensor([3, 3])
    later_inputs = torch.tensor([[0.1, -0.2], [0.1, -0.2]])
    output, states = model(labels, lengths, later_inputs, return_states=True)

    expected = []
    for label in labels.tolist():
        state = torch.tanh(torch.tensor(-0.3 if label == 0 else 0.4))
        sequence = [state]
        for value in (0.1, -0.2):
            state = torch.tanh(torch.tensor(0.7) * state + value)
            sequence.append(state)
        expected.append(torch.stack(sequence))
    expected_states = torch.stack(expected)
    assert torch.allclose(states, expected_states, atol=1e-6, rtol=1e-6)
    assert torch.allclose(output, expected_states[:, -1], atol=1e-6, rtol=1e-6)


def test_fixed_dataset_is_balanced_reusable_and_noise_conditions_are_paired():
    off = generate_fixed_dataset(
        size=20, max_length=10, noise_amplitude=0.0, data_seed=17, split="train"
    )
    on = generate_fixed_dataset(
        size=20, max_length=10, noise_amplitude=0.2, data_seed=17, split="train"
    )
    repeat = generate_fixed_dataset(
        size=20, max_length=10, noise_amplitude=0.0, data_seed=17, split="train"
    )

    assert torch.bincount(off.labels, minlength=2).tolist() == [10, 10]
    assert torch.equal(off.labels, repeat.labels)
    assert torch.equal(off.lengths, repeat.lengths)
    assert torch.equal(off.later_inputs, repeat.later_inputs)
    assert torch.equal(off.labels, on.labels)
    assert torch.equal(off.lengths, on.lengths)
    assert torch.count_nonzero(off.later_inputs) == 0
    assert torch.count_nonzero(on.later_inputs) > 0


def test_noisy_latch_inputs_are_uniformly_bounded_and_padding_is_zero():
    data = generate_fixed_dataset(
        size=200, max_length=12, noise_amplitude=0.2, data_seed=23, split="validation"
    )
    valid = torch.arange(1, 12).unsqueeze(0) < data.lengths.unsqueeze(1)
    valid_values = data.later_inputs[valid]
    assert torch.all(valid_values >= -0.2)
    assert torch.all(valid_values <= 0.2)
    assert float(valid_values.abs().max()) > 0.15
    assert torch.count_nonzero(data.later_inputs[~valid]) == 0


def test_historical_initialization_is_seeded_and_stays_within_documented_bounds():
    first = HistoricalLatch(seed=31)
    second = HistoricalLatch(seed=31)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    values = torch.cat([parameter.detach().reshape(-1) for parameter in first.parameters()])
    assert torch.all(values >= -0.5)
    assert torch.all(values <= 0.5)
    assert not torch.equal(
        values,
        torch.cat([parameter.detach().reshape(-1) for parameter in HistoricalLatch(seed=32).parameters()]),
    )


def test_temporal_gradients_agree_with_central_finite_differences():
    model = HistoricalLatch(seed=8)
    labels = torch.tensor([0, 1])
    lengths = torch.tensor([4, 4])
    later_inputs = torch.tensor(
        [[0.05, -0.07, 0.11], [-0.04, 0.08, -0.09]], dtype=torch.float32
    )
    outputs = model(labels, lengths, later_inputs)
    objective = _loss(outputs, labels)
    objective.backward()
    analytic = {name: parameter.grad.item() for name, parameter in model.named_parameters()}

    epsilon = 1e-3
    for name, parameter in model.named_parameters():
        with torch.no_grad():
            parameter.add_(epsilon)
        plus = _loss(model(labels, lengths, later_inputs), labels).item()
        with torch.no_grad():
            parameter.sub_(2 * epsilon)
        minus = _loss(model(labels, lengths, later_inputs), labels).item()
        with torch.no_grad():
            parameter.add_(epsilon)
        numerical = (plus - minus) / (2 * epsilon)
        assert analytic[name] == pytest.approx(numerical, rel=2e-3, abs=2e-4)

    diagnostic = temporal_gradient_diagnostics(
        model, label=1, later_inputs=later_inputs[1]
    )
    assert len(diagnostic["fp32"]["input_gradient"]) == later_inputs.shape[1] + 1
    assert len(diagnostic["fp32"]["state_gradient"]) == later_inputs.shape[1] + 2


def test_plain_sgd_single_step_is_exact_gradient_descent():
    model = HistoricalLatch(seed=2)
    labels = torch.tensor([1])
    lengths = torch.tensor([3])
    later_inputs = torch.tensor([[0.06, -0.02]])
    learning_rate = 0.037
    objective = _loss(model(labels, lengths, later_inputs), labels)
    objective.backward()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0)
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert torch.allclose(
            parameter,
            before[name] - learning_rate * gradients[name],
            atol=1e-7,
            rtol=1e-6,
        )


class _MockRun:
    def __init__(self) -> None:
        self.records: list[tuple[int | None, dict[str, object]]] = []

    def log(self, record: dict[str, object], step: int | None = None) -> None:
        self.records.append((step, record))


def test_train_honors_5000_sequence_presentation_budget_without_online_logging():
    args = parse_args(
        [
            "--protocol",
            "bsf1994-latch-v1",
            "--max-length",
            "2",
            "--noise-amplitude",
            "0.0",
            "--seed",
            "3",
            "--data-seed",
            "3",
            "--learning-rate",
            "0.01",
            "--presentations",
            "5000",
            "--train-size",
            "2",
            "--validation-size",
            "2",
            "--log-every",
            "1000",
            "--device",
            "cpu",
            "--stage",
            "engineering-smoke",
        ]
    )
    run = _MockRun()
    result = train(
        args,
        device=torch.device("cpu"),
        wandb_run=run,
        resolved_condition_id="test-condition",
    )
    assert result.presentations == 5000
    assert [step for step, _ in run.records] == [0, 1, 1000, 2000, 3000, 4000, 5000]
    assert run.records[0][1]["presentation"] == 0
    assert result.diagnostic_evaluation_sequences == 7 * (2 + 2 + 4)
    assert run.records[-1][1]["presentation"] == 5000
    assert result.terminal_outcome == "completed"
