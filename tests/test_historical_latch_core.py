import argparse
import math

import pytest
import torch

from optimizer_resurrection.historical_latch import (
    HistoricalLatch,
    condition_id,
    generate_fixed_dataset,
    parse_args,
    resolved_config,
    temporal_gradient_diagnostics,
)


def test_model_has_exact_historical_parameterization_and_initialization():
    first = HistoricalLatch(seed=7)
    second = HistoricalLatch(seed=7)
    parameters = dict(first.named_parameters())
    assert set(parameters) == {"w", "u_negative", "u_positive"}
    assert sum(parameter.numel() for parameter in parameters.values()) == 3
    assert all(parameter.dtype == torch.float32 for parameter in parameters.values())
    assert all(-0.5 <= float(parameter.detach()) <= 0.5 for parameter in parameters.values())
    assert first.parameter_values() == second.parameter_values()


def test_recurrence_matches_scalar_definition_with_variable_lengths():
    model = HistoricalLatch(seed=0)
    with torch.no_grad():
        model.w.fill_(0.7)
        model.u_negative.fill_(-0.4)
        model.u_positive.fill_(0.3)
    labels = torch.tensor([0, 1])
    lengths = torch.tensor([2, 4])
    noise = torch.tensor([[0.1, 99.0, 99.0], [-0.1, 0.2, 0.05]])
    outputs, states = model(labels, lengths, noise, return_states=True)

    negative_1 = math.tanh(-0.4)
    negative_2 = math.tanh(0.7 * negative_1 + 0.1)
    positive = math.tanh(0.3)
    for value in (-0.1, 0.2, 0.05):
        positive = math.tanh(0.7 * positive + value)
    torch.testing.assert_close(outputs, torch.tensor([negative_2, positive]))
    assert states.shape == (2, 4)
    assert float(states[0, -1].detach()) == pytest.approx(negative_2)


def test_fixed_data_are_balanced_bounded_and_paired_across_noise_conditions():
    clean = generate_fixed_dataset(
        size=100,
        max_length=11,
        noise_amplitude=0.0,
        data_seed=3,
        split="train",
    )
    noisy = generate_fixed_dataset(
        size=100,
        max_length=11,
        noise_amplitude=0.2,
        data_seed=3,
        split="train",
    )
    repeat = generate_fixed_dataset(
        size=100,
        max_length=11,
        noise_amplitude=0.2,
        data_seed=3,
        split="train",
    )
    assert int(clean.labels.sum()) == 50
    assert int(clean.lengths.min()) >= 6
    assert int(clean.lengths.max()) <= 11
    assert torch.equal(clean.labels, noisy.labels)
    assert torch.equal(clean.lengths, noisy.lengths)
    assert torch.count_nonzero(clean.later_inputs) == 0
    assert float(noisy.later_inputs.abs().max()) <= 0.2
    assert torch.equal(noisy.later_inputs, repeat.later_inputs)
    assert torch.equal(noisy.labels, repeat.labels)


def test_full_span_gradients_match_finite_differences():
    model = HistoricalLatch(seed=1)
    with torch.no_grad():
        model.w.fill_(0.8)
        model.u_positive.fill_(0.15)
    later = torch.tensor([0.04, -0.03, 0.02], dtype=torch.float32)
    diagnostics = temporal_gradient_diagnostics(model, label=1, later_inputs=later)
    gradients = diagnostics["fp32"]["input_gradient"]
    assert len(gradients) == 4
    assert len(diagnostics["fp32"]["state_gradient"]) == 5

    inputs = torch.cat((model.u_positive.detach().reshape(1), later)).double()

    def loss_at(values):
        state = torch.zeros((), dtype=torch.float64)
        for value in values:
            state = torch.tanh(model.w.detach().double() * state + value)
        return 0.5 * (state - 0.8) ** 2

    epsilon = 1e-6
    finite_difference = []
    for index in range(inputs.numel()):
        plus, minus = inputs.clone(), inputs.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference.append(float((loss_at(plus) - loss_at(minus)) / (2 * epsilon)))
    assert gradients == pytest.approx(finite_difference, rel=2e-4, abs=2e-6)


def test_plain_sgd_one_step_matches_manual_update():
    model = HistoricalLatch(seed=4)
    labels = torch.tensor([1])
    lengths = torch.tensor([3])
    later = torch.tensor([[0.1, -0.2]])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.03, momentum=0.0, weight_decay=0.0
    )
    output = model(labels, lengths, later)
    loss = 0.5 * (output - 0.8).square().mean()
    loss.backward()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name] - 0.03 * gradients[name])


def test_condition_id_uses_scientific_config_and_assertion_is_strict():
    argv = [
        "--protocol", "bsf1994-latch-v1",
        "--max-length", "10",
        "--noise-amplitude", "0.2",
        "--seed", "0",
        "--data-seed", "0",
        "--learning-rate", "0.01",
    ]
    first = resolved_config(parse_args([*argv, "--run-name", "attempt-a"]))
    second = resolved_config(parse_args([*argv, "--run-name", "attempt-b"]))
    assert first["condition_id"] == second["condition_id"]
    assert condition_id(first) == first["condition_id"]

    parsed = parse_args([*argv, "--condition-id", "incorrect"])
    with pytest.raises(ValueError, match="does not match"):
        resolved_config(parsed)


def test_unknown_or_nonhistorical_training_controls_are_rejected():
    base = [
        "--protocol", "bsf1994-latch-v1",
        "--max-length", "10",
        "--noise-amplitude", "0",
        "--seed", "0",
        "--data-seed", "0",
        "--learning-rate", "0.01",
    ]
    with pytest.raises(SystemExit):
        parse_args([*base, "--momentum", "0.9"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--protocol", "other"])
