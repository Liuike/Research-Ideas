from __future__ import annotations

import math

import pytest
import torch

from optimizer_resurrection.historical_sequence_tasks import (
    PROTOCOLS,
    FixedSequenceDataset,
    Parity,
    TwoSequence,
    dataset_digest,
    generate_fixed_dataset,
)


def test_protocols_and_fixed_dataset_shapes_are_explicit():
    assert PROTOCOLS == ("bsf1994-two-sequence-v1", "bsf1994-parity-v1")
    for protocol in PROTOCOLS:
        dataset = generate_fixed_dataset(protocol, 20, 11, 0.2, 7, "train")
        assert isinstance(dataset, FixedSequenceDataset)
        assert dataset.labels.shape == (20,)
        assert dataset.lengths.shape == (20,)
        assert dataset.inputs.shape == (20, 11)
        assert dataset.labels.dtype == torch.long
        assert dataset.lengths.dtype == torch.long
        assert dataset.inputs.dtype == torch.float32
        assert torch.bincount(dataset.labels, minlength=2).tolist() == [10, 10]
        assert int(dataset.lengths.min()) >= 6
        assert int(dataset.lengths.max()) <= 11
        assert torch.count_nonzero(dataset.inputs[torch.arange(11) >= dataset.lengths[:, None]]) == 0


def test_two_sequence_pairing_and_templates_are_reproducible():
    clean_train = generate_fixed_dataset(PROTOCOLS[0], 30, 12, 0.0, 19, "train")
    noisy_train = generate_fixed_dataset(PROTOCOLS[0], 30, 12, 0.2, 19, "train")
    repeat = generate_fixed_dataset(PROTOCOLS[0], 30, 12, 0.0, 19, "train")
    clean_validation = generate_fixed_dataset(PROTOCOLS[0], 30, 12, 0.0, 19, "validation")
    assert torch.equal(clean_train.labels, noisy_train.labels)
    assert torch.equal(clean_train.lengths, noisy_train.lengths)
    assert torch.equal(clean_train.inputs, repeat.inputs)
    assert torch.count_nonzero(clean_train.inputs) > 0
    # Same templates are used for both splits, so each class's clean prefix is
    # identical even though labels and lengths are independently sampled.
    for label in (0, 1):
        train_row = clean_train.inputs[clean_train.labels == label][0]
        validation_row = clean_validation.inputs[clean_validation.labels == label][0]
        common = min(
            int(clean_train.lengths[clean_train.labels == label][0]),
            int(clean_validation.lengths[clean_validation.labels == label][0]),
        )
        assert torch.equal(train_row[:common], validation_row[:common])


def test_parity_targets_are_clean_prefix_parities_and_noise_does_not_change_them():
    clean = generate_fixed_dataset(PROTOCOLS[1], 100, 15, 0.0, 3, "train")
    noisy = generate_fixed_dataset(PROTOCOLS[1], 100, 15, 0.2, 3, "train")
    active = torch.arange(15).unsqueeze(0) < clean.lengths.unsqueeze(1)
    clean_bits = (clean.inputs > 0).long()
    parity = (clean_bits * active).sum(dim=1) % 2
    assert torch.equal(parity, clean.labels)
    assert torch.equal(clean.labels, noisy.labels)
    assert torch.equal(clean.lengths, noisy.lengths)
    assert torch.count_nonzero(noisy.inputs - clean.inputs) > 0
    assert torch.count_nonzero(noisy.inputs[~active]) == 0


def test_digests_are_stable_and_sensitive_to_exact_data():
    first = generate_fixed_dataset(PROTOCOLS[0], 20, 10, 0.2, 9, "train")
    second = generate_fixed_dataset(PROTOCOLS[0], 20, 10, 0.2, 9, "train")
    third = generate_fixed_dataset(PROTOCOLS[0], 20, 10, 0.2, 10, "train")
    assert dataset_digest(first) == dataset_digest(second)
    assert dataset_digest(first) != dataset_digest(third)


def test_two_sequence_recurrence_and_trace_match_hand_calculation():
    model = TwoSequence(seed=0)
    with torch.no_grad():
        model.W.zero_()
        model.W[4, 0] = 0.7
        model.W[4, 4] = 0.2
    inputs = torch.tensor([[0.3, -0.1, 0.4], [0.2, 9.0, 9.0]])
    lengths = torch.tensor([3, 1])
    outputs, states = model(inputs, lengths, return_states=True)
    first = math.tanh(0.7 * math.tanh(0.3))
    first = math.tanh(0.7 * math.tanh(-0.1) + 0.2 * first)
    assert float(outputs[0].detach()) == pytest.approx(first)
    assert float(outputs[1].detach()) == pytest.approx(0.0)
    assert states.shape == (2, 3, 5)
    assert torch.equal(states[1, 1], states[1, 2])
    trace = model.trace(inputs[:1])
    assert len(trace) == 4
    assert trace[0].shape == (1, 5)
    assert bool(trace[0].requires_grad)
    torch.testing.assert_close(trace[-1][:, 4], outputs[:1])


def test_parity_wiring_and_finite_difference_gradient():
    model = Parity(seed=1)
    assert sum(parameter.numel() for parameter in model.parameters()) == 7
    assert set(dict(model.named_parameters())) == set("abcdefg")
    with torch.no_grad():
        for name, value in zip("abcdefg", (0.2, 0.3, 0.1, 0.4, 0.5, 0.6, 0.05)):
            getattr(model, name).fill_(value)
    inputs = torch.tensor([[1.0, -1.0, 0.5]], requires_grad=True)
    lengths = torch.tensor([3])
    output = model(inputs, lengths)
    trace = model.trace(inputs)
    assert len(trace) == 4
    assert trace[0].shape == (1, 2)
    assert bool(trace[0].requires_grad)
    output.square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert torch.isfinite(inputs.grad).all()

    expected = torch.zeros((), dtype=torch.float64)
    previous = expected
    for value in (0.2, -0.3, 0.4):
        hidden = torch.tanh(torch.tensor(0.2, dtype=torch.float64) * value
                            + torch.tensor(0.3, dtype=torch.float64) * previous
                            + torch.tensor(0.1, dtype=torch.float64))
        previous = torch.tanh(torch.tensor(0.4, dtype=torch.float64) * value
                              + torch.tensor(0.5, dtype=torch.float64) * previous
                              + torch.tensor(0.6, dtype=torch.float64) * hidden
                              + torch.tensor(0.05, dtype=torch.float64))
    parity_inputs = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32)
    parity_output, parity_states = model(parity_inputs, torch.tensor([3]), return_states=True)
    assert float(parity_output.detach()) == pytest.approx(float(previous), rel=1e-6, abs=1e-6)
    torch.testing.assert_close(model.trace(parity_inputs)[-1][:, 1], parity_output)
    assert parity_states.shape == (1, 3, 2)


def _central_difference(model, inputs, lengths, target, parameter, index=None, epsilon=1e-6):
    def objective():
        output = model(inputs, lengths)
        return 0.5 * (output - target).square().sum()

    with torch.no_grad():
        if index is None:
            parameter.add_(epsilon)
        else:
            parameter[index] += epsilon
    plus = float(objective().detach())
    with torch.no_grad():
        if index is None:
            parameter.sub_(2 * epsilon)
        else:
            parameter[index] -= 2 * epsilon
    minus = float(objective().detach())
    with torch.no_grad():
        if index is None:
            parameter.add_(epsilon)
        else:
            parameter[index] += epsilon
    return (plus - minus) / (2 * epsilon)


@pytest.mark.parametrize("model_type", [TwoSequence, Parity])
def test_all_model_and_input_gradients_match_float64_central_differences(model_type):
    model = model_type(seed=13).to(dtype=torch.float64)
    inputs = torch.tensor([[0.17, -0.23, 0.31]], dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([3])
    target = torch.tensor([0.19], dtype=torch.float64)
    loss = 0.5 * (model(inputs, lengths) - target).square().sum()
    loss.backward()

    for parameter in model.parameters():
        gradient = parameter.grad.detach().clone()
        if parameter.ndim == 0:
            numerical = _central_difference(model, inputs, lengths, target, parameter)
            assert float(gradient) == pytest.approx(numerical, rel=2e-5, abs=2e-7)
        else:
            for index in torch.cartesian_prod(*[torch.arange(size) for size in parameter.shape]):
                index = tuple(int(value) for value in index)
                numerical = _central_difference(model, inputs, lengths, target, parameter, index)
                assert float(gradient[index]) == pytest.approx(numerical, rel=2e-5, abs=2e-7)

    for index in range(inputs.numel()):
        numerical = _central_difference(model, inputs, lengths, target, inputs.reshape(-1), (index,))
        assert float(inputs.grad.reshape(-1)[index]) == pytest.approx(numerical, rel=2e-5, abs=2e-7)
