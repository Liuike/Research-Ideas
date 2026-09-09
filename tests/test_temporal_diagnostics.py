import torch

from optimizer_resurrection.diagnostics import temporal_jacobian_statistics


def test_temporal_diagnostics_use_each_sequence_length():
    weight = torch.eye(2)
    states = torch.zeros(2, 6, 2)
    # Padded states are deliberately saturated; they must not affect results.
    states[0, 3:] = 0.999
    lengths = torch.tensor([3, 6])
    stats = temporal_jacobian_statistics(weight, states, lengths, lags=(3,))
    assert stats["3"]["examples"] == 2
    assert abs(stats["3"]["spectral_norm"] - 1.0) < 1e-6

