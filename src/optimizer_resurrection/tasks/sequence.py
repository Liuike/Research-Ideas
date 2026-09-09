from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SequenceBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    lengths: torch.Tensor


def _rand(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, generator=generator)


def make_sequence_batch(
    task: str,
    batch_size: int,
    max_length: int,
    generator: torch.Generator,
    variable_length: bool = True,
) -> SequenceBatch:
    """Generate Latch, 2-Sequence, or Parity with padded variable lengths."""
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    low = (max_length + 1) // 2 if variable_length else max_length
    lengths = torch.randint(low, max_length + 1, (batch_size,), generator=generator)

    if task == "latch":
        targets = torch.randint(0, 2, (batch_size,), generator=generator)
        x = torch.randn(batch_size, max_length, 2, generator=generator) * 0.2
        x[:, 0, 0] = targets.float().mul(2).sub(1)
        x[:, 0, 1] = 1.0
    elif task == "two_sequence":
        targets = torch.randint(0, 2, (batch_size,), generator=generator)
        x = torch.randn(batch_size, max_length, 1, generator=generator) * (0.2**0.5)
        # Opposite two-symbol prefixes; everything after the prefix is noise.
        signs = targets.float().mul(2).sub(1)
        x[:, 0, 0] = signs
        x[:, 1, 0] = -signs
    elif task == "parity":
        x = torch.zeros(batch_size, max_length, 1)
        targets = torch.empty(batch_size, dtype=torch.long)
        for i, length in enumerate(lengths.tolist()):
            bits = torch.randint(0, 2, (length,), generator=generator)
            x[i, :length, 0] = bits.float().mul(2).sub(1)
            targets[i] = int(bits.sum().item() % 2)
    else:
        raise ValueError(f"unknown sequence task: {task}")

    valid = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)
    x = x * valid.unsqueeze(2)
    return SequenceBatch(x, targets, lengths)

