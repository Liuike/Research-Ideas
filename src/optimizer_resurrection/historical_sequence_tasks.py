"""Small fixed-data reconstructions of the historical sequence benchmarks.

The historical source identifies the Two-Sequence task as a five-unit tanh
recurrent system with one input-to-state connection and the Parity task as a
two-unit, seven-parameter recurrent graph.  It does not fully specify the
input templates or every wiring detail, so this module makes those choices
explicit instead of presenting them as recovered historical settings.

Both generators produce a fixed trial dataset.  Labels and lengths are paired
across noise conditions, while noise is an independent stream.  Two-Sequence
uses two fixed uniform templates, shared by the train and validation splits;
the class chooses which template is used.  Parity creates balanced labels by
choosing the target parity and flipping the final active clean bit when the
sampled bits do not match it.  Padded inputs are always zero.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


PROTOCOLS = ("bsf1994-two-sequence-v1", "bsf1994-parity-v1")
_HIDDEN_SIZE = 5


@dataclass(frozen=True)
class FixedSequenceDataset:
    """A fixed, padded binary sequence dataset for one historical trial."""

    labels: torch.Tensor
    lengths: torch.Tensor
    inputs: torch.Tensor

    def __post_init__(self) -> None:
        size = int(self.labels.numel())
        if self.labels.dtype != torch.long or self.labels.shape != (size,):
            raise ValueError("labels must be one-dimensional torch.long")
        if self.lengths.dtype != torch.long or self.lengths.shape != (size,):
            raise ValueError("lengths must be one-dimensional torch.long and match labels")
        if self.inputs.dtype != torch.float32 or self.inputs.ndim != 2:
            raise ValueError("inputs must be a two-dimensional torch.float32 tensor")
        if self.inputs.shape[0] != size or size <= 0 or self.inputs.shape[1] < 1:
            raise ValueError("inputs must have shape [examples, max_length]")
        if not bool(torch.isfinite(self.inputs).all()):
            raise ValueError("inputs must be finite")
        if bool((self.labels < 0).any()) or bool((self.labels > 1).any()):
            raise ValueError("labels must be binary")
        if bool((self.lengths < 1).any()) or bool((self.lengths > self.inputs.shape[1]).any()):
            raise ValueError("lengths must lie within the represented sequence span")

    def __len__(self) -> int:
        return int(self.labels.numel())


def _validate_protocol(protocol: str) -> None:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported historical sequence protocol: {protocol!r}")


def _stream_seed(protocol: str, data_seed: int, split: str, purpose: str) -> int:
    """Return a domain-separated seed without consuming global RNG state."""
    payload = f"{protocol}:{data_seed}:{split}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _balanced_labels(size: int, generator: torch.Generator) -> torch.Tensor:
    if size <= 0 or size % 2:
        raise ValueError("dataset size must be a positive even integer for exact balance")
    labels = torch.cat(
        (torch.zeros(size // 2, dtype=torch.long), torch.ones(size // 2, dtype=torch.long))
    )
    return labels[torch.randperm(size, generator=generator)]


def _lengths(size: int, max_length: int, generator: torch.Generator) -> torch.Tensor:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    minimum_length = (max_length + 1) // 2
    return torch.randint(minimum_length, max_length + 1, (size,), generator=generator)


def _add_noise(
    clean: torch.Tensor,
    lengths: torch.Tensor,
    noise_amplitude: float,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.empty(clean.shape, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    result = clean + float(noise_amplitude) * noise
    valid = torch.arange(clean.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return result.masked_fill(~valid, 0.0).to(dtype=torch.float32)


def _generate_two_sequence(
    *,
    size: int,
    max_length: int,
    noise_amplitude: float,
    data_seed: int,
    split: str,
) -> FixedSequenceDataset:
    # The templates deliberately omit the split from their stream key.  The
    # same two templates therefore define both train and validation trials.
    label_generator = torch.Generator().manual_seed(
        _stream_seed(PROTOCOLS[0], data_seed, split, "labels")
    )
    length_generator = torch.Generator().manual_seed(
        _stream_seed(PROTOCOLS[0], data_seed, split, "lengths")
    )
    template_generator = torch.Generator().manual_seed(
        _stream_seed(PROTOCOLS[0], data_seed, "shared", "templates")
    )
    noise_generator = torch.Generator().manual_seed(
        _stream_seed(PROTOCOLS[0], data_seed, split, "noise")
    )
    labels = _balanced_labels(size, label_generator)
    lengths = _lengths(size, max_length, length_generator)
    templates = torch.empty(2, max_length, dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=template_generator
    )
    clean = templates[labels]
    inputs = _add_noise(clean, lengths, noise_amplitude, noise_generator)
    return FixedSequenceDataset(labels=labels, lengths=lengths, inputs=inputs)


def _generate_parity(
    *,
    size: int,
    max_length: int,
    noise_amplitude: float,
    data_seed: int,
    split: str,
) -> FixedSequenceDataset:
    label_generator = torch.Generator().manual_seed(
        _stream_seed(PROTOCOLS[1], data_seed, split, "labels")
    )
    length_generator = torch.Generator().manual_seed(
        _stream_seed(PROTOCOLS[1], data_seed, split, "lengths")
    )
    template_generator = torch.Generator().manual_seed(
        _stream_seed(PROTOCOLS[1], data_seed, split, "templates")
    )
    noise_generator = torch.Generator().manual_seed(
        _stream_seed(PROTOCOLS[1], data_seed, split, "noise")
    )
    labels = _balanced_labels(size, label_generator)
    lengths = _lengths(size, max_length, length_generator)

    # Sample clean +/-1 symbols, then force the active prefix parity to equal
    # the already-shuffled target label.  Noise is added only after this step.
    bits = torch.randint(0, 2, (size, max_length), generator=template_generator)
    for index, length in enumerate(lengths.tolist()):
        parity = int(bits[index, :length].sum().item() % 2)
        if parity != int(labels[index]):
            bits[index, length - 1] = 1 - bits[index, length - 1]
    clean = bits.to(dtype=torch.float32).mul(2.0).sub(1.0)
    inputs = _add_noise(clean, lengths, noise_amplitude, noise_generator)
    return FixedSequenceDataset(labels=labels, lengths=lengths, inputs=inputs)


def generate_fixed_dataset(
    protocol: str,
    size: int,
    max_length: int,
    noise_amplitude: float,
    data_seed: int,
    split: str,
) -> FixedSequenceDataset:
    """Generate one deterministic fixed trial for a registered protocol."""
    _validate_protocol(protocol)
    if not math.isfinite(noise_amplitude) or noise_amplitude < 0:
        raise ValueError("noise_amplitude must be finite and nonnegative")
    if not split:
        raise ValueError("split must be nonempty")
    if protocol == PROTOCOLS[0]:
        return _generate_two_sequence(
            size=size,
            max_length=max_length,
            noise_amplitude=noise_amplitude,
            data_seed=data_seed,
            split=split,
        )
    return _generate_parity(
        size=size,
        max_length=max_length,
        noise_amplitude=noise_amplitude,
        data_seed=data_seed,
        split=split,
    )


def dataset_digest(dataset: FixedSequenceDataset) -> str:
    """Hash the exact tensors used by a trial for provenance."""
    digest = hashlib.sha256()
    for name, tensor in (
        ("labels", dataset.labels),
        ("lengths", dataset.lengths),
        ("inputs", dataset.inputs),
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _as_batch(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] < 1:
        raise ValueError("inputs must have shape [batch, max_length]")
    return inputs


def _validate_forward_inputs(inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = _as_batch(inputs)
    if lengths.ndim != 1 or lengths.shape[0] != inputs.shape[0]:
        raise ValueError("lengths must have shape [batch] matching inputs")
    if bool((lengths < 1).any()) or bool((lengths > inputs.shape[1]).any()):
        raise ValueError("lengths must lie within the represented sequence span")
    return inputs, lengths


class TwoSequence(nn.Module):
    """Five-unit tanh recurrence with one input connection and 25 weights."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.empty(_HIDDEN_SIZE, _HIDDEN_SIZE, dtype=torch.float32).uniform_(
            -0.5, 0.5, generator=generator
        )
        self.W = nn.Parameter(initial)

    def trace(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Return the initial zero state and every unpadded recurrent state."""
        inputs = _as_batch(inputs).to(device=self.W.device, dtype=self.W.dtype)
        state = torch.zeros(
            inputs.shape[0], _HIDDEN_SIZE, device=self.W.device, dtype=self.W.dtype,
            requires_grad=True,
        )
        states = [state]
        for step in range(inputs.shape[1]):
            drive = torch.cat((inputs[:, step : step + 1], torch.zeros_like(inputs[:, step : step + 1]).expand(-1, 4)), dim=1)
            state = torch.tanh(state @ self.W.transpose(0, 1) + drive)
            states.append(state)
        return states

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        inputs, lengths = _validate_forward_inputs(inputs, lengths)
        inputs = inputs.to(device=self.W.device, dtype=self.W.dtype)
        lengths = lengths.to(device=self.W.device)
        state = torch.zeros(inputs.shape[0], _HIDDEN_SIZE, device=self.W.device, dtype=self.W.dtype)
        states: list[torch.Tensor] = []
        for step in range(inputs.shape[1]):
            drive = torch.cat((inputs[:, step : step + 1], torch.zeros_like(inputs[:, step : step + 1]).expand(-1, 4)), dim=1)
            proposed = torch.tanh(state @ self.W.transpose(0, 1) + drive)
            state = torch.where((lengths > step).unsqueeze(1), proposed, state)
            states.append(state)
        output = state[:, 4]
        if return_states:
            return output, torch.stack(states, dim=1)
        return output

    def parameter_values(self) -> dict[str, Any]:
        return {"W": self.W.detach().cpu().tolist()}


class Parity(nn.Module):
    """Seven-parameter two-unit tanh graph used for the parity reconstruction."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.empty(7, dtype=torch.float32).uniform_(-0.5, 0.5, generator=generator)
        self.a, self.b, self.c, self.d, self.e, self.f, self.g = (
            nn.Parameter(value.clone()) for value in initial
        )

    def trace(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Return the initial zero state and every unpadded recurrent state."""
        inputs = _as_batch(inputs).to(device=self.a.device, dtype=self.a.dtype)
        state = torch.zeros(inputs.shape[0], 2, device=self.a.device, dtype=self.a.dtype, requires_grad=True)
        states = [state]
        for step in range(inputs.shape[1]):
            value = inputs[:, step]
            hidden = torch.tanh(self.a * value + self.b * state[:, 1] + self.c)
            output = torch.tanh(self.d * value + self.e * state[:, 1] + self.f * hidden + self.g)
            state = torch.stack((hidden, output), dim=1)
            states.append(state)
        return states

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        inputs, lengths = _validate_forward_inputs(inputs, lengths)
        inputs = inputs.to(device=self.a.device, dtype=self.a.dtype)
        lengths = lengths.to(device=self.a.device)
        state = torch.zeros(inputs.shape[0], 2, device=self.a.device, dtype=self.a.dtype)
        states: list[torch.Tensor] = []
        for step in range(inputs.shape[1]):
            value = inputs[:, step]
            hidden = torch.tanh(self.a * value + self.b * state[:, 1] + self.c)
            output = torch.tanh(self.d * value + self.e * state[:, 1] + self.f * hidden + self.g)
            proposed = torch.stack((hidden, output), dim=1)
            state = torch.where((lengths > step).unsqueeze(1), proposed, state)
            states.append(state)
        result = state[:, 1]
        if return_states:
            return result, torch.stack(states, dim=1)
        return result

    def parameter_values(self) -> dict[str, float]:
        return {name: float(getattr(self, name).detach()) for name in "abcdefg"}
