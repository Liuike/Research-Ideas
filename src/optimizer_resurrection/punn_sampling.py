"""Raw loss-landscape sampling for the repository's product-unit networks.

The returned walks and uniform point sets are deliberately raw.  This module
does not calculate paper-level fitness-landscape metrics or write artifacts.
In particular, its progressive-walk rule is an explicit reconstruction choice
because the 2024 PUNN study describes the bias direction but does not specify
all implementation details in the article text.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .models import ProductUnitNetwork


_PARAMETER_LAYOUT = ("exponents", "output.weight", "output.bias")
_PRW_TOWARD_PROBABILITY = 0.75


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed_value(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if seed >= 2**63:
        raise ValueError("seed must be less than 2**63")
    return seed


def _validate_model(model: nn.Module) -> tuple[int, int, int]:
    if not isinstance(model, ProductUnitNetwork):
        raise TypeError("model must be a ProductUnitNetwork")
    named_parameters = list(model.named_parameters())
    names = tuple(name for name, _ in named_parameters)
    if names != _PARAMETER_LAYOUT:
        raise ValueError(
            "model parameter order must be exponents, output.weight, output.bias"
        )
    if model.exponents.ndim != 2 or model.output.weight.ndim != 2:
        raise ValueError("model has an invalid product-unit parameter shape")
    hidden_units, input_dim = model.exponents.shape
    output_dim, output_hidden_units = model.output.weight.shape
    if output_hidden_units != hidden_units or model.output.bias.shape != (output_dim,):
        raise ValueError("model has inconsistent output parameter shapes")
    return input_dim, hidden_units, output_dim


def _device(device: str | torch.device) -> torch.device:
    try:
        target = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device must be a CPU or CUDA device") from exc
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("device must be a CPU or CUDA device")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return target


def _validate_data(
    model: ProductUnitNetwork,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_dim, _, output_dim = _validate_model(model)
    x = torch.as_tensor(train_x)
    y = torch.as_tensor(train_y)
    if x.ndim != 2 or x.shape[1] != input_dim or x.shape[0] == 0:
        raise ValueError(f"train_x must have shape [N, {input_dim}] with N > 0")
    if y.ndim != 2 or y.shape != (x.shape[0], output_dim):
        raise ValueError(f"train_y must have shape [N, {output_dim}]")
    if x.is_complex() or y.is_complex():
        raise ValueError("train data must be real-valued")
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("train data must be finite")
    if model.input_domain == "positive":
        if (x <= 0).any():
            raise ValueError("positive product units require strictly positive inputs")
    elif (x == 0).any():
        raise ValueError("real_complex product units require nonzero inputs")
    return x, y


def _validate_vectors(vectors: torch.Tensor, dimension: int) -> torch.Tensor:
    values = torch.as_tensor(vectors)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"vectors must have shape [N, {dimension}]")
    if values.is_complex():
        raise ValueError("parameter vectors must be real-valued")
    values = values.to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("parameter vectors must be finite")
    return values


def evaluate_vectors(
    model: ProductUnitNetwork,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    vectors: torch.Tensor,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> torch.Tensor:
    """Evaluate full-training-set MSE for parameter vectors in supplied order.

    Vectors follow ``torch.nn.utils.parameters_to_vector(model.parameters())``:
    flattened exponents, output weights, then output biases. ``batch_size`` is
    the number of candidate parameter vectors evaluated together. Data rows
    are streamed in chunks of at most 512. The result is a CPU float32 tensor;
    nonfinite model outputs remain nonfinite in the corresponding loss.

    The supplied model is never modified. Evaluation uses the network's exact
    positive-input or real-part-of-complex-power formula and the mean squared
    error over every supplied training pattern and output.
    """
    batch_size = _positive_int(batch_size, "batch_size")
    input_dim, hidden_units, output_dim = _validate_model(model)
    target_device = _device(device)
    x, y = _validate_data(model, train_x, train_y, target_device)

    pieces: list[torch.Tensor] = []
    values = _validate_vectors(vectors, sum(p.numel() for p in model.parameters()))
    if values.shape[0] == 0:
        return torch.empty(0, dtype=torch.float32)

    offsets: list[tuple[int, int]] = []
    offset = 0
    for _, parameter in model.named_parameters():
        size = parameter.numel()
        offsets.append((offset, offset + size))
        offset += size
    data_batch_size = 512

    with torch.no_grad():
        for vector_batch_cpu in values.split(batch_size):
            vector_batch = vector_batch_cpu.to(device=target_device)
            count = vector_batch.shape[0]
            exponents = vector_batch[:, offsets[0][0]:offsets[0][1]].reshape(
                count, hidden_units, input_dim
            )
            output_weight = vector_batch[:, offsets[1][0]:offsets[1][1]].reshape(
                count, output_dim, hidden_units
            )
            output_bias = vector_batch[:, offsets[2][0]:offsets[2][1]].reshape(
                count, output_dim
            )

            squared_error_sum = torch.zeros(count, device=target_device, dtype=torch.float32)
            for start in range(0, x.shape[0], data_batch_size):
                xb = x[start : start + data_batch_size]
                yb = y[start : start + data_batch_size]
                log_magnitude = torch.log(xb.abs())
                log_activation = torch.einsum("bd,vhd->vbh", log_magnitude, exponents)
                if model.input_domain == "positive":
                    activation = torch.exp(log_activation)
                else:
                    negative_indicators = (xb < 0).to(dtype=torch.float32)
                    phase = torch.einsum("bd,vhd->vbh", negative_indicators, exponents)
                    activation = torch.exp(log_activation) * torch.cos(math.pi * phase)
                prediction = torch.einsum("vbh,voh->vbo", activation, output_weight)
                prediction = prediction + output_bias[:, None, :]
                squared_error_sum += (prediction - yb[None, :, :]).square().sum(dim=(1, 2))

            pieces.append((squared_error_sum / (x.shape[0] * output_dim)).cpu())

    return torch.cat(pieces)


def orthonormal_directions(dim: int, seed: int) -> torch.Tensor:
    """Return two seeded orthonormal CPU directions, shaped ``[2, dim]``."""
    dim = _positive_int(dim, "dim")
    if dim < 2:
        raise ValueError("dim must be at least 2 for two orthonormal directions")
    seed = _seed_value(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    basis, triangular = torch.linalg.qr(torch.randn(dim, 2, generator=generator,
                                                   dtype=torch.float64))
    signs = torch.where(torch.diag(triangular) < 0, -1.0, 1.0)
    return (basis * signs[None, :]).T.to(dtype=torch.float32).contiguous()


def _generator(seed: int, stream: int) -> torch.Generator:
    # Separate streams make a sampler's output independent of the sizes and
    # execution order of the other sample sets.
    stream_seed = (seed + 0x1_0000_01B3 * (stream + 1)) % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(stream_seed)


def _corner_vectors(dimension: int, bound: float) -> torch.Tensor:
    total = 1 << dimension
    indices = torch.arange(dimension, dtype=torch.int64) * total // dimension
    bit_positions = torch.arange(dimension, dtype=torch.int64)
    bits = (indices[:, None] >> bit_positions[None, :]) & 1
    return (bits.to(torch.float32) * 2 - 1) * bound


def _reflect(values: torch.Tensor, bound: float) -> torch.Tensor:
    width = 2.0 * bound
    phase = torch.remainder(values + bound, 2.0 * width)
    reflected = torch.where(phase <= width, phase, 2.0 * width - phase)
    return reflected - bound


def _progressive_walk(
    start: torch.Tensor,
    *,
    steps: int,
    step_size: float,
    bound: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """One all-coordinate walk, biased 75% toward the opposite start corner."""
    current = start.clone()
    destination = -start
    path = [current.clone()]
    dimension = start.numel()
    for _ in range(steps):
        toward = torch.sign(destination - current)
        toward = torch.where(toward == 0, torch.ones_like(toward), toward)
        choose_toward = torch.rand(dimension, generator=generator) < _PRW_TOWARD_PROBABILITY
        direction = torch.where(choose_toward, toward, -toward)
        magnitude = torch.rand(dimension, generator=generator) * step_size
        current = _reflect(current + direction * magnitude, bound)
        path.append(current.clone())
    return torch.stack(path)


def _manhattan_walk(
    start: torch.Tensor,
    *,
    steps: int,
    step_size: float,
    bound: float,
    generator: torch.Generator,
) -> torch.Tensor:
    current = start.clone()
    path = [current.clone()]
    dimension = start.numel()
    for _ in range(steps):
        coordinate = int(torch.randint(dimension, (1,), generator=generator))
        delta = (2.0 * torch.rand((), generator=generator) - 1.0) * step_size
        current = current.clone()
        current[coordinate] = _reflect(current[coordinate] + delta, bound)
        path.append(current.clone())
    return torch.stack(path)


def _uniform_vectors(count: int, dimension: int, bound: float,
                     generator: torch.Generator) -> torch.Tensor:
    return (torch.rand(count, dimension, generator=generator) * (2.0 * bound)) - bound


def _sample_record(model: ProductUnitNetwork, train_x: torch.Tensor, train_y: torch.Tensor,
                   vectors: torch.Tensor, *, device: torch.device, batch_size: int) -> dict[str, torch.Tensor]:
    point_shape = vectors.shape[:-1]
    flat_vectors = vectors.reshape(-1, vectors.shape[-1])
    losses = evaluate_vectors(model, train_x, train_y, flat_vectors,
                              device=device, batch_size=batch_size).reshape(point_shape)
    return {"vectors": vectors.contiguous(), "mse": losses,
            "nonfinite": ~torch.isfinite(losses)}


def _finite_summary(losses: torch.Tensor) -> dict[str, int | float | None]:
    finite = losses[torch.isfinite(losses)].double()
    return {
        "point_count": int(losses.numel()),
        "finite_count": int(finite.numel()),
        "nonfinite_count": int(losses.numel() - finite.numel()),
        "finite_min": float(finite.min()) if finite.numel() else None,
        "finite_mean": float(finite.mean()) if finite.numel() else None,
        "finite_max": float(finite.max()) if finite.numel() else None,
    }


def _mrw_gradient_summary(vectors: torch.Tensor, losses: torch.Tensor) -> dict[str, int | float | None]:
    per_walk: list[dict[str, int | float | None]] = []
    slope_parts: list[torch.Tensor] = []
    nonfinite_step_count = 0
    for walk_vectors, walk_losses in zip(vectors, losses):
        finite_pairs = torch.isfinite(walk_losses[:-1]) & torch.isfinite(walk_losses[1:])
        nonfinite_step_count += int((~finite_pairs).sum())
        l1_steps = (walk_vectors[1:].double() - walk_vectors[:-1].double()).abs().sum(dim=1)
        valid = finite_pairs & (l1_steps > 0)
        slopes = (walk_losses[1:].double() - walk_losses[:-1].double()).abs()[valid] / l1_steps[valid]
        slope_parts.append(slopes)
        per_walk.append({
            "valid_step_count": int(slopes.numel()),
            "nonfinite_step_count": int((~finite_pairs).sum()),
            "mean": float(slopes.mean()) if slopes.numel() else None,
            "population_std": float(slopes.std(unbiased=False)) if slopes.numel() else None,
        })
    slopes = torch.cat(slope_parts) if slope_parts else torch.empty(0, dtype=torch.float64)
    return {
        "definition": "abs(delta full-training MSE) divided by the step's L1 parameter distance",
        "valid_step_count": int(slopes.numel()),
        "nonfinite_step_count": nonfinite_step_count,
        "mean": float(slopes.mean()) if slopes.numel() else None,
        "population_std": float(slopes.std(unbiased=False)) if slopes.numel() else None,
        "per_walk": per_walk,
    }


def sample_global(
    model: ProductUnitNetwork,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    seed: int,
    bound: float,
    prw_steps: int = 1000,
    mrw_steps: int = 2000,
    uniform_per_dim: int = 500,
    dispersion_samples: int = 2000,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> dict[str, Any]:
    """Sample raw global loss-landscape points for one supplied PUNN and data.

    Coordinates use ``[-bound, bound]`` in each parameter dimension. PRW micro
    and macro coordinate displacements are 1% and 10% of the full coordinate
    width, respectively; MRW uses 1% and changes one uniformly selected
    coordinate per step. Reflection handles boundary crossings. Each PRW
    starts from a seeded deterministic corner from the evenly spread corner
    set, and each step chooses its direction toward the opposite starting
    corner with probability 0.75 (otherwise away), with an independent
    uniform magnitude in ``[0, step_size]`` per coordinate. These are explicit
    sampling assumptions, not a claim to reproduce the paper's exact PRW code.

    ``uniform`` and ``dispersion`` are independent uniform samples. The latter
    retains the points needed for later dispersion analysis; no fitness-
    landscape metric is inferred here. All generated parameter arrays and
    returned loss arrays are CPU float32 tensors, even when evaluation uses
    CUDA. This function does not alter global random generators or model
    parameters.
    """
    seed = _seed_value(seed)
    input_dim, hidden_units, output_dim = _validate_model(model)
    del input_dim, hidden_units, output_dim
    if isinstance(bound, bool) or not isinstance(bound, (float, int)) or not math.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be finite and positive")
    bound = float(bound)
    prw_steps = _positive_int(prw_steps, "prw_steps")
    mrw_steps = _positive_int(mrw_steps, "mrw_steps")
    uniform_per_dim = _positive_int(uniform_per_dim, "uniform_per_dim")
    dispersion_samples = _positive_int(dispersion_samples, "dispersion_samples")
    batch_size = _positive_int(batch_size, "batch_size")
    target_device = _device(device)
    _validate_data(model, train_x, train_y, target_device)

    dimension = sum(parameter.numel() for parameter in model.parameters())
    corners = _corner_vectors(dimension, bound)
    uniform_generator = _generator(seed, 30_000)
    dispersion_generator = _generator(seed, 40_000)

    width = 2.0 * bound
    prw_micro_walks = []
    prw_macro_walks = []
    mrw_walks = []
    for walk_index, start in enumerate(corners):
        micro_generator = _generator(seed, walk_index)
        macro_generator = _generator(seed, 10_000 + walk_index)
        mrw_generator = _generator(seed, 20_000 + walk_index)
        prw_micro_walks.append(_progressive_walk(
            start, steps=prw_steps, step_size=0.01 * width, bound=bound,
            generator=micro_generator,
        ))
        prw_macro_walks.append(_progressive_walk(
            start, steps=prw_steps, step_size=0.10 * width, bound=bound,
            generator=macro_generator,
        ))
        mrw_walks.append(_manhattan_walk(
            start, steps=mrw_steps, step_size=0.01 * width, bound=bound,
            generator=mrw_generator,
        ))
    generated: dict[str, torch.Tensor] = {
        "corners": corners,
        "prw_micro": torch.stack(prw_micro_walks),
        "prw_macro": torch.stack(prw_macro_walks),
        "mrw": torch.stack(mrw_walks),
        "uniform": _uniform_vectors(uniform_per_dim * dimension, dimension, bound,
                                    uniform_generator),
        "dispersion": _uniform_vectors(dispersion_samples, dimension, bound,
                                       dispersion_generator),
    }
    samples = {
        name: _sample_record(model, train_x, train_y, vectors,
                             device=target_device, batch_size=batch_size)
        for name, vectors in generated.items()
    }
    summaries = {name: _finite_summary(record["mse"]) for name, record in samples.items()}
    summaries["mrw_gradient_estimates"] = _mrw_gradient_summary(
        samples["mrw"]["vectors"], samples["mrw"]["mse"]
    )

    return {
        "metadata": {
            "sampler": "punn_global_raw_v1",
            "seed": seed,
            "bound": [-bound, bound],
            "dimension": dimension,
            "parameter_layout": list(_PARAMETER_LAYOUT),
            "objective": "full supplied training-set mean squared error in FP32",
            "evaluation_device": str(target_device),
            "parameter_vector_batch_size": batch_size,
            "coordinate_width": width,
            "corners": "dimension-many deterministic vertices at binary corner indices floor(k*2**dimension/dimension), k=0..dimension-1",
            "prw_assumption": (
                "dimension-many independent walks per scale, one from each listed corner; every coordinate moves each step; "
                "each coordinate is biased 0.75 toward the opposite start corner, otherwise away; "
                "magnitude is independent uniform [0, step_size]; boundaries reflect"
            ),
            "prw_micro_step": 0.01 * width,
            "prw_macro_step": 0.10 * width,
            "mrw_assumption": "dimension-many independent walks, one from each listed corner; one uniformly selected coordinate moves per step; signed uniform step; boundaries reflect",
            "walk_count": dimension,
            "walk_corner_indices": [int(k * (1 << dimension) // dimension) for k in range(dimension)],
            "mrw_step": 0.01 * width,
            "uniform_point_count": uniform_per_dim * dimension,
            "dispersion_point_count": dispersion_samples,
            "losses_are_raw": True,
            "nonfinite_loss_policy": "retain raw nonfinite MSE and mark it in each nonfinite mask",
        },
        "samples": samples,
        "summary": summaries,
    }


def sample_slice(
    model: ProductUnitNetwork,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    center: torch.Tensor,
    *,
    directions: torch.Tensor,
    radius: float = 1.0,
    points: int = 31,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> dict[str, Any]:
    """Evaluate an unclipped two-direction parameter-space MSE slice.

    ``directions`` must have shape ``[2, D]``; they may be generated with
    :func:`orthonormal_directions`. The grid contains the center because
    ``points`` must be odd. No search-bound clipping is applied to slice points.
    """
    dimension = sum(parameter.numel() for parameter in model.parameters())
    _validate_model(model)
    target_device = _device(device)
    _validate_data(model, train_x, train_y, target_device)
    batch_size = _positive_int(batch_size, "batch_size")
    points = _positive_int(points, "points")
    if points < 3 or points % 2 == 0:
        raise ValueError("points must be an odd integer of at least 3")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    center_vector = torch.as_tensor(center)
    if center_vector.ndim != 1 or center_vector.numel() != dimension or center_vector.is_complex():
        raise ValueError(f"center must be a real vector with shape [{dimension}]")
    center_vector = center_vector.to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(center_vector).all():
        raise ValueError("center must be finite")
    direction_vectors = torch.as_tensor(directions)
    if direction_vectors.shape != (2, dimension) or direction_vectors.is_complex():
        raise ValueError(f"directions must be a real tensor with shape [2, {dimension}]")
    direction_vectors = direction_vectors.to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(direction_vectors).all() or (direction_vectors.norm(dim=1) == 0).any():
        raise ValueError("directions must be finite and nonzero")

    coordinates = torch.linspace(-float(radius), float(radius), points, dtype=torch.float32)
    first, second = torch.meshgrid(coordinates, coordinates, indexing="ij")
    grid_coordinates = (
        center_vector[None, None, :]
        + first[:, :, None] * direction_vectors[0]
        + second[:, :, None] * direction_vectors[1]
    )
    flattened = grid_coordinates.reshape(points * points, dimension)
    mse = evaluate_vectors(model, train_x, train_y, flattened,
                           device=target_device, batch_size=batch_size).reshape(points, points)
    nonfinite = ~torch.isfinite(mse)
    return {
        "metadata": {
            "sampler": "punn_slice_raw_v1",
            "dimension": dimension,
            "parameter_layout": list(_PARAMETER_LAYOUT),
            "objective": "full supplied training-set mean squared error in FP32",
            "evaluation_device": str(target_device),
            "radius": float(radius),
            "point_count_per_axis": points,
            "grid_order": "first direction varies along axis 0; second direction along axis 1",
            "bounds_clipped": False,
            "losses_are_raw": True,
        },
        "coordinates_1d": coordinates,
        "grid_coordinates": grid_coordinates,
        "mse": mse,
        "nonfinite": nonfinite,
        "summary": _finite_summary(mse.flatten()),
    }
