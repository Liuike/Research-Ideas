"""Small deterministic population optimizers for product-unit experiments.

The defaults follow the PSO and DE coefficients reported by Engelbrecht and
Gouldie (2024). Population size, dimensional bounds, initialization and
boundary handling remain explicit because the paper does not specify all of
those implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import torch


Objective = Callable[[torch.Tensor], float | torch.Tensor]
BatchObjective = Callable[[torch.Tensor], torch.Tensor]
EpochCallback = Callable[
    [int, torch.Tensor, float, torch.Tensor, torch.Tensor], None
]
Bounds = tuple[float | torch.Tensor, float | torch.Tensor]
Boundary = Literal["clip", "reflect"]


@dataclass(frozen=True)
class PopulationResult:
    """Best candidate found by a population optimizer."""

    best_parameters: torch.Tensor
    best_loss: float
    evaluations: int


def particle_swarm(
    objective: Objective,
    *,
    dimension: int,
    bounds: Bounds,
    seed: int,
    epochs: int = 500,
    population_size: int = 30,
    initial_vector: torch.Tensor | None = None,
    boundary: Boundary = "clip",
    on_epoch: EpochCallback | None = None,
    batch_objective: BatchObjective | None = None,
) -> PopulationResult:
    """Minimize a scalar objective with the paper's global-best PSO settings.

    The initial swarm is sampled uniformly from ``bounds``. If supplied,
    ``initial_vector`` replaces member zero, allowing optimizers to share a
    model initialization. ``epochs`` counts swarm updates, after the initial
    population evaluation. Inputs passed to ``objective`` are flat CPU tensors.
    ``batch_objective``, when supplied, instead receives the whole population
    matrix and must return one float32 CPU loss per member. ``on_epoch`` is
    called after each update with the 1-based epoch and detached snapshots.
    """
    _validate_common(dimension, bounds, seed, epochs, population_size, boundary)
    lower, upper = _materialize_bounds(bounds, dimension)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions = _initial_population(
        lower, upper, population_size, generator, initial_vector
    )
    span = upper - lower
    velocities = (torch.rand(positions.shape, generator=generator) * 2.0 - 1.0) * span
    if initial_vector is not None:
        velocities[0].zero_()

    evaluations = 0
    losses = torch.empty(population_size, dtype=positions.dtype)
    if batch_objective is None:
        for index in range(population_size):
            losses[index] = _evaluate(objective, positions[index])
            evaluations += 1
    else:
        losses = _evaluate_batch(batch_objective, positions)
        evaluations += population_size

    personal_best = positions.clone()
    personal_losses = losses.clone()
    best_index = int(torch.argmin(personal_losses).item())
    global_best = personal_best[best_index].clone()
    global_loss = float(personal_losses[best_index].item())

    # Standard global-best PSO coefficients reported in the 2024 study.
    inertia = 0.7298
    cognitive = 1.496
    social = 1.496
    for epoch in range(1, epochs + 1):
        r_personal = torch.rand(positions.shape, generator=generator)
        r_global = torch.rand(positions.shape, generator=generator)
        velocities = (
            inertia * velocities
            + cognitive * r_personal * (personal_best - positions)
            + social * r_global * (global_best.unsqueeze(0) - positions)
        )
        positions, velocities = _apply_boundary(
            positions + velocities, velocities, lower, upper, boundary
        )

        if batch_objective is None:
            generation_losses = torch.empty_like(losses)
            for index in range(population_size):
                loss = _evaluate(objective, positions[index])
                generation_losses[index] = loss
                evaluations += 1
                if loss < float(personal_losses[index].item()):
                    personal_losses[index] = loss
                    personal_best[index] = positions[index]
                    if loss < global_loss:
                        global_loss = loss
                        global_best = positions[index].clone()
        else:
            generation_losses = _evaluate_batch(batch_objective, positions)
            evaluations += population_size
            for index in range(population_size):
                loss = float(generation_losses[index].item())
                if loss < float(personal_losses[index].item()):
                    personal_losses[index] = loss
                    personal_best[index] = positions[index]
                    if loss < global_loss:
                        global_loss = loss
                        global_best = positions[index].clone()

        if on_epoch is not None:
            on_epoch(
                epoch,
                global_best.detach().clone(),
                global_loss,
                positions.detach().clone(),
                generation_losses.detach().clone(),
            )

    return PopulationResult(global_best, global_loss, evaluations)


def differential_evolution(
    objective: Objective,
    *,
    dimension: int,
    bounds: Bounds,
    seed: int,
    epochs: int = 500,
    population_size: int = 30,
    initial_vector: torch.Tensor | None = None,
    boundary: Boundary = "clip",
    on_epoch: EpochCallback | None = None,
    batch_objective: BatchObjective | None = None,
) -> PopulationResult:
    """Minimize with DE/rand/1/bin and the paper's beta/crossover settings.

    At least four population members are required by DE/rand/1. Each epoch is
    one complete generation: all trial vectors are formed from the current
    population, then selected together. ``boundary`` explicitly controls how
    out-of-range trial coordinates are handled. ``batch_objective``, when
    supplied, receives the whole population or trial matrix and must return
    one float32 CPU loss per member. ``on_epoch`` is called after each
    generation with the 1-based epoch and detached snapshots.
    """
    _validate_common(dimension, bounds, seed, epochs, population_size, boundary)
    if population_size < 4:
        raise ValueError("differential_evolution requires population_size >= 4")
    lower, upper = _materialize_bounds(bounds, dimension)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    population = _initial_population(
        lower, upper, population_size, generator, initial_vector
    )
    losses = torch.empty(population_size, dtype=population.dtype)
    evaluations = 0
    if batch_objective is None:
        for index in range(population_size):
            losses[index] = _evaluate(objective, population[index])
            evaluations += 1
    else:
        losses = _evaluate_batch(batch_objective, population)
        evaluations += population_size

    beta = 0.7
    crossover_probability = 0.3
    for epoch in range(1, epochs + 1):
        trials = torch.empty_like(population)
        trial_losses = torch.empty_like(losses)
        for target_index in range(population_size):
            choices = [i for i in range(population_size) if i != target_index]
            order = torch.randperm(len(choices), generator=generator)[:3].tolist()
            donor_indices = [choices[i] for i in order]
            a, b, c = (population[index] for index in donor_indices)
            mutant = a + beta * (b - c)

            mask = torch.rand(dimension, generator=generator) < crossover_probability
            mask[int(torch.randint(dimension, (1,), generator=generator).item())] = True
            trials[target_index] = torch.where(mask, mutant, population[target_index])

        trials, _ = _apply_boundary(
            trials, torch.zeros_like(trials), lower, upper, boundary
        )
        if batch_objective is None:
            for index in range(population_size):
                trial_losses[index] = _evaluate(objective, trials[index])
                evaluations += 1
        else:
            trial_losses = _evaluate_batch(batch_objective, trials)
            evaluations += population_size

        accepted = trial_losses <= losses
        population = torch.where(accepted.unsqueeze(1), trials, population)
        losses = torch.where(accepted, trial_losses, losses)

        if on_epoch is not None:
            best_index = int(torch.argmin(losses).item())
            on_epoch(
                epoch,
                population[best_index].detach().clone(),
                float(losses[best_index].item()),
                population.detach().clone(),
                losses.detach().clone(),
            )

    best_index = int(torch.argmin(losses).item())
    return PopulationResult(
        population[best_index].clone(), float(losses[best_index].item()), evaluations
    )


def _validate_common(
    dimension: int,
    bounds: Bounds,
    seed: int,
    epochs: int,
    population_size: int,
    boundary: Boundary,
) -> None:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if population_size < 1:
        raise ValueError("population_size must be positive")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if boundary not in ("clip", "reflect"):
        raise ValueError("boundary must be 'clip' or 'reflect'")
    if len(bounds) != 2:
        raise ValueError("bounds must be a (lower, upper) pair")


def _materialize_bounds(bounds: Bounds, dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
    lower = _bound_vector(bounds[0], dimension, "lower")
    upper = _bound_vector(bounds[1], dimension, "upper")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError("bounds must be finite")
    if torch.any(lower > upper):
        raise ValueError("each lower bound must be <= its upper bound")
    return lower, upper


def _bound_vector(value: float | torch.Tensor, dimension: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().flatten()
    if tensor.numel() == 1:
        return tensor.expand(dimension).clone()
    if tensor.numel() != dimension:
        raise ValueError(f"{name} bound must be scalar or have {dimension} entries")
    return tensor.clone()


def _initial_population(
    lower: torch.Tensor,
    upper: torch.Tensor,
    population_size: int,
    generator: torch.Generator,
    initial_vector: torch.Tensor | None,
) -> torch.Tensor:
    dimension = lower.numel()
    random_values = torch.rand((population_size, dimension), generator=generator)
    population = lower + random_values * (upper - lower)
    if initial_vector is not None:
        vector = torch.as_tensor(initial_vector, dtype=lower.dtype, device="cpu").detach().flatten()
        if vector.numel() != dimension:
            raise ValueError(f"initial_vector must have {dimension} entries")
        if not torch.isfinite(vector).all():
            raise ValueError("initial_vector must be finite")
        if torch.any(vector < lower) or torch.any(vector > upper):
            raise ValueError("initial_vector must lie within bounds")
        population[0] = vector
    return population


def _evaluate(objective: Objective, candidate: torch.Tensor) -> float:
    value = objective(candidate.detach().clone())
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("objective must return a scalar")
        scalar = float(value.detach().cpu().item())
    else:
        scalar = float(value)
    return scalar if math.isfinite(scalar) else math.inf


def _evaluate_batch(
    batch_objective: BatchObjective, population: torch.Tensor
) -> torch.Tensor:
    values = batch_objective(population.detach().clone())
    if not isinstance(values, torch.Tensor):
        raise TypeError("batch_objective must return a torch.Tensor")
    if values.shape != (population.shape[0],):
        raise ValueError(
            "batch_objective must return a one-dimensional tensor with one value "
            f"per candidate (expected {(population.shape[0],)}, got {tuple(values.shape)})"
        )
    if values.device.type != "cpu" or values.dtype != torch.float32:
        raise TypeError("batch_objective must return a float32 CPU tensor")
    values = values.detach().clone()
    return torch.where(torch.isfinite(values), values, torch.full_like(values, math.inf))


def _apply_boundary(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    boundary: Boundary,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = upper - lower
    fixed = width == 0
    if boundary == "clip":
        clipped = torch.maximum(torch.minimum(positions, upper), lower)
        adjusted_velocities = torch.where(
            (positions != clipped) | fixed, torch.zeros_like(velocities), velocities
        )
        return clipped, adjusted_velocities

    safe_width = torch.where(fixed, torch.ones_like(width), width)
    phase = torch.remainder((positions - lower) / safe_width, 2.0)
    folded = torch.where(phase <= 1.0, phase, 2.0 - phase)
    reflected = lower + folded * width
    direction = torch.where(phase <= 1.0, 1.0, -1.0)
    adjusted_velocities = torch.where(fixed, torch.zeros_like(velocities), velocities * direction)
    return reflected, adjusted_velocities
