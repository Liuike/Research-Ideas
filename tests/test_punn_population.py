import math

import pytest
import torch

from optimizer_resurrection.punn_population import (
    differential_evolution,
    particle_swarm,
)


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
def test_population_optimizer_is_seeded_bounded_and_counts_evaluations(optimizer):
    calls = 0

    def objective(parameters):
        nonlocal calls
        calls += 1
        return ((parameters - torch.tensor([0.25, -0.5])) ** 2).sum()

    result = optimizer(
        objective,
        dimension=2,
        bounds=(-1.0, 1.0),
        seed=17,
        epochs=20,
        population_size=10,
        boundary="reflect",
    )

    assert result.evaluations == calls == 10 * 21
    assert result.best_parameters.shape == (2,)
    assert torch.all(result.best_parameters >= -1.0)
    assert torch.all(result.best_parameters <= 1.0)
    assert result.best_loss < 0.1


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
def test_population_optimizer_repeats_exactly_with_same_seed(optimizer):
    kwargs = dict(
        objective=lambda parameters: (parameters.square()).sum(),
        dimension=3,
        bounds=(-2.0, 2.0),
        seed=5,
        epochs=8,
        population_size=8,
    )
    first = optimizer(**kwargs)
    second = optimizer(**kwargs)

    assert torch.equal(first.best_parameters, second.best_parameters)
    assert first.best_loss == second.best_loss
    assert first.evaluations == second.evaluations


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
def test_supplied_initial_vector_is_evaluated_first(optimizer):
    seen = []
    initial = torch.tensor([0.2, -0.3])

    def objective(parameters):
        seen.append(parameters.clone())
        return (parameters - initial).square().sum()

    result = optimizer(
        objective,
        dimension=2,
        bounds=(-1.0, 1.0),
        seed=0,
        epochs=0,
        population_size=5,
        initial_vector=initial,
    )

    assert torch.equal(seen[0], initial)
    assert torch.equal(result.best_parameters, initial)
    assert result.best_loss == 0.0
    assert result.evaluations == 5


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
def test_nonfinite_objective_values_are_treated_as_infinity(optimizer):
    result = optimizer(
        lambda parameters: torch.tensor(float("nan")),
        dimension=1,
        bounds=(-1.0, 1.0),
        seed=1,
        epochs=0,
        population_size=4,
    )
    assert math.isinf(result.best_loss)
    assert result.evaluations == 4


def test_differential_evolution_requires_four_members():
    with pytest.raises(ValueError, match="population_size >= 4"):
        differential_evolution(
            lambda parameters: parameters.square().sum(),
            dimension=1,
            bounds=(-1.0, 1.0),
            seed=0,
            epochs=0,
            population_size=3,
        )
