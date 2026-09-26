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


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
def test_population_optimizer_epoch_callback_reports_each_generation(optimizer):
    generations = []

    def on_epoch(epoch, best_parameters, best_loss, population, losses):
        generations.append(
            (epoch, best_parameters, best_loss, population, losses)
        )

    result = optimizer(
        lambda parameters: parameters.square().sum(),
        dimension=2,
        bounds=(-1.0, 1.0),
        seed=23,
        epochs=4,
        population_size=6,
        on_epoch=on_epoch,
    )

    assert [generation[0] for generation in generations] == [1, 2, 3, 4]
    for _, best_parameters, best_loss, population, losses in generations:
        assert best_parameters.shape == (2,)
        assert population.shape == (6, 2)
        assert losses.shape == (6,)
        assert best_loss <= float(losses.min().item())
    assert result.evaluations == 6 * 5


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
def test_population_optimizer_epoch_callback_cannot_mutate_optimizer_state(optimizer):
    kwargs = dict(
        objective=lambda parameters: parameters.square().sum(),
        dimension=3,
        bounds=(-2.0, 2.0),
        seed=29,
        epochs=7,
        population_size=8,
    )
    baseline = optimizer(**kwargs)

    def mutate_callback(_epoch, best_parameters, _best_loss, population, losses):
        best_parameters.fill_(1000.0)
        population.fill_(1000.0)
        losses.fill_(1000.0)

    with_callback = optimizer(**kwargs, on_epoch=mutate_callback)

    assert torch.equal(with_callback.best_parameters, baseline.best_parameters)
    assert with_callback.best_loss == baseline.best_loss
    assert with_callback.evaluations == baseline.evaluations


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
def test_population_optimizer_batch_objective_matches_scalar_and_counts_rows(optimizer):
    scalar_calls = 0
    batch_rows = 0

    def scalar_objective(parameters):
        nonlocal scalar_calls
        scalar_calls += 1
        return parameters.square().sum()

    def batch_objective(population):
        nonlocal batch_rows
        batch_rows += population.shape[0]
        return population.square().sum(dim=1)

    kwargs = dict(
        dimension=3,
        bounds=(-2.0, 2.0),
        seed=31,
        epochs=9,
        population_size=8,
    )
    scalar_result = optimizer(scalar_objective, **kwargs)
    batch_result = optimizer(scalar_objective, **kwargs, batch_objective=batch_objective)

    assert torch.equal(batch_result.best_parameters, scalar_result.best_parameters)
    assert batch_result.best_loss == scalar_result.best_loss
    assert batch_result.evaluations == scalar_result.evaluations == 8 * 10
    assert scalar_calls == 8 * 10
    assert batch_rows == 8 * 10


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
@pytest.mark.parametrize("shape", [(4, 1), (3,)])
def test_population_optimizer_rejects_invalid_batch_objective_shape(optimizer, shape):
    with pytest.raises(ValueError, match="one-dimensional tensor with one value"):
        optimizer(
            lambda parameters: parameters.square().sum(),
            dimension=2,
            bounds=(-1.0, 1.0),
            seed=37,
            epochs=0,
            population_size=4,
            batch_objective=lambda population: torch.zeros(shape, dtype=torch.float32),
        )


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
def test_population_optimizer_requires_float32_batch_objective(optimizer):
    with pytest.raises(TypeError, match="float32 CPU tensor"):
        optimizer(
            lambda parameters: parameters.square().sum(),
            dimension=2,
            bounds=(-1.0, 1.0),
            seed=41,
            epochs=0,
            population_size=4,
            batch_objective=lambda population: torch.zeros(
                population.shape[0], dtype=torch.float64
            ),
        )


@pytest.mark.parametrize("optimizer", [particle_swarm, differential_evolution])
def test_population_optimizer_maps_nonfinite_batch_losses_to_infinity(optimizer):
    result = optimizer(
        lambda parameters: parameters.square().sum(),
        dimension=2,
        bounds=(-1.0, 1.0),
        seed=43,
        epochs=0,
        population_size=4,
        batch_objective=lambda population: torch.full(
            (population.shape[0],), float("nan"), dtype=torch.float32
        ),
    )
    assert math.isinf(result.best_loss)
    assert result.evaluations == 4
