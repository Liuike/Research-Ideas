import torch

from optimizer_resurrection.diagnostics import stiefel_residual
from optimizer_resurrection.optim import ManifoldMuon, RiemannianSGD
from optimizer_resurrection.optim.manifold_muon import manifold_muon_direction, retract_stiefel


def test_retraction_handles_tall_and_wide_matrices():
    for shape in ((10, 4), (4, 10), (8, 8)):
        matrix = retract_stiefel(torch.randn(shape), scale=2.0)
        assert stiefel_residual(matrix, scale=2.0) < 1e-5


def test_convolution_retraction_matches_explicit_output_channel_flattening():
    for shape in ((8, 1, 2, 2), (4, 3, 2, 2)):
        kernel = torch.randn(shape, generator=torch.Generator().manual_seed(sum(shape)))
        flattened = retract_stiefel(kernel.reshape(shape[0], -1), scale=1.5)
        retracted = retract_stiefel(kernel, scale=1.5)

        assert torch.equal(retracted.reshape(shape[0], -1), flattened)
        assert stiefel_residual(retracted.reshape(shape[0], -1), scale=1.5) < 1e-5


def test_manifold_muon_convolution_update_matches_explicit_matrix_update():
    for shape in ((8, 1, 2, 2), (4, 3, 2, 2)):
        generator = torch.Generator().manual_seed(sum(shape) + 10)
        initial = retract_stiefel(torch.randn(shape, generator=generator))
        kernel = torch.nn.Parameter(initial.clone())
        matrix = torch.nn.Parameter(initial.reshape(shape[0], -1).clone())
        conv_optimizer = ManifoldMuon(
            [kernel], lr=0.002, momentum=0.9, nesterov=True, max_iterations=3
        )
        matrix_optimizer = ManifoldMuon(
            [matrix], lr=0.002, momentum=0.9, nesterov=True, max_iterations=3
        )

        for _ in range(2):
            gradient = torch.randn(shape, generator=generator)
            kernel.grad = gradient.clone()
            matrix.grad = gradient.reshape(shape[0], -1).clone()
            conv_optimizer.step()
            matrix_optimizer.step()

        assert torch.equal(kernel.reshape(shape[0], -1), matrix)
        assert conv_optimizer.state[kernel]["momentum_buffer"].shape == shape
        assert stiefel_residual(kernel.reshape(shape[0], -1)) < 1e-5


def test_manifold_muon_two_dimensional_step_matches_matrix_reference():
    generator = torch.Generator().manual_seed(31)
    initial = retract_stiefel(torch.randn(5, 3, generator=generator))
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = ManifoldMuon(
        [parameter], lr=0.003, momentum=0.9, nesterov=True, max_iterations=4
    )
    reference = initial.clone()
    reference_momentum = torch.zeros_like(reference)

    for _ in range(2):
        gradient = torch.randn(reference.shape, generator=generator)
        parameter.grad = gradient.clone()
        reference_momentum.mul_(0.9).add_(gradient)
        update = gradient.add(reference_momentum, alpha=0.9)
        direction, _, _ = manifold_muon_direction(
            reference, update, dual_lr=0.01, max_iterations=4, scale=1.0
        )
        reference.add_(direction, alpha=-0.003)
        reference.copy_(retract_stiefel(reference))
        optimizer.step()

    assert torch.equal(parameter, reference)


def test_manifold_muon_no_momentum_does_not_create_history_for_convolution():
    parameter = torch.nn.Parameter(retract_stiefel(torch.randn(8, 1, 2, 2)))
    optimizer = ManifoldMuon([parameter], momentum=0.0, max_iterations=2)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    assert "momentum_buffer" not in optimizer.state[parameter]
    assert optimizer.state[parameter]["inner_iterations"] == 2
    assert stiefel_residual(parameter.reshape(parameter.shape[0], -1)) < 1e-5


def test_manifold_muon_rejects_unsupported_parameter_dimensions():
    for shape in ((4,), (2, 3, 4)):
        parameter = torch.nn.Parameter(torch.randn(shape))
        optimizer = ManifoldMuon([parameter])
        parameter.grad = torch.randn_like(parameter)
        try:
            optimizer.step()
        except ValueError as error:
            assert "2D matrices or 4D convolution kernels" in str(error)
        else:
            raise AssertionError(f"expected unsupported shape {shape} to fail")


def test_manifold_muon_preserves_constraint():
    # Primary study matrices are square; the published initialization solves
    # this case immediately.
    parameter = torch.nn.Parameter(retract_stiefel(torch.randn(8, 8)))
    optimizer = ManifoldMuon([parameter], lr=0.02)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    assert stiefel_residual(parameter) < 1e-5


def test_manifold_muon_uses_ten_dual_iterations_by_default():
    parameter = torch.nn.Parameter(torch.eye(4))
    optimizer = ManifoldMuon([parameter])
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    assert optimizer.max_iterations == 10
    assert optimizer.state[parameter]["inner_iterations"] == 10


def test_manifold_muon_records_unconverged_residual_without_aborting():
    parameter = torch.nn.Parameter(retract_stiefel(torch.randn(8, 5, generator=torch.Generator().manual_seed(0))))
    optimizer = ManifoldMuon([parameter], lr=0.02, max_iterations=1)
    parameter.grad = torch.randn(8, 5, generator=torch.Generator().manual_seed(1))
    optimizer.step()

    assert optimizer.state[parameter]["inner_iterations"] == 1
    assert optimizer.state[parameter]["tangent_residual"] > 1e-6
    assert stiefel_residual(parameter) < 1e-5


def test_manifold_muon_handles_exploratory_matrix_size():
    generator = torch.Generator().manual_seed(7)
    parameter = torch.nn.Parameter(retract_stiefel(torch.randn(64, 64, generator=generator)))
    optimizer = ManifoldMuon([parameter], lr=0.001)
    parameter.grad = torch.randn(64, 64, generator=generator)
    optimizer.step()

    assert optimizer.state[parameter]["inner_iterations"] == 10
    assert torch.isfinite(parameter).all()
    assert stiefel_residual(parameter) < 1e-5


def test_riemannian_sgd_preserves_constraint():
    parameter = torch.nn.Parameter(retract_stiefel(torch.randn(8, 5)))
    optimizer = RiemannianSGD([parameter], lr=0.05)
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    assert stiefel_residual(parameter) < 1e-5
