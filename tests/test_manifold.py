import torch

from optimizer_resurrection.diagnostics import stiefel_residual
from optimizer_resurrection.optim import ManifoldMuon, RiemannianSGD
from optimizer_resurrection.optim.manifold_muon import retract_stiefel


def test_retraction_handles_tall_and_wide_matrices():
    for shape in ((10, 4), (4, 10), (8, 8)):
        matrix = retract_stiefel(torch.randn(shape), scale=2.0)
        assert stiefel_residual(matrix, scale=2.0) < 1e-5


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
