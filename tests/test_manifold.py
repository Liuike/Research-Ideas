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

    assert optimizer.max_iterations == 10


def test_manifold_muon_fails_closed_on_unconverged_dual_solve():
    parameter = torch.nn.Parameter(retract_stiefel(torch.randn(8, 5, generator=torch.Generator().manual_seed(0))))
    optimizer = ManifoldMuon([parameter], lr=0.02, max_iterations=1, tolerance=1e-12)
    parameter.grad = torch.randn(8, 5, generator=torch.Generator().manual_seed(1))
    try:
        optimizer.step()
    except RuntimeError as error:
        assert "failed to reach tangent tolerance" in str(error)
    else:
        raise AssertionError("unconverged dual solve must not be applied")


def test_riemannian_sgd_preserves_constraint():
    parameter = torch.nn.Parameter(retract_stiefel(torch.randn(8, 5)))
    optimizer = RiemannianSGD([parameter], lr=0.05)
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    assert stiefel_residual(parameter) < 1e-5
