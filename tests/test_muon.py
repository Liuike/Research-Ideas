import torch

from optimizer_resurrection.optim import Muon, matrix_sign_ns, matrix_sign_svd


def test_exact_matrix_sign_is_semi_orthogonal():
    matrix = torch.randn(12, 7, generator=torch.Generator().manual_seed(1))
    polar = matrix_sign_svd(matrix)
    assert torch.allclose(polar.T @ polar, torch.eye(7), atol=2e-5, rtol=2e-5)


def test_ns5_points_toward_exact_polar_direction():
    matrix = torch.randn(16, 16, generator=torch.Generator().manual_seed(2))
    exact = matrix_sign_svd(matrix)
    approximate = matrix_sign_ns(matrix, steps=5)
    cosine = torch.nn.functional.cosine_similarity(exact.flatten(), approximate.flatten(), dim=0)
    assert float(cosine) > 0.95


def test_rectangular_muon_uses_reference_aspect_scaling():
    parameter = torch.nn.Parameter(torch.zeros(8, 2))
    parameter.grad = torch.eye(8, 2)
    optimizer = Muon([parameter], lr=1.0, momentum=0.0, nesterov=False, backend="svd")
    optimizer.step()
    # sqrt(rows / cols) = 2 for this tall matrix.
    assert torch.allclose(parameter[:2], -2 * torch.eye(2), atol=1e-6)
