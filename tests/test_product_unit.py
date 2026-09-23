import math

import pytest
import torch

from optimizer_resurrection.models import ProductUnitNetwork


def test_positive_product_unit_recovers_monomial_and_gradients() -> None:
    model = ProductUnitNetwork(2, 1, input_domain="positive").double()
    with torch.no_grad():
        model.exponents.copy_(torch.tensor([[2.0, -1.0]], dtype=torch.float64))
        model.output.weight.fill_(1.0)
        model.output.bias.zero_()
    x = torch.tensor([[3.0, 2.0]], dtype=torch.float64)
    assert torch.allclose(model(x), torch.tensor([[4.5]], dtype=torch.float64))
    model(x).sum().backward()
    expected = torch.tensor([[4.5 * math.log(3), 4.5 * math.log(2)]], dtype=torch.float64)
    assert torch.allclose(model.exponents.grad, expected)


def test_signed_product_unit_matches_real_part_of_complex_power() -> None:
    model = ProductUnitNetwork(2, 1, input_domain="real_complex").double()
    with torch.no_grad():
        model.exponents.copy_(torch.tensor([[0.3, 2.0]], dtype=torch.float64))
        model.output.weight.fill_(1.0)
        model.output.bias.zero_()
    x = torch.tensor([[-4.0, 3.0]], dtype=torch.float64, requires_grad=True)
    expected = ((-4.0 + 0j) ** 0.3 * (3.0 + 0j) ** 2).real
    assert torch.allclose(model(x), torch.tensor([[expected]], dtype=torch.float64), atol=1e-12)
    assert torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=1e-5)
    model(x).sum().backward()
    amplitude = (4.0**0.3) * 9.0
    expected_derivative = amplitude * (math.log(4.0) * math.cos(0.3 * math.pi)
                                       - math.pi * math.sin(0.3 * math.pi))
    assert math.isclose(float(model.exponents.grad[0, 0]), expected_derivative,
                        rel_tol=1e-10, abs_tol=1e-10)


@pytest.mark.parametrize("domain,invalid", [("positive", -1.0), ("positive", 0.0), ("real_complex", 0.0)])
def test_invalid_input_domain_is_explicit(domain: str, invalid: float) -> None:
    model = ProductUnitNetwork(1, 1, input_domain=domain)
    with pytest.raises(ValueError):
        model(torch.tensor([[invalid]]))
