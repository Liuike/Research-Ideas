import math

import torch
from torch.nn import functional as F

from optimizer_resurrection.models.pure_resnet import (
    PUReBasicBlock,
    ProductUnitConv2d,
    pure_resnet18_cifar10,
)


def test_product_unit_conv_matches_explicit_multichannel_product() -> None:
    layer = ProductUnitConv2d(2, 2, kernel_size=2, padding=1).double()
    with torch.no_grad():
        layer.theta.fill_(math.log(math.expm1(0.5 - 1e-7)))
        layer.weight.copy_(torch.tensor([
            [[[0.2, -0.1], [0.3, 0.1]], [[-0.2, 0.15], [0.05, 0.25]]],
            [[[-0.1, 0.2], [0.15, -0.05]], [[0.25, -0.1], [0.2, 0.1]]],
        ], dtype=torch.double))
    x = torch.tensor(
        [[[[0.8, 1.1], [0.6, 1.4]], [[1.2, 0.7], [0.9, 1.3]]]],
        dtype=torch.double,
    )
    floor = F.softplus(layer.theta) + 1e-7
    clipped = torch.maximum(x, floor)
    expected = torch.ones((1, 2, 3, 3), dtype=torch.double)
    # Evaluate each product directly. Locations outside the input contribute
    # the multiplicative identity, matching zero padding after the logarithm.
    for out_channel in range(2):
        for out_y in range(3):
            for out_x in range(3):
                product = torch.ones((), dtype=torch.double)
                for in_channel in range(2):
                    for kernel_y in range(2):
                        for kernel_x in range(2):
                            input_y = out_y - 1 + kernel_y
                            input_x = out_x - 1 + kernel_x
                            if 0 <= input_y < 2 and 0 <= input_x < 2:
                                product = product * clipped[0, in_channel, input_y, input_x] ** layer.weight[
                                    out_channel, in_channel, kernel_y, kernel_x
                                ]
                expected[0, out_channel, out_y, out_x] = product
    torch.testing.assert_close(layer(x), expected)


def test_padding_is_zero_in_log_space() -> None:
    layer = ProductUnitConv2d(1, 1, kernel_size=3, padding=1).double()
    with torch.no_grad():
        layer.theta.fill_(-10.0)
        layer.weight.fill_(1.0)
    # The eight padded positions contribute log(1)=0, so only the center x=2
    # contributes to the product at the sole output position.
    actual = layer(torch.tensor([[[[2.0]]]], dtype=torch.double))
    torch.testing.assert_close(actual, torch.tensor([[[[2.0]]]], dtype=torch.double))


def test_product_unit_gradients_pass_double_precision_finite_differences() -> None:
    layer = ProductUnitConv2d(1, 1, kernel_size=2, padding=1).double()
    with torch.no_grad():
        layer.theta.fill_(0.1)
        layer.weight.copy_(torch.tensor([[[[0.08, -0.04], [0.03, 0.06]]]], dtype=torch.double))
    x = torch.tensor([[[[-0.8, 0.9], [1.2, 2.0]]]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda input_, weight, theta: torch.exp(
            F.conv2d(
                torch.log(torch.maximum(input_, F.softplus(theta) + 1e-7)),
                weight,
                padding=1,
            )
        ),
        (x, layer.weight, layer.theta),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_negative_and_zero_inputs_remain_finite_and_differentiable() -> None:
    layer = ProductUnitConv2d(2, 3, kernel_size=3, padding=1).double()
    x = torch.tensor(
        [[[[0.0, -1.0], [0.2, 1.0]], [[-0.3, 0.0], [1.2, -0.7]]]],
        dtype=torch.double,
        requires_grad=True,
    )
    output = layer(x)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.theta.grad is not None and torch.isfinite(layer.theta.grad)


def test_pure_resnet18_cifar10_architecture_and_parameter_count() -> None:
    model = pure_resnet18_cifar10()
    layers = [module for module in model.modules() if isinstance(module, ProductUnitConv2d)]
    assert len(layers) == 8
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)
    assert [len(stage) for stage in (model.layer1, model.layer2, model.layer3, model.layer4)] == [2, 2, 2, 2]
    assert [stage[0].conv1.out_channels for stage in (model.layer1, model.layer2, model.layer3, model.layer4)] == [64, 128, 256, 512]
    assert model.fc.out_features == 10
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_173_970
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 10)


def test_pure_block_passes_negative_bn1_values_to_product_unit() -> None:
    block = PUReBasicBlock(2, 2).eval()
    block.conv1 = torch.nn.Identity()
    block.bn1 = torch.nn.Identity()
    captured = []
    hook = block.conv2.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
    x = -torch.ones(1, 2, 4, 4)
    try:
        output = block(x)
    finally:
        hook.remove()
    assert len(captured) == 1
    assert torch.all(captured[0] < 0)
    assert torch.all(output >= 0)

