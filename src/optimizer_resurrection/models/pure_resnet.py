"""PURe residual networks for CIFAR-sized images.

The product convolution applies an affine map in log space after enforcing a
learned positive input floor. Its spatial convolution padding is consequently
zero in log space (the multiplicative identity in the original domain).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, ResNet


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("spatial parameters must be integers or pairs")
        return value
    return (value, value)


class ProductUnitConv2d(nn.Module):
    """Full-channel product-unit convolution with one learned input floor.

    Given exponent kernels ``W`` and scalar parameter ``theta``, this computes

    ``exp(conv2d(log(max(x, softplus(theta) + 1e-7)), W))``.

    ``torch.nn.functional.conv2d`` applies padding to the logged input, so the
    padded values are zero in log space. The layer intentionally does not clamp
    its exponentiated output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.theta = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match torchvision's convolution initialization convention.
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.theta)

    @property
    def threshold(self) -> Tensor:
        """The positive floor applied before taking the logarithm."""
        return F.softplus(self.theta) + 1e-7

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(x.shape)}"
            )
        logged = torch.log(torch.maximum(x, self.threshold))
        return torch.exp(
            F.conv2d(
                logged,
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        )


class PUReBasicBlock(BasicBlock):
    """Torchvision basic residual block with a product-unit second operator."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        if groups != 1 or base_width != 64:
            raise ValueError("PUReBasicBlock supports torchvision's standard width/groups")
        super().__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )
        conv2 = self.conv2
        self.conv2 = ProductUnitConv2d(
            conv2.in_channels,
            conv2.out_channels,
            kernel_size=conv2.kernel_size,
            stride=conv2.stride,
            padding=conv2.padding,
            dilation=conv2.dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # PURe removes the intermediate ReLU between the two branch operators.
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


def pure_resnet18_cifar10() -> ResNet:
    """Build the paper-inspired PURe ResNet-18 adapted to CIFAR-10 inputs."""
    model = ResNet(PUReBasicBlock, [2, 2, 2, 2], num_classes=10)
    # CIFAR stem: preserve 32x32 resolution until the first residual downsample.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model

