from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset, IterableDataset


_PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


class ShapeSetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic online-style 32x32 Shapeset-3x2 approximation.

    Classes 0..2 are single triangle/parallelogram/ellipse. Classes 3..8 are
    unordered two-object combinations (with repetition), yielding nine classes.
    """

    def __init__(self, size: int = 100_000, seed: int = 0) -> None:
        self.size = size
        self.seed = seed
        coords = torch.linspace(-1.0, 1.0, 32)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")

    def __len__(self) -> int:
        return self.size

    def _object(self, kind: int, g: torch.Generator) -> tuple[torch.Tensor, float]:
        angle = float(torch.rand((), generator=g) * (2 * math.pi))
        scale = float(torch.empty(()).uniform_(0.25, 0.55, generator=g))
        tx, ty = torch.empty(2).uniform_(-0.45, 0.45, generator=g).tolist()
        c, s = math.cos(angle), math.sin(angle)
        x = ((self.xx - tx) * c + (self.yy - ty) * s) / scale
        y = (-(self.xx - tx) * s + (self.yy - ty) * c) / scale
        if kind == 0:  # triangle
            mask = (y > -0.65) & (y < 0.8) & (x.abs() < (0.8 - y) * 0.72)
        elif kind == 1:  # parallelogram
            mask = (y.abs() < 0.62) & ((x + 0.35 * y).abs() < 0.78)
        else:  # ellipse
            mask = x.square() / 0.9**2 + y.square() / 0.62**2 <= 1
        shade = float(torch.empty(()).uniform_(0.35, 1.0, generator=g))
        return mask, shade

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed + int(index) * 1_000_003)
        label = int(torch.randint(0, 9, (), generator=g))
        kinds = [label] if label < 3 else list(_PAIRS[label - 3])
        image = torch.empty(32, 32).uniform_(0.0, 0.08, generator=g)
        for kind in kinds:
            mask, shade = self._object(kind, g)
            # Later objects overwrite earlier ones, creating partial occlusion.
            image = torch.where(mask, torch.tensor(shade), image)
        image += torch.randn((32, 32), generator=g) * 0.02
        return image.clamp(0, 1).unsqueeze(0), torch.tensor(label)


class OnlineShapeSet(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Infinite deterministic Shapeset stream; no training example is recycled."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.dataset = ShapeSetDataset(size=1, seed=seed)

    def __iter__(self):
        index = 0
        while True:
            yield self.dataset[index]
            index += 1
