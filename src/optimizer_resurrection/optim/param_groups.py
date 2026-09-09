from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .manifold_muon import ManifoldMuon, retract_stiefel
from .muon import Muon
from .riemannian_sgd import RiemannianSGD


def split_parameters(
    model: nn.Module, assignment: str = "primary"
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    if assignment not in {"primary", "standard"}:
        raise ValueError("assignment must be 'primary' or 'standard'")
    compared = list(
        model.compared_weights
        if assignment == "primary"
        else model.standard_muon_weights
    )
    compared_ids = {id(p) for p in compared}
    auxiliary = [p for p in model.parameters() if id(p) not in compared_ids]
    return compared, auxiliary


@dataclass
class OptimizerBundle:
    primary: torch.optim.Optimizer
    auxiliary: torch.optim.Optimizer | None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.primary.zero_grad(set_to_none=set_to_none)
        if self.auxiliary is not None:
            self.auxiliary.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.primary.step()
        if self.auxiliary is not None:
            self.auxiliary.step()


def build_optimizer(
    model: nn.Module,
    condition: str,
    lr: float,
    aux_lr: float,
    weight_decay: float = 0.0,
    manifold_scale: float = 1.0,
    muon_backend: str = "ns",
    assignment: str = "primary",
) -> OptimizerBundle:
    compared, auxiliary = split_parameters(model, assignment)
    if condition == "bptt_sgd":
        return OptimizerBundle(torch.optim.SGD(model.parameters(), lr=lr), None)
    if condition.startswith("sgd"):
        return OptimizerBundle(
            torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay),
            None,
        )
    if condition.startswith("adam"):
        groups = [{"params": compared, "lr": lr, "weight_decay": weight_decay}]
        if auxiliary:
            groups.append({"params": auxiliary, "lr": aux_lr, "weight_decay": 0.0})
        return OptimizerBundle(torch.optim.AdamW(groups), None)
    if not compared:
        # Depth-one MLPs have no repeated hidden transform; this is intentional.
        primary = torch.optim.AdamW(auxiliary, lr=aux_lr, weight_decay=0.0)
        return OptimizerBundle(primary, None)
    else:
        primary_params = compared

    if condition.startswith("muon"):
        primary = Muon(primary_params, lr=lr, backend=muon_backend)
    elif condition == "rsgd":
        with torch.no_grad():
            for p in primary_params:
                p.copy_(retract_stiefel(p, manifold_scale))
        primary = RiemannianSGD(primary_params, lr=lr, scale=manifold_scale)
    elif condition in {"mm", "scaled_mm"}:
        with torch.no_grad():
            for p in primary_params:
                p.copy_(retract_stiefel(p, manifold_scale))
        primary = ManifoldMuon(primary_params, lr=lr, scale=manifold_scale)
    else:
        raise ValueError(f"unknown optimizer condition: {condition}")

    aux = (
        torch.optim.AdamW(auxiliary, lr=aux_lr, weight_decay=0.0)
        if auxiliary
        else None
    )
    return OptimizerBundle(primary, aux)
