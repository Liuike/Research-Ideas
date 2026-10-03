"""PURe routing for the existing exact-SVD Manifold Muon DA-10 recipe."""
from __future__ import annotations

import hashlib
import math
import time

import torch

from .models.pure_resnet import ProductUnitConv2d
from .optim.manifold_muon import ManifoldMuon, retract_stiefel
from .optim.param_groups import OptimizerBundle


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class PUReManifoldBundle(OptimizerBundle):
    """Runner facade; schedulers act on both underlying torch optimizers."""

    @property
    def optimizers(self):
        return [self.primary, self.auxiliary]

    @property
    def param_groups(self):
        return self.primary.param_groups + self.auxiliary.param_groups

    @property
    def state(self):
        return {**self.primary.state, **self.auxiliary.state}

    def begin_epoch(self):
        self.residual_sum = 0.0
        self.residual_max = 0.0
        self.measurements = 0
        self.step_count = 0

    def step(self):
        super().step()
        self.step_count += 1
        for state in self.primary.state.values():
            residual = state["tangent_residual"]
            if not math.isfinite(residual) or state["inner_iterations"] != 10:
                raise FloatingPointError("invalid_manifold_dual_state")
            self.residual_sum += residual
            self.residual_max = max(self.residual_max, residual)
            self.measurements += 1

    @torch.no_grad()
    def constraint_metrics(self):
        metrics = {}
        for name, p in self.named_exponents:
            matrix = p.reshape(p.shape[0], -1) / self.primary.param_groups[0]["scale"]
            gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
            error = gram - torch.eye(gram.shape[0], device=p.device, dtype=p.dtype)
            metrics[f"manifold/constraint/{name}/rms"] = float(torch.linalg.vector_norm(error.double()) / math.sqrt(error.numel()))
            metrics[f"manifold/constraint/{name}/max_abs"] = float(error.abs().max())
        return metrics

    def epoch_metrics(self):
        return {**self.constraint_metrics(), "manifold/minibatch_count": self.step_count,
                "manifold/dual_measurement_count": self.measurements,
                "manifold/dual_iterations": 10,
                "manifold/tangent_residual_mean": self.residual_sum / self.measurements if self.measurements else None,
                "manifold/tangent_residual_max": self.residual_max,
                "auxiliary_learning_rate": self.auxiliary.param_groups[0]["lr"]}


def build_manifold_optimizer(model, args):
    named = [(name + ".weight", layer.weight) for name, layer in model.named_modules()
             if isinstance(layer, ProductUnitConv2d)]
    if not named:
        raise ValueError("Manifold PURe routing requires product exponent kernels")
    ids = {id(p) for _, p in named}
    auxiliary = [(name, p) for name, p in model.named_parameters() if id(p) not in ids]
    if not auxiliary or len(ids) != len(named):
        raise ValueError("invalid Manifold PURe parameter ownership")
    unprojected = parameter_digest(model)
    start = time.perf_counter()
    with torch.no_grad():
        for _, p in named:
            p.copy_(retract_stiefel(p, args.manifold_scale))
    projection_seconds = time.perf_counter() - start
    primary = ManifoldMuon([p for _, p in named], lr=args.learning_rate,
                          momentum=args.momentum, nesterov=args.momentum > 0,
                          dual_lr=args.dual_learning_rate, max_iterations=args.dual_iterations,
                          scale=args.manifold_scale)
    aux = torch.optim.AdamW([p for _, p in auxiliary], lr=args.aux_learning_rate,
                           betas=(.9, .95), eps=1e-8, weight_decay=0)
    bundle = PUReManifoldBundle(primary, aux)
    bundle.named_exponents = named
    bundle.begin_epoch()
    metadata = {"unprojected_initialization_digest": unprojected,
                "initialization_digest": parameter_digest(model),
                "manifold_projection_seconds": projection_seconds,
                "manifold_parameter_names": [name for name, _ in named],
                "manifold_matrix_shapes": {name: [p.shape[0], p.numel() // p.shape[0]] for name, p in named},
                "auxiliary_parameter_names": [name for name, _ in auxiliary],
                "manifold_nesterov": args.momentum > 0,
                "manifold_sign_backend": "exact_svd",
                "auxiliary_optimizer": "AdamW; betas=(0.9,0.95); eps=1e-8; weight_decay=0",
                "manifold_initial_constraints": bundle.constraint_metrics()}
    return bundle, metadata
