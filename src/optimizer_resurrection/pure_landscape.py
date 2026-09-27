"""Online gradient and local-loss diagnostics for PURe CIFAR training.

The module deliberately writes no artifacts.  It reports minibatch
aggregates, activation summaries from one training batch per epoch, and
fixed-probe local geometry.  The fixed probe is supplied by the caller and
must come from the normalized, non-augmented training set.
"""
from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


PROBE_SEED = 314159
PROBE_SIZE = 32
STANDARD_PROBE_EPOCHS = (0, 1, 40, 80, 81, 120, 121, 160)
SMOKE_PROBE_EPOCHS = (0, 1)
POWER_ITERATIONS = 8
HUTCHINSON_SAMPLES = 4
SLICE_ALPHAS = (-0.10, -0.05, 0.0, 0.05, 0.10)
_EPS = 1e-30


def fixed_probe_indices(n: int, size: int = PROBE_SIZE,
                        seed: int = PROBE_SEED) -> list[int]:
    """Return a deterministic subset of training indices without global RNG use."""
    if n < 1 or size < 1 or size > n:
        raise ValueError("probe size must be between one and the dataset size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(n, generator=generator)[:size].tolist()


def probe_epochs(args: Any) -> tuple[int, ...]:
    """Pre-registered probe schedule: full recipe or small engineering run."""
    stage = getattr(args, "stage", "")
    epochs = int(getattr(args, "epochs", 160))
    if stage in {"test", "engineering-smoke"} or epochs < 160:
        return tuple(dict.fromkeys((*SMOKE_PROBE_EPOCHS, epochs)))
    return STANDARD_PROBE_EPOCHS


def _finite_number(value: Tensor | float) -> float | None:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("expected scalar tensor")
        value = float(value.detach().item())
    else:
        value = float(value)
    return value if math.isfinite(value) else None


def _flat(parts: list[Tensor]) -> Tensor:
    if not parts:
        return torch.empty(0)
    return torch.cat([part.reshape(-1) for part in parts])


def _norm(tensor: Tensor) -> Tensor:
    # Diagnostic-only scalar reduction in FP64 avoids FP32 square overflow for
    # finite large gradients. Forward/backward, SGD and HVP remain FP32.
    return torch.linalg.vector_norm(tensor.detach().reshape(-1).to(torch.float64))


def _cosine(a: Tensor, b: Tensor) -> float | None:
    a64, b64 = a.reshape(-1).to(torch.float64), b.reshape(-1).to(torch.float64)
    anorm, bnorm = _norm(a64), _norm(b64)
    denominator = anorm * bnorm
    if not bool(torch.isfinite(denominator)) or float(denominator) == 0:
        return None
    return _finite_number(torch.dot(a64, b64) / denominator)


def _tensor_stats(tensor: Tensor) -> dict[str, float | None]:
    tensor = tensor.detach().float().reshape(-1)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        return {key: None for key in ("l2", "rms", "max_abs", "zero_fraction")}
    return {
        "l2": _finite_number(_norm(tensor)),
        "rms": _finite_number(_norm(tensor) / math.sqrt(tensor.numel())),
        "max_abs": _finite_number(tensor.abs().max()),
        "zero_fraction": _finite_number((tensor == 0).float().mean()),
    }


class LandscapeDiagnostics:
    """Collect gradient, activation, and fixed-probe landscape summaries.

    ``before_step`` is called after ``loss.backward()`` and immediately before
    the optimizer step; ``after_step`` is called immediately afterwards.  The
    returned epoch dictionary contains only scalar summaries.  Probe slice
    points are returned as named scalars so they can be logged online without
    creating local result files.
    """

    def __init__(self, model: nn.Module, args: Any,
                 probe: tuple[Tensor, Tensor]) -> None:
        self.model = model
        self.args = args
        self.weight_decay = float(getattr(args, "weight_decay", 0.0))
        self.seed = int(getattr(args, "seed", 0))
        self.probe_x, self.probe_y = probe
        if self.probe_x.shape[0] != self.probe_y.shape[0] or self.probe_x.shape[0] == 0:
            raise ValueError("fixed probe inputs and labels must have matching nonzero batches")
        if not self.weight_decay >= 0 or not math.isfinite(self.weight_decay):
            raise ValueError("weight decay must be finite and nonnegative")

        self._named_parameters = [(name, p) for name, p in model.named_parameters()
                                  if p.requires_grad]
        self._scopes = self._make_scopes()
        self._series: dict[str, list[float]] = defaultdict(list)
        self._batch_count = 0
        self._valid_batches = 0
        self._nonfinite_batches = 0
        self._previous_gradients: dict[str, Tensor] = {}
        self._step_snapshots: dict[str, list[tuple[str, nn.Parameter, Tensor]]] = {}
        self._activation_values: dict[str, dict[str, float | None]] = {}
        self._activation_handles: list[Any] = []
        self._capture_activations = False
        self._activation_epoch: int | None = None
        self._max_locations: dict[str, tuple[float, int]] = {}
        self._diagnostic_seconds = 0.0

    @property
    def diagnostic_seconds(self) -> float:
        """Time spent inside diagnostic methods (activation-hook work is excluded)."""
        return self._diagnostic_seconds

    def _make_scopes(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        from .models.pure_resnet import ProductUnitConv2d

        modules = dict(self.model.named_modules())
        scopes: dict[str, list[tuple[str, nn.Parameter]]] = {
            "global": list(self._named_parameters),
        }
        groups: dict[str, list[tuple[str, nn.Parameter]]] = defaultdict(list)
        layers: dict[str, list[tuple[str, nn.Parameter]]] = defaultdict(list)
        for name, parameter in self._named_parameters:
            module_name, _, local_name = name.rpartition(".")
            module = modules.get(module_name)
            if isinstance(module, ProductUnitConv2d):
                group = "pu_theta" if local_name == "theta" else "pu_weight"
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                group = "bn"
            elif isinstance(module, nn.Linear) and module_name == "fc":
                group = "head"
            elif isinstance(module, nn.Conv2d):
                group = "conv"
            else:
                group = "other"
            groups[group].append((name, parameter))
            layers[module_name or "root"].append((name, parameter))
        for group, values in groups.items():
            scopes[f"group/{group}"] = values
        for layer, values in layers.items():
            scopes[f"layer/{layer}"] = values
        return scopes

    def begin_epoch(self, epoch: int) -> None:
        """Arm activation capture for the first product-unit forward this epoch."""
        self.close_epoch_hooks()
        self._activation_values = {}
        self._activation_epoch = int(epoch)
        self._capture_activations = True
        from .models.pure_resnet import ProductUnitConv2d

        for name, module in self.model.named_modules():
            if isinstance(module, ProductUnitConv2d):
                self._activation_handles.append(
                    module.register_forward_hook(self._activation_hook(name))
                )

    def _activation_hook(self, name: str):
        from .models.pure_resnet import ProductUnitConv2d

        def capture(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
            if not self._capture_activations or not isinstance(module, ProductUnitConv2d):
                return
            x = inputs[0].detach().float()
            threshold = module.threshold.detach().float()
            with torch.no_grad():
                finite_input = bool(torch.isfinite(x).all())
                clamped = torch.maximum(x, threshold)
                log_input = torch.log(clamped)
                pre_exp = F.conv2d(
                    log_input, module.weight.detach().float(), bias=None,
                    stride=module.stride, padding=module.padding,
                    dilation=module.dilation,
                )
                out = output.detach().float()
                finite_pre = pre_exp[torch.isfinite(pre_exp)]
                finite_out = out[torch.isfinite(out)]
                values: dict[str, float | None] = {
                    "input_floor_fraction": _finite_number((x <= threshold).float().mean())
                    if finite_input else None,
                    "log_input_min": _finite_number(log_input.min()) if finite_input else None,
                    "log_input_max": _finite_number(log_input.max()) if finite_input else None,
                    "pre_exp_min": _finite_number(finite_pre.min()) if finite_pre.numel() else None,
                    "pre_exp_max": _finite_number(finite_pre.max()) if finite_pre.numel() else None,
                    "pre_exp_overflow_fraction": _finite_number(
                        (pre_exp > math.log(torch.finfo(torch.float32).max)).float().mean()
                    ),
                    "pre_exp_underflow_risk_fraction": _finite_number(
                        (pre_exp < (-149.0 * math.log(2.0))).float().mean()
                    ),
                    "output_finite_fraction": _finite_number(torch.isfinite(out).float().mean()),
                    "output_zero_fraction": _finite_number((out == 0).float().mean()),
                    "output_max_finite": _finite_number(finite_out.max()) if finite_out.numel() else None,
                }
                values["finite"] = float(finite_input and bool(torch.isfinite(pre_exp).all())
                                          and bool(torch.isfinite(out).all()))
                self._activation_values[name] = values

        return capture

    def close_epoch_hooks(self) -> None:
        self._capture_activations = False
        for handle in self._activation_handles:
            handle.remove()
        self._activation_handles.clear()

    def close(self) -> None:
        """Remove any pending activation hooks; safe to call repeatedly."""
        self.close_epoch_hooks()

    def _grad_parts(self, entries: list[tuple[str, nn.Parameter]], effective: bool
                    ) -> list[Tensor]:
        parts = []
        for _, parameter in entries:
            grad = parameter.grad
            if grad is None:
                parts.append(torch.zeros_like(parameter, memory_format=torch.preserve_format))
            else:
                value = grad.detach()
                if effective:
                    value = value + self.weight_decay * parameter.detach()
                parts.append(value)
        return parts

    def _record_metric(self, key: str, value: float | None) -> None:
        if value is not None and math.isfinite(value):
            self._series[key].append(value)

    def _record_location(self, key: str, value: float | None, batch_index: int) -> None:
        if value is None or not math.isfinite(value):
            return
        previous = self._max_locations.get(key)
        if previous is None or value > previous[0]:
            self._max_locations[key] = (value, batch_index)

    def before_step(self) -> None:
        started = time.perf_counter()
        self.close_epoch_hooks()
        self._batch_count += 1
        finite = True
        current_vectors: dict[str, Tensor] = {}
        for scope, entries in self._scopes.items():
            data = _flat(self._grad_parts(entries, effective=False))
            effective = _flat(self._grad_parts(entries, effective=True))
            params = _flat([p.detach() for _, p in entries])
            grad_finite = bool(torch.isfinite(data).all())
            effective_finite = bool(torch.isfinite(effective).all())
            finite &= grad_finite and effective_finite
            prefix = f"{scope}/"
            self._record_metric(prefix + "data_finite", float(grad_finite))
            for metric, value in _tensor_stats(data).items():
                self._record_metric(prefix + "data_" + metric, value)
                if scope == "global" and metric == "l2":
                    self._record_location("global/data_l2", value, self._batch_count - 1)
            for metric, value in _tensor_stats(effective).items():
                self._record_metric(prefix + "effective_" + metric, value)
            if scope == "global" and bool(torch.isfinite(params).all()):
                self._record_metric(prefix + "parameter_l2", _finite_number(_norm(params)))
            ratio = _finite_number(_norm(data) / (_norm(params) + _EPS)) \
                if grad_finite and bool(torch.isfinite(params).all()) else None
            self._record_metric(prefix + "data_grad_to_param_ratio", ratio)
            effective_ratio = _finite_number(_norm(effective) / (_norm(params) + _EPS)) \
                if effective_finite and bool(torch.isfinite(params).all()) else None
            self._record_metric(prefix + "effective_grad_to_param_ratio", effective_ratio)
            vector = data.detach().reshape(-1)
            track_cosine = scope == "global" or scope.startswith("group/")
            previous = self._previous_gradients.get(scope) if track_cosine else None
            cosine = None
            if grad_finite:
                if previous is not None and previous.numel() == vector.numel():
                    cosine = _cosine(vector, previous)
                if track_cosine:
                    current_vectors[scope] = vector.clone()
            self._record_metric(prefix + "consecutive_data_gradient_cosine", cosine)

        if finite:
            self._valid_batches += 1
            self._previous_gradients = {scope: value for scope, value in current_vectors.items()}
        else:
            self._nonfinite_batches += 1
            self._previous_gradients.clear()

        # Scopes overlap, so store one old value per parameter only.
        self._step_snapshots = {
            "parameters": [(name, parameter, parameter.detach().clone())
                           for name, parameter in self._named_parameters]
        }
        self._diagnostic_seconds += time.perf_counter() - started

    def after_step(self) -> None:
        started = time.perf_counter()
        deltas = {name: parameter.detach() - before
                  for name, parameter, before in self._step_snapshots.get("parameters", [])}
        for scope, entries in self._scopes.items():
            delta = _flat([deltas[name] for name, _ in entries])
            value = _finite_number(_norm(delta)) if bool(torch.isfinite(delta).all()) else None
            key = f"{scope}/actual_update_l2"
            self._record_metric(key, value)
            if scope == "global":
                self._record_location(key, value, max(0, self._batch_count - 1))
        self._step_snapshots.clear()
        self._diagnostic_seconds += time.perf_counter() - started

    def epoch_metrics(self) -> dict[str, int | float | None]:
        """Return and clear the per-epoch minibatch aggregate."""
        self.close_epoch_hooks()
        result: dict[str, int | float | None] = {
            "landscape/train/minibatch_count": self._batch_count,
            "landscape/train/gradient_valid_batches": self._valid_batches,
            "landscape/train/nonfinite_gradient_batches": self._nonfinite_batches,
            "landscape/train/activation_epoch": self._activation_epoch,
        }
        for key, values in self._series.items():
            output_key = "landscape/train/" + key
            result[output_key + "_mean"] = sum(values) / len(values) if values else None
            result[output_key + "_max"] = max(values) if values else None
        for metric, (_value, batch_index) in self._max_locations.items():
            result[f"landscape/train/{metric}_max_batch"] = batch_index
        for module, metrics in self._activation_values.items():
            for metric, value in metrics.items():
                result[f"landscape/activation/{module}/{metric}"] = value
        self._series.clear()
        self._batch_count = self._valid_batches = self._nonfinite_batches = 0
        self._previous_gradients.clear()
        self._max_locations.clear()
        self._activation_values = {}
        self._activation_epoch = None
        return result

    def _model_and_rng_snapshot(self) -> dict[str, Any]:
        parameters = [(p, p.detach().clone(), p.grad,
                       p.grad.detach().clone() if p.grad is not None else None)
                      for p in self.model.parameters()]
        buffers = [(b, b.detach().clone()) for b in self.model.buffers()]
        return {
            "parameters": parameters,
            "buffers": buffers,
            "training": [(module, module.training) for module in self.model.modules()],
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and torch.cuda.is_initialized() else None,
        }

    def _restore_model_and_rng(self, state: dict[str, Any]) -> None:
        with torch.no_grad():
            for parameter, value, original_grad, grad_value in state["parameters"]:
                parameter.copy_(value)
                parameter.grad = original_grad
                if original_grad is not None and grad_value is not None:
                    original_grad.copy_(grad_value)
            for buffer, value in state["buffers"]:
                buffer.copy_(value)
        # Assign individual flags to preserve mixed train/eval submodule modes.
        for module, training in state["training"]:
            module.training = training
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def _objective(self, x: Tensor, y: Tensor) -> Tensor:
        logits = self.model(x)
        ce = F.cross_entropy(logits, y)
        penalty = sum(parameter.square().sum() for _, parameter in self._named_parameters)
        return ce + (self.weight_decay / 2.0) * penalty

    def _bn_comparison(self, x: Tensor, y: Tensor) -> dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            eval_logits = self.model(x)
            eval_loss = F.cross_entropy(eval_logits, y)
        buffers = [(b, b.detach().clone()) for b in self.model.buffers()]
        try:
            self.model.train()
            with torch.no_grad():
                batch_logits = self.model(x)
                batch_loss = F.cross_entropy(batch_logits, y)
        finally:
            with torch.no_grad():
                for buffer, value in buffers:
                    buffer.copy_(value)
            self.model.eval()
        eval_finite = bool(torch.isfinite(eval_logits).all()) and bool(torch.isfinite(eval_loss))
        batch_finite = bool(torch.isfinite(batch_logits).all()) and bool(torch.isfinite(batch_loss))
        finite = eval_finite and batch_finite
        return {
            "landscape/probe/bn_comparison_valid": finite,
            "landscape/probe/bn_eval_finite": eval_finite,
            "landscape/probe/bn_batch_stats_finite": batch_finite,
            "landscape/probe/bn_comparison_status": "valid" if finite else "invalid_numerical",
            "landscape/probe/bn_eval_ce_loss": _finite_number(eval_loss) if eval_finite else None,
            "landscape/probe/bn_batch_stats_ce_loss": _finite_number(batch_loss) if batch_finite else None,
            "landscape/probe/bn_eval_vs_batch_logit_max_abs":
                _finite_number((eval_logits - batch_logits).abs().max()) if finite else None,
            "landscape/probe/bn_prediction_disagreement_fraction":
                _finite_number((eval_logits.argmax(1) != batch_logits.argmax(1)).float().mean())
                if finite else None,
            "landscape/probe/bn_misclassification_count_delta":
                int((batch_logits.argmax(1) != y).sum()) - int((eval_logits.argmax(1) != y).sum())
                if finite else None,
        }

    def _unflatten(self, vector: Tensor) -> tuple[Tensor, ...]:
        result, offset = [], 0
        for _, parameter in self._named_parameters:
            count = parameter.numel()
            result.append(vector[offset:offset + count].view_as(parameter))
            offset += count
        return tuple(result)

    def _hvp(self, gradients: tuple[Tensor, ...], vector: Tensor) -> Tensor:
        product = torch.autograd.grad(
            gradients, [p for _, p in self._named_parameters],
            grad_outputs=self._unflatten(vector), retain_graph=True,
            allow_unused=True, create_graph=False,
        )
        values = [torch.zeros_like(parameter) if part is None else part
                  for (_, parameter), part in zip(self._named_parameters, product)]
        return _flat(values)

    def _local_geometry(self, x: Tensor, y: Tensor, epoch: int) -> dict[str, Any]:
        prefix = "landscape/probe/"
        result: dict[str, Any] = {
            prefix + "hessian_valid": False,
            prefix + "hessian_status": "invalid_numerical",
            prefix + "hessian_reason": "",
            prefix + "slice_valid": False,
            prefix + "slice_status": "invalid_numerical",
        }
        self.model.eval()
        parameters = [p for _, p in self._named_parameters]
        with torch.enable_grad():
            objective = self._objective(x, y)
            if not bool(torch.isfinite(objective.detach())):
                result[prefix + "hessian_reason"] = "nonfinite_probe_objective"
                self._null_geometry(result)
                return result
            gradients = torch.autograd.grad(objective, parameters, create_graph=True,
                                            retain_graph=True, allow_unused=True)
            gradients = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, gradients))
            grad_finite = all(bool(torch.isfinite(g.detach()).all()) for g in gradients)
            regularized_vector = _flat([g.detach() for g in gradients])
            parameter_vector = _flat([p.detach() for p in parameters])
            data_vector = regularized_vector - self.weight_decay * parameter_vector
            result.update({
                prefix + "probe_data_gradient_l2": _finite_number(_norm(data_vector))
                if bool(torch.isfinite(data_vector).all()) else None,
                prefix + "probe_objective_gradient_l2": _finite_number(_norm(regularized_vector))
                if grad_finite else None,
                prefix + "probe_data_gradient_to_parameter_ratio":
                    _finite_number(_norm(data_vector) / (_norm(parameter_vector) + _EPS))
                    if bool(torch.isfinite(data_vector).all()) and bool(torch.isfinite(parameter_vector).all())
                    else None,
            })
            if not grad_finite:
                result[prefix + "hessian_reason"] = "nonfinite_probe_gradient"
                self._null_geometry(result)
                return result
            device = parameters[0].device
            count = sum(p.numel() for p in parameters)
            rng = torch.Generator(device="cpu").manual_seed(PROBE_SEED + self.seed + 17011)
            vector = torch.randn(count, generator=rng, dtype=torch.float32).to(device)
            norm = _norm(vector)
            if not bool(torch.isfinite(norm)) or float(norm) == 0:
                result[prefix + "hessian_reason"] = "invalid_power_initial_vector"
                self._null_geometry(result)
                return result
            vector = vector / norm
            hv = None
            for _ in range(POWER_ITERATIONS):
                hv = self._hvp(gradients, vector)
                hv_norm = _norm(hv)
                if not bool(torch.isfinite(hv_norm)) or not bool(torch.isfinite(hv).all()):
                    result[prefix + "hessian_reason"] = "nonfinite_hessian_vector_product"
                    self._null_geometry(result)
                    return result
                if float(hv_norm) <= _EPS:
                    vector = torch.zeros_like(vector)
                    break
                vector = hv / hv_norm
            hv = self._hvp(gradients, vector)
            eigenvalue = torch.dot(vector, hv)
            residual = _norm(hv - eigenvalue * vector)
            if not bool(torch.isfinite(eigenvalue)) or not bool(torch.isfinite(residual)):
                result[prefix + "hessian_reason"] = "nonfinite_power_iteration_result"
                self._null_geometry(result)
                return result
            result.update({
                prefix + "hessian_valid": True,
                prefix + "hessian_status": "valid",
                prefix + "hessian_reason": "",
                prefix + "hessian_power_iterations": POWER_ITERATIONS,
                prefix + "hessian_signed_rayleigh": _finite_number(eigenvalue),
                prefix + "hessian_largest_magnitude_estimate": _finite_number(eigenvalue.abs()),
                prefix + "hessian_residual_norm": _finite_number(residual),
                prefix + "hessian_relative_residual": _finite_number(residual / (eigenvalue.abs() + _EPS)),
                prefix + "hessian_rayleigh_data": _finite_number(eigenvalue - self.weight_decay),
            })
            estimates = []
            for sample in range(HUTCHINSON_SAMPLES):
                trace_rng = torch.Generator(device="cpu").manual_seed(
                    PROBE_SEED + self.seed + 23003 + sample
                )
                rademacher = torch.randint(0, 2, (count,), generator=trace_rng,
                                           dtype=torch.int8).to(device=device, dtype=torch.float32)
                rademacher.mul_(2).sub_(1)
                trace_hv = self._hvp(gradients, rademacher)
                estimate = torch.dot(rademacher, trace_hv)
                number = _finite_number(estimate)
                if number is None:
                    result[prefix + "hessian_valid"] = False
                    result[prefix + "hessian_status"] = "invalid_numerical"
                    result[prefix + "hessian_reason"] = "nonfinite_hutchinson_estimate"
                    self._null_geometry(result, keep_hessian_reason=True)
                    return result
                estimates.append(number)
            result[prefix + "hessian_hutchinson_samples"] = HUTCHINSON_SAMPLES
            result[prefix + "hessian_trace_estimate"] = sum(estimates) / len(estimates)
            result[prefix + "hessian_trace_sample_sd"] = (
                float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None
            )
            result[prefix + "hessian_trace_standard_error"] = (
                result[prefix + "hessian_trace_sample_sd"] / math.sqrt(len(estimates))
                if len(estimates) > 1 else None
            )
            result[prefix + "hessian_trace_data_estimate"] = (
                result[prefix + "hessian_trace_estimate"]
                - self.weight_decay * sum(p.numel() for p in parameters)
            )

        return result

    def _loss_slices(self, x: Tensor, y: Tensor, epoch: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        parameter_entries = self._named_parameters
        direction_rngs = [torch.Generator(device="cpu").manual_seed(
            PROBE_SEED + self.seed + 31013 + i
        ) for i in range(2)]
        original = [(p, p.detach().clone()) for _, p in parameter_entries]
        all_directions: list[list[Tensor]] = []
        valid = True
        try:
            for direction_index, generator in enumerate(direction_rngs, start=1):
                directions = []
                for _, parameter in parameter_entries:
                    random_direction = torch.randn(parameter.shape, generator=generator,
                                                   dtype=torch.float32)
                    base = parameter.detach().float().cpu()
                    if parameter.ndim >= 2:
                        random_matrix = random_direction.reshape(parameter.shape[0], -1)
                        random_norm = torch.linalg.vector_norm(random_matrix, dim=1).clamp_min(_EPS)
                        base_norm = torch.linalg.vector_norm(base.reshape(parameter.shape[0], -1), dim=1)
                        scaled = random_matrix * (base_norm / random_norm).unsqueeze(1)
                        direction = scaled.reshape(parameter.shape)
                    else:
                        random_norm = torch.linalg.vector_norm(random_direction).clamp_min(_EPS)
                        base_norm = torch.linalg.vector_norm(base)
                        direction = random_direction * (base_norm / random_norm)
                    directions.append(direction.to(device=parameter.device, dtype=parameter.dtype))
                all_directions.append(directions)
                for alpha in SLICE_ALPHAS:
                    with torch.no_grad():
                        for (parameter, base), direction in zip(original, directions):
                            parameter.copy_(base + alpha * direction)
                    with torch.no_grad():
                        logits = self.model(x)
                        ce = F.cross_entropy(logits, y)
                        penalty = (self.weight_decay / 2.0) * sum(
                            parameter.square().sum() for _, parameter in parameter_entries
                        )
                        objective = ce + penalty
                    ce_number, objective_number = _finite_number(ce), _finite_number(objective)
                    point_valid = ce_number is not None and objective_number is not None
                    valid &= point_valid
                    key = f"landscape/probe/slice{direction_index}"
                    result[f"{key}/valid_alpha_{alpha:+.2f}"] = point_valid
                    result[f"{key}/ce_alpha_{alpha:+.2f}"] = ce_number
                    result[f"{key}/objective_alpha_{alpha:+.2f}"] = objective_number
                with torch.no_grad():
                    for parameter, base in original:
                        parameter.copy_(base)

            # The joint local slice evaluates all 25 combinations of the two
            # fixed filter-normalized directions.
            first, second = all_directions
            for alpha in SLICE_ALPHAS:
                for beta in SLICE_ALPHAS:
                    with torch.no_grad():
                        for (parameter, base), d1, d2 in zip(original, first, second):
                            parameter.copy_(base + alpha * d1 + beta * d2)
                        logits = self.model(x)
                        ce = F.cross_entropy(logits, y)
                        penalty = (self.weight_decay / 2.0) * sum(
                            parameter.square().sum() for _, parameter in parameter_entries
                        )
                        objective = ce + penalty
                    ce_number, objective_number = _finite_number(ce), _finite_number(objective)
                    point_valid = ce_number is not None and objective_number is not None
                    valid &= point_valid
                    key = f"landscape/probe/slice2d/"
                    suffix = f"a_{alpha:+.2f}_b_{beta:+.2f}"
                    result[key + "valid_" + suffix] = point_valid
                    result[key + "ce_" + suffix] = ce_number
                    result[key + "objective_" + suffix] = objective_number
        finally:
            with torch.no_grad():
                for parameter, base in original:
                    parameter.copy_(base)
        result["_slice_valid"] = valid
        return result

    def _slice_metrics(self, x: Tensor, y: Tensor, epoch: int) -> dict[str, Any]:
        prefix = "landscape/probe/"
        try:
            values = self._loss_slices(x, y, epoch)
            valid = values.pop("_slice_valid")
            values.update({
                prefix + "slice_valid": valid,
                prefix + "slice_status": "valid" if valid else "invalid_numerical",
                prefix + "slice_reason": "" if valid else "one_or_more_nonfinite_slice_points",
            })
            return values
        except FloatingPointError as exc:
            values: dict[str, Any] = {
                prefix + "slice_valid": False,
                prefix + "slice_status": "invalid_numerical",
                prefix + "slice_reason": f"{type(exc).__name__}: {exc}",
            }
            for direction in (1, 2):
                for alpha in SLICE_ALPHAS:
                    values[f"{prefix}slice{direction}/valid_alpha_{alpha:+.2f}"] = False
                    values[f"{prefix}slice{direction}/ce_alpha_{alpha:+.2f}"] = None
                    values[f"{prefix}slice{direction}/objective_alpha_{alpha:+.2f}"] = None
            for alpha in SLICE_ALPHAS:
                for beta in SLICE_ALPHAS:
                    base = f"{prefix}slice2d/"
                    suffix = f"a_{alpha:+.2f}_b_{beta:+.2f}"
                    values[base + "valid_" + suffix] = False
                    values[base + "ce_" + suffix] = None
                    values[base + "objective_" + suffix] = None
            return values

    @staticmethod
    def _null_geometry(result: dict[str, Any], keep_hessian_reason: bool = False) -> None:
        prefix = "landscape/probe/"
        for key in (
            "hessian_signed_rayleigh", "hessian_largest_magnitude_estimate",
            "hessian_rayleigh_data", "hessian_residual_norm", "hessian_relative_residual",
            "hessian_trace_estimate", "hessian_trace_data_estimate",
            "hessian_trace_sample_sd", "hessian_trace_standard_error",
            "hessian_hutchinson_samples",
            "hessian_power_iterations",
        ):
            result[prefix + key] = None
        for direction in (1, 2):
            for alpha in SLICE_ALPHAS:
                result[f"{prefix}slice{direction}/valid_alpha_{alpha:+.2f}"] = False
                result[f"{prefix}slice{direction}/ce_alpha_{alpha:+.2f}"] = None
                result[f"{prefix}slice{direction}/objective_alpha_{alpha:+.2f}"] = None
        for alpha in SLICE_ALPHAS:
            for beta in SLICE_ALPHAS:
                base = f"{prefix}slice2d/"
                suffix = f"a_{alpha:+.2f}_b_{beta:+.2f}"
                result[base + "valid_" + suffix] = False
                result[base + "ce_" + suffix] = None
                result[base + "objective_" + suffix] = None
        result.setdefault(prefix + "hessian_reason", "")
        result[prefix + "slice_valid"] = False
        result.setdefault(prefix + "slice_status", "invalid_numerical")
        result.setdefault(prefix + "slice_reason", "")
        if not keep_hessian_reason:
            result[prefix + "hessian_valid"] = False
            result[prefix + "hessian_status"] = "invalid_numerical"

    def probe(self, epoch: int) -> dict[str, Any]:
        """Measure fixed-probe BN behavior, curvature, and two local loss slices.

        This method restores parameters, buffers, mode flags, existing gradients,
        and Python/NumPy/CPU/CUDA RNG state even when a probe calculation fails.
        CUDA OOM and unexpected runtime errors are re-raised to avoid disguising
        infrastructure or implementation faults as a scientific invalidity.
        """
        started = time.perf_counter()
        state = self._model_and_rng_snapshot()
        result: dict[str, Any] = {
            "landscape/probe/epoch": int(epoch),
            "landscape/probe/valid": False,
            "landscape/probe/status": "invalid_numerical",
            "landscape/probe/reason": "",
        }
        try:
            device = next(self.model.parameters()).device
            x, y = self.probe_x.to(device), self.probe_y.to(device)
            self.model.eval()
            self._activation_values = {}
            self._capture_activations = True
            from .models.pure_resnet import ProductUnitConv2d
            for name, module in self.model.named_modules():
                if isinstance(module, ProductUnitConv2d):
                    self._activation_handles.append(
                        module.register_forward_hook(self._activation_hook(name))
                    )
            with torch.no_grad():
                logits = self.model(x)
                ce = F.cross_entropy(logits, y)
                objective = ce + (self.weight_decay / 2.0) * sum(
                    p.detach().square().sum() for _, p in self._named_parameters
                )
            self.close_epoch_hooks()
            for module, metrics in self._activation_values.items():
                for metric, value in metrics.items():
                    result[f"landscape/probe/activation/{module}/{metric}"] = value
            result["landscape/probe/parameter_count"] = sum(
                p.numel() for _, p in self._named_parameters
            )
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(objective)):
                result["landscape/probe/reason"] = "nonfinite_probe_logits_or_objective"
                result["landscape/probe/hessian_reason"] = "nonfinite_probe_logits_or_objective"
                result["landscape/probe/slice_reason"] = "nonfinite_probe_logits_or_objective"
                self._null_geometry(result)
                result.update(self._bn_comparison(x, y))
                return result
            result.update({
                "landscape/probe/valid": True,
                "landscape/probe/status": "valid",
                "landscape/probe/reason": "",
                "landscape/probe/ce_loss": _finite_number(ce),
                "landscape/probe/objective": _finite_number(objective),
                "landscape/probe/logit_max_abs": _finite_number(logits.abs().max()),
                "landscape/probe/logit_rms": _finite_number(_norm(logits) / math.sqrt(logits.numel())),
                "landscape/probe/accuracy": _finite_number((logits.argmax(1) == y).float().mean()),
            })
            result.update(self._bn_comparison(x, y))
            try:
                geometry = self._local_geometry(x, y, int(epoch))
            except (FloatingPointError, NotImplementedError) as exc:
                geometry = {
                    "landscape/probe/hessian_valid": False,
                    "landscape/probe/hessian_status": "unsupported",
                    "landscape/probe/hessian_reason": f"{type(exc).__name__}: {exc}",
                    "landscape/probe/slice_valid": False,
                    "landscape/probe/slice_status": "unsupported",
                }
                self._null_geometry(geometry, keep_hessian_reason=True)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message or "cuda error" in message:
                    raise
                if any(token in message for token in (
                    "derivative", "not implemented", "does not require grad",
                    "differentiated tensor", "backward through the graph",
                )):
                    geometry = {
                        "landscape/probe/hessian_valid": False,
                        "landscape/probe/hessian_status": "unsupported",
                        "landscape/probe/hessian_reason": f"RuntimeError: {exc}",
                        "landscape/probe/slice_valid": False,
                        "landscape/probe/slice_status": "unsupported",
                    }
                    self._null_geometry(geometry, keep_hessian_reason=True)
                else:
                    raise
            result.update(geometry)
            # Slices are independent of Hessian-vector-product validity; a
            # higher-derivative failure must not discard finite loss surfaces.
            result.update(self._slice_metrics(x, y, int(epoch)))
            return result
        finally:
            self.close_epoch_hooks()
            self._restore_model_and_rng(state)
            self._diagnostic_seconds += time.perf_counter() - started

