"""Reconstruct the latch experiment from Bengio, Simard, and Frasconi (1994).

The paper's Sections 3 and 5.5 specify a single tanh recurrent unit,

    x_t = tanh(w * x_{t-1} + h_t),  x_0 = 0,

with three adaptive scalars: ``w`` and one class-specific initial input for
each class.  All three are initialized uniformly in [-0.5, 0.5].  The target
is -0.8 or +0.8 at the final time, later inputs are either zero or uniform
noise in [-0.2, 0.2], examples are fixed within a trial, and lengths are
uniform integers from ceil(T / 2) through T.

The paper does not report the SGD learning rate, number of examples in a
trial, ordering of examples, or an exact presentation budget.  Protocol
``bsf1994-latch-v1`` therefore registers the following reconstruction choices:
100 balanced training examples, 1,000 balanced validation examples,
sequential cyclic batch-one SGD, constant learning rate, no momentum, weight
decay, or clipping, FP32 training, and 5,000 sequence presentations.  These
choices are explicit in every W&B config and must not be described as recovered
historical settings.  Likewise, the operational success rule (train and
validation mean loss at most 0.1 and sign accuracy at least 0.95) is a registered
reconstruction criterion, not a criterion reported by the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .tracking import load_wandb_credentials, require_online_wandb
from .train import (
    _determinism_metadata,
    _environment_metadata,
    _git_metadata,
    _require_reproducible_source,
    _sanitized_invocation,
    _source_tree_digest,
    seed_everything,
)


PROTOCOL = "bsf1994-latch-v1"
TARGET_MAGNITUDE = 0.8
SUCCESS_LOSS = 0.1
SUCCESS_ACCURACY = 0.95
_RUNTIME_KEYS = {"condition_id", "device", "dry_run", "run_group", "run_name", "stage"}


@dataclass(frozen=True)
class FixedLatchDataset:
    """A fixed, balanced set of latch sequences for one trial."""

    labels: torch.Tensor
    lengths: torch.Tensor
    later_inputs: torch.Tensor

    def __post_init__(self) -> None:
        size = int(self.labels.numel())
        if self.labels.shape != (size,) or self.lengths.shape != (size,):
            raise ValueError("labels and lengths must be one-dimensional and equally sized")
        if self.later_inputs.ndim != 2 or self.later_inputs.shape[0] != size:
            raise ValueError("later_inputs must have shape [examples, max_length - 1]")

    def __len__(self) -> int:
        return int(self.labels.numel())


@dataclass(frozen=True)
class LatchRunResult:
    terminal_outcome: Literal["completed"]
    presentations: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    condition_id: str
    success: bool
    diagnostic_evaluation_sequences: int
    train_dataset_digest: str
    validation_dataset_digest: str


class HistoricalLatch(nn.Module):
    """The paper's one-unit, three-parameter latching system."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.empty(3, dtype=torch.float32).uniform_(
            -0.5, 0.5, generator=generator
        )
        self.w = nn.Parameter(initial[0].clone())
        self.u_negative = nn.Parameter(initial[1].clone())
        self.u_positive = nn.Parameter(initial[2].clone())

    def forward(
        self,
        labels: torch.Tensor,
        lengths: torch.Tensor,
        later_inputs: torch.Tensor,
        *,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return each sequence's final state, optionally with padded states.

        ``later_inputs[:, t - 1]`` is h_t for recurrence step t+1.  Inactive
        padded steps preserve the preceding state, so the returned final state
        corresponds to each example's own length.
        """
        if labels.ndim != 1 or lengths.shape != labels.shape:
            raise ValueError("labels and lengths must have matching one-dimensional shapes")
        if later_inputs.ndim != 2 or later_inputs.shape[0] != labels.shape[0]:
            raise ValueError("later_inputs must have shape [batch, max_length - 1]")
        if int(lengths.min()) < 1 or int(lengths.max()) > later_inputs.shape[1] + 1:
            raise ValueError("length is outside the represented sequence span")

        labels = labels.to(device=self.w.device)
        lengths = lengths.to(device=self.w.device)
        later_inputs = later_inputs.to(device=self.w.device, dtype=self.w.dtype)
        initial_input = torch.where(labels.bool(), self.u_positive, self.u_negative)
        state = torch.tanh(self.w * torch.zeros_like(initial_input) + initial_input)
        states = [state]
        for step in range(2, later_inputs.shape[1] + 2):
            proposed = torch.tanh(self.w * state + later_inputs[:, step - 2])
            state = torch.where(lengths >= step, proposed, state)
            states.append(state)
        if return_states:
            return state, torch.stack(states, dim=1)
        return state

    def parameter_values(self) -> dict[str, float]:
        return {
            "w": float(self.w.detach()),
            "u_negative": float(self.u_negative.detach()),
            "u_positive": float(self.u_positive.detach()),
        }


def _stream_seed(data_seed: int, split: str, purpose: str) -> int:
    """Domain-separate deterministic data streams without consuming global RNG."""
    payload = f"{PROTOCOL}:{data_seed}:{split}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def generate_fixed_dataset(
    *,
    size: int,
    max_length: int,
    noise_amplitude: float,
    data_seed: int,
    split: str,
) -> FixedLatchDataset:
    """Generate fixed paired data using independent label/length/noise streams.

    Multiplying a common base-noise stream by ``noise_amplitude`` means an
    amplitude-zero and amplitude-0.2 trial with the same seed have identical
    labels and lengths.  Noise generation cannot perturb either stream.
    """
    if size <= 0 or size % 2:
        raise ValueError("dataset size must be a positive even integer for exact balance")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if not math.isfinite(noise_amplitude) or noise_amplitude < 0:
        raise ValueError("noise_amplitude must be finite and nonnegative")
    if not split:
        raise ValueError("split must be nonempty")

    label_generator = torch.Generator().manual_seed(_stream_seed(data_seed, split, "labels"))
    length_generator = torch.Generator().manual_seed(_stream_seed(data_seed, split, "lengths"))
    noise_generator = torch.Generator().manual_seed(_stream_seed(data_seed, split, "noise"))

    labels = torch.cat(
        (torch.zeros(size // 2, dtype=torch.long), torch.ones(size // 2, dtype=torch.long))
    )
    labels = labels[torch.randperm(size, generator=label_generator)]
    minimum_length = (max_length + 1) // 2
    lengths = torch.randint(
        minimum_length, max_length + 1, (size,), generator=length_generator
    )
    base_noise = torch.empty(size, max_length - 1, dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=noise_generator
    )
    later_inputs = base_noise.mul(float(noise_amplitude))
    # Padded values do not affect the recurrence, but zeroing them makes the
    # represented dataset unambiguous in diagnostics and provenance checks.
    steps = torch.arange(2, max_length + 1).unsqueeze(0)
    later_inputs = later_inputs.masked_fill(steps > lengths.unsqueeze(1), 0.0)
    return FixedLatchDataset(labels=labels, lengths=lengths, later_inputs=later_inputs)


def dataset_digest(dataset: FixedLatchDataset) -> str:
    """Hash the exact fixed tensors used by a trial."""
    digest = hashlib.sha256()
    for name, tensor in (
        ("labels", dataset.labels),
        ("lengths", dataset.lengths),
        ("later_inputs", dataset.later_inputs),
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _targets(labels: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    signed = labels.to(device=device, dtype=dtype).mul(2.0).sub(1.0)
    return signed.mul(TARGET_MAGNITUDE)


def _losses(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = _targets(labels, dtype=outputs.dtype, device=outputs.device)
    return 0.5 * (outputs - targets).square()


@torch.no_grad()
def evaluate(
    model: HistoricalLatch,
    dataset: FixedLatchDataset,
    device: torch.device,
) -> tuple[float, float]:
    outputs = model(
        dataset.labels.to(device),
        dataset.lengths.to(device),
        dataset.later_inputs.to(device),
    )
    losses = _losses(outputs, dataset.labels)
    predictions = outputs >= 0
    return float(losses.mean()), float((predictions == dataset.labels.to(device).bool()).float().mean())


def _temporal_gradients_at_dtype(
    model: HistoricalLatch,
    *,
    label: int,
    later_inputs: torch.Tensor,
    dtype: torch.dtype,
) -> dict[str, list[float] | float]:
    """Differentiate one final loss through every input and state in time."""
    device = model.w.device
    w = model.w.detach().to(dtype=dtype)
    class_input = (
        model.u_positive if label == 1 else model.u_negative
    ).detach().to(dtype=dtype)
    inputs = torch.cat((class_input.reshape(1), later_inputs.to(device=device, dtype=dtype))).detach()
    inputs.requires_grad_(True)
    state = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
    states = [state]
    for value in inputs:
        state = torch.tanh(w * state + value)
        states.append(state)
    target = torch.tensor(
        TARGET_MAGNITUDE if label == 1 else -TARGET_MAGNITUDE,
        device=device,
        dtype=dtype,
    )
    loss = 0.5 * (state - target).square()
    input_gradient, *state_gradients = torch.autograd.grad(loss, (inputs, *states))
    return {
        "loss": float(loss.detach()),
        "input_gradient": [float(value) for value in input_gradient.detach().cpu()],
        "state_gradient": [float(value) for value in torch.stack(state_gradients).detach().cpu()],
    }


def temporal_gradient_diagnostics(
    model: HistoricalLatch,
    *,
    label: int,
    later_inputs: torch.Tensor,
) -> dict[str, Any]:
    """Return full-span FP32 gradients and a float64 reference on underflow.

    The reference is diagnostic only: it uses detached parameter values and
    never participates in an optimizer update.
    """
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    if later_inputs.ndim != 1:
        raise ValueError("later_inputs must be one-dimensional")
    fp32 = _temporal_gradients_at_dtype(
        model, label=label, later_inputs=later_inputs, dtype=torch.float32
    )
    reference = _temporal_gradients_at_dtype(
        model, label=label, later_inputs=later_inputs, dtype=torch.float64
    )
    fp32_values = fp32["input_gradient"] + fp32["state_gradient"]
    reference_values = reference["input_gradient"] + reference["state_gradient"]
    if not all(math.isfinite(value) for value in fp32_values):
        raise FloatingPointError("FP32 temporal gradients are not finite")
    if not math.isfinite(float(fp32["loss"])):
        raise FloatingPointError("FP32 diagnostic loss is not finite")
    underflow = any(a == 0.0 and b != 0.0 for a, b in zip(fp32_values, reference_values))
    result: dict[str, Any] = {"fp32": fp32, "fp32_underflow": underflow}
    if underflow:
        if not all(math.isfinite(value) for value in reference_values):
            raise FloatingPointError("float64 temporal-gradient reference is not finite")
        result["float64_reference"] = reference
    return result


def _diagnostic_examples(dataset: FixedLatchDataset) -> dict[int, tuple[int, torch.Tensor]]:
    result: dict[int, tuple[int, torch.Tensor]] = {}
    for label in (0, 1):
        label_lengths = dataset.lengths[dataset.labels == label]
        longest_for_label = int(label_lengths.max())
        matches = torch.nonzero(
            (dataset.labels == label) & (dataset.lengths == longest_for_label),
            as_tuple=False,
        ).flatten()
        index = int(matches[0])
        length = int(dataset.lengths[index])
        result[label] = (length, dataset.later_inputs[index, : length - 1])
    return result


def _require_finite_model(model: HistoricalLatch, *, context: str) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"non-finite parameter {name!r} {context}")
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"non-finite gradient {name!r} {context}")


def train(
    args: argparse.Namespace,
    *,
    device: torch.device,
    wandb_run: Any,
    resolved_condition_id: str,
    train_data: FixedLatchDataset | None = None,
    validation_data: FixedLatchDataset | None = None,
) -> LatchRunResult:
    """Run exactly ``presentations`` cyclic batch-one plain-SGD updates."""
    train_data = train_data or generate_fixed_dataset(
        size=args.train_size,
        max_length=args.max_length,
        noise_amplitude=args.noise_amplitude,
        data_seed=args.data_seed,
        split="train",
    )
    validation_data = validation_data or generate_fixed_dataset(
        size=args.validation_size,
        max_length=args.max_length,
        noise_amplitude=args.noise_amplitude,
        data_seed=args.data_seed,
        split="validation",
    )
    model = HistoricalLatch(args.seed).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != 3:
        raise AssertionError("historical Latch model must have exactly three parameters")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.0, weight_decay=0.0
    )
    probes = _diagnostic_examples(validation_data)
    final_loss = float("nan")
    diagnostic_evaluation_sequences = 0
    last_metrics: tuple[float, float, float, float] | None = None

    def log_snapshot(presentation: int, example_train_loss: float | None) -> None:
        nonlocal diagnostic_evaluation_sequences, last_metrics
        _require_finite_model(model, context=f"at presentation {presentation}")
        train_loss, train_accuracy = evaluate(model, train_data, device)
        validation_loss, validation_accuracy = evaluate(model, validation_data, device)
        metrics = (train_loss, train_accuracy, validation_loss, validation_accuracy)
        if not all(math.isfinite(value) for value in metrics):
            raise FloatingPointError(f"non-finite evaluation metric at presentation {presentation}")
        temporal: dict[str, Any] = {}
        for label, name in ((0, "negative"), (1, "positive")):
            length, inputs = probes[label]
            temporal[name] = {
                "sequence_length": length,
                **temporal_gradient_diagnostics(
                    model, label=label, later_inputs=inputs.to(device)
                ),
            }
        # Each temporal probe is evaluated in both FP32 and float64.
        evaluated_now = len(train_data) + len(validation_data) + 2 * len(probes)
        diagnostic_evaluation_sequences += evaluated_now
        last_metrics = metrics
        record = {
            "presentation": presentation,
            "step": presentation,
            "optimizer_sequence_presentations": presentation,
            "diagnostic_evaluation_sequences": diagnostic_evaluation_sequences,
            "diagnostic_evaluation_sequences_this_snapshot": evaluated_now,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "parameters": model.parameter_values(),
            "temporal_gradients": temporal,
        }
        if example_train_loss is not None:
            record["example_train_loss"] = example_train_loss
        wandb_run.log(record, step=presentation)

    # Capture initialization before the first parameter update.  This is the
    # reference point for determining whether temporal gradients already vanish.
    log_snapshot(0, None)

    for presentation in range(1, args.presentations + 1):
        index = (presentation - 1) % len(train_data)
        label = train_data.labels[index : index + 1].to(device)
        length = train_data.lengths[index : index + 1].to(device)
        later_inputs = train_data.later_inputs[index : index + 1].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(label, length, later_inputs)
        loss = _losses(output, label).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss at presentation {presentation}")
        loss.backward()
        _require_finite_model(model, context=f"before update {presentation}")
        optimizer.step()
        _require_finite_model(model, context=f"after update {presentation}")
        final_loss = float(loss.detach())

        if presentation == 1 or presentation % args.log_every == 0 or presentation == args.presentations:
            log_snapshot(presentation, final_loss)

    if last_metrics is None:
        raise AssertionError("final diagnostic snapshot was not recorded")
    train_loss, train_accuracy, validation_loss, validation_accuracy = last_metrics
    return LatchRunResult(
        terminal_outcome="completed",
        presentations=args.presentations,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        validation_loss=validation_loss,
        validation_accuracy=validation_accuracy,
        condition_id=resolved_condition_id,
        success=(
            train_loss <= SUCCESS_LOSS
            and validation_loss <= SUCCESS_LOSS
            and train_accuracy >= SUCCESS_ACCURACY
            and validation_accuracy >= SUCCESS_ACCURACY
        ),
        diagnostic_evaluation_sequences=diagnostic_evaluation_sequences,
        train_dataset_digest=dataset_digest(train_data),
        validation_dataset_digest=dataset_digest(validation_data),
    )


def scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key not in _RUNTIME_KEYS}


def condition_id(config: dict[str, Any]) -> str:
    """Hash scientific fields from either a scientific or fully resolved config."""
    scientific = {key: value for key, value in config.items() if key not in _RUNTIME_KEYS}
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return all resolved CLI values and verify an asserted condition ID."""
    resolved = dict(vars(args))
    if resolved["run_group"] is None:
        resolved["run_group"] = f"{resolved['stage']}/{resolved['recipe_version']}"
    computed = condition_id(resolved)
    asserted = resolved.get("condition_id")
    if asserted is not None and asserted != computed:
        raise ValueError(
            f"asserted condition_id {asserted!r} does not match resolved config {computed!r}"
        )
    resolved["condition_id"] = computed
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=[PROTOCOL], required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--noise-amplitude", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--presentations", type=int, default=5_000)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", default="historical-reproduction")
    parser.add_argument("--run-group")
    parser.add_argument("--run-name")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--recipe-version", default="historical-latch-initial-v1")
    parser.add_argument("--condition-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if args.noise_amplitude < 0 or not math.isfinite(args.noise_amplitude):
        parser.error("--noise-amplitude must be finite and nonnegative")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        parser.error("--learning-rate must be finite and positive")
    if args.presentations <= 0:
        parser.error("--presentations must be positive")
    if args.train_size <= 0 or args.train_size % 2:
        parser.error("--train-size must be a positive even integer")
    if args.validation_size <= 0 or args.validation_size % 2:
        parser.error("--validation-size must be a positive even integer")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    return args


def main(argv: list[str] | None = None) -> LatchRunResult | None:
    args = parse_args(argv)
    resolved = resolved_config(args)
    args = argparse.Namespace(**resolved)
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return None

    project_root = Path(__file__).resolve().parents[2]
    credentials = load_wandb_credentials(project_root)
    require_online_wandb()
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    git_metadata = _git_metadata(project_root)
    _require_reproducible_source(args.stage, git_metadata)
    invocation_argv = [
        sys.executable,
        "-m",
        "optimizer_resurrection.historical_latch",
        *(sys.argv[1:] if argv is None else argv),
    ]
    train_data = generate_fixed_dataset(
        size=args.train_size,
        max_length=args.max_length,
        noise_amplitude=args.noise_amplitude,
        data_seed=args.data_seed,
        split="train",
    )
    validation_data = generate_fixed_dataset(
        size=args.validation_size,
        max_length=args.max_length,
        noise_amplitude=args.noise_amplitude,
        data_seed=args.data_seed,
        split="validation",
    )
    data_digests = {
        "train": dataset_digest(train_data),
        "validation": dataset_digest(validation_data),
    }
    provenance = {
        "invocation": _sanitized_invocation(invocation_argv, credentials),
        "git": git_metadata,
        "source_tree": _source_tree_digest(project_root),
        "environment": _environment_metadata(device),
        "determinism": _determinism_metadata(args.seed, args.data_seed),
        "historical_source": {
            "citation": "Bengio, Simard, and Frasconi (1994), Sections 3 and 5.5",
            "url": "https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Bengio_94.pdf",
        },
        "fixed_dataset_digests": data_digests,
        "reconstruction_assumptions": {
            "train_size": args.train_size,
            "validation_size": args.validation_size,
            "example_order": "sequential cyclic",
            "batch_size": 1,
            "optimizer": "plain SGD",
            "learning_rate": args.learning_rate,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "gradient_clipping": None,
            "dtype": "float32",
            "presentations": args.presentations,
            "operational_success_rule": {
                "maximum_train_loss": SUCCESS_LOSS,
                "maximum_validation_loss": SUCCESS_LOSS,
                "minimum_train_accuracy": SUCCESS_ACCURACY,
                "minimum_validation_accuracy": SUCCESS_ACCURACY,
                "historical_setting": False,
            },
        },
    }
    run_config = {**vars(args), "provenance": provenance}
    run_name = args.run_name or (
        f"historical-latch__T{args.max_length}__noise{args.noise_amplitude:g}__"
        f"seed{args.seed}__{args.condition_id[:8]}"
    )

    import wandb

    wandb_run = wandb.init(
        project=credentials["WANDB_PROJECT"],
        entity=credentials["WANDB_ENTITY"],
        name=run_name,
        group=args.run_group,
        config=run_config,
        mode="online",
        settings=wandb.Settings(mode="online"),
        save_code=False,
    )
    if wandb_run is None:
        raise RuntimeError("W&B online initialization returned no run")
    identity = {
        "id": str(wandb_run.id),
        "name": str(wandb_run.name),
        "entity": str(getattr(wandb_run, "entity", credentials["WANDB_ENTITY"])),
        "project": str(getattr(wandb_run, "project", credentials["WANDB_PROJECT"])),
        "group": args.run_group,
        "url": str(wandb_run.url),
    }
    wandb_run.config.update(
        {"provenance": {**provenance, "wandb": identity}}, allow_val_change=True
    )
    try:
        result = train(
            args,
            device=device,
            wandb_run=wandb_run,
            resolved_condition_id=args.condition_id,
            train_data=train_data,
            validation_data=validation_data,
        )
        payload = asdict(result)
        wandb_run.summary.update({**payload, "run_result": payload})
    except BaseException:
        wandb_run.summary.update({"terminal_outcome": "failed"})
        wandb_run.finish(exit_code=1)
        raise
    else:
        wandb_run.finish(exit_code=0)
    print(json.dumps(asdict(result), sort_keys=True))
    return result


if __name__ == "__main__":
    main()
