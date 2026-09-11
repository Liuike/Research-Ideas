"""Runnable, explicitly reconstructed Section 5.5 Two-Sequence and Parity baselines.

See docs/historical_sequence_protocols.md for recovered facts and assumptions.
Existing historical Latch identities and behavior remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import torch
from .historical_latch import (
    condition_id, resolved_config, _require_finite_model,
    SUCCESS_ACCURACY, SUCCESS_LOSS,
)
from .historical_sequence_tasks import (
    PROTOCOLS, TwoSequence, Parity, generate_fixed_dataset, dataset_digest,
)
from .tracking import load_wandb_credentials, require_online_wandb
from .train import (
    _determinism_metadata, _environment_metadata, _git_metadata,
    _require_reproducible_source, _sanitized_invocation, _source_tree_digest,
    seed_everything,
)

TASK_ASSUMPTIONS = {
    PROTOCOLS[0]: "Two fixed trial-specific uniform templates shared across splits; "
        "prefixes selected by balanced class; additive input unit 0; output unit 4; "
        "zero initial state; target +/-0.8; no bias. Template process and readout "
        "index are reconstruction assumptions.",
    PROTOCOLS[1]: "Seven-parameter reconstructed graph: h=tanh(a*u+b*y_prev+c), "
        "y=tanh(d*u+e*y_prev+f*h+g), y0=0. Exact historical wiring is unresolved. "
        "Balanced clean-symbol parity; target +/-1; additive noise after labels.",
}

def target_magnitude(protocol):
    return 1.0 if protocol == PROTOCOLS[1] else 0.8

def losses(outputs, labels, protocol):
    target = (2 * labels.to(outputs) - 1) * target_magnitude(protocol)
    return 0.5 * (outputs - target).square()

@dataclass(frozen=True)
class SequenceRunResult:
    terminal_outcome: str
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

@torch.no_grad()
def evaluate(model, data, device, protocol):
    output = model(data.inputs.to(device), data.lengths.to(device))
    return (float(losses(output, data.labels, protocol).mean()),
            float(((output >= 0) == data.labels.to(device).bool()).float().mean()))

def temporal_diagnostics(model, inputs, label, protocol):
    records = {}
    for dtype, name in ((torch.float32, 'fp32'), (torch.float64, 'float64_reference')):
        reference = copy.deepcopy(model).to(dtype=dtype)
        probe = inputs.detach().to(next(model.parameters()).device, dtype).reshape(1,-1)
        probe.requires_grad_(True)
        states = reference.trace(probe)
        loss = losses(states[-1][:,-1], torch.tensor([label]), protocol).sum()
        gradients = torch.autograd.grad(loss, (probe, *states), allow_unused=True)
        filled = [torch.zeros_like(x) if g is None else g
                  for g, x in zip(gradients, (probe, *states))]
        if not all(bool(torch.isfinite(g).all()) for g in filled) or not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite temporal diagnostic')
        records[name] = {'input_gradient': filled[0].detach().cpu().flatten().tolist(),
                         'state_gradient': torch.stack(filled[1:], dim=1)[0].detach().cpu().tolist(),
                         'loss': float(loss.detach())}
    def flattened(record):
        return record['input_gradient'] + [value for state in record['state_gradient'] for value in state]
    a = torch.tensor(flattened(records['fp32']), dtype=torch.float64)
    b = torch.tensor(flattened(records['float64_reference']), dtype=torch.float64)
    records['fp32_underflow'] = bool(((a == 0) & (b != 0)).any())
    return records

def train(args, *, device, wandb_run, resolved_condition_id,
          train_data=None, validation_data=None):
    def data(size, split):
        return generate_fixed_dataset(protocol=args.protocol, size=size,
            max_length=args.max_length, noise_amplitude=args.noise_amplitude,
            data_seed=args.data_seed, split=split)
    train_data = train_data if train_data is not None else data(args.train_size, 'train')
    validation_data = validation_data if validation_data is not None else data(args.validation_size, 'validation')
    model = (TwoSequence if args.protocol == PROTOCOLS[0] else Parity)(args.seed).to(device)
    expected = 25 if args.protocol == PROTOCOLS[0] else 7
    assert sum(p.numel() for p in model.parameters()) == expected
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.0, weight_decay=0.0)
    count = 0
    last = None
    def snapshot(step):
        nonlocal count, last
        _require_finite_model(model, context=f'at presentation {step}')
        last = (*evaluate(model, train_data, device, args.protocol),
                *evaluate(model, validation_data, device, args.protocol))
        if not all(math.isfinite(v) for v in last):
            raise FloatingPointError('nonfinite evaluation')
        temporal = {}
        for label in (0,1):
            indices = torch.nonzero(validation_data.labels == label).flatten()
            index = int(indices[validation_data.lengths[indices].argmax()])
            length = int(validation_data.lengths[index])
            temporal[str(label)] = temporal_diagnostics(model,
                validation_data.inputs[index,:length], label, args.protocol)
        increment = len(train_data.labels) + len(validation_data.labels) + 4
        count += increment
        wandb_run.log(dict(presentation=step, step=step,
            optimizer_sequence_presentations=step,
            diagnostic_evaluation_sequences=count,
            diagnostic_evaluation_sequences_this_snapshot=increment,
            train_loss=last[0], train_accuracy=last[1], validation_loss=last[2],
            validation_accuracy=last[3], parameters=model.parameter_values(),
            temporal_gradients=temporal), step=step)
    snapshot(0)
    for presentation in range(1, args.presentations + 1):
        index = (presentation - 1) % len(train_data.labels)
        optimizer.zero_grad(set_to_none=True)
        output = model(train_data.inputs[index:index+1].to(device),
                       train_data.lengths[index:index+1].to(device))
        loss = losses(output, train_data.labels[index:index+1], args.protocol).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite training loss')
        loss.backward()
        _require_finite_model(model, context=f'before update {presentation}')
        optimizer.step()
        _require_finite_model(model, context=f'after update {presentation}')
        if presentation == 1 or presentation % args.log_every == 0 or presentation == args.presentations:
            snapshot(presentation)
    return SequenceRunResult('completed', args.presentations, *last,
        resolved_condition_id,
        last[0] <= SUCCESS_LOSS and last[2] <= SUCCESS_LOSS and
        last[1] >= SUCCESS_ACCURACY and last[3] >= SUCCESS_ACCURACY,
        count, dataset_digest(train_data), dataset_digest(validation_data))

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
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
    parser.add_argument("--recipe-version", default="historical-sequences-reconstruction-v1")
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


def main(argv: list[str] | None = None) -> SequenceRunResult | None:
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
        "optimizer_resurrection.historical_sequences",
        *(sys.argv[1:] if argv is None else argv),
    ]
    train_data = generate_fixed_dataset(
        protocol=args.protocol,
        size=args.train_size,
        max_length=args.max_length,
        noise_amplitude=args.noise_amplitude,
        data_seed=args.data_seed,
        split="train",
    )
    validation_data = generate_fixed_dataset(
        protocol=args.protocol,
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
            "citation": "Bengio, Simard, and Frasconi (1994), Section 5.5",
            "url": "https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Bengio_94.pdf",
        },
        "fixed_dataset_digests": data_digests,
        "reconstruction_assumptions": {
            "task_definition": TASK_ASSUMPTIONS[args.protocol],
            "target_magnitude": target_magnitude(args.protocol),
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
        f"{args.protocol}__T{args.max_length}__noise{args.noise_amplitude:g}__"
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
    try:
        wandb_run.config.update(
            {"provenance": {**provenance, "wandb": identity}}, allow_val_change=True
        )
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
        try:
            wandb_run.summary.update({"terminal_outcome": "failed"})
        finally:
            wandb_run.finish(exit_code=1)
        raise
    else:
        wandb_run.finish(exit_code=0)
    print(json.dumps(asdict(result), sort_keys=True))
    return result


if __name__ == "__main__":
    main()
