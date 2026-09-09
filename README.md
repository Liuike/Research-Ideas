# Optimizer Resurrection

This repository implements the verification study “Can Muon Resurrect
Historically Optimization-Limited Architectures?” It compares SGD, AdamW,
Muon, Riemannian SGD, and Stiefel Manifold Muon on deep sigmoid/tanh MLPs and
plain tanh RNN long-dependency tasks.

The scientific targets are trainability boundaries (`D50` and `T50`), not small
final-loss differences. The implementation keeps update geometry,
initialization, and maintained weight constraints separate.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[analysis,test]"
.\scripts\test.ps1
```

The default test suite keeps optimizer mathematics, task generators,
configuration expansion, and reproducibility checks, while skipping the
mocked-W&B training integration test. Run the complete suite with:

```powershell
.\scripts\test.ps1 -Full
```

Both commands use a reusable directory under Windows system temp, keyed to the
actual process account. This avoids the Codex sandbox's collision with Luke's
private `pytest-of-Luke` directory and does not leave pytest roots in the repo.

Fill `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT` in `.secrets/env`.
The file is ignored by Git. Training requires online W&B and creates no custom
raw config, metric, result, checkpoint, or `outputs/` artifacts. W&B's own
managed local logs and cache are allowed. Scientific stages require a clean
committed revision; `engineering-smoke` runs may use a dirty tree. Then run a
short smoke test:

```powershell
$env:PYTHONPATH="src"
python -m optimizer_resurrection.train --track rnn --task latch --optimizer adam_o --hidden-size 16 --max-length 20 --steps 5 --batch-size 8 --device cuda --stage engineering-smoke
```

Use `scripts/local_smoke.ps1` for the 4060 Ti verification run. Long MVP arrays
are in `scripts/oscar_mlp_mvp.sbatch` and `scripts/oscar_rnn_mvp.sbatch`.
On Brown's secure campus Wi-Fi, use the campus SSH endpoint directly without a
VPN. See `AGENTS.md` and `docs/experiment_protocol.md`.

## Orchestration

Committed configs are executable rather than informal notes:

```powershell
# Inspect a plan without running it.
.venv\Scripts\python -m optimizer_resurrection.experiment plan configs\mlp\mvp.yaml --max-runs 3

# Run an expanded plan locally (use --max-runs for bounded pilots).
.venv\Scripts\python -m optimizer_resurrection.experiment plan configs\mlp\mvp.yaml --run --max-runs 1

# Select frozen learning rates from completed W&B calibration runs.
.venv\Scripts\python -m optimizer_resurrection.experiment select-recipes --run-group calibration-v1

# Full-study expansion refuses to proceed without the frozen recipe artifact.
.venv\Scripts\python -m optimizer_resurrection.experiment plan configs\full_study.yaml --recipes configs\frozen_recipes.json

# This queries W&B and exits nonzero until reproduction Gates A-C all pass.
.venv\Scripts\python -m optimizer_resurrection.experiment check-gates --run-group reproduction-v1
```

For long plans, add `--plan-file plans/full.jsonl`, sync the repository, then
submit with an explicit array matching the plan line count, for example
`sbatch --array=0-199 scripts/oscar_plan.sbatch plans/mvp.jsonl`.
`scripts/oscar_plan.sbatch` maps exactly one JSON command to each array task and
supports `DRY_RUN=1`. Plan files are orchestration inputs, not result logs;
metrics and terminal results exist only in W&B.

`configs/oracle.yaml` expands the separately labelled equal-budget, per-task
oracle sweep. `analysis/critical_boundary.py` groups by task/model, learning
rates, assignment, scale, and recipe version; use its expected-axis/seed flags
with `--strict` so missing hard runs cannot silently inflate a boundary.
Track every launched study and its W&B group in `docs/experiments.md`.

## Layout

- `src/optimizer_resurrection`: models, tasks, optimizers, diagnostics, training
- `configs`: frozen calibration/MVP recipes
- `scripts`: local and Oscar launchers
- `analysis`: critical-boundary estimation
- `tests`: numerical, generator, assignment, and reproducibility checks
