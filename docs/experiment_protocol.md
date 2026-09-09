# Experiment protocol

## Stages

1. Correctness: run all tests, then one CPU step for every optimizer.
2. Reproduction: verify increasing logistic depth raises saturation (Gate A),
   ordinary BPTT degrades with length (Gate B), and every maintained Stiefel
   matrix has normalized orthogonality error below `1e-4` (Gate C).
3. Calibration: use exactly 16 learning-rate trials per optimizer only at MLP
   depths 4 and 8 and RNN maximum lengths 50 and 100.
4. Fixed recipe: freeze one recipe per optimizer, then run the full depth/length
   sweeps without retuning.
5. Oracle: run a separately labelled equal-budget per-task sweep.

Use `optimizer_resurrection.experiment` to expand committed configs, select
calibration recipes from completed W&B runs, and enforce the reproduction
gates. `full_study.yaml` cannot be expanded without the selected recipe JSON.
The analysis command queries W&B; its
`--strict --expect-axis-values ... --expect-seeds ...` mode rejects incomplete
curves rather than silently dropping interrupted or failed jobs.

The MVP uses MNIST at depths 4, 8, 16, 32 for logistic and tanh, plus Latch at
lengths 50, 100, 200, 500, 1000 with hidden size 64. Optimizers are Adam-O,
Muon-O, RSGD, and MM. Use five MLP seeds and ten RNN seeds.

## Success thresholds

Thresholds are configuration, not post-hoc analysis choices. A run succeeds
only when both `train_loss <= success_loss` and
`validation_accuracy >= success_accuracy`. Record every seed, including failed
or interrupted runs. Fit a binomial logistic curve against depth or log length;
bootstrap seeds for confidence intervals.

## Credentials, logging, and Oscar

Store W&B and Oscar identifiers in `.secrets/env`. Never store a password or Duo
token. Assume this device is on Brown's secure campus Wi-Fi and connect directly
to `sshcampus.ccv.brown.edu` with the configured SSH key; ask Luke for an
interactive session only if key authentication fails and Duo is required.

W&B is the authoritative run store. Training must fail before the first step
unless online W&B initialization succeeds. W&B-managed local logs and cache are
allowed, but project code must not create custom config, metric, result,
checkpoint, or `outputs/` artifacts. Each W&B run must contain its resolved scientific config,
exact sanitized command, condition ID, seeds, source-tree digest, git state,
software/hardware metadata, scheduler identifiers, step history, and terminal
summary.

Engineering smoke tests may record a dirty source tree, but reproduction,
calibration, MVP, oracle, and full-study runs must fail before training unless
the repository has a clean committed revision.

After connecting, sync the repository, create/activate the environment, run
`bash -n`, representative `DRY_RUN=1` tasks, `sbatch --test-only`, and only then
submit the arrays. Actual Slurm stdout/stderr is discarded because console and
metric logging belong to W&B; dry runs remain visible in the invoking terminal.

The Oscar MVP scripts accept optimizer-specific frozen recipes via
`LR_ADAM_O`, `LR_MUON_O`, `LR_RSGD`, and `LR_MM`. Before a fixed-recipe sweep,
set those to calibration-selected values, set a semantic `RECIPE_VERSION`, and
change `LOG_GROUP` from its default `*-pilot` label. The defaults are smoke/pilot
values and must not be reported as equal-budget calibrated results.
