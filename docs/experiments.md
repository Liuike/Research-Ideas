# Experiment ledger

This is the source-controlled index of experiments for the Optimizer
Resurrection study. W&B is the authoritative run store; this document records
intent, provenance, status, and decisions without copying metric values out of
W&B.

## Recording rules

1. Add an entry before launching a new experiment group.
2. Use a unique semantic W&B group and record the committed config path.
3. Record the git revision and source-tree digest shown in W&B. A dirty-tree run
   may be used for engineering smoke tests but not for scientific claims.
4. Record every expected condition and seed in the plan. Treat duplicate
   finished condition IDs, missing cells, crashed runs, and excluded seeds as
   explicit failures until resolved.
5. Link the W&B group, gate-check run, recipe-selection run, and boundary-analysis
   run when they exist. Do not paste console summaries or create custom local
   result files; W&B-managed local logs and cache are allowed.
6. Update `Status` and `Decision` after review; never rewrite the original
   purpose or protocol.

## Experiment index

| Experiment ID | Purpose | Config | W&B group | Expected scope | Status | Decision / follow-up |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy-local-smokes-2026-09-05` | Early implementation checks created before W&B-only logging | ad hoc | none | Small MLP/RNN optimizer checks | Superseded and cleaned | Raw local records were deleted without migration; they were engineering smoke data and are not claimable. |
| `wandb-pipeline-smoke-2026-09-08` | Verify mandatory online logging and absence of new raw output files | ad hoc one-step Latch/Adam-O CPU run | `implementation-validation-2026-09-08` | One temporary verification run | Passed and cleaned | W&B API confirmed a finished run with condition ID, provenance, seeds, terminal metrics, and gate-summary fields. The remote run and local W&B cache were then deleted to establish a clean experimental slate. |
| `local-gpu-smoke-v1` | Verify the full training/logging path on the RTX 4060 Ti | [`scripts/local_smoke.ps1`](../scripts/local_smoke.ps1) | `local-gpu-smoke` | One ShapeSet/Adam-O run and one Latch/MM run, 10 steps each | Planned | Require finished W&B runs, matching condition IDs, deterministic metadata, finite losses, and no project-local artifacts. |
| `exploratory-2xl40s-smoke-v1` | Verify the committed code, CUDA, W&B, and all four candidate optimizers on Oscar | [`configs/exploratory/smoke_mlp_2xl40s.yaml`](../configs/exploratory/smoke_mlp_2xl40s.yaml), [`configs/exploratory/smoke_rnn_2xl40s.yaml`](../configs/exploratory/smoke_rnn_2xl40s.yaml) | `exploratory-2xl40s-smoke-v1` | 8 disposable runs, 25 steps each | Passed and cleaned | Oscar job `6117431` completed all eight cells on revision `7964bb7`; W&B coverage/provenance was verified and the group was deleted. |
| `exploratory-2xl40s-tests-v1` | Reduced qualitative probes for Gates A-C | [`configs/exploratory/gates_2xl40s.yaml`](../configs/exploratory/gates_2xl40s.yaml) | `exploratory-2xl40s-tests-v1` | 30 runs | Completed after retry | Array `6117578` completed 26/30 gate cells but lost tasks 9, 10, 11, and 13 without retained stderr. Revision `dfe0166` enabled ignored Slurm logs; retry array `6122331` reran only those cells, and all four exited `0:0` with complete W&B summaries. This remains exploratory: two seeds cannot establish formal gate passage. |
| `exploratory-2xl40s-lr-scout-v1` | Equal three-candidate LR scout at one anchor per track | [`configs/exploratory/lr_scout_2xl40s.yaml`](../configs/exploratory/lr_scout_2xl40s.yaml) | `exploratory-2xl40s-lr-scout-v1` | 24 runs | Completed and selected | Array `6117578`, tasks 38-61, completed 24/24. Replacement CPU selector `6122332` completed `0:0` after retry `6122331`. |
| `exploratory-2xl40s-boundary-v1` | Two-seed ShapeSet-depth and Latch-length signal check using scout-selected recipes | [`configs/exploratory/boundary_mlp_2xl40s.yaml`](../configs/exploratory/boundary_mlp_2xl40s.yaml), [`configs/exploratory/boundary_rnn_2xl40s.yaml`](../configs/exploratory/boundary_rnn_2xl40s.yaml) | `exploratory-2xl40s-boundary-v1` | 32 runs | Running | Replacement Oscar array `6122333` was released by selector `6122332`; `%2`, one GPU per task, L40S, revision `dfe0166`, W&B logging, and retained ignored Slurm logs are enforced. |
| `reproduction-v1` | Evaluate historical failure modes and Stiefel preservation (Gates A-C) | [`configs/gates.yaml`](../configs/gates.yaml) | `reproduction-v1` | Every configured gate condition and seed | Blocked on smoke validation | `headline_ready=true` is mandatory before calibration or headline experiments. |
| `historical-b0` | Historical BPTT-SGD generator and difficulty baseline without Muon claims | [`configs/rnn/historical_b0.yaml`](../configs/rnn/historical_b0.yaml) | `historical-b0` | 480 configured runs | Deferred; execution fields incomplete | Do not launch until steps, batch size, and variable-length behavior are explicit. |
| `calibration-v1` | Equal-budget learning-rate calibration | [`configs/calibration.yaml`](../configs/calibration.yaml) | `calibration-v1` | 16 trials per optimizer at only the registered calibration axes | Blocked on Gates A-C | Record the recipe-selection W&B run and frozen recipe digest. |
| `mlp-mvp-v1` | Bounded MNIST depth sweep using frozen recipes | [`configs/mlp/mvp.yaml`](../configs/mlp/mvp.yaml) | `mlp-mnist-mvp-v1` | 160 runs | Blocked on calibration | No pilot defaults may be reported as calibrated results. |
| `rnn-mvp-v1` | Bounded Latch length sweep using frozen recipes | [`configs/rnn/mvp.yaml`](../configs/rnn/mvp.yaml) | `rnn-latch-mvp-v1` | 200 runs | Blocked on calibration | No pilot defaults may be reported as calibrated results. |
| `full-study-v1` | Fixed-recipe boundary evaluation | [`configs/full_study.yaml`](../configs/full_study.yaml) | `full-study-v1` | Full registered MLP/RNN matrix | Blocked on MVP review | Run strict completeness checks before D50/T50 analysis. |
| `oracle-v1` | Separately labelled per-task oracle tuning | [`configs/oracle.yaml`](../configs/oracle.yaml) | `oracle-v1` | Equal-budget oracle cells | Blocked on primary study | Keep separate from fixed-recipe headline comparisons. |

## Current protocol notes

- Manifold Muon's dual ascent always uses exactly 10 iterations. The final
  tangent residual is retained as a diagnostic rather than an abort condition.
- Primary comparisons use constant learning rate, FP32, zero weight decay, no
  AMP, and no gradient clipping.
- W&B credentials and project identity come from `.secrets/env`; secret values
  never belong in this document.
- Oscar jobs run from `/users/ezhan153/random-idea-1` and use the Brown campus
  SSH endpoint directly while this device is on secure campus Wi-Fi.
