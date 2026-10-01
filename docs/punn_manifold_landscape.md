# DA-10 recorded landscape study

The recorded study completed all 480 logical conditions: 116 on CUDA and 364
on CPU, following Luke's request to stop the GPU sweep and transfer remaining
conditions to CPU. Both ordinary parameter slices and feasible manifold slices
were retained. The initial frozen record is
[`manifold_landscape_gpu.yaml`](../configs/product_unit/manifold_landscape_gpu.yaml):
the same 16 task/architecture cells, 30 seeds each, and 500 epochs (480 runs).
The continuation uses `manifold_landscape_cpu_continuation.yaml`. Training and
landscape evaluation use the same explicit device within each run, in FP32.
The original CPU outcomes are retained as a separate study. Plain DA-10's
learning rates, momentum, ten dual iterations, scale 1, auxiliary AdamW head,
batch-one sample order, and explicit classification L2 settings are unchanged.
There is no tuning, clipping or mixed precision; device assignments are explicit
in the registered GPU and CPU configurations.

Each run records copied parameter states at initialization, every completed
epoch, and terminal outcome into W&B-owned artifacts. At initialization,
epoch 1, every 50 epochs, and terminal, it records 31-by-31 slices with radius
1 in each coordinate:

- **Ambient:** two seeded orthonormal directions in the complete flattened
  parameter space; this can leave the Stiefel constraint and measures
  sensitivity in the ordinary PUNN parameter domain.
- **Manifold:** project the corresponding exponent-direction blocks onto the
  tangent space at that center, keep the output head free, orthonormalize,
  then polar-retract displaced exponent matrices. This measures sensitivity
  within the feasible model domain. Projection/normalization changes the
  directions; equal coordinate radius is not equal realized displacement.

Raw MSE and the L2 training objective are distinguished. Nonfinite samples,
nonfinite terminal states, and unavailable slices must be retained explicitly;
they cannot disappear from the scientific denominator. Slice centers, direction
vectors, data identity, masks, and feasibility diagnostics must be retained so
the reported neighborhoods can be independently checked. Diagnostic RNG and
evaluation must not alter training parameters, sample order, or optimizer state.

The f1 small exponent matrix is 1-by-1 and its Stiefel tangent space has
dimension zero: its feasible slices vary the output head only. Stability,
two-dimensional sensitivity, and final prediction error are different
measurements. Neither view establishes full-space flatness or a causal
explanation for DA-10's numerical stability. Prior SGD/AdamW/Moonlight Muon
landscape artifacts cover f1/f4 only and use different optimizer recipes;
any eventual comparison must check exact datasets/directions and disclose
the CPU-versus-CUDA training difference and DA-10's restricted capacity.

## Execution and monitoring

Before the scientific launch, run unit tests and the registered 16-cell CPU
smoke, then the 16-cell CUDA smoke. Both engineering plans have two epochs,
five-point slices, and seed 999; they are excluded from scientific results.
Verify both artifact views and exact scalar-reference centers before launch.

The durable launcher is `scripts/punn_manifold_landscape_local.ps1`. It uses
the pinned UV environment and dispatches training through
`uv run --frozen --no-sync python -m optimizer_resurrection.experiment plan`.
An online W&B controller captures planner stdout inside W&B's managed logs.
Scientific source must remain a clean committed isolated checkout throughout
the sweep. W&B group: `punn-manifold-landscape-gpu-v1`; controller group:
`punn-manifold-landscape-gpu-v1-controller`.

The hourly chat heartbeat is `check-da-10-gpu-landscapes-hourly`. It checks
W&B outcomes/artifacts and active processes, reports changing progress or
failures, and pauses after final completion or an unrecoverable failure.
A fresh planner restart would repeat all cells. The launcher holds an exclusive
Windows mutex for the controller lifetime and rejects fresh launches into an
existing group. Use explicit `-Resume` only after reconciling process state;
the planner then rejects active/duplicate conditions and skips uniquely
accepted outcomes with complete uploaded artifacts from the same clean source.
`-DryRun -Resume` queries W&B without training; ordinary `-DryRun` only expands
the frozen plan.

## Recording preflight

The full repository unit suite and focused geometry, observer, pipeline, resume,
and Windows mutex tests passed. Both online engineering smoke plans completed
all 16 cells (two epochs each). Independent readback verified every uploaded
artifact: 64 parameter states and 128 slices per device, exact centers, nonfinite
masks, selected corner losses, and feasible-grid Stiefel residuals.

- [CPU recording audit](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/kingph3v):
  maximum relative scalar-reference difference 1.55e-6; maximum feasible residual
  9.72e-7.
- [CUDA recording audit](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/fxyxm6nl):
  maximum relative scalar-reference difference 2.53e-5; maximum feasible residual
  3.33e-6. Centers were within 1.03e-6 relative difference against CPU references.

These audits are engineering checks, not optimizer-performance results.
## Original GPU sweep

Started October 1, 2026 at 04:44 UTC (00:44 EDT), using four concurrent local
CUDA workers. The exact frozen planner expanded 480 distinct conditions.
Startup verification checked the first four online scientific runs against
every resolved config field and the same clean revision/source bytes;
`nvidia-smi` observed 91% GPU utilization and 3,458 MiB of 16,380 MiB allocated.
There were zero completed scientific outcomes at this startup check.

- [Online controller `y911ni46`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/y911ni46)
  was stopped at the user's request on October 1. Query it directly if filtered W&B indexing is delayed.
- Scientific revision: `c55600e4755adf687753e6dee01b63462bbc6f65`.
- Clean checkout: `.cache/da10-landscape-gpu-c55600e` within the original repository.
- Raw source SHA256: `853afa71ba5f0e4df8f1a78e4967d573c5f476e6ef35522c176e5639768946c6`.
- Initial launcher PID: 48344; controller Python PID: 16840; owned planner UV
  PID: 39468; planner Python PID: 64864. Windows Python redirectors remain
  alongside the real processes and must not be counted as duplicate runs.
- Initial scientific runs: `h0zggvfh` (seed 0), `v7phtojo` (seed 1), `7n0bes34`
  (seed 2), `zp1dzin3` (seed 3), all f1-small, CUDA training and diagnostics.
- W&B-managed controller log:
  `wandb/wandb/run-20261001_004400-y911ni46/files/output.log`.

The hourly heartbeat has been updated with these exact identities. It validates
recorded outcomes, tracks progress/failures, and performs the landscape comparison
after all registered outcomes and artifacts are complete. A successful 500-epoch
run records 502 parameter states and 13 slices per view (26 total); terminal and
epoch-500 slices are retained separately. No completed landscape result is claimed
at launch.

The startup guard was corrected and independently reviewed before this launch:
it exempts only the verified immediate Python redirector of the same controller
invocation. Seven focused guard/mutex tests, seven resume tests, and a physical
Windows redirector probe passed. Other controller/planner/worker processes still
block a second launch. The earlier startup attempt stopped before creating any
scientific W&B run.

## CPU continuation

The user requested stopping CUDA and completing the remaining cells on CPU on
October 1, 2026. The owned GPU launcher/controller/planner/worker process tree
was stopped at approximately 13:58 UTC. W&B preserves 116 completed outcomes:
30 f1-small, 30 f1-oversized, 30 f4-small and 26 f4-oversized. All 116 immutable
landscape manifests were verified (502 states, 13 slices per view, 31x31 grids).
Four interrupted f4-oversized attempts (seeds 26–29, runs `cxt9kgys`, `596i9vdm`,
`fjz4zlas`, `n765jbot`) remain crashed and explicitly excluded; they restart
from initialization on CPU. No numerical failures occurred among retained runs.

The registered `manifold_landscape_cpu_continuation.yaml` retains all 480 logical
cells and every training/recording setting, changing only training and diagnostic
devices and the run group. Strict online reconciliation maps cells by task,
architecture, method, model seed and data seed, checks every other recipe value,
and skips the 116 recorded CUDA outcomes, leaving 364 CPU conditions. The source
GPU config is checked against revision `c55600e`; accepted source records require
that exact clean revision and raw source digest. CPU outcomes require the new
clean continuation source. Active or duplicate accepted cells fail closed.
The controller holds both CPU and source GPU mutexes throughout dispatch.

Both devices remain explicit in provenance and analysis: the completed study
will contain 116 CUDA and 364 CPU outcomes, not a homogeneous GPU rerun.
The separate CPU transfer engineering preflight covers all 16 cells with
2 epochs; these checks are excluded from scientific results. The hourly
heartbeat now follows CPU continuation and must never restart the GPU group.


CPU transfer readback audit:
[`vq6hdhop`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/vq6hdhop)
verified all 16 engineering outcomes, 64 states and 128 slices. Maximum relative
scalar-reference difference was 1.55e-6 and maximum feasible-grid Stiefel residual
was 9.72e-7. The repository suite passed, followed by focused continuation,
geometry and Windows mutex tests after final guard changes.


The CPU continuation started at approximately 14:19 UTC (10:19 EDT) on October 1,
2026, with eight one-thread workers. Controller
[`qm8ich4f`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/qm8ich4f)
is running; its W&B-managed output is
`wandb/wandb/run-20261001_101944-qm8ich4f/files/output.log`.

- Clean scientific checkout: `.cache/da10-landscape-cpu-59b53d1`.
- Revision: `59b53d1ac33b7d8edc836a55f530889296f96746`.
- Raw source SHA256: `116c85bc9c09d2cf9e5e02d07b7c657948af206797dafa986fc475cd87a26e3f` (135 files).
- Hidden launcher PID28996, actual controller Python63088 (redirector21904),
  planner UV65592 and Python37276 (redirector10332).
- First CPU f4-oversized runs: `cgwx2mq6` (seed26), `1xg4eehb` (seed27),
  `t2l5i0ql` (seed28), `iyymfvfm` (seed29).
- First CPU XOR-small seeds0–3: `v2smnpui`, `irjnk9zt`, `fyrh064q`, `68fenqi2`.
  All four finished 500 epochs and uploaded recordings. Initial eight CPU run
  configs and clean-source provenance were verified against the frozen plan.

The first hidden CPU launcher stopped during its device preflight before any
controller or scientific run was created: Windows PowerShell removed internal
string quotes. The quoting fix was physically verified before this launch.
The final launcher also loads allowlisted root credentials before online dry-run
reconciliation. The scientific checkout remains unchanged; these startup
engineering failures contributed no scientific outcomes.

The existing hourly automation ID `check-da-10-gpu-landscapes-hourly` now has
name “Check DA-10 CPU landscapes hourly” and the exact CPU identities above.
It verifies combined logical-cell coverage and artifact completeness, preserves
the 116 CUDA outcomes, and never restarts CUDA science. Final landscape analysis
and graph updates follow all 480 recorded outcomes, with the device split stated.


Immutable GPU-to-CPU transfer audit:
[`t6eu6glr`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/t6eu6glr),
artifact `punn-da10-hardware-transfer-t6eu6glr:v0`. It retains all 116 source
run/artifact references and maps their GPU condition IDs to the distinct CPU
condition IDs for the same logical cells. It records four excluded interruptions,
364 selected CPU jobs, the clean CPU revision/digest and the preflight audit.
No training run was added by this audit.


Post-launch snapshot: W&B reported79 finished CPU records with recording-complete
markers and7 running records during the query (dispatch/upload transitions may
change the instantaneous worker count). The first four full CPU immutable
artifacts were read back and verified:502 states and13 ambient plus13 manifold
31x31 slices each. The interrupted f4 seeds had reached epoch400. This snapshot
is progress only; all remaining artifacts still require final verification.


### Hourly check, October 1 at 15:35 UTC

380 unique recorded outcomes passed online recipe/provenance and immutable
manifest checks:116 retained CUDA plus264 CPU. The verified manifests include
502-state trajectories,13 slices per view,31x31 MSE/objective grids, nonfinite
masks, feasibility residuals and realized displacement arrays. No numerical
failures were observed. The scientific CPU checkout remained clean with its
registered135-file source digest; eight CPU workers were active and no DA-10
GPU processes remained. A refreshed snapshot had reached diabetes-small,
seeds0-7, around initialization through epoch100.

One incomplete CPU attempt is excluded:
[`4mpnp4hr`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/4mpnp4hr),
iris-oversized seed21/data_seed100021, condition
`a67999a71a39900e681413871136e173334ceb024ccf6f2a593e27ebc8203beb`.
It failed before recording/training with a W&B service busy/timeout error
reported as `AuthenticationError`; the controller traceback places the error
in dataset artifact logging. Its summary records `infrastructure_failure`,
`failed_before_recording` and `landscape_recording_complete=false`. This is not
a numerical outcome. No standalone output log was available for this failed
attempt; its online failure summary and the controller traceback are retained.

The original controller continues other conditions. The hourly monitor retains
this failed attempt and will use the exact clean source/config with strict
`-DryRun -Resume` reconciliation, then `-Resume`, once the current dispatcher
has exited and its workers are gone. No concurrent retry or recipe change is
permitted. The four interrupted GPU attempts remain excluded as documented.


### Hourly check, October 1 at 16:33 UTC

440 unique recorded outcomes passed strict online recipe/provenance checks and
immutable artifact manifest/layout verification:116 CUDA plus324 CPU. All
verified successful recordings retain502 states and13 ambient plus13 manifold
31x31 slices, with MSE/objective grids, masks and geometry diagnostics. No
numerical failures or additional infrastructure failures were observed. The
known iris-oversized seed21 W&B timeout remains excluded and awaits safe retry
after the current dispatcher exits; the four GPU interruptions remain excluded.

The controller remains running. A refreshed snapshot showed eight CPU workers
on diabetes-oversized seeds24-29 and diabetes-regularized seeds0-1, ranging from
initialization to epoch400. No duplicate controller or concurrent retry was
started. The frozen source and recipe are unchanged; hourly monitoring continues.

## Final verification and comparisons, October 1, 2026

All 480 uniquely accepted runs completed 500 epochs with complete immutable
recordings and **zero numerical failures**:116 CUDA and 364 CPU. The original
dispatcher ended with `execution_failure` because of the documented W&B timeout.
After its workers exited, strict `-DryRun -Resume` accepted479 conditions and
selected only iris-oversized seed21. Retry controller
[`gnxjbr6w`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/gnxjbr6w)
completed successfully; accepted retry
[`5d8b3j1y`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/5d8b3j1y)
retains the full recording. Four GPU interruptions and CPU attempt `4mpnp4hr`
remain preserved and excluded rather than counted as outcomes.

The final audit checked exact frozen configs, each device's clean revision/raw
source digest, and the immutable hardware-transfer mapping. All 480 raw payloads
and datasets were read back from W&B with live entry-MD5 verification. Each run
has 502 states and 13 slices per view:240,960 states and 12,480 slices in total.
Epoch500 and terminal remain separate. Checks cover dataset/seed/order identity,
phase-to-state indices, parameter digests, finite masks, seeded/projected
directions, centers, independently recomputed L2, objective=MSE+L2, displacement
and feasibility. Independent CPU evaluations checked all centers, terminal
train/test MSE, and two terminal corners per view. Maximum center relative
difference:3.82e-6; corner difference:4.66e-6; feasible Stiefel residual:2.04e-6.
All recorded MSE grid values were finite. Completion does not imply low loss:
the terminal performance graph retains large finite outliers.

Authoritative raw audit:
[`kct323zf`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/kct323zf).
Final visually checked graph bundle and machine-readable report:
[`e00u5oe9`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/e00u5oe9),
artifact `punn-da10-recorded-landscape-comparison-e00u5oe9:v0`. Plotting used clean
analysis revision `0d28fae`; scientific checkouts remained unchanged. It contains:

- `da10_terminal_landscape_sensitivity.png`: all 16 cells,30 seeds, paired
  ambient/feasible terminal sensitivity with full individual tails.
- `da10_landscape_sensitivity_over_epochs.png`: median/IQR at every retained
  checkpoint, with the distinct terminal snapshot marked at 500.
- `optimizer_baseline_landscape_sensitivity.png`: f1/f4 small models, seeds 0–9,
  SGD/AdamW/Moonlight Muon/DA-10. Colored bars use each method's available
  endpoints; diamonds use the same cohort completed by all four methods.
- `da10_terminal_prediction_mse.png`: terminal train and held-out prediction
  MSE for all six tasks/architectures, separate from the L2 objective.
- `da10_four_method_terminal_surfaces.png`: actual retained grids for the lowest
  shared completed seed (f1 seed1, f4 seed2), selected without inspecting surface
  values. Each task uses one common color scale.

### Ambient versus feasible neighborhoods

The statistic is the 95th percentile of `|MSE(candidate)-MSE(center)|` over 961
grid points, then the median across 30 seeds. It measures sampled 2D sensitivity,
not Hessian sharpness or full-space flatness. Feasible medians are lower in 11
of 16 cells; exceptions are both f1 architectures and all three XOR architectures.

| Task | Architecture | Ambient median | Feasible median |
| --- | --- | ---: | ---: |
| f1 | small | 1.192 | 1.310 |
| f1 | oversized | 1.962 | 2.325 |
| f4 | small | 10.939 | 2.693 |
| f4 | oversized | 1.413 | 0.766 |
| XOR | small | 1.259 | 1.345 |
| XOR | oversized | 1.113 | 1.376 |
| XOR | regularized | 1.163 | 1.392 |
| Iris | small | 140.163 | 7.646 |
| Iris | oversized | 21143.230 | 475.501 |
| Iris | regularized | 17409.707 | 361.102 |
| Wine | small | 3.283 | 2.127 |
| Wine | oversized | 0.607 | 0.466 |
| Wine | regularized | 0.646 | 0.273 |
| Diabetes | small | 5.164 | 0.832 |
| Diabetes | oversized | 271.960 | 52.266 |
| Diabetes | regularized | 59.025 | 22.829 |

Equal coordinate radius does not imply equal displacement after polar retraction.
Ambient 95th-percentile displacement is 1.229; feasible cell medians range 1.004–1.229.
The f1-small exponent tangent dimension is zero, so feasible slices vary the head
only. Lower feasible sensitivity does not establish an unconstrained optimizer
advantage or explain stability causally.

### Matching earlier optimizer recordings

All 60 baseline attempts were checked against their registered recipes, clean
sources and immutable artifacts. Only f1/f4 small models and seeds 0–9 match:
dataset tensors, ambient directions, preprojection initialization and order
seeds are exact. Historical SGD numerical failures are failed W&B runs with
complete diagnostic artifacts. Five remain in the attempted denominator:
f1 seeds 0,3 and f4 seeds 0,1,4. AdamW, Moonlight Muon and DA-10 have no numerical
failures in this subset. On old failed runs, a terminal null update can leave
an earlier finite MSE in W&B's summary. The immutable artifact result is used;
the report identifies stale summary fields.

Completion by all four methods gives 8 shared f1 seeds and 7 f4 seeds. Ambient
sensitivity medians on these identical cohorts are:

| Task | Common completed seeds | SGD | AdamW | Moonlight Muon | DA-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| f1 | 8 | 0.998 | 5.101 | 2.716 | 1.072 |
| f4 | 7 | 0.120 | 2.968 | 0.857 | 17.716 |

This cohort conditions on SGD survival and does not score its failed seeds.
DA-10's f1 median over all 10 available endpoints is 34.185, compared with 1.072
on the 8 common seeds. Available-endpoint medians alone cannot establish a broad
ranking. For f4, DA-10's common-cohort ambient sensitivity remains substantially
larger than the other methods, including Moonlight Muon. Neither comparison
establishes statistical significance, full-space flatness, or causation of error.

Recipes differ and DA-10 projection restricts capacity. Its f1/f4 small runs
trained/evaluated on CUDA; baselines trained on CPU and retain CPU-reference
grids. Independent center/corner checks bound observed arithmetic differences
but do not imply bitwise equality of every CUDA/CPU grid value. Feasible axes
change with the center and are not the ambient directions used by baselines.
This exploratory PUNN comparison does not declare passage of Gates A–C in the
broader Muon study.

The hourly heartbeat is paused after final readback and documentation. No
scientific training remains active or is scheduled for restart.
