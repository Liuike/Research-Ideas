# DA-10 GPU landscape rerun

Luke requested a GPU rerun of the completed six-task DA-10 study with both
ordinary parameter slices and feasible manifold slices. The frozen record is
[`manifold_landscape_gpu.yaml`](../configs/product_unit/manifold_landscape_gpu.yaml):
the same 16 task/architecture cells, 30 seeds each, and 500 epochs (480 runs).
Training and landscape evaluation both use the local RTX 4060 Ti, FP32.
The original CPU outcomes are retained as a separate study. Plain DA-10's
learning rates, momentum, ten dual iterations, scale 1, auxiliary AdamW head,
batch-one sample order, and explicit classification L2 settings are unchanged.
There is no tuning, clipping, mixed precision, or CPU fallback.

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
