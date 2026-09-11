# Historical reproduction progress

## Authorized scope and stopping point

Implement the shared historical protocol infrastructure and the 1994 Section 5.5
Latch benchmark, validate, and submit the initial 12 Oscar jobs. Stop after
scheduler acceptance and this handoff; do not wait for scientific completion.
The follow-up request explicitly authorizes implementing Two-Sequence and Parity.
Their implementation and validation continue below; scientific submission is
not part of that follow-up. MLP, Muon comparisons, and later budget stages remain
deferred. On 2026-09-11 the user authorized starting all three prepared groups and
monitoring until the first W&B scientific log. This authorizes diagnostic
Latch submission despite the failed control and proceeding with the explicitly
documented Two-Sequence/Parity reconstructions. The failed screen is preserved.

## Two-Sequence and Parity follow-up

- Scope confirmed by user: implement both remaining recurrent benchmarks.
- Protocol and unresolved assumptions: [historical_sequence_protocols.md](historical_sequence_protocols.md).
- Added isolated models/data generators, plain-SGD runner, strict planner,
  held-out scout selection, full-span FP32/float64 diagnostics, W&B provenance,
  and separate CPU smoke/scout configs. Existing Latch recipes are unchanged.
- Independent agent implemented core/data tests and reviewed the parent-owned
  runner. Primary reviewed the core and integrated fixes. Review found and
  resolved W&B initialization-failure cleanup and state-gradient underflow
  detection gaps.
- Full suite passed 81 tests without warnings. A subsequently added failure-path
  regression passed with all 12 runner tests (82 tests in the resulting suite).
  Central finite differences cover every parameter/input of both models;
  recurrence hand checks, trace equivalence, pairing, parity, exact SGD,
  budget accounting, strict selection and diagnostic isolation also pass.
- Four online CPU engineering smokes completed: Two-Sequence `uhps9j76` (noise0)
  and `80ocoqnu` (noise0.2); Parity `yfcwmvnp` (noise0) and `5m4w7teo` (noise0.2).
  All ran 10 presentations, with snapshots at0/1/5/10 and 136 diagnostic
  sequence evaluations. API read-back verified identities, finite full-span
  gradients, data digests, terminal outcomes and dirty engineering provenance
  based on `ea03532`. These are retained in the ledger's smoke groups; they
  are not scientific learning evidence. Final review fixes were unit-tested
  after these smokes. No GPU validation or new Slurm submission occurred.
- Scientific runs and comparison recipe selection remain unexecuted. Do not
  report implementation tests as evidence of historical training failure.

Commands from the project root (use `.venv/Scripts/uv.exe` on this Windows
machine, or the pinned `uv` executable in a prepared environment):

```bash
uv run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/historical/two_sequence_smoke.yaml configs/historical/parity_smoke.yaml
uv run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/historical/two_sequence_scout.yaml configs/historical/parity_scout.yaml
```

Add `--run` only when intentionally executing the registered conditions. After
completed calibration, use the following selector separately for each task;
it validates completeness and writes an analysis record to online W&B:

```bash
uv run --frozen --no-sync python -m optimizer_resurrection.historical_sequence_protocol --config configs/historical/two_sequence_scout.yaml
uv run --frozen --no-sync python -m optimizer_resurrection.historical_sequence_protocol --config configs/historical/parity_scout.yaml
```

Before an Oscar launch, review assumptions, synchronize a clean committed
revision, recreate the pinned environment, validate a task-appropriate durable
Slurm script and online GPU smoke, and retain logs. Delete disposable smoke
W&B groups only after verification and before releasing scientific jobs.
Review every scout outcome before freezing a comparison recipe. No initial
comparison matrix or selected LR is implied by the registered scouts.

## Fidelity record

Source: Bengio, Simard, and Frasconi (1994), Section 5.5:
https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Bengio_94.pdf
(author preprint; DOI https://doi.org/10.1109/72.279181).

- Documented: one recurrent tanh unit; three adaptive parameters (recurrent
  weight and positive/negative initial input); parameter initialization uniform
  in [-0.5, 0.5]; fixed generated samples per trial; sequence lengths uniform
  between T/2 and T; no input noise or uniform noise in [-0.2, 0.2].
- Recurrence: x[0]=0; x[t]=tanh(w*x[t-1]+input[t]); the first input is the
  trainable value for the sequence class. The remaining inputs are fixed noise.
  Final squared error uses targets -0.8/+0.8; classification uses output sign.
- Reconstruction assumptions: 100 balanced training sequences and 1,000 balanced
  validation sequences; independent validation stream; fixed shuffled cyclic
  sample order; one sequence per plain SGD update; constant LR, FP32, no
  momentum, clipping, decay, or early termination. Integer lengths include both
  ceil(T/2) and T. These details are explicit rather than claims of exact replay.
- Use independent random streams for labels/order, lengths, and noise so noise
  conditions share labels and lengths. Model and data seeds are recorded.
- Do not mix this benchmark with Section 3's Gaussian-noise experiment, fixed
  recurrent weight, or multiple trainable early inputs.
- No verified historical plain-SGD learning rate was recovered. Register a
  short-task scout: T=10, seed/data_seed=1000, both noise settings, LR in
  {0.001, 0.01, 0.1}, each 5,000 presentations. Select one LR by mean validation
  loss across noise settings; ties choose the smaller LR. Require the selected
  recipe to learn both easy conditions before releasing the main array.
  Operational easy-control thresholds are train and validation accuracy >=0.95
  and both mean half-squared losses <=0.10. These are new screening thresholds,
  not reported historical thresholds; sign accuracy alone is insufficient when
  outputs are almost zero.

## Run plan

- Scout: 6 runs, separate from evaluation seeds.
- Evaluation: lengths 10/100 x noise 0/0.2 x seeds 0/1/2 = 12 runs.
- Each evaluation run: exactly 5,000 sequence presentations (batch size one).
- Diagnostics: whole-dataset train/validation loss and accuracy; full-span
  gradients back to the initial input; float64 diagnostic references distinguish
  FP32 underflow from small nonzero gradients. Diagnostic work is counted
  separately and must not advance the training random stream.
- Scientific records, metrics, and artifacts live only in online W&B. This
  document stores protocol decisions, provenance, status, and commands.

## Current progress

- [x] Audited original protocol against current code; current generic Latch is
  retained as a different, earlier experiment.
- [x] Verified Oscar campus SSH using the existing RSA key; remote checkout was
  clean at dfe0166 and no jobs were present at initial inspection.
- [x] Implement and independently test the historical protocol.
- [x] Review correctness, planner identity, and launch mapping. Fixed initial
  snapshot test and diagnostic accounting identified by the independent review.
- [x] CPU and online local GPU smoke validation. Full suite: 61 tests passed.
  Smoke histories contain steps 0,1,5,10, full-span gradients, dataset digests,
  terminal outcome, source digest, git metadata, and hardware metadata. Verified
  CPU run xpjf4qa5 and RTX 4060 Ti run drnak59m, then deleted both disposable
  W&B groups and confirmed zero remaining records with fresh API queries.
  Initial attempts pyn1usks/j5xychih lacked git revision because of Windows Git
  ownership checking; discarded and rerun using process-local safe.directory.
- [x] Commit and synchronize clean code: f671263 on branch
  `codex/historical-latch-reproduction-v1`. Direct incremental Git bundle to
  Oscar; GitHub push was rejected by automatic approval review as an unapproved
  export destination, and was not performed. Pinned environment recreated with
  `bash scripts/oscar_setup_uv.sh`: 60 tests passed, 1 integration test deselected
  (the complete 61-test suite passed locally).
- [x] Validate and clean disposable Oscar smoke: job6196206 completed 0:0;
  W&B run ow9y640m verified clean f671263, L40S hardware, Slurm identity and full
  trajectory history, then deleted; fresh API query confirmed empty group.
- [x] Finish LR scout and freeze selected recipe: all 6 tasks completed 0:0.
  W&B selection run `9i7yjdzi` verified exact condition identities, full budgets,
  finite summaries, and completeness. Selected LR=0.1 by the registered mean
  validation-loss rule. Frozen config: `configs/historical/latch_initial.yaml`.
  **Easy-control screen failed:** the noisy T10 cell learned; the noise-free T10
  cell did not learn at any candidate LR. Selection therefore returned
  `easy_controls_pass=false` and exit1, correctly preserving the failed screen.
  No runs or seeds were excluded. Awaiting user's decision on submitting the
  requested 12 cells as a diagnostic comparison despite this failed screen.
- [ ] Submit 12-run evaluation array and record scheduler acceptance.

## Handoff (submission pending)

Config paths: `configs/historical/latch_scout.yaml`,
`configs/historical/latch_smoke.yaml`, `configs/historical/latch_gpu_smoke.yaml`,
and `configs/historical/latch_initial.yaml` (frozen in commit `7fd6a08`).
Final local full-suite validation: **62 passed**. Oscar remains on the clean
scout revision `f671263`; synchronize the final evaluation revision and repeat
launcher validation before any evaluation submission. No initial evaluation
jobs have been submitted. Launcher:
`scripts/oscar_historical_latch.sbatch`. Remote project root:
`/users/ezhan153/random-idea-1`. Connect directly to `sshcampus.ccv.brown.edu`
using the existing RSA key (never embed key material in commands or documents).

Useful commands inside the Oscar project (replace JOB_ID with recorded job ID):

```bash
squeue -j JOB_ID -o '%.20i %.12T %.30j %.30R'
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList
ls logs/slurm/historical-latch-JOB_ID_*.out logs/slurm/historical-latch-JOB_ID_*.err
.uv-bootstrap/bin/uv run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/historical/latch_scout.yaml
.uv-bootstrap/bin/uv run --frozen --no-sync python -m optimizer_resurrection.historical_protocol --config configs/historical/latch_scout.yaml
```

The last command validates completed scout cells and records a W&B selection
analysis run; it does not launch training. Use only if a new selection analysis
is wanted; the original selection ID will be recorded below.

Read-only W&B inspection (from the pinned project environment):

```bash
.uv-bootstrap/bin/uv run --frozen --no-sync python - <<'PY'
import wandb
from optimizer_resurrection.tracking import load_wandb_credentials
credentials = load_wandb_credentials()
project = credentials['WANDB_ENTITY'] + '/' + credentials['WANDB_PROJECT']
api = wandb.Api(timeout=30)
for group in ('historical-latch-lr-scout-v1', 'historical-latch-initial-v1'):
    for run in api.runs(project, filters={'group': group}):
        print(group, run.id, run.state, dict(run.config), dict(run.summary))
PY
```

Do not redirect scientific summaries into project artifacts. Inspect them in
W&B or terminal output. An empty initial group is expected before submission.

Oscar smoke: job `6196206`, submitted on f671263 after `bash -n`, planner dry-run,
and `sbatch --test-only` passed; observed running on gpu2708. Slurm's test-only
start-time estimate was pessimistic and did not reflect the actual immediate
start. Retained streams: `logs/slurm/historical-latch-6196206_0.{out,err}`.

Scout: array `6196236`, six tasks with `%2`, committed config
`configs/historical/latch_scout.yaml`, revision f671263. All six mappings passed
DRY_RUN and `sbatch --test-only` passed before submission. Group:
`historical-latch-lr-scout-v1`. Retained streams:
`logs/slurm/historical-latch-6196236_TASK_ID.{out,err}`.

Scout W&B training IDs: `l6lrxf2u`, `wqvbnki8`, `3jqrci5m`, `e0ouls8c`,
`kpp9sixk`, `b2164mfu`. All completed; no failed training attempts or excluded
cells. Tasks2/3 experienced startup filesystem waits (observed through a short
read-only diagnostic step in allocation6196277), then completed normally.

The source-controlled easy-control screen was an operational safeguard, not a
historical result or a reason to erase failures. The failure remains part of the
record. Numerical gradients, one-step updates, data pairing, and the noisy T10
learning trajectory support continued investigation, but do not establish the
cause of the noise-free failure. The frozen rate is the registered selector's
choice, not a claim that both controls passed.

Next agent: inspect every expected Latch condition in W&B and the retained Slurm
streams. Separate interrupted jobs from completed training that missed its
success criterion. Do not claim historical reproduction solely from a finished
run or an accuracy difference. Review the trajectories and easy-task controls
before proposing the 20,000/100,000-presentation stages. Existing Gates A-C and
headline readiness remain independent and unchanged.


## 2026-09-11 authorized launch

User requested starting all prepared runs and monitoring until W&B receives
its first scientific log. Latch is now labeled `historical-diagnostic`; only
that runtime label changed, not the frozen LR, data, budgets, or condition IDs.
The earlier pending-decision notes describe prior status and are superseded by
this authorization. Two-Sequence/Parity retain their documented assumptions.

Launcher: `scripts/oscar_historical_reproduction.sbatch`, scientific mode,
array0-23%2: tasks0-11 Latch initial,12-17 Two-Sequence scout,18-23 Parity scout.
All tasks request one L40S; the combined array caps total concurrency at two.
Smoke mode uses array0-3%2 and the new committed GPU smoke configs for both tasks.
Source will be synchronized directly to Oscar with a Git bundle. No GitHub push.
Preflight, job IDs, execution revision and first-log evidence will be recorded
below. Keep the remote execution checkout unchanged while jobs are queued.

Smoke attempt array6240725 failed all four tasks with exit127 before training:
Slurm spooled the dispatcher, whose SCRIPT_DIR pointed into /var/spool/slurmd.
Fixed nested-launcher lookup to PROJECT_ROOT/scripts. Retain failed scheduler
logs `logs/slurm/historical-reproduction-6240725_TASK.{out,err}`. No scientific
runs were submitted and no W&B training records were created by this attempt.


### Submitted scientific array

Execution revision: `15d3027f85834dfdccc493676168fa16aa7897bc` (clean on Oscar).
Local full suite82 passed; Oscar pinned rebuild81 passed,1 integration deselected.
All28 mappings passed dry-run; corrected dispatcher smoke mappings and six
scientific boundaries passed again. Both scripts passed bash-n; scientific
sbatch-test-only passed immediately before submission.

GPU smoke retry6241277 completed4/4 exit0:0 on NVIDIA L40S. W&B IDs euv4plws,
sn8mp1bd, etkzf94l,8fqhvz14 verified complete online histories, clean revision,
Slurm IDs, exact budgets and finite diagnostics. Those two GPU smoke groups
were deleted and fresh queries confirmed empty. Retain both smoke arrays'
Slurm logs. The environment rebuild initially hit a transient directory-removal
error, then succeeded on retry; this was setup failure, not training failure.

Scientific array **6241516**, accepted2026-09-11 14:28EDT, array0-23%2,
one L40S/task,2CPU,8GB,2h/task. Scheduler confirmed ArrayTaskThrottle=2.
Tasks0-11: `configs/historical/latch_initial.yaml`, group historical-latch-initial-v1.
Tasks12-17: `configs/historical/two_sequence_scout.yaml`, group historical-two-sequence-scout-v1.
Tasks18-23: `configs/historical/parity_scout.yaml`, group historical-parity-scout-v1.
All24 conditions retain exactly5000 sequence presentations. No other tasks or
budget extensions launched. All three groups were empty before submission.
Logs: `/users/ezhan153/random-idea-1/logs/slurm/historical-reproduction-6241516_TASK.{out,err}`.

Inspect on Oscar:

```bash
cd /users/ezhan153/random-idea-1
squeue -j 6241516 -o '%.22i %.12T %.30R'
sacct -j 6241516 --format=JobID,State,ExitCode,Elapsed,NodeList
```

Use the read-only W&B query above with these three group names. Compare each
record's global Slurm task ID against the ranges above; the dispatched local
config index is separate and does not overwrite scheduler provenance. Leave
the remote checkout at15d3027 while jobs remain queued/running. Local handoff
commits after submission do not change that execution revision.


First scientific W&B history verified: run `ab5m284z`,
https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/ab5m284z
The returned history included presentation0 with finite train/validation loss
and accuracy. API confirmed clean execution revision15d3027 and Slurm array6241516.
That run already reported finished at read-back; this does not imply completion
of the entire array or a successful historical reproduction. Monitoring stopped
at the user's requested first-log milestone. Next agent must inspect every one
of the24 outcomes, preserve the failed Latch easy-control screen, distinguish
execution failures from completed-but-unlearned runs, and review calibration
before freezing new recipes or extending any budgets.
