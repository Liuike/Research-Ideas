# Manifold Muon DA-10 without momentum

This preregistration evaluates the effect of removing momentum from the
Manifold Muon DA-10 exponent update. It covers the same six tasks and 16
task/architecture cells as the recorded DA-10 landscape study, with 30 model
seeds per cell and 500 epochs (480 scientific runs). The fixed local CPU plan
is [`manifold_landscape_no_momentum_cpu.yaml`](../configs/product_unit/manifold_landscape_no_momentum_cpu.yaml).

## Update rule

The exponent optimizer uses the DA-10 matrix-sign direction computed from the
current exponent gradient, with the same scale, ten dual-ascent iterations,
learning rate, and Stiefel polar retraction as the momentum DA-10 reference.
Its momentum coefficient is exactly zero: there is no accumulated momentum
buffer and no Nesterov contribution. In the reference notation,

\[
m_t = \beta m_{t-1} + g_t,\qquad
h_t = g_t + \beta m_t,
\]

the preregistered setting is \(\beta=0\), so \(h_t=g_t\). The matrix-sign
projection and parameter retraction remain active; this is not ordinary
gradient descent on the exponents. The output head continues to use the same
auxiliary AdamW optimizer and learning rate as the reference. All other
optimizer settings are fixed to the existing DA-10 recipe. There is no
hyperparameter tuning, gradient clipping, mixed precision, or weight decay.

The comparison pairs model initialization seed, dataset seed, and minibatch
order and ambient random directions with the reference DA-10 study.
Feasible directions are projected separately
at each model center. The recorded reference is a mixed-device study (116 CUDA
and 364 CPU outcomes); the new study is entirely CPU, so device differences
must be disclosed in the paired comparison. Only the regularized classification
cells use the pre-registered \(\lambda=0.0001\) L2 penalty. Small and oversized
cells, including f1 and f4, retain \(\lambda=0\).
Initialization uses the same polar Stiefel retraction before training.

## Recorded outcomes

The scientific configuration uses CPU for both FP32 batch-one training and
landscape evaluation, 31-by-31 grids with radius 1, and both ambient and
feasible manifold views. It records initialization, epoch 1, every 50 epochs,
and the terminal state; epoch 500 and the terminal state are retained
separately. Raw MSE and the L2 training objective remain separate, and
nonfinite samples and failed runs remain visible in the run denominator.

The final comparison will show run failures, terminal train/test performance,
and loss-landscape sensitivity. Failure means a recorded numerical failure
during training or evaluation; high but finite loss is performance, not a
failure. The landscape results are local two-dimensional slices around each
recorded center. Ambient directions may violate the exponent constraint;
feasible directions are projected onto its tangent space and retracted. Equal
grid coordinates therefore need not represent equal realized displacement.

Existing SGD, AdamW, and Moonlight Muon landscapes cover only f1/f4 and use
different optimizer recipes. Compare them only on shared tasks, architectures,
seeds, data, and directions, and show the available cohort and failures. The
new CPU DA-10 results may be compared directly with the prior CPU DA-10 study
under its matching cells and seeds. Do not infer full-space flatness, causal
mechanisms, or a general optimizer ranking from these slices or this
fixed-recipe study.

## Engineering smoke

[`manifold_landscape_no_momentum_smoke.yaml`](../configs/product_unit/manifold_landscape_no_momentum_smoke.yaml)
defines a separate 16-cell, two-epoch CPU engineering check with three-point
slices. Its `engineering-smoke` stage and run group keep it out of scientific
outcomes. It must verify that both views and the expected loss references are
retained before the scientific plan is launched.

## Execution, October 3, 2026

The 16 engineering cases completed and passed independent raw recording and
dataset readback in [audit 9txrb9ph](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/9txrb9ph).
Engineering outcomes are excluded from the scientific comparison. The original
73 recording/optimizer/resume tests and three no-momentum tests passed; an
independent review found no blocking update-rule or recipe issue.

The full 480-condition CPU plan was expanded through the frozen experiment
planner and launched with eight one-thread workers. The online controller is
[404egm3t](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/404egm3t),
group `punn-manifold-landscape-no-momentum-cpu-v1`. Scientific training uses the
clean isolated checkout `.cache/da10-no-momentum-cpu-fb8f9b1`, revision
`fb8f9b1afaaf9a0dbdc7e003d55892edb626a876`, raw source SHA256
`15123dbfdb7ae923f6bbfe6e946c9cf57aaa502364fdee4ae3a5c737db76b65f`
(141 files). This checkout must remain unchanged during training and resume.

At launch, hourly monitoring was active. Interim counts checked online recipes/source and
immutable manifests; final completion requires full raw artifact readback.
The final graph bundle was deferred until all registered outcomes and
diagnostics are retained. Failures and incomplete infrastructure attempts
must remain separate.

Initial scientific full readback verified 56 completed outcomes with zero
numerical failures and no excluded attempts in
[audit 0gz4pgvk](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/0gz4pgvk).
This was an interim verification of those retained raw payloads while the
remaining sweep was running.

The final publisher is `optimizer_resurrection.punn_no_momentum_publish`.
Run it from the original repository with `PYTHONPATH` pointing to the original
`src` directory, after committing the analysis source, using the shared frozen
environment and `--scientific-checkout .cache/da10-no-momentum-cpu-fb8f9b1`.
It requires all 480 raw recordings, verifies immutable historical references
and seed/data/order pairing, and writes its report and figures to a W&B
artifact. The figures include numerical failures, terminal train/test MSE
across five methods and six tasks, and the available ambient/feasible landscape
comparisons. The analysis, acceptance and historical-reference checks add 19
passing tests (95 focused tests including the earlier checks). Synthetic
fixtures were used only for visual layout checks and were not published as
scientific results.

At the October 3 hourly check, 413 outcomes passed exact recipe/source and
immutable manifest reconciliation. Run `phz8hu28` (iris regularized, seed 25)
had a missing online terminal summary despite its complete immutable recording.
Independent raw/data/source and center/corner checks verified its completed
500 epochs, 502 states and 13 slices per view. Its missing terminal metadata
was backfilled from that retained snapshot in
[recovery audit 6uuja5kb](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/6uuja5kb),
bringing the verified coverage to 414/480 with zero numerical failures. The
audit preserves the original summary, checksums and recovery script; no
training was rerun and no source or recipe changed. Its retained `wall_seconds`
is explicitly marked as pre-upload training/recording time, rather than an
inferred total duration. Other jobs were still active at this interim check.

## Final audited result, October 3, 2026

All **480/480 CPU outcomes completed 500 epochs**, with zero numerical failures,
zero excluded attempts, no missing cells and no active scientific workers.
The exact clean scientific revision and 141-file digest above remain unchanged.
[Full raw audit e361vpb0](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/e361vpb0)
verified immutable entry checksums, datasets, all parameter/slice schedules,
finite masks, center MSE and L2 objectives, constraint residuals, realized
displacements and independent center/corner evaluations. The recordings retain
240,960 parameter states and 12,480 landscape grids (13 per view per run).
The metadata recovery for `phz8hu28` remains documented and was included in
this final raw audit; it did not create a replacement scientific outcome.

The [final report and six figures](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/2446ajff)
are retained in immutable artifact
`enyan_zhang1-brown-university/optimizer-resurrection/punn-da10-no-momentum-comparison-2446ajff:v0`.
Every report/image entry passed live checksum readback and visual review.
The initial publication `dw7cta11` is preserved; the final rendering corrects
labels and surface spacing only, with identical report data and no training
added. The six existing report tests passed after the formatting corrections;
95 distinct focused acceptance/optimizer/publication checks passed overall.

Every one of the 480 recorded pairs passed dataset, preprojection
initialization, minibatch-order and ambient-direction equality checks.
Projected initialization is bitwise equal for the 364 CPU/CPU pairs; the 116
CUDA/CPU pairs passed the declared tolerance (maximum absolute difference
7.75e-7). The separate legacy CPU performance study was independently anchored
through all 480 DA-10 outcomes. The five-method performance/failure figures use
those historical CPU cohorts. Direct recorded landscape/performance figures
use the prior 116-CUDA/364-CPU DA-10 reference versus the new all-CPU study.
The prior recorded reference's five incomplete attempts and the legacy CPU
reference's 13 declared DA-10 infrastructure attempts remain preserved outside
their accepted logical-condition denominators.

| Method | Numerical failures / 480 logical conditions |
| --- | ---: |
| SGD | 235 |
| AdamW | 4 |
| Moonlight Muon | 21 |
| DA-10, momentum 0.95 | 0 |
| DA-10, no momentum | 0 |

Removing matrix momentum raised median held-out MSE in **12 of 16 cells**
relative to the momentum-enabled CPU control, lowered it in the three XOR
cells, and left f1-small exactly equal. For example, iris-oversized median
held-out MSE increased from 0.2224 to 20.2026, and wine-regularized from 0.1253
to 3.5865. High finite losses remain in the performance distributions and are
not numerical failures. These are descriptive comparisons of fixed recipes,
not significance tests or general optimizer rankings.

The same twelve cells with worse held-out medians have larger median paired
terminal p95 absolute MSE changes in both ambient and feasible slices.
Classification cells show particularly large changes. The graph retains all
30 terminal grids per method/view/cell, with no nonfinite terminal grid points
in either DA-10 study. Actual f1/f4-small surfaces use the lowest shared completed
seed (seed 0), with one color scale per task across both methods/views.
Feasible directions depend on each center, and coordinate radius does not
equal realized displacement after retraction. In f1-small, the exponent
tangent dimension is zero: its feasible slice varies the head only. These
sampled 2D comparisons do not establish full-space flatness or a causal
mechanism. Earlier SGD/AdamW/Moonlight landscapes retain their restricted
f1/f4-small, seeds 0-9 cohort and five failed SGD attempts in denominators.

Hourly monitoring is paused after verified completion, graph publication and
documentation. The old recorded DA-10 studies remain unchanged.
