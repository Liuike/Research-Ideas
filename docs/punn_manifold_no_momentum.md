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

Hourly monitoring is active. Interim counts check online recipes/source and
immutable manifests; final completion requires full raw artifact readback.
The final graph bundle remains pending until all registered outcomes and
diagnostics are retained. Failures and incomplete infrastructure attempts
must remain separate.

Initial scientific full readback verified 56 completed outcomes with zero
numerical failures and no excluded attempts in
[audit 0gz4pgvk](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/0gz4pgvk).
This verifies those retained raw payloads; the rest of the sweep is still
running and no final comparison is claimed.

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
inferred total duration. The other jobs remain active; final graphs are pending.
