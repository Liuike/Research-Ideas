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
must be disclosed in the paired comparison. The classification cells retain the
pre-registered \(\lambda=0.0001\) L2 penalty; f1 and f4 retain \(\lambda=0\).
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
