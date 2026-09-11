# Two-Sequence and Parity reconstruction

Source: Bengio, Simard, and Frasconi (1994), Section 5.5, preprint pages 21–22,
[author preprint](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Bengio_94.pdf).
This implementation continues the historical plain-SGD baseline, without
altering Latch or introducing optimizer comparisons. Exact historical replay
is not established: the original samples, RNG, learning rate, and full Parity
wiring have not been recovered.

## Recovered settings

Both tasks use trial-specific fixed samples and initial parameters uniform in
[-0.5,0.5], variable sequence lengths between T/2 and T, and optional uniform
input noise in [-0.2,0.2]. Outputs are supervised once, at sequence end.
Two-Sequence uses five fully recurrent units, no bias, one additive external
input and 25 trainable parameters. Class information comes from prefixes of
two sequences generated uniformly in [-1,1]. Parity uses two units and seven
parameters; positive output denotes an odd number of clean +1 symbols.

## Explicit reconstruction decisions

- Two-Sequence draws two length-T templates once per data seed, shared across
  training and validation. A balanced class selects a prefix; additive noise
  is independently drawn for each fixed sample. Templates do not regenerate
  each epoch or split. Input enters unit 0 and unit 4 is read out. The recurrent
  update is `x[t]=tanh(W*x[t-1]+e0*u[t])`, with zero initial state.
- Parity uses `h[t]=tanh(a*u[t]+b*y[t-1]+c)` and
  `y[t]=tanh(d*u[t]+e*y[t-1]+f*h[t]+g)`, with `y[0]=0`.
  This seven-parameter topology is an explicit candidate reconstruction,
  **not a recovered historical graph**. Biases and the within-step hidden
  connection are assumptions. Each desired class is balanced; randomly drawn
  symbols have their final active symbol flipped when needed to match parity.
  Labels are determined before additive noise. No noisy threshold relabeling.
- Two-Sequence targets are +/-0.8 (following the Latch objective convention);
  Parity targets are +/-1. Both use mean half-squared final-output loss and
  sign classification. No intermediate supervision is added.
- Lengths are inclusive integers from ceil(T/2) to T. Training uses 100 samples,
  validation 1,000, with independent split RNG streams and cyclic fixed order.
  These sizes, balancing, ordering, and integer convention are assumptions.
- Plain SGD uses FP32, constant learning rate, batch one, no clipping,
  momentum, weight decay, AMP, curriculum, or early termination. Diagnostic
  evaluations are counted separately from optimizer sequence presentations.
- Smoke configs run ten presentations on CPU for both noise settings. Scout
  configs register T10, held-out seed1000, LR0.001/0.01/0.1, both noise settings,
  and 5,000 presentations. These rates and budgets are new calibration choices;
  no comparison recipe has been selected or frozen for either task.

The operational learning criterion (both train/validation loss <=0.1 and
accuracy >=0.95) is a diagnostic label, not a historical threshold. A completed
run that fails it is a training failure, distinct from an execution failure.

## Identity, diagnostics, and next steps

Protocol IDs are `bsf1994-two-sequence-v1` and `bsf1994-parity-v1`. Scientific
condition hashes include task, seeds, noise, length, LR, budget, sample sizes,
and recipe version. Dataset digests identify the actual fixed tensors.
Full input and recurrent-state gradients are evaluated on the longest
validation example of each class at initialization and logged checkpoints,
in FP32 and float64. The float64 copy never changes optimizer state.

Before scientific launch, review the unresolved template and Parity topology
assumptions, commit source/configs, validate online smoke and pinned Oscar
environment, and validate the durable launcher. Run calibration first; use
equal budgets and held-out calibration seeds. Freeze selected recipes before
any length comparison. Neither new task has scientific results merely because
its implementation or CPU tests pass. MLP remains deferred.
