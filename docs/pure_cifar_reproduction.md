# PURe ResNet-18 CIFAR-10 reconstruction

This implements the accepted adaptation of [arXiv:2505.04397v3](https://arxiv.org/html/2505.04397v3).
The paper's CIFAR table uses ResNet20 and larger CIFAR models, so there is no
published CIFAR ResNet-18 accuracy to reproduce. This experiment includes only
five SGD runs of the four-stage PURe ResNet-18, without tuning or baselines.

## Architecture and assumptions

The model reuses pinned torchvision ResNet/BasicBlock machinery. A 3x3, stride-1
stem replaces the ImageNet stem; max pooling is removed. Four stages contain
two blocks each at widths 64/128/256/512. Each branch is Conv3x3, BatchNorm,
ProductUnit3x3, BatchNorm; the unchanged shortcut is added before ReLU. The
intermediate branch ReLU is removed. Adaptive pooling feeds a ten-class head.
There are 11,173,970 parameters, including eight scalar thresholds.

Product units compute `exp(conv2d(log(max(x, softplus(theta)+1e-7)), W))`.
Padding is zero in log space, all channels mix, and there is no bias or output
clamp. Initialization assumptions: theta=0, Kaiming normal fan-out kernels,
BatchNorm scale/bias=1/0, and torchvision's default classifier initialization.
No public official implementation was located; initialization, padding and
threshold granularity are explicit reconstruction choices.

## Frozen recipe

`configs/pure_cifar/defaults.yaml` records five seeds (0-4), 160 epochs, batch
128, SGD LR 0.01/momentum 0.9/decay 0.001 on all parameters, no Nesterov, and
cross entropy without smoothing. LR drops by 0.1 after completed epochs 80 and
120. FP32, no AMP/TF32, no clipping and no exponent projection. This scheduled,
decayed paper reconstruction is marked secondary to the main optimizer study.

Use all 50,000 official training examples and 10,000 official test examples.
Train augmentation is RandomCrop(32,padding=4) and horizontal flip p=0.5.
Normalize with mean (0.4914,0.4822,0.4465), SD (0.2470,0.2435,0.2616).
Preprocessing, normalization, decay factor and final-model selection are
assumptions where the CIFAR description is incomplete. No validation split:
evaluate the final model once after epoch160; retain the final partial batch.

The model seed is recorded separately from data seed (seed+100000). Shuffling
uses data_seed+1000003; worker initialization uses data_seed+2000003 and PyTorch's
per-iterator worker seeds. With zero workers, augmentation uses the explicit
data_seed+3000003 parent stream. Worker Python/NumPy RNGs are also initialized.
Changing worker count can change the augmentation realization and is logged.

## Execution and analysis

Download/checksum-verify CIFAR once before launch; batch jobs do not download.
Expand every config with `uv run --frozen --no-sync python -m
optimizer_resurrection.experiment plan`. Run CPU/local GPU engineering smoke,
then the disposable Oscar smoke; verify and delete its W&B group before the
scientific array. Oscar uses one L40S per seed, array 0-4%2, four CPUs, 32GB,
twelve hours. Validate Bash syntax, DRY_RUN=1 and sbatch --test-only first.

Scientific source must be clean/committed. Online W&B stores commands, config,
source/dependency/environment identity, initialization/data digests, scheduler
identity, metrics, thresholds, runtime and terminal outcomes. There are no
custom result/checkpoint files; final evaluation uses the model in memory.
Ignored Slurm stdout/stderr logs are retained. A killed job must be recorded
and any full restart identified explicitly, without silently replacing seeds.

Analyze with `uv run --frozen --no-sync python -m
optimizer_resurrection.pure_analysis --config configs/pure_cifar/defaults.yaml`.
Analysis rejects missing/duplicate conditions, invalid terminal states/budgets,
dirty or inconsistent source, and inconsistent datasets. Report individual
test accuracies, mean and sample SD, time, parameter count, and all failures.
Numerical failures remain counted separately from finite-run statistics.
Completion requires five finite 160-epoch runs; no accuracy threshold is set.

## Status

All 130 unit tests passed locally and on Oscar (one integration test deselected).
Live CPU smoke `o9bmqchf`, local GPU smoke `xmz4e9pw`, and Oscar L40S smoke
array `6728167` / run `ydjyeemd` trained their one epoch with finite values but
recorded `nonfinite_test_output` during final evaluation. Local GPU training
took 2.04 seconds for eight batches, with 1.21GB peak allocated memory; Oscar
took 3.37 seconds including its worker startup. The Oscar record's clean source,
data, workers, device and Slurm identity were verified and its disposable group
deleted. These are execution checks with an explicit stability failure, not
successful scientific fitting.

The frozen five-seed scientific array `6728199` completed from clean revision
`9e56993c91c98caf602ac24ccf8fce873b609a28` in
`/oscar/scratch/ezhan153/pure-cifar-9e56993`, two L40S tasks at a time. All five
seeds trained 160 full-data epochs and evaluated the final model successfully.
No frozen hyperparameters were changed in response to smoke failures. There
were no numerical failures, scheduler failures, excluded seeds, or replacements
in the scientific study. The smoke evaluation failures did not recur after
full training; their cause was not established by this study.

Data transport used a mirror after the original Windows downloads failed with
TLS integrity errors. The archive MD5 matched torchvision's official
`c58f30108f718f92721af3b95e74349a`; all extracted training/test batch checksums
were verified by torchvision. A supported alternative SSH cipher allowed the
verified archive to be transferred locally. Data content and recipe are unchanged.

## Final results (2026-09-26, America/New_York)

The [strict W&B analysis](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/aekcnz2i)
verified all five expected conditions, exact resolved recipes, clean and
consistent source, dataset identity, data/order/worker seeds, parameter count,
terminal states, 160-epoch budgets, and evaluation of all 10,000 test examples.
Accuracy is the final-model top-1 test accuracy; no test-based selection was used.

| Seed | Test accuracy (%) | Test cross-entropy | Training time (min) | W&B run |
| --- | ---: | ---: | ---: | --- |
| 0 | 94.22 | 0.240884 | 37.49 | [66303l7q](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/66303l7q) |
| 1 | 93.82 | 0.246212 | 37.57 | [0g8c4ohz](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/0g8c4ohz) |
| 2 | 93.55 | 0.256549 | 38.92 | [peb6bcwp](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/peb6bcwp) |
| 3 | 93.56 | 0.261568 | 41.30 | [toytuo9u](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/toytuo9u) |
| 4 | 94.00 | 0.247697 | 37.41 | [3wi45i49](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/3wi45i49) |

Mean test accuracy: **93.83%**, sample SD **0.28827 percentage points** (ddof=1).
Mean training time: **2312.26 seconds / 38.54 minutes per seed**, excluding data
preparation, initialization and final test evaluation. Peak allocated CUDA
memory was 1,210,786,304 bytes for every seed. Parameter count: **11,173,970**.
These results establish successful execution of the accepted reconstruction.
The paper has no published CIFAR ResNet-18 number, and this study includes no
baseline comparison. The initialization, preprocessing, padding and selection
assumptions above remain limitations.

Source-tree SHA256:
`3cfd0d038193da5e7518e2cdadef09fb244541a85381ffdd26cc298c8502b40e`.
Dataset SHA256:
`6b3883dca6c867f1e58def548c063d1865569b8575fcbce06b3e468a89f9895a`.

## Plain SGD seed-0 follow-up

`configs/pure_cifar/plain_sgd_seed0.yaml` registers the user-requested exploratory
condition `pure-cifar10-resnet18-plain-sgd-v1`. It repeats seed 0 with momentum
zero, retaining weight decay 0.001, initial learning rate 0.01, the 80/120 decay
milestones, 160 epochs, batch size 128, architecture, initialization and data
seeds. The paired momentum-SGD seed-0 reference is 94.22% (run `66303l7q`).
This is a single-seed optimizer ablation; it does not estimate across-seed
variation or reproduce a paper-reported plain-SGD result. CPU, local GPU and
disposable Oscar smoke configs precede scientific submission.

Submitted Oscar array `6738291` with `--array=0-0%1`, one L40S, four CPUs,
32 GB RAM and a twelve-hour limit, from clean revision
`46805f57fcb5ade32d7f256c4f8b72407f60c322` in
`/oscar/scratch/ezhan153/pure-plain-sgd-46805f5`. All 229 tests passed on
Oscar. The disposable Oscar smoke `6738040` completed finite; its online
record verified source/data/model/worker/Slurm/L40S identity, and its group
was deleted before scientific submission. CPU smoke recorded a final-evaluation
numerical failure; local GPU smoke completed finite.

### Plain SGD final result (2026-09-27)

The [strict online W&B analysis](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/ux11tiwa)
verified the single expected condition, clean source, data identity, all 160
epochs and final evaluation on all 10,000 test examples. Slurm job `6738291_0`
completed with exit code 0; retained logs show successful W&B synchronization.
There were no numerical failures, infrastructure failures, incomplete records
or excluded seeds in this scientific condition.

| Seed-0 condition | Final test accuracy (%) | Test cross-entropy | Training time (min) | W&B run |
| --- | ---: | ---: | ---: | --- |
| Plain SGD, momentum 0 | 87.08 | 33.992001 | 35.32 | [kr2n6lak](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/kr2n6lak) |
| SGD, momentum 0.9 | 94.22 | 0.240884 | 37.49 | [66303l7q](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/66303l7q) |

Plain SGD's accuracy was **7.14 percentage points lower** in this paired seed.
Online records match the model initialization digest, full dataset digest,
model seed 0, data seed 100000, order seed 1100003, worker seed 2100003 and
augmentation seed 3100003. All recipe fields except momentum match. Both used
the same pinned Python/PyTorch/torchvision/CUDA/cuDNN environment and L40S GPU
type; source revisions, source-tree digests and scheduler jobs/nodes differ.
Its training time was 2118.981654 seconds, excluding preparation, initialization
and final evaluation; scheduler wall time was 35 minutes 47 seconds. Parameter
count was **11,173,970**, and peak allocated CUDA memory was 1,166,607,360 bytes.
Final training accuracy was 98.166%, with training loss 0.060588. Test loss was
large despite finite outputs; these records do not establish its cause.
One seed provides no across-seed standard deviation or optimizer-ranking claim.
This remains a CIFAR ResNet-18 architecture adaptation without a published
accuracy target, and plain SGD is a user-requested change to the paper recipe.

Scientific source-tree SHA256:
`04f501b6bd54b42c0bd214b2c7648df3c0014b0cccbdbf4b83596100c5184462`.


## Instrumented plain-SGD seed-0 rerun

`configs/pure_cifar/landscape_sgd_seed0.yaml` preregisters protocol/group
`pure-cifar10-resnet18-plain-sgd-landscape-v1`. The user requested gradient
landscape measurements after the plain-SGD result. Scientific training settings
and seed pairing stay identical to `plain_sgd_seed0.yaml`; no tuning is added.
All diagnostic data go to online W&B, without custom checkpoints or result files.

Every minibatch contributes global and named-parameter gradient, effective
weight-decayed gradient, and actual update statistics, aggregated per epoch.
Product-layer activation diagnostics sample the first training minibatch each
epoch. A separate fixed 32-example training subset uses normalization without
augmentation, seed 314159, and recorded indices/content digest. It never samples
the test set or consumes the training loader's shuffle/worker generators.

Fixed probes are scheduled at epochs 0, 1, 40, 80, 81, 120, 121 and 160.
They measure the evaluation-mode cross-entropy and L2 objective, approximate
Hessian curvature, and local loss slices in two fixed random directions.
Training-mode BatchNorm probes on the same inputs help expose differences from
running-statistics evaluation. Probe parameters, buffers, gradients, module
modes and RNG states are restored afterward. Invalid/nonfinite diagnostic
measurements are recorded explicitly and do not silently change training.

This is a sampled local characterization of the objective. It does not recover
the complete high-dimensional landscape or provide a population estimate of
curvature. Loss slices and Hessian estimates refer to a fixed training subset
in evaluation mode; minibatch gradients refer to augmented training data with
training-mode BatchNorm. Instrumentation adds runtime, so instrumented total
runtime is not an optimizer speed comparison.


Frozen diagnostic reconstruction details: eight Hessian power iterations estimate
largest eigenvalue magnitude, with signed Rayleigh quotient and residual;
four seeded Rademacher vectors estimate trace and its sample standard error.
The trace and Rayleigh records also subtract the known L2 Hessian contribution.
Two seeded random directions are normalized per output filter (whole-tensor
normalization for vectors/scalars). Their alpha/beta coordinates are
`[-0.10, -0.05, 0, 0.05, 0.10]`; CE, regularized objective and per-point validity
are logged for both 1D axes and the 25-point 2D grid. These directions use
current parameter norms; they are a relative perturbation scale, not a claim
of exhaustive sharpness. Failed Hessian estimates do not discard loss slices.
Numerical validity does not imply power-iteration convergence; inspect residuals.

Model forward/backward, training, and Hessian-vector products remain FP32.
Diagnostic scalar norm/cosine/Rayleigh/trace reductions use FP64 to avoid
false overflow on large but finite FP32 values. Product-layer records include
input-floor fraction, logged-base/exponent ranges, overflow/underflow risk,
finite-output and zero fractions. The underflow-risk threshold is below
`log(2**-149)`; observed zeros are logged separately. Final evaluation additionally
records loss quantiles, maximum logit magnitude, and confidence/loss split by
correct versus incorrect predictions during the existing one-pass test evaluation.

Engineering CPU record `8gl204kb` trained finite and recorded final-evaluation
failure. Its probe data exposed the need for stable scalar reductions and
independent loss slices; it remains an engineering record. The final CPU smoke
uses separate group `pure-cifar-landscape-cpu-smoke-v2` to preserve that record.


The final CPU smoke `zrbn4kfs` recorded all gradient/activation/probe budgets;
training remained finite while final evaluation was nonfinite, matching the
previous reduced CPU behavior. Initial/epoch-1 invalid curvature or slice
measurements are explicit. Local RTX 4060 Ti smoke `p89bs1li` completed finite,
with identical initialization/data identity and training/test loss and accuracy
to the uninstrumented smoke `qwrijmhn`. Instrumented total runtime was
8.285135 seconds, first-epoch time 3.454299 seconds and peak allocated CUDA
memory 1,335,751,680 bytes. Its initial Hessian probe was numerically invalid;
epoch-1 Hessian and forward probe were valid. All required diagnostic history
fields and slice point validity masks were verified online. Unit checks include
closed-form scalar logistic Hessian, state/RNG restoration, large-finite-gradient
reductions, invalid-eval/finite-batch-stat behavior, unavailable-HVP/valid-slices,
and exact baseline-versus-instrumented training replay.


Oscar validation passed all 244 tests. Shell syntax, both dry runs, and
`sbatch --test-only` passed. Disposable L40S array `6740194` completed finite;
its online source/data/model/worker/Slurm/device and substantive diagnostic
history were verified, including eight measured minibatches, two probes and
all loss-grid point masks. Its epoch-1 Hessian was valid; initial Hessian and
some initial slice points were explicitly invalid. Total instrumented smoke
runtime was 9.320236 seconds, epoch time 1.156456 seconds, and peak allocated
CUDA memory 1,335,751,680 bytes. Training/test metrics matched the previous
uninstrumented Oscar smoke. The disposable W&B group was deleted before launch.

Scientific array `6740232` was submitted with `--array=0-0%1`, one L40S, four
CPUs, 32 GB RAM and a twelve-hour limit. Clean scientific revision:
`aa064504d03d396443c3e85427049dee4077d89d`; source-tree digest:
`fef52000a3e685559dd628076bb0b8f62738b3e7541d0001b00deda630b512f0`.
Checkout: `/oscar/scratch/ezhan153/pure-sgd-landscape-v1`.
### Completed landscape run

Array `6740232_0` completed with exit `0:0` in 1:36:03. Retained scheduler
logs show successful W&B synchronization and no infrastructure failure.
[Strict online analysis `fn39wxj6`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/fn39wxj6)
verified the unique expected cell, clean source/data identity, all 160 epochs,
62,560 gradient minibatches, eight probes, and all 35 slice masks per probe.
No epochs or seeds were excluded. The authoritative scientific run is
[`p5gr1t41`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/p5gr1t41).

Final test accuracy was **87.08%**, cross-entropy **33.99200115118027**;
training accuracy was 98.166% and loss 0.06058786812782287. Training and final
evaluation remained finite. Parameters: **11,173,970**. Instrumented runtime
was 5,750.916 seconds (95.85 minutes), including 3,562.493 seconds attributed
to diagnostic calls; peak allocated CUDA memory was 1,334,112,768 bytes.
The diagnostic timing excludes activation-hook overhead and is not an isolated
GPU benchmark. Uninstrumented runtime was 2,118.982 seconds.

Online pairing against uninstrumented run `kr2n6lak` verified exactly one
training row for every epoch 1–160 in both runs. Training loss, accuracy and
learning rate matched exactly in all 160 pairs (zero mismatches, maximum
absolute difference zero); final training/test metrics also matched exactly.
Initialization and full dataset digests matched the recorded reference, as
did order/worker/augmentation seeds 1100003/2100003/3100003 and the frozen
training settings. Instrumentation therefore preserved the recorded trajectory
for this execution; this is observed replay, not a guarantee across hardware.

#### Stochastic training gradients

These are means/maxima of per-minibatch L2 norms, not norms of averaged gradients.
Consecutive alignment is the mean cosine between adjacent minibatch gradients.

| Epoch | Mean gradient L2 | Maximum gradient L2 | Mean adjacent cosine | Mean actual update L2 |
|---|---:|---:|---:|---:|
| 1 | 15.2737 | 45.3330 | -0.00374 | 0.152742 |
| 40 | 6.3573 | 8.3552 | -0.02981 | 0.063580 |
| 80 | 6.4073 | 11.0427 | -0.02054 | 0.064078 |
| 81 | 5.3328 | 8.9867 | 0.00222 | 0.005333 |
| 120 | 5.1342 | 8.9504 | -0.00253 | 0.005135 |
| 121 | 5.0089 | 10.1029 | 0.00084 | 0.000501 |
| 160 | 4.8543 | 8.6155 | 0.00129 | 0.000486 |

Across epochs, mean gradient norms ranged 4.7226–15.2737; the largest measured
minibatch norm was 45.3330 (epoch 1, zero-based batch 8). Epoch mean alignment
ranged -0.04530–0.00332: successive stochastic gradients were nearly orthogonal
or weakly opposed. Global mean exact-zero fractions ranged 8.01e-9–2.12e-6,
providing no evidence of widespread exactly zero parameter gradients.
This does not establish absence of weak gradients in particular directions.
Weight-decay effective norms were close to data-gradient norms; at epoch 160
they averaged 4.8550 versus 4.8543. Update magnitudes fell at the scheduled LR
changes. Final group mean gradient norms were conv 3.7442, product exponents
3.0485, BN 0.4141, head 0.2600, and threshold parameters 0.0790. Group norms
are not normalized for their different parameter counts.

Final learned input floors ranged 0.4582–0.5690, down from softplus(0)+1e-7.
All eight threshold gradients were nonzero in every final-epoch minibatch.
In the first-batch activation samples across all epochs/layers, 73.63–90.04%
of inputs hit the learned floor. Pre-exponential values ranged -10.2419–11.3116;
outputs were all finite, with no observed exact zeros or overflow/underflow-risk
entries. These are activation samples, not checks of every training activation.

#### Fixed-subset geometry and BatchNorm

The following measurements use the same 32 normalized, unaugmented training
examples in evaluation mode. They differ from the stochastic training objective.
Rayleigh values refer to the regularized objective; trace values remove the L2
contribution of 11,173.97. SE is the four-vector Hutchinson standard error.

| Epoch | Eval CE | Batch-stat CE | Signed Rayleigh | Relative residual | Data trace ± SE | Valid slice points |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.414e28 | 2.4413 | invalid | — | invalid | 14/35 |
| 1 | 1.9567 | 1.9523 | 741.71 | 0.3131 | 2,953 ± 2,881 | 35/35 |
| 40 | 498.6186 | 0.4477 | -3.409e13 | 0.0513 | -1.431e13 ± 1.677e13 | 35/35 |
| 80 | 2.3910 | 0.1645 | -8.273e8 | 0.00224 | -3.795e8 ± 1.991e8 | 35/35 |
| 81 | 0.019970 | 0.1343 | 8,333.52 | 0.000323 | 20,274 ± 4,358 | 35/35 |
| 120 | 0.003623 | 0.1018 | 4,183.73 | 0.0000798 | 7,311 ± 1,655 | 35/35 |
| 121 | 0.003140 | 0.1007 | 3,317.51 | 0.000850 | 6,570 ± 1,335 | 35/35 |
| 160 | 0.002461 | 0.1117 | 2,403.24 | 0.00338 | 5,720 ± 1,129 | 35/35 |

Initial Hutchinson estimation was nonfinite; 21 initial slice points were
invalid. Those measurements are explicitly unavailable. Epoch-1 power residual
is large, so its estimate is poorly converged. Later finite estimates include
negative dominant-magnitude curvature at epochs 40/80; power iteration does not
estimate the largest algebraic eigenvalue or the full spectrum. The noisy
epoch-40 trace cannot establish its sign reliably. Threshold max operations
also make this a piecewise-smooth objective; HVPs describe local autograd branches.

The two sampled axes/grid show substantial sensitivity: at epoch 40 their
finite CE values ranged 0.5721–9.958e13; at epoch 80, 0.1336–30.6337. At epoch
160 they ranged 0.002461–3.9435 over the registered relative perturbations.
All points after initialization were valid. These two directions do not
characterize the complete landscape or prove optimizer superiority.

BatchNorm mode differences were large at initialization and epoch 40, despite
finite training gradients. At epoch 40 the fixed-subset eval gradient norm was
4.100e7 versus the stochastic epoch mean 6.3573, while eval/batch-stat CE was
498.62/0.4477. Thus large evaluation-mode geometry must not be mistaken for
training-gradient explosion. At epoch 160 eval/batch-stat CE was 0.00246/0.11166,
with one of 32 predictions differing. These subset comparisons do not identify
the cause of test outliers.

#### Final test loss tail

Median per-example CE was 0.000479, 90th percentile 1.4544, 99th percentile
8.1676, and maximum **306,684.8125**. The largest example alone contributes
**90.22%** of total test loss, explaining the large mean despite 87.08% accuracy.
Maximum absolute logit was 644,352.4375. Incorrect predictions had mean CE
262.8236 and confidence 0.7789; correct predictions had mean CE 0.04040 and
confidence 0.9665. This demonstrates an extreme loss tail, not its mechanism.
No checkpoint or per-example identity was retained for further attribution.

## Instrumented momentum-SGD seed-0 comparison

`configs/pure_cifar/landscape_momentum_sgd_seed0.yaml` registers a separate
exploratory run using the original five-seed study's **momentum 0.9** recipe,
restricted to seed 0. LR 0.01, weight decay 0.001, milestones 80/120, batch
size 128, 160 epochs, data transformations and model initialization stay
frozen. The instrumentation, fixed 32-image training subset, probe schedule,
filter-normalized slice coordinates, curvature estimators, and final test-tail
diagnostics are identical to the plain-SGD landscape protocol above. The new
protocol/group is `pure-cifar10-resnet18-momentum-sgd-landscape-v1`.

The primary comparison will pair the momentum run with uninstrumented
momentum seed-0 run `66303l7q` by initialization and dataset digests, RNG seeds,
all 160 training losses/accuracies/LRs and final test metrics. It will then
compare measured stochastic gradient/update trajectories, product activation
statistics, fixed-subset geometry, local loss slices, BatchNorm mode differences
and test loss tails with instrumented plain-SGD run `p5gr1t41`. The two
optimizer runs follow different parameter paths: equal probe coordinates and
subset permit a controlled diagnostic protocol, but not pointwise comparison
of the same weights. The comparison is one paired seed, with no optimizer
ranking or causal attribution from the local slices.

The completed result and paired diagnostic comparison follow the launch record.

Scientific array `6749126` was submitted as `0-0%1`, one L40S, four CPUs,
32 GB RAM and a twelve-hour limit from clean revision
`83a02620023a7b333633ba6e4975b3d715cbc7ef`. Its isolated checkout is
`/oscar/scratch/ezhan153/pure-momentum-landscape-83a0262`. Shell syntax,
both dry-run plans and `sbatch --test-only` passed. The scientific job uses
the extracted official CIFAR-10 archive (MD5
`c58f30108f718f92721af3b95e74349a`) in that checkout.

Engineering validation: 245 tests passed locally (one integration test
deselected). CPU smoke `nlubdl3h` recorded four finite-gradient minibatches;
training stayed finite and the final reduced evaluation was nonfinite. Local
RTX 4060 Ti smoke `g2xaavna` recorded eight finite-gradient minibatches and
two scheduled probes, with peak allocated memory 1,301,975,040 bytes. Its
training loss 2.350838363170624 and accuracy 0.111328125 exactly matched
the prior uninstrumented momentum smoke `xmz4e9pw`, as did initialization,
data digest and order/worker/augmentation seeds. Both recorded the same
nonfinite final evaluation after their reduced one-epoch budget. The full
160-epoch momentum baseline completed finite; reduced-smoke evaluation
failure is retained as an engineering observation, not altered by tuning.
Oscar passed the same 245 tests. The first disposable job `6749005` stopped
before training because its shared extracted-data target was empty; its log
retains the infrastructure failure. The official archive was extracted only
into the new isolated checkout. Corrected disposable job `6749031` completed
with exit `0:0`, eight gradient minibatches, two probes and all 35 slice masks
per probe. It recorded finite training and the same reduced-budget final
evaluation failure. Both disposable W&B records were checked online and their
group deleted before scientific submission.

### Completed momentum landscape result

Oscar task `6749126_0` completed `0:0` on gpu3106 in 1:37:05; retained logs
show successful online W&B synchronization and no infrastructure error.
[Strict analysis `jw3a3qd9`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/jw3a3qd9)
verified one expected cell, clean source/data identity, all 160 epochs,
62,560 gradient minibatches, eight probes, and all 35 validity masks per
probe. Its [scientific run `rle3qp4d`](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/rle3qp4d)
had no numerical failure or exclusion. Source-tree digest:
`8ea8075f38a4c3429f41db0c773b924275f2b31fd20d4ecd6b759108dd915a39`.
Online pairing against original momentum seed-0 run `66303l7q` found exactly
one row for each epoch 1–160 in both runs. Training loss, accuracy and LR
matched exactly at every epoch (zero mismatches, maximum absolute difference
zero); final training and test metrics also matched exactly. Initialization,
full dataset, order, worker and augmentation identities matched. This is
observed exact replay on the tested platform, not a general determinism claim.

| Seed-0 SGD condition | Momentum | Final train loss / accuracy | Final test CE / accuracy | Instrumented training time |
|---|---:|---:|---:|---:|
| Plain SGD `p5gr1t41` | 0 | 0.060588 / 98.166% | 33.9920 / 87.08% | 5,750.916 s |
| Momentum SGD `rle3qp4d` | 0.9 | 0.005843 / 99.906% | 0.240884 / 94.22% | 5,800.576 s |

Momentum improved paired seed-0 test accuracy by **7.14 percentage points**
and avoided the extreme test-loss tail observed without momentum. Both runs
used 11,173,970 parameters, the same initial model digest
`4b937bf65d6c6aa3038b317686b7571aacd83379e2efa19eb610439f82d0f4c1`,
full dataset digest
`6b3883dca6c867f1e58def548c063d1865569b8575fcbce06b3e468a89f9895a`,
and order/worker/augmentation seeds 1100003/2100003/3100003. The momentum
run's measured diagnostic time was 3,520.182 s, and peak allocated CUDA
memory 1,383,927,296 bytes. Instrumented runtimes include diagnostic work;
they are not a clean optimizer-speed comparison.

#### Stochastic training gradients and product layers

All 62,560 minibatch data gradients were finite. The table shows per-epoch
means of minibatch gradient L2 norms, adjacent-gradient cosine, and actual
parameter-update L2 norms. These gradients use augmented examples and
training-mode BatchNorm. At a given epoch the two optimizers occupy different
weight states; the numbers are not gradients at a common parameter point.

| Epoch | Momentum gradient L2 / cosine / update L2 | Plain gradient L2 / cosine / update L2 |
|---|---:|---:|
| 1 | 4.391 / 0.1055 / 0.09989 | 15.274 / -0.0037 / 0.15274 |
| 40 | 3.422 / 0.0154 / 0.07890 | 6.357 / -0.0298 / 0.06358 |
| 80 | 3.553 / 0.0115 / 0.08213 | 6.407 / -0.0205 / 0.06408 |
| 81 | 2.667 / 0.0033 / 0.00643 | 5.333 / 0.0022 / 0.00533 |
| 120 | 1.148 / 0.0014 / 0.00337 | 5.134 / -0.0025 / 0.00513 |
| 121 | 1.221 / 0.0021 / 0.00036 | 5.009 / 0.0008 / 0.00050 |
| 160 | 0.878 / 0.0002 / 0.00028 | 4.854 / 0.0013 / 0.00049 |

Momentum epoch-mean gradient norms ranged 0.776–4.391, versus 4.723–15.274
for plain SGD. Maximum observed minibatch norms were 42.927 and 45.333,
respectively, both in epoch 1. Momentum's mean adjacent-gradient cosine was
positive early (0.1055 at epoch 1), then near zero; plain SGD was near zero
or weakly negative. Epoch-160 weight-decay-effective gradient L2 averaged
0.879 for momentum versus 4.855 for plain, close to their respective raw
norms. Actual updates reflect the momentum buffer and learning-rate schedule,
so they are not simply LR times the current gradient.

At epoch 160 the global exact-zero gradient fraction averaged **3.54%** with
momentum versus approximately **1.0e-8** without. Momentum's zeros were
concentrated in ordinary convolution (8.09%) and BN (3.70%) parameter groups;
product-exponent group zero fraction was about 3.4e-7 and all eight learned
threshold gradients remained nonzero. Exact zeros do not alone diagnose dead
units. Final group mean gradient norms with momentum were conv 0.652, product
exponents 0.524, BN 0.264, head 0.0201, thresholds 0.0332; the corresponding
plain values were 3.744, 3.048, 0.414, 0.260, 0.0790. Groups differ in size.

Final learned floors spanned 0.0893–0.9705 with momentum, compared with
0.4582–0.5690 without it. First-training-batch product-layer input-floor
fractions across epochs/layers spanned 75.45–99.36% with momentum versus
73.63–90.04% plain. Sampled pre-exponential ranges were -11.548–15.128
versus -10.242–11.312. All sampled product outputs were finite, with no
observed exact-zero, overflow or underflow-risk entries. These samples do not
cover every activation from training or testing.

#### Fixed-subset geometry

Both runs used the same 32 unaugmented normalized training images and relative
filter-normalized perturbation coordinates. Probes use evaluation-mode BN and
different current model parameters. The signed Rayleigh value is the
eight-iteration dominant-*magnitude* Hessian estimate of the regularized
objective; residual measures approximation quality. Trace is four-vector
Hutchinson with L2 curvature removed, reported with its standard error.

| Epoch | Momentum eval CE / batch-stat CE | Plain eval CE / batch-stat CE | Momentum signed Rayleigh (residual) | Plain signed Rayleigh (residual) |
|---|---:|---:|---:|---:|
| 0 | 1.414e28 / 2.441 | same | invalid | invalid |
| 1 | 1.496 / 1.579 | 1.957 / 1.952 | 3,510 (0.0155) | 742 (0.3131) |
| 40 | 0.391 / 0.214 | 498.619 / 0.448 | 10,885 (0.00884) | -3.409e13 (0.0513) |
| 80 | 2.120 / 0.339 | 2.391 / 0.165 | 65,399 (0.00360) | -8.273e8 (0.00224) |
| 81 | 0.12264 / 0.18877 | 0.01997 / 0.13435 | 4,730 (0.0182) | 8,334 (0.000323) |
| 120 | 0.001214 / 0.14418 | 0.003623 / 0.10184 | 405 (0.00155) | 4,184 (0.0000798) |
| 121 | 0.001109 / 0.13928 | 0.003140 / 0.10068 | 314 (0.0251) | 3,318 (0.000850) |
| 160 | 0.001139 / 0.13106 | 0.002461 / 0.11166 | 505 (0.000356) | 2,403 (0.00338) |

The shared initial Hessian trace was numerically invalid and only 14/35
initial loss-slice points were finite. Every later slice point was valid for
both optimizers. At epoch 40, finite slice CE ranged 0.391–8.376 with
momentum versus 0.572–9.958e13 plain. At epoch 160, the ranges were
0.001139–11.253 with momentum versus 0.002461–3.944 plain. Thus the momentum
endpoint has a smaller estimated dominant curvature along its power direction,
yet a larger maximum CE over these two sampled relative directions. No single
probe establishes globally flatter geometry.

At epoch 160, data-only trace estimates were **1,694 ± 517** (momentum) and
**5,720 ± 1,129** (plain). The four-vector uncertainty is large; the 32-image
subset and changed weights limit interpretation. At epoch 40 the plain trace
was -1.431e13 ± 1.677e13, so its sign is uncertain. The large plain-SGD
epoch-40 evaluation-mode CE and curvature coexist with ordinary finite
training-mode gradients, and BatchNorm mode is one observed difference; the
measurements do not identify a single cause. By epoch 160, eval/batch-stat CE
remained different for both models (0.00114/0.13106 momentum,
0.00246/0.11166 plain), with 1/32 probe predictions differing in each.

#### Final test loss tails

| Metric | Momentum | Plain |
|---|---:|---:|
| Median per-example CE | 0.00106 | 0.000479 |
| 90th percentile CE | 0.09483 | 1.45439 |
| 99th percentile CE | 6.3963 | 8.1676 |
| Maximum per-example CE | **10.49** | **306,684.81** |
| Maximum absolute logit | 11.53 | 644,352.44 |
| Incorrect-example mean CE | 3.865 | 262.824 |

The momentum model has no comparable extreme test-loss outlier. Its incorrect
predictions still averaged confidence 0.8248, versus 0.7789 plain; this
single-seed result does not establish calibrated probabilities. Final
accuracy and cross-entropy are measured on the full 10,000-image test set
once, with no test-based selection. This is a paired descriptive comparison
of two trajectories under the same protocol, not evidence that momentum
generally avoids PURe landscape failures across seeds or architectures.
