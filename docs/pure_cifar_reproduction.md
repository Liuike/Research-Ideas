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
Full-run outcome and landscape interpretation pending.
