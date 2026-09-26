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

Implementation and 130 unit tests passed; live CPU/GPU smoke verification and
the five-seed scientific array are pending. The experiment ledger records
subsequent job/run IDs and outcomes.
