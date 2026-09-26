# Recorded 2024 PUNN replication

`configs/product_unit/landscape_recorded_references.yaml` separately registers a rerun of
the original 180 basic cells: f1/f4, SGD/PSO/DE, seeds 0–29, 500 epochs. The
original `landscape_basic.yaml` and its W&B group are preserved. This request
does not include the separate gradient-optimizer comparison.

Training retains the original scalar CPU FP32 objective, model initialization,
data generation/split, SGD minibatch order, optimizer settings, population size,
and population bounds. The small three-/ten-parameter models train on CPU;
batched landscape measurements run on the local RTX 4060 Ti. Landscape RNGs
are independent and sampling never mutates the trained model. Training and
diagnostic evaluation counts are distinct. TF32/AMP are disabled.

Each run uploads an immutable `loss-landscape` W&B artifact and records its
version in the run summary. Numerical training failures also retain landscape
records; failed parameters remain raw and nonfinite values are explicitly
marked. Recording/upload failure fails the run and cannot count as complete.
Files are created only through W&B's artifact-managed temporary files:

- `dataset.npz`: exact train/test patterns and their digest.
- `training_landscapes.npz`: initialization, best/model parameter states after
  each complete epoch, terminal state, population snapshots at measurement
  epochs, common orthonormal slice directions, and raw slice coordinates,
  parameter points, MSE, and nonfinite masks. Slices use a 31×31 grid with
  radius 1 at initialization, epoch 1, every 50 epochs, and terminal state when
  its parameters are finite. Slices are not clipped to training bounds.
  Every point also retains the original scalar-candidate CPU FP32 model's MSE
  and its nonfinite mask, independent of the batched CUDA diagnostic.
- `loss_slices.table.json`: browsable W&B table with epoch, coordinates, MSE,
  and nonfinite flag. It accompanies the raw arrays.
- `global_bound_1.npz` plus `global_bound_3.npz` for f1 or
  `global_bound_7.npz` for f4: raw global landscape samples and metadata. These
  wider bounds are diagnostic only; optimizer training still uses the original
  ±1 bounds for population methods.
  Global points likewise retain scalar CPU reference losses and a summary of
  CPU/CUDA finiteness and finite-loss disagreements.
- `provenance.json`: resolved config, source revision/digest, environment,
  parameter ordering, seeds, and terminal training result.

Global measurements use dimension-many walks from evenly spaced corners:
1000-step progressive random walks at 1% and 10% of domain width, 2000-step
Manhattan random walks at 1%, 500×dimension uniform samples, and 2000 separate
uniform dispersion samples. The sampled parameter coordinates and raw losses
are retained, so future analysis can recompute landscape statistics. Sampling
choices not fully specified in the paper are recorded explicitly. The initial
finite-difference summaries and local slices are **not** claims of exact
replication of the paper's gradient, entropy, neutrality, or FDC metrics. No
SUNN baseline or additional training architecture is introduced here.

GPU smoke uncovered severe FP32 cancellation sensitivity at wider-bound
corners where equal product activations have opposite output weights. For one
f4 point, the original CPU model and CUDA sampler gave MSE about 48.2, but the
batched CPU sampler gave infinity. Batch shape and accumulation kernels matter
even on the same device. Both arrays are retained; comparisons tied to the
historical training objective should use `cpu_reference_mse`, obtained with
the original CPU `ProductUnitNetwork.forward` separately for each candidate.
CPU reference evaluation uses a deep copy and cannot mutate trained models or
consume training RNGs. No losses are clamped or substituted.

The first `landscape_recorded.yaml` registration and initial smoke configs are
preserved. The reference config adds this measurement without changing the
optimizer recipe; its group is `punn-landscape-recorded-reference-v1`.

The landscape seed is `data_seed + 2000000`, identical across methods within a
task/seed pair. Slice directions are also shared. Training losses are now
logged at the same measurement epochs for PSO/DE as for SGD. Mean test losses
must continue to exclude and separately count numerical failures.

Validation order is unit tests, six online CPU smoke cells, six online GPU
smoke cells, and inspection of uploaded artifacts before the scientific run.
Scientific runs require a clean committed source. All commands are expanded
by the registered plan; local subprocess concurrency affects scheduling only:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.cache/uv'
$env:WANDB_DIR = Join-Path (Get-Location) 'wandb'
$env:WANDB_DATA_DIR = $env:WANDB_DIR
$env:WANDB_CONFIG_DIR = $env:WANDB_DIR
$env:WANDB_CACHE_DIR = $env:WANDB_DIR
# Windows sandbox/user ownership differs here; trust only this checkout for this process.
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = (Get-Location).Path.Replace('\', '/')
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/landscape_recorded_references.yaml --run --workers 4
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.punn_analysis --config configs/product_unit/landscape_recorded_references.yaml
```

Analysis rejects missing/duplicate cells, incomplete uploaded-landscape status,
broken dataset/initialization/order pairing, or inconsistent clean source
provenance. The analysis W&B run identifies all 180 source runs.
