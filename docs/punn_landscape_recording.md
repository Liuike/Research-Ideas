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

## Completed local rerun, 2026-09-26

The registered 180-cell plan completed locally with four workers from clean
revision `af2d7cf72991aa28f0066ec3988e7c867f455f32`, source-tree digest
`ff8e533d42b37929f2e4c49aec50b4313b50e375b953805b466bf3f81e34c40c`.
The [strict paired analysis](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/iyogzx2n)
accepted all 180 unique conditions and verified source consistency, shared
data/initializations/minibatch order, and identical raw global landscape
samples across methods within each task/seed. All 180 immutable artifact
manifests contain the required files. In total they retain **84,936 parameter
states and 2,053 landscape slices**, including the available states/slices for
failed training runs. There were no recording or upload failures.

| Task | Method | Finite completions / 30 | Numerical failures | Mean test MSE of finite completions |
| --- | --- | ---: | ---: | ---: |
| f1 | SGD | 27 | 3 | 0.072275 |
| f1 | PSO | 30 | 0 | 0.045026 |
| f1 | DE | 30 | 0 | 0.045263 |
| f4 | SGD | 22 | 8 | 0.260680 |
| f4 | PSO | 30 | 0 | 0.015858 |
| f4 | DE | 30 | 0 | 0.016789 |

The excluded numerical-failure seeds are f1 SGD **0, 3, 12** and f4 SGD
**0, 1, 4, 18, 20, 21, 25, 29**. They remain in the 180-run accounting and
retain artifacts; they are excluded only from finite-loss means. SGD failed
or had higher test MSE than both population methods in 19/30 f1 and 27/30 f4
pairs. Finite completion does not by itself establish successful learning.

Compared with the earlier Oscar replication, population-method mean test
losses are nearly unchanged. SGD outcomes differ: the old study had one f1
and eleven f4 numerical failures. For f1 seed 3, running the original
`6a370b1` training source locally without recording reproduced the new failure
bit for bit, with the same initial loss and dataset as the Oscar record. This
rules out the recorder as the cause for that case; the precise cause of the
Windows/Linux arithmetic difference remains unverified. Local Python is
3.11.15 versus 3.11.11 on Oscar, within the registered Python 3.11 pin; locked
dependency versions match. Treat this as a new local replication rather than
an exact replay of Oscar's SGD outcomes.

Validation included 76 focused unit tests, six online CPU reference smoke
cells, six online GPU reference smoke cells, direct scalar-model checks of
downloaded f1/f4 artifacts, and complete manifest verification. One finished
f4 PSO record (`h9zo1u97`) returned an empty online summary despite a committed
complete artifact. Its original W&B-managed `wandb-summary.json` matched the
artifact's provenance and results; that exact summary was restored and the
recovery recorded in W&B before strict analysis passed. No experiment was
replaced or rerun to change an outcome.

The analysis run also contains `punn-landscape-example-iyogzx2n:v0`: an
illustrative 2×4 comparison of the shared initialization and final SGD/PSO/DE
slices for paired seed 2, using original CPU reference losses. Directions are
shared; each plane is centered on its own model. This single-seed figure is
not an aggregate landscape statistic.

The results support the paper's broad population-optimizer reliability
observation on these two regression cases. Raw walks and wider-domain losses
are now available for landscape comparisons, but the paper's complete
gradient/entropy/neutrality/FDC analysis and SUNN/oversized-architecture
baselines have not been replicated. Reconstructed sampling choices and
unavailable author datasets still prevent an exact numeric landscape-study
replication claim.
