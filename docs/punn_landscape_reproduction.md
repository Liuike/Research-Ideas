# Basic PUNN landscape reconstruction

Target: Engelbrecht and Gouldie, *Fitness Landscape Analysis of Product Unit
Neural Networks*, Algorithms 17(6):241 (2024), Sections 2, 6, and 8:
https://www.mdpi.com/1999-4893/17/6/241

This is a focused reconstruction of the paper's core optimizer observation:
momentum SGD failed to train its shallow PUNNs, while PSO and differential
evolution trained them. It does not reproduce the paper's full nine-task suite,
classification datasets, landscape random walks, or all architecture/boundary
conditions. It is separate from the Muon study; no Muon inference follows from
these runs.

## Grounded specification

- Model: one hidden learned-exponent product layer, then a linear output with
  bias. The `f1` model has 1 input and 1 hidden unit (3 trainable parameters);
  `f4` has 2 inputs and 3 hidden units (10 parameters), matching Table 1.
- Signed nonzero inputs: each hidden activation is the real part of the
  principal complex power in the paper's Section 2, implemented as
  `exp(V @ log(abs(x))) * cos(pi * V @ 1[x < 0])`. A zero input raises an error;
  generated exact zeros are resampled. No positive shift or exponential clamp
  is applied.
- The complex-number expression handles negative inputs. The paper's term
  *complex PUNN landscape* instead refers to an oversized architecture;
  that architecture is outside this basic reconstruction.
- Functions: `f1(x)=x²`, 50 sampled patterns; `f4(x,y)=y⁷x³−0.5x⁶`, 300
  sampled patterns. Inputs are uniform on `[-1,1]`. Each seed uses a 75/25
  train/test split shared across methods.
- Weights start uniform on `[-1,1]`. Training uses FP32, MSE, 500 epochs,
  SGD learning rate 0.1 and momentum 0.9, PSO inertia 0.7298 and both
  accelerations 1.496, or DE mutation factor 0.7 and crossover rate 0.3.
  The registered scientific plan has 30 seeds per task and method.

The authors offer their datasets on request but do not publish the sampled
patterns or seeds. Their text also does not fully specify minibatch order,
population size, population update variant, or boundary handling. The local
protocol records its choices: seeded reshuffle and batch-one SGD; population
size 20; global-best PSO and DE/rand/1/bin; clip at `[-1,1]` for population
methods. The baseline gives PSO/DE more model evaluations per epoch than SGD,
as the paper's epoch convention does; the run records evaluation counts. These
choices prevent an exact numeric Table 5 reproduction claim. The initial
`[-1,1]` search bounds cannot represent `f1=x²` exactly because its exponent
would need to equal 2. The full study should therefore also inspect the
paper's wider-bound condition before interpreting representational limits.

## Run the registered plan

Run unit tests and the CPU smoke first. Use the locked environment and expand
all configs through `experiment plan`:

```powershell
.\scripts\test.ps1 tests/test_product_unit.py tests/test_punn_landscape.py tests/test_punn_population.py
.venv\Scripts\uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs\product_unit\landscape_smoke.yaml --max-runs 6
.venv\Scripts\uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs\product_unit\landscape_basic.yaml --max-runs 6
```

The full plan contains 180 cells: two tasks, three methods, 30 paired seeds.
Scientific cells require a clean committed revision and online W&B. The smoke
config is labelled `engineering-smoke` and may run on a dirty tree. Both plans
use CPU so their small-model arithmetic has stable deterministic behavior.
Set `WANDB_DIR`, `WANDB_DATA_DIR`, `WANDB_CONFIG_DIR`, and `WANDB_CACHE_DIR`
to an ignored workspace `wandb/` directory if the default user profile
directories are not writable. Never set W&B offline mode.

The 180-cell array ran on Oscar as job `6649298` from clean revision
`6a370b1`; the isolated worktree is pinned to that commit. Analyze the full
W&B group with:

```powershell
.venv\Scripts\uv.exe run --frozen --no-sync python -m optimizer_resurrection.punn_analysis --config configs\product_unit\landscape_basic.yaml
```

Analysis rejects missing or duplicate conditions, broken seed/data pairing,
inconsistent source revisions, and incomplete successful runs. Numerical
failures are counted separately from completed MSE values.

## Initial engineering smoke, 2026-09-23

The 5-epoch, seed-0 smoke is a software check, not a 30-run reproduction.
All six W&B runs shared the same dataset digest within each task. Both SGD
cells became nonfinite during epoch 1. PSO and DE stayed finite and reduced
training MSE from the shared initialization.

| Task | SGD | PSO train MSE | DE train MSE |
| --- | --- | ---: | ---: |
| f1 | nonfinite at pattern 33 of epoch 1 | 0.0508 | 0.0905 |
| f4 | nonfinite at pattern 207 of epoch 1 | 0.0321 | 0.2142 |

W&B run IDs: f1 SGD `fwfljisc`, PSO `ctxqx9wz`, DE `y4orwy33`;
f4 SGD `9cbjj01o`, PSO `99v7ad1j`, DE `5hbaaock`. The `f4` DE smoke has
only 5 generations and is not at the paper's reported performance level.

## 500-epoch engineering pilot, 2026-09-23

The paired seed-0 pilot reached the registered 500-epoch budget for PSO and
DE. Both SGD runs failed numerically during epoch 1 (f1 after 34 pattern
evaluations, f4 after 207). The completed population runs had final train/test
MSE as follows; these single-seed, dirty-source runs are not scientific
estimates or exact Table 5 values.

| Task | PSO train / test MSE | DE train / test MSE |
| --- | ---: | ---: |
| f1 | 0.04452 / 0.03646 | 0.04452 / 0.03646 |
| f4 | 0.01310 / 0.00868 | 0.01398 / 0.01001 |

The six source runs and completeness check are recorded in the
[pilot W&B analysis](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/vm53fh3l).

## 30-seed scientific result, 2026-09-23

All 180 Slurm parent tasks completed. The [paired W&B analysis](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/msv5dnqu)
accepted all 180 unique condition IDs, matched every dataset digest across
methods for each task and seed, and verified the same clean source revision
and source-tree digest across runs. The table gives mean test MSE for
*completed* runs only; numerical failures are shown separately and do not
contribute to those means.

| Function | Method | Completed / 30 | Numerical failures | Mean test MSE | Paper Table 5, first oPUNN row |
| --- | --- | ---: | ---: | ---: | ---: |
| f1 | SGD | 29 | 1 | 0.09754 | not reported |
| f1 | PSO | 30 | 0 | 0.04503 | 0.062 |
| f1 | DE | 30 | 0 | 0.04526 | 0.059 |
| f4 | SGD | 19 | 11 | 0.37335 | not reported |
| f4 | PSO | 30 | 0 | 0.01591 | 0.031 |
| f4 | DE | 30 | 0 | 0.01679 | 0.033 |

On paired test sets, SGD either failed numerically or had greater test MSE
than *both* population methods in 21/30 f1 seeds and 29/30 f4 seeds. Thus,
the reconstruction supports the paper's broad observation that PSO and DE
train these PUNNs more reliably than SGD, especially for f4. It does not
replicate a literal failure of SGD in every run: 29 f1 and 19 f4 SGD runs
remained finite through 500 epochs. The paper does not give a numeric SGD
failure threshold or publish its SGD results, so finite completion alone
cannot establish that those runs trained successfully.

The population MSEs are lower than the paper's first oPUNN rows, rather than
matching them numerically. The sampled patterns and seeds are unavailable,
and population size, update details, minibatch order, and boundary policy
are reconstructed choices. This is a directional replication of two basic
regression cases, not an exact recreation of Table 5 or the full landscape
analysis. The paper's larger-bound oPUNN rows remain untested here.
