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
