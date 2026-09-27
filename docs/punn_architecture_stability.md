# Plain SGD stability across PUNN architectures

This exploratory extension tests numerical stability, separately from learning
quality, on f1/f4 regression and the 2024 paper's XOR, Iris, Wine and Diabetes
classification tasks. The original signed-input product activation and linear
output are retained. Source: Engelbrecht and Gouldie (2024), Sections 2 and 6,
<https://www.mdpi.com/1999-4893/17/6/241>.

| Task | Inputs | Small hidden units | Oversized hidden units | Outputs |
| --- | ---: | ---: | ---: | ---: |
| f1 | 1 | 1 | 2 | 1 |
| f4 | 2 | 3 | 8 | 1 |
| XOR | 2 | 1 | 2 | 1 |
| Iris | 4 | 2 | 4 | 3 |
| Wine | 13 | 2 | 10 | 3 |
| Diabetes | 8 | 1 | 8 | 1 |

The oversized widths follow Table 1's SUNN widths, as specified for oversized
PUNNs in Section 6.2. Regularized models use the same oversized width and an
explicit objective `MSE + 0.0001 * sum(parameter**2)`, including output bias.
The penalty is added to the loss; optimizer weight decay is zero. Including
bias is a recorded reconstruction choice because the paper does not clarify
it. Regularized conditions are limited to classification, following Section
6.2's stated scope. There is no regularized regression claim.

`architecture_stability.yaml` freezes 16 task/architecture conditions, each
with 30 seeds (480 runs). Plain SGD means zero momentum, learning rate 0.1,
FP32, batch one, constant learning rate, no clipping, no AMP and no parameter
projection. The 500-epoch budget and initialization in [-1,1] match the
historical reconstruction. Zero momentum differs from the paper and earlier
repository SGD runs, which used momentum 0.9; the new small controls avoid
confounding that change with width. No hyperparameter search is performed.

CPU scalar arithmetic preserves the prior local training convention. This
study needs no GPU landscape sampling: it records finite-value diagnostics
and first failure locations directly in online W&B. A large finite loss is
not a numerical failure. The primary outcome is the number reaching all 500
epochs with finite training and test MSE, finite objective, gradients and
parameters. Failure reasons and failed epochs remain in the full attempted
seed accounting; finite-loss averages are secondary.

Data, split and minibatch-order seeds are paired across architectures.
Oversized and regularized models also share exact initialization. Small
models have different parameter dimensions, so initialization digests cannot
match the larger models. Author datasets/preprocessing are unavailable; the
public classification data and explicitly recorded preprocessing are a
reconstruction, not an exact numerical paper replication.

Iris and Wine use pinned UCI raw files (150 and 178 rows). Diabetes uses the
768-row Pima dataset from a commit-pinned public mirror, because the historical
UCI endpoint returns 404; its numeric rows were independently cross-checked.
Raw bytes are SHA-256 verified before parsing. XOR is the four-pattern truth
table with inputs -1/+1 and targets 0/1. Iris and Wine targets are one-hot;
Diabetes targets are 0/1. The loss remains MSE with linear outputs, rather than
cross entropy or a sigmoid output.

For Iris/Wine/Diabetes, scaling is fit to each training split's feature minima
and maxima, mapping the training range to [-1,1]. Test values use those same
training extrema and are not clipped. Only exact normalized zeros are replaced
by +1e-6, because the principal-power formula is undefined at zero; every
nonzero value, including tiny inputs, remains unchanged. Replacement counts,
scaler extrema, split indices, raw-source URL/hash and label mapping are stored
with the exact train/test tensors in W&B dataset artifacts. Pima's original
zero sentinels are preserved as observed raw values, without imputation. This
explicit near-zero policy can affect stability and is part of the recipe.

Validation uses focused unit tests and all 16 conditions at seed 0 for two
epochs as online CPU engineering smokes. After a clean source commit, run the
registered scientific plan and strict analysis:

```powershell
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/architecture_stability_cpu_smoke.yaml --run --workers 4
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/architecture_stability.yaml --run --workers 4
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.punn_architecture_analysis --config configs/product_unit/architecture_stability.yaml
```

## Completed results (2026-09-26)

All 480 registered runs finished: 245 finite completions and 235 numerical
failures. There were no infrastructure failures or excluded seeds. Each cell
contains all 30 seeds, including failures. Strict online analysis:
[jwu371nr](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/jwu371nr).
The W&B analysis artifact `punn-architecture-stability-jwu371nr:v0` contains
`stability_report.json` (including every run ID, failed seed and failure epoch)
and `numerical_stability.png`.

| Task | Small failures / 30 | Oversized failures / 30 | Regularized oversized failures / 30 |
| --- | ---: | ---: | ---: |
| f1 | 0 | 2 | Not run |
| f4 | 4 | 24 | Not run |
| XOR | 0 | 0 | 0 |
| Iris | 19 | 24 | 21 |
| Wine | 11 | 30 | 30 |
| Diabetes | 11 | 29 | 30 |

Oversizing increased observed numerical failures on every task except XOR.
At lambda=0.0001, L2 was not a dependable stability remedy. On Iris it rescued
seeds 7, 9, 20, 25 and 28, but caused failures in seeds 12 and 15, for a net
three additional finite completions. It rescued no Wine seed. On Diabetes it
made seed 7 fail at epoch 69; that seed was the only finite oversized run.
These are descriptive paired observations, not a claim of statistical
significance or an exact replication of the authors' classification numbers.

All initial full-training objectives were finite. Of 235 failures, 233 first
occurred in the training forward pass, one during backward computation, and
one during final test evaluation (small Iris seed 17, after 500 finite training
epochs). There were 221 first-epoch failures. Every oversized f4/Wine failure
and all 29 oversized Diabetes failures occurred in epoch 1. Later failures
also occurred, up to epoch 429 during training, so a brief smoke alone would
miss some instability.

Finite completion does not mean good learning or well-behaved magnitudes.
For example, oversized f1 seed 0 completed with a maximum finite gradient
norm of 1.36e15 and parameter magnitude of 1.34e14; its test MSE was 0.06747.
Both this run (`yj5xwwtk`) and small f1 seed 0 (`t9t2fj11`) were independently
replayed from their archived dataset tensors. Outcomes, initialization,
processed-example counts and numerical metrics matched, allowing one
floating-point serialization step in a gradient maximum.

All scientific cells used clean committed source
`504b9ce37ff01f57ce5a4a79063d588b18c035fc`, with source-tree digest
`f947b110fbeffea6af503dd0570ea5651f5b5bc5cc5ca657b59d57a8042a4008`.
Before launch, 56 focused unit tests passed and all 16 online CPU engineering
smokes were checked. Afterward, all 480 dataset artifact manifests and all
180 unique task/data-seed tensor datasets passed readback/digest checks;
failure sample locations and dedicated failure counters were verified.
The two-file analysis artifact was downloaded and its figure visually checked.

An audit found that W&B's delayed history reductions can replace generic
terminal summary metrics on runs failing after a logged epoch with that
earlier history value. Failure outcome, reason, phase, epoch, sample location
and `failure_examples_processed` are unaffected, as are the final metrics of
finite completions (epoch 500 is always logged). Use the dedicated failure
counter for failure progress. Generic failed-run `examples_processed`, last
finite MSE/objective and maxima must not be treated as exact terminal values;
recovering them would require replay. The comparison above does not use those
affected fields. Subsequent code separates history into `training/` metrics
and stores a separate `terminal_result` snapshot; the completed sweep retains
its original scientific revision and records.
The logging fix and strict analysis tests passed together (22 tests), including
a simulated delayed history reduction that preserves the terminal snapshot.
