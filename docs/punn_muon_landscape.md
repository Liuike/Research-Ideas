# SGD, AdamW and Moonlight Muon on PUNN landscapes

`configs/product_unit/muon_landscape.yaml` registers 20 exploratory cells:
f1/f4, seeds 0–9, 500 epochs. It retains the existing untuned recipe from
`gradient_defaults.yaml`: Moonlight-scaled Muon on the exponent matrix with
learning rate 0.02 and momentum 0.95, plus an AdamW output weight/bias head
with learning rate 0.001, betas (0.9, 0.95), epsilon 1e-8. Both have zero
decay. Moonlight uses five quintic Newton–Schulz iterations and shape scale
`0.2 * sqrt(max(rows, cols))`, in FP32. This is the repository's existing
hybrid setup, not Muon on every parameter.

The model exponent matrices are only 1×1 (f1) and 3×2 (f4). This untuned
exploration does not establish a broad optimizer ranking or the main
study's headline Muon/MM gates. No calibration or retuning occurs.

CPU FP32 batch-one training preserves the scalar objective and seed/order
conventions used for the earlier local SGD/AdamW records. The RTX 4060 Ti
evaluates the same registered raw landscape diagnostics, with original CPU
reference losses. No clipping, AMP or exponent projection is used. The new
protocol/group are `punn-muon-recorded-v1` / `punn-muon-landscape-v1`.
The signed-input PUNN models, data, initializations, diagnostic seeds, bounds
and measurement budgets are unchanged. The underlying MSE surface is shared;
the optimizer changes the training path and endpoint. Landscape loss is
unregularized MSE, including for the secondary AdamW baseline with decay .01.

Run focused tests, the registered two-cell CPU and GPU smoke plans, then the
clean-source scientific plan via `experiment plan`. Use the workspace W&B/UV
environment documented in `punn_landscape_recording.md`:

```powershell
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/muon_landscape_cpu_smoke.yaml --run
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/muon_landscape_gpu_smoke.yaml --run
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/muon_landscape.yaml --run --workers 4
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.punn_analysis --config configs/product_unit/muon_landscape.yaml
```

The final three-way comparison must validate all 20 new Muon cells and pair
them to the 20 local AdamW records and the first ten seeds per task in the
local SGD replication. Check dataset, initialization, minibatch-order seed,
raw global-sample digest, and all diagnostic settings across groups. Keep
numerical failures separate from finite loss means. Compare all-seed curves
and outcomes alongside illustrative slices; a single plane/seed does not
establish failure frequency or full-space flatness.

## Completed local comparison

All 20 Muon cells completed with finite terminal losses and complete landscape
artifacts on clean source `3bac6584875b261e9b1f4674cd4350a7df933451`, source-tree
digest `fe90d026e6910d47927123087673357cb6aa3ad9b6877ded1954cebeba002b8c`.
The [strict W&B analysis](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/hlgg24bk)
accepted every expected cell. The 20 manifests contain 10,040 parameter states
and 240 slices. No Muon numerical, recording or upload failures occurred; no
seeds were excluded. The summed per-run wall time was 1,347 seconds, with four
local workers (this sum is not elapsed sweep time).

The comparison uses seeds 0–9 for each task and optimizer, 60 runs total.
SGD records come from `punn-landscape-recorded-reference-v1` at `af2d7cf`,
AdamW records from `punn-adamw-landscape-v1` at `7150cb1`. Means below use only
finite completions; failure counts always retain all ten attempted seeds.

| Task | Optimizer | Mean test MSE | Median test MSE | Numerical failures / attempts |
| --- | --- | ---: | ---: | ---: |
| f1 | SGD | 0.057088 | 0.045495 | 2 / 10 |
| f1 | AdamW | 0.048525 | 0.056816 | 0 / 10 |
| f1 | Muon + AdamW head | 0.040963 | 0.044811 | 0 / 10 |
| f4 | SGD | 0.059490 | 0.031583 | 3 / 10 |
| f4 | AdamW | 0.014283 | 0.014515 | 0 / 10 |
| f4 | Muon + AdamW head | 0.010529 | 0.008896 | 0 / 10 |

SGD failed numerically on f1 seeds 0 and 3, and f4 seeds 0, 1 and 4.
Failure means nonfinite loss, gradients, parameters or terminal evaluation;
it does not mean merely a high finite MSE. The AdamW group had one recovered
landscape-recording failure, disclosed in `punn_adamw_landscape.md`; its
training was finite and the exact registered retry completed the same run.

Validation included 73 focused unit tests, two CPU smoke cells followed by two
GPU smoke cells, and readback of all four smoke artifacts. Independent checks
of full Muon f1/f4 seed-0 artifacts confirmed the terminal parameter vector,
CPU-reference center losses and direct CPU evaluations at the plane corners.
Raw global samples were exactly equal across all three optimizers for those
representative seeds at both registered bounds.

All 60 manifests were verified. Cross-group checks matched dataset,
initialization, minibatch-order seed, global-sample digest, sampling settings
and the two slice directions for every paired seed. Every finite final slice
is at epoch 500 and its CPU-reference center equals the recorded training
loss at FP32 precision (W&B decimal serialization can differ in the last
double-precision digit).

| Task | Muon lower test MSE than finite SGD pairs | Muon lower test MSE than AdamW |
| --- | ---: | ---: |
| f1 | 4 / 8 | 4 / 10 |
| f4 | 7 / 7 | 8 / 10 |

Thus the strongest paired result is on f4. On f1 the mean improves but Muon
does not beat AdamW on most seeds, and several finite SGD runs fit nearly
exactly. Including SGD breakdowns as unsuccessful attempts, Muon has a lower
finite test MSE or avoids the baseline's failure on 6/10 f1 and 10/10 f4 seeds.
That count keeps numerical failures distinct from finite-error comparisons.

The underlying objective is identical across optimizers; the recorded
neighborhoods have different endpoint centers. A post-hoc sensitivity check
uses the 95th percentile MSE across the final radius-1, 31×31 CPU-reference
plane, minus center MSE. On the intersection of seeds where all three methods
complete, the median increases are:

| Task | Paired finite seeds | SGD | AdamW | Muon + AdamW head |
| --- | ---: | ---: | ---: | ---: |
| f1 | 8 | 0.9978 | 5.1011 | 2.7161 |
| f4 | 7 | 0.1198 | 2.9680 | 0.8574 |

No plane points were nonfinite in these paired endpoint comparisons. Muon's
increase exceeds SGD on 7/8 f1 and 7/7 f4 pairs; it exceeds AdamW on 5/8 f1
and 1/7 f4 pairs. The f1 median and paired counts therefore tell different
stories. Better test error does not imply a flatter endpoint under these
perturbations. This check depends on parameter scale and two directions and
does not measure full-space curvature or establish a causal explanation.

The W&B artifact `punn-three-optimizer-comparison-hlgg24bk:v0` contains
`all_seed_three_way.png`, `three_way_seed_2_slices.png`,
`three_way_slice_sensitivity.png`, `paired_comparison.json`, and the standalone
`comparison_analysis.py`. The all-seed figure retains early failed traces and
explicit failure counts; the seed-2 figure is illustrative. Publication
initially timed out in W&B image logging; retrying via the artifact upload
path succeeded without rerunning training.

These results compare the existing untuned recipes. SGD uses learning rate
.1 and momentum .9; AdamW uses learning rate .001 and decay .01; Muon uses
the zero-decay hybrid described above. They do not isolate optimizer effects
from learning rate, decay or parameter assignment, nor pass the main study's
headline gates. The 2024 landscape study supplies the PUNN problem and
diagnostic framework; AdamW and Muon are exploratory extensions.
