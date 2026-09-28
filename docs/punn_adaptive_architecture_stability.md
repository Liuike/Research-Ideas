# AdamW and Muon across PUNN architectures

This exploratory extension repeats all 16 task/architecture conditions from
the [plain-SGD stability study](punn_architecture_stability.md) with AdamW and
Moonlight-scaled Muon. The frozen plan is
`configs/product_unit/adaptive_architecture_stability.yaml`: 16 conditions,
two optimizers, 30 seeds and up to 500 epochs, totaling 960 attempted runs.

The widths, exact data-generation and preprocessing rules, seed offset
100000, scalar example order, FP32 CPU arithmetic, initialization in [-1,1],
and finite-value checks are unchanged. Each optimizer is paired by dataset,
initialization and minibatch-order seed with the earlier SGD run at the same
task, architecture and seed. No tuning is performed.

Recipes reuse the previous optimizer comparison's frozen defaults:

| Optimizer | Learning rate | Momentum / beta1 | Optimizer weight decay | Assignment |
| --- | ---: | ---: | ---: | --- |
| AdamW | 0.001 | 0.9 | 0.01 | All parameters; beta2=0.999, epsilon=1e-8 |
| Muon | 0.02 | 0.95 | 0 | Product exponents only; existing FP32 Moonlight scaling, Nesterov, five Newton-Schulz steps |
| Muon output head | 0.001 | 0.9 | 0 | AdamW on output weights/bias; beta2=0.95, epsilon=1e-8 |

AdamW is explicitly secondary because its existing recipe includes nonzero
decoupled weight decay. The explicit regularized architecture still adds
`0.0001 * sum(parameter**2)` to MSE, including output bias. Thus regularized
AdamW has both this L2 objective term and the separately recorded optimizer
decay; those are different operations. Muon uses zero optimizer decay in both
architecture variants. Regularization remains limited to XOR, Iris, Wine and
Diabetes; f1/f4 retain only small and oversized variants.

The primary observation remains any first NaN/Inf in predictions, loss,
gradients, parameters or optimizer state through final test evaluation. Large
finite error or parameter magnitude alone is not a numerical failure. Failed
seeds remain in the denominator, and infrastructure failures are not silently
excluded. Online W&B records preserve clean scientific provenance, exact
dataset artifacts, failure location and a separate terminal-result snapshot.

The criterion covers these tracked tensors, not every temporary inside an
optimizer. In particular, the existing FP32 Muon implementation can overflow
its normalization norm for a very large finite update and produce a zero
direction while its parameters and state remain finite. Such a run can still
meet this finite-completion criterion. This is a limitation of that binary
measure; completion is not evidence of good learning or universally finite
internal arithmetic.

Run the registered two-epoch, seed-zero smoke across all 32 cells before the
clean committed scientific sweep. Use the frozen UV environment:

```powershell
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/adaptive_architecture_stability_cpu_smoke.yaml --run --workers 8
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/adaptive_architecture_stability.yaml --run --workers 12
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.punn_architecture_analysis --config configs/product_unit/adaptive_architecture_stability.yaml
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.punn_architecture_comparison --adaptive-config configs/product_unit/adaptive_architecture_stability.yaml --baseline-config configs/product_unit/architecture_stability.yaml
```

After validation and the clean scientific commit,
`scripts/punn_adaptive_stability_local.ps1 -Workers 12` runs the full plan and
both analyses in order, stopping on any infrastructure or analysis error.

The comparison must accept all 960 adaptive and 480 SGD terminal records,
reject duplicate/incomplete cells, verify pairing across the separately
recorded source revisions, and publish its JSON and figure as W&B artifacts.
Classification preprocessing remains a documented public-data reconstruction.
Results are descriptive PUNN stability observations, not headline Muon/MM
claims for the repository's separate gated full study.

## Preflight validation (2026-09-27)

The combined focused suite passed 75 tests. This includes recipe and matrix
expansion, assignment of Muon and its output optimizer, actual FP32 AdamW
state overflow with finite gradients/parameters, pairing, dotted W&B snapshot
readback, snapshot identity rejection, and incomplete/duplicate comparison
rejection. Independent review found all 480 scientific and 16 smoke plain-SGD
commands, parsed configurations and condition IDs identical to commit
`28cf664`; the adaptive plan expands 960 unique conditions.

All 32 two-epoch online engineering smokes completed finitely. Strict analysis
and readback verified all 32 terminal snapshots and dataset manifests, with
six unique task/data-seed tensor datasets. The smoke group is
`punn-adaptive-architecture-cpu-smoke-v1`. These short runs validate the
implementation; they do not establish 500-epoch numerical stability.

## Scientific execution

The sweep began from clean commit `25e26cf41160bab22c5115859489402333ccca3f`,
with source-tree SHA-256
`11fdecaf92fc81a984e1e54b8d7563d5db980d2944c54a3c3e93cb7837b2cf7b`
(111 files, including the existing ignored package metadata).
After 63 completed runs, unrelated CIFAR changes in the shared checkout caused
the clean-source gate to block the other 897 launches before training or W&B
run creation. Those are launch interruptions, not numerical failures.

Execution resumed in `.cache/adaptive-stability-worktree`, a detached checkout
of the same commit with the exact original source fingerprint. The recovery
expanded the unchanged config through `experiment plan`, checked all existing
records were finished and unique, and skipped their 63 condition IDs. It used
the same frozen environment and 12 local workers for the remaining 897 commands.
No recipe, dataset, seed or numerical criterion changed.

Independent in-memory CPU replays reproduced all checked terminal metrics and
identity/recipe fields exactly for oversized f1 seed zero: AdamW `iyd4bcsn` and
Muon `f2hqm87k`. Both used archived dataset
`punn-architecture-data-f1-100000:v1`, completed 500 epochs and processed 18,500
examples. The completed strict analysis subsequently verified the entire
960-run matrix.

One online summary, `wid8pa9d` (oversized f4 AdamW seed 10), remained at epoch
400 after its SDK stream had recorded successful 500-epoch completion. Native
W&B sync replayed that original stream to the same run ID. Direct API readback
then matched every terminal field against the SDK snapshot and retained the
original condition ID and clean scientific provenance. The run's
`record_recovery` summary records stream SHA-256
`545992ada1c008b446842af91f5d4de9af4909116899bb10fbcc94aa6286c4f6`;
training was not repeated. Incremental readback now allows bounded ingestion
retries but never accepts a missing terminal snapshot.

An independent replay of oversized Iris AdamW seed 2 (`8oypylyz`) reproduced
the epoch-one failure at update 52, sample index 97. Six `exp_avg_sq` entries
were Inf (four product exponents and two output weights), while parameters
and gradients remained finite; maximum gradient magnitude was
`3.704903121954941e22`. Failure location, counters and state pattern matched the
saved record, with only a final decimal serialization difference in the
parameter maximum.

The paired Muon record `jjx9yjk4` completed all 500 epochs finitely, with test
MSE `0.05923113599419594`. A separate first-epoch replay inspected 112 unchanged
Newton-Schulz inputs: no nonfinite inputs, no finite inputs with an infinite
FP32 normalization norm, and no nonzero inputs giving an entirely zero
direction. Its maximum gradient norm `2.787470588955501e18` matched the saved
500-epoch maximum. This probe found no normalization overflow in that epoch;
it does not rule it out in other epochs or cells. These were in-memory
readback probes, not additional scientific conditions.

## Completed results (2026-09-27)

All 960 adaptive scientific conditions finished and were accepted by the
strict W&B analysis: 30 distinct seeds in each of the 32 task/architecture/
optimizer cells. The comparison also accepted all 480 prior SGD conditions,
with the same dataset, initialization and example-order seed for each paired
method. There are no excluded scientific seeds. The 897 initially blocked
launches never became runs and were later completed from the isolated checkout.
An independent readback checked all 960 terminal snapshots and all 180
task/data-seed artifacts: artifact metadata, finite FP32 NPZ tensors,
content SHA-256, raw-source/data-seed metadata, and cross-architecture/
optimizer pairing. It found 935 finite completions, 25 scientific numerical
failures, and no missing snapshots, dataset mismatches or infrastructure
outcomes.

Numerical failures out of 30 paired seeds per row:

| Task | Architecture | SGD | AdamW | Muon |
| --- | --- | ---: | ---: | ---: |
| f1 | small | 0 | 0 | 0 |
| f1 | oversized | 2 | 0 | 0 |
| f4 | small | 4 | 0 | 0 |
| f4 | oversized | 24 | 0 | 0 |
| XOR | small | 0 | 0 | 0 |
| XOR | oversized | 0 | 0 | 0 |
| XOR | regularized | 0 | 0 | 0 |
| Iris | small | 19 | 0 | 0 |
| Iris | oversized | 24 | 2 | 1 |
| Iris | regularized | 21 | 2 | 1 |
| Wine | small | 11 | 0 | 0 |
| Wine | oversized | 30 | 0 | 1 |
| Wine | regularized | 30 | 0 | 2 |
| Diabetes | small | 11 | 0 | 0 |
| Diabetes | oversized | 29 | 0 | 13 |
| Diabetes | regularized | 30 | 0 | 3 |
| **Total / 480** | | **235** | **4** | **21** |

All four AdamW and 21 Muon failures were first recorded at the optimizer
update, with a nonfinite optimizer state. None first appeared during the
forward pass, backward pass, or final test. Every adaptive failed seed was
also an SGD failed seed. AdamW completed finitely on 231 seeds that failed
under SGD, and Muon on 214. This is an observation for the frozen recipes;
AdamW's 0.01 decoupled decay means the three-way comparison does not isolate
optimizer mechanics alone.

For oversized Diabetes, explicit L2 reduced Muon failures from 13 to 3 of
30, rescuing 10 of the 13 failed seeds. AdamW had no numerical failures in
either architecture. That result is specific to this dataset reconstruction,
initialization range and fixed optimizer settings. In Wine, L2 increased Muon
failures from one to two; it did not uniformly improve finite completion.
The binary failure measure also does not evaluate predictive quality or
detect every temporary FP32 overflow inside Muon.

The adaptive [W&B analysis run](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/1hew3ft6)
contains the full 960-run JSON report and numerical-stability figure. The
strict [three-way comparison run](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/j9tmiup4)
contains the 16-row paired-seed report and a comparison figure. The reports
record each failed seed and its run IDs; all source revisions and source-tree
digests are retained with the W&B records.
