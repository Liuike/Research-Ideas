# Plain DA-10 Manifold Muon on the six PUNN tasks

This exploratory numerical-stability extension adds **one** optimizer condition
to the completed SGD/AdamW/Muon architecture comparison. It covers f1, f4,
XOR, Iris, Wine, and Diabetes across the same 16 task/architecture cells and
30 seeds per cell: 480 registered Manifold Muon runs, up to 500 epochs each.
The frozen record is
[`manifold_architecture_stability.yaml`](../configs/product_unit/manifold_architecture_stability.yaml).
No scale sweep, optimizer controls, or learning-rate tuning are part of this
request.

“Plain DA-10” uses the repository's `ManifoldMuon` defaults: ten SVD-based
dual-ascent iterations per scalar update, dual learning rate 0.01, outer
learning rate 0.01, momentum 0.95 with Nesterov, Stiefel scale 1, and polar
retraction after each update. It applies to the PUNN exponent matrix. The
linear output weights and bias use the same auxiliary AdamW as the prior Muon
condition: learning rate 0.001, betas (0.9, 0.95), epsilon 1e-8, zero weight
decay. The explicit classification L2 term remains 0.0001 in the registered
regularized architecture and zero elsewhere.

Data, raw-source pins, 75/25 split, task widths, seed mapping, minibatch order,
CPU FP32 batch-one updates, no clipping, and first-nonfinite detection match
the previous architecture study. Each Manifold condition uses the same
initialization *draw* as the SGD/AdamW/Muon seed, then projects only the
exponent matrix to the Stiefel manifold before training. W&B records both the
preprojection and projected initialization digests. Projected initialization
and constrained updates change the model's expressive family; numerical
failure counts are descriptive and cannot isolate an optimizer-update benefit.
An SVD failure caused by finite but overflowing DA-10 intermediates is also
recorded as a scientific numerical failure at the optimizer update, so it
cannot silently abort or disappear from the 480-run denominator. Completed
runs must end with normalized Stiefel residual at most 1e-4.

The f1 small model has a 1×1 exponent matrix, which projection fixes to +1 or
-1. It is included to satisfy the six-task numerical-stability request but is
capacity-restricted and must not be interpreted as an ordinary trainability
comparison. More generally, finite completion does not establish good fit.
The explicit L2 penalty on a fixed-scale Stiefel exponent matrix is constant
along that manifold, although it still acts on the output head.

The separate headline Muon/MM study's Gates A–C have not been satisfied for
this PUNN extension. These runs should not be reported as headline MM results.

## Preflight

The registered plan expands to 480 distinct condition IDs. The full
repository test suite and 60 focused runner/comparison tests passed. All 16 two-epoch
engineering smokes finished online in W&B with unique condition IDs, complete
dataset artifacts, and finite terminal outcomes. Their maximum initial and
final normalized Stiefel residuals were 6.17e-7 and 6.54e-7, respectively.
The smoke group is `punn-manifold-architecture-cpu-smoke-v1`; scientific runs
use the separate `punn-manifold-architecture-stability-v1` group from a clean
committed source revision.

The registered execution and strict four-method analysis are:

```powershell
uv run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/manifold_architecture_stability.yaml --run --workers 12
uv run --frozen --no-sync python -m optimizer_resurrection.punn_manifold_comparison --manifold-config configs/product_unit/manifold_architecture_stability.yaml --adaptive-config configs/product_unit/adaptive_architecture_stability.yaml --baseline-config configs/product_unit/architecture_stability.yaml
```

The comparison must verify all 480 Manifold, 960 AdamW/Muon, and 480 SGD
terminal records, reject incomplete or duplicate cells, match dataset and
order seeds, verify preprojection initialization against SGD, and report
numerical failures separately from finite but high losses.
