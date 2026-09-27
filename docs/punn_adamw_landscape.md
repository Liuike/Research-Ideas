# AdamW on the recorded PUNN landscapes

`configs/product_unit/adamw_landscape.yaml` registers 20 untuned cells: f1/f4,
AdamW, seeds 0–9, 500 epochs. The recipe matches the existing
`gradient_defaults.yaml`: learning rate 0.001, betas (0.9, 0.999), epsilon
1e-8, weight decay 0.01 on all parameters. Decay makes this a labelled
secondary comparison. This extends the 2024 PUNN reconstruction; AdamW was
not an optimizer in that paper's experiments.

Training retains CPU FP32, batch-one seeded order, no clipping/AMP, and
unprojected exponents. Landscape diagnostics use the local RTX 4060 Ti and
retain original scalar CPU reference losses. Data, initialization, order,
slice directions, diagnostic seeds, and sampling budgets match the earlier
recorded SGD/PSO/DE runs for seeds 0–9. The ten seeds were selected to match
the existing AdamW study, before observing new outcomes; no retuning occurs.

AdamW does not change the model's underlying training-set MSE surface.
Decoupled weight decay influences the update; the recorded landscapes show
the unregularized MSE, not an MSE-plus-penalty objective. Differences in
optimizer trajectories and the neighborhoods around their endpoints are the
comparison of interest. Two-dimensional slices cannot show all directions
in the three-/ten-dimensional parameter space.

Artifacts use the same schema as
[`punn_landscape_recording.md`](punn_landscape_recording.md): initial/epoch/
terminal parameter states, 31×31 slices at initialization, epoch 1, every 50
epochs and terminal, global PRW/MRW/uniform/dispersion samples at bounds 1
and the paper-wide bounds 3 (f1)/7 (f4), exact dataset, explicit nonfinite
masks, and full clean-source/environment provenance. Online W&B is the
authoritative store; no project result/checkpoint files are generated.

Run focused unit tests and both registered two-cell smoke plans before the
scientific plan. With the workspace W&B/UV environment described in the
recording document:

```powershell
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/adamw_landscape_cpu_smoke.yaml --run
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/adamw_landscape_gpu_smoke.yaml --run
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.experiment plan configs/product_unit/adamw_landscape.yaml --run --workers 4
.venv/Scripts/uv.exe run --frozen --no-sync python -m optimizer_resurrection.punn_analysis --config configs/product_unit/adamw_landscape.yaml
```

The separate protocol `punn-adamw-recorded-v1` and W&B group
`punn-adamw-landscape-v1` preserve the original registered replication.
Scientific cells require a clean committed revision. Final comparisons must
verify every cell/artifact and explicitly match the historical dataset,
initialization, minibatch order and global-sample digests across groups.

## Completed results, 2026-09-26

All 20 cells completed from clean revision
`7150cb1c71d68a58c926f01986a5c09d587de046`, source-tree digest
`0081b3bebee15b17d4c3f535ed9e20f9fddf959cf6661d0dd83952c88968f582`.
The [strict W&B analysis](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/qvndvhpq)
accepted every condition and all 20 artifact manifests were verified.
AdamW recorded **10,040 parameter states and 240 landscape slices**. No AdamW
run had numerical training failure, and no seed was excluded from its means.

The comparison uses only the same ten seeds from the earlier local recorded
replication, not its full 30-seed aggregates. All 20 AdamW cells were matched
against 60 SGD/PSO/DE cells by dataset, initialization, minibatch-order seed,
landscape seed, raw global-sample digest, and every diagnostic budget/bound.
The optimizer does not change the MSE surface: paired global arrays are the
same. The optimizers' trajectories and final slice centers differ.

| Task | Method | Finite / 10 | Numerical failures | Mean test MSE of finite runs | Median test MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| f1 | SGD | 8 | 2 | 0.057088 | 0.045495 |
| f1 | AdamW | 10 | 0 | 0.048525 | 0.056816 |
| f1 | PSO | 10 | 0 | 0.048671 | 0.049686 |
| f1 | DE | 10 | 0 | 0.050168 | 0.050377 |
| f4 | SGD | 7 | 3 | 0.059490 | 0.031583 |
| f4 | AdamW | 10 | 0 | 0.014283 | 0.014515 |
| f4 | PSO | 10 | 0 | 0.012605 | 0.011432 |
| f4 | DE | 10 | 0 | 0.014144 | 0.014413 |

The baseline numerical-failure seeds are f1 SGD 0, 3 and f4 SGD 0, 1, 4.
Among pairs where both stayed finite, AdamW had lower test MSE in 2/8 f1 and
5/7 f4 pairs. Including baseline failures gives 4/10 and 8/10 respectively.
Several finite SGD runs fit f1 almost exactly; AdamW is not uniformly better
on f1, despite its lower mean and absence of breakdown. f4 AdamW improves
stability and typical finite test error relative to this SGD recipe, while
its mean is similar to DE and higher than PSO. These are untuned recipes with
different learning rates; no equal-budget tuned optimizer ranking or claim
isolating adaptivity from learning rate/decay follows.

### Measured endpoint neighborhoods

The comparison artifact `punn-adamw-comparison-qvndvhpq:v0` contains:

- `all_seed_comparison.png`: sampled CPU-reference learning curves, numerical
  failure counts and paired final test errors for all ten seeds, including
  failures in the accounting.
- `adamw_seed_2_slices.png`: shared initialization and final SGD/AdamW slices
  for seed 2, which stays finite under both optimizers. Directions are shared,
  each plane is centered at its own model, and colors saturate outside log10
  MSE [-4, 3]. It is an illustration, not evidence of failure frequency.
- `paired_comparison.json` and `comparison_analysis.py`: all matched source
  records, descriptive results and the code producing the comparison.

As a post-hoc descriptive check, compute the 95th percentile of MSE across
each final 31×31 radius-1 slice, minus its center MSE. Use only pairs where
both optimizers stayed finite: eight f1 pairs and seven f4 pairs. Every such
AdamW slice has a greater increase than its paired SGD slice. All points in
these final slices are finite.

| Task | Finite pairs | AdamW greater slice loss increase | Median SGD increase | Median AdamW increase |
| --- | ---: | ---: | ---: | ---: |
| f1 | 8 | 8/8 | 0.9978 | 5.1011 |
| f4 | 7 | 7/7 | 0.1198 | 2.9680 |

For seed 2 specifically, the f4 center MSE is 0.02821 under SGD and 0.01501
under AdamW, while the slice's 95th-percentile MSE is 0.1372 versus 7.1945.
AdamW reaches lower loss in that example while having steeper walls at this
perturbation scale. This is sensitivity in a sampled plane at a specified
parameter scale, not an infinitesimal curvature estimate, a full-space
flatness measure, or a replication of the paper's entropy/neutrality/FDC
statistics. It does not establish how the optimizer avoided breakdown.
Raw per-seed summaries are in `punn-adamw-slice-summary-qvndvhpq:v0`; the
analysis summary retains both the paired plane statistics and their scope.

### Validation and recording recovery

63 focused unit tests passed. Both registered two-cell CPU/GPU smoke plans
passed; all four smoke artifacts were read back, with exact CPU-reference
agreement across devices. Independent checks on full f1/f4 seed-2 artifacts
verified terminal slice centers against saved train MSE, sixteen direct
scalar CPU reference points, and exact paired global arrays. All previously
registered replication commands/condition IDs remain unchanged.

The first f1 seed-7 recording attempt (`fq94u4qk`) failed after epoch 500,
before an artifact was committed; its recorded train MSE was finite. The
original exception text was unavailable. A local in-memory replay of the
recorder completed, and the exact command expanded from the registered plan
succeeded when retried under the same W&B ID and clean source revision.
Train MSE was identical (0.05978836864233017). The failed attempt and recovery
are disclosed in that run's `recording_retry` summary. This was one recovered
recording failure, not a numerical training failure or an excluded seed.
