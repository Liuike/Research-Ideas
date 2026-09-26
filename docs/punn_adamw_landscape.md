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
