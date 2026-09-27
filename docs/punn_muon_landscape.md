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
