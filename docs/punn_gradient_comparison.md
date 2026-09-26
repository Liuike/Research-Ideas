# PUNN gradient optimizer exploration

This is an untuned comparison on the same reconstructed f1 and f4 models,
data generator, signed-input forward expression, and uniform [-1,1]
initialization as the 2024 basic reconstruction. Ten paired seeds (0–9),
500 epochs, batch size one, a seeded permutation per epoch, FP32, constant
learning rates, no clipping, and no exponent projection are frozen in
`configs/product_unit/gradient_defaults.yaml`. All methods have zero tuning
budget. This exploratory subset does not establish the main study's Muon
headline gates or show best achievable performance.

| Method | Learning rate | Momentum / Adam betas | Weight decay | Assignment |
| --- | ---: | --- | ---: | --- |
| SGD | 0.1 | momentum 0.9 | 0 | all parameters |
| Adam | 0.001 | (0.9, 0.999) | 0 | all parameters |
| AdamW | 0.001 | (0.9, 0.999) | 0.01 | all parameters; secondary condition |
| Moonlight Muon | 0.02 | momentum 0.95, Nesterov | 0 | exponent matrix |
| Muon's AdamW head | 0.001 | (0.9, 0.95) | 0 | output weight and bias |

AdamW's nonzero decay distinguishes it from Adam and is explicitly secondary.
The Moonlight matrix update is ported from the [official pinned toy implementation](https://github.com/MoonshotAI/Moonlight/blob/c2ad5b20c605086526a179d36901bfc41b52b44b/examples/toy_train.py):
five quintic Newton–Schulz iterations and shape scale
`0.2 * sqrt(max(rows, cols))`. Orthogonalization uses FP32 instead of the
source's BF16, in accordance with the project protocol. The separate head
learning rate and zero decay are experimental choices. The head uses PyTorch
AdamW rather than the source's custom hybrid AdamW arithmetic.

Muon receives only the exponent matrix (1×1 for f1 and 3×2 for f4); the output
head is excluded. This tiny-matrix, batch-one experiment is far from the
large-layer setting for which Moonlight was designed. Numerical failures
include nonfinite loss, gradient, parameters, optimizer state, or final MSE;
finite completion is not a claim of successful fitting. Online W&B is the
authoritative result store, and analysis rejects missing/duplicate cells and
checks data, initialization, minibatch-order seeds, and source consistency.
