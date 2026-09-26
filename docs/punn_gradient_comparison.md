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

## Results (2026-09-26)

Oscar CPU array `6721431` ran all 80 cells from clean revision
`ee7979eeb8529baf270077012c9bcf4dfba8c527`. The [validated W&B analysis](https://wandb.ai/enyan_zhang1-brown-university/optimizer-resurrection/runs/z5g4h498)
contains the full resolved experiment record and source run IDs. Analysis
verified all expected cells, one source revision/tree digest, paired datasets
and initializations, order seeds, and the full 500-epoch budget for finite
completions. There were no scheduler failures in this scientific array.

MSE statistics below include only finite completions; failures are counted
separately and are not assigned a finite error. Standard deviations describe
variation across seeds, not uncertainty intervals.

| Function | Method | Finite / attempted | Numerical failures | Mean train MSE | Mean test MSE ± SD | Median test MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f1 | SGD momentum | 9 / 10 | 1 | 0.078325 | 0.088335 ± 0.074170 | 0.058545 |
| f1 | Adam | 10 / 10 | 0 | 0.044748 | 0.047922 ± 0.019739 | 0.056277 |
| f1 | AdamW | 10 / 10 | 0 | 0.045543 | 0.048525 ± 0.019931 | 0.056816 |
| f1 | Moonlight Muon | 10 / 10 | 0 | 0.038258 | 0.040961 ± 0.024534 | 0.044811 |
| f4 | SGD momentum | 5 / 10 | 5 | 0.027396 | 0.089209 ± 0.140645 | 0.034131 |
| f4 | Adam | 10 / 10 | 0 | 0.014367 | 0.013734 ± 0.005319 | 0.014723 |
| f4 | AdamW | 10 / 10 | 0 | 0.015192 | 0.014283 ± 0.004456 | 0.014515 |
| f4 | Moonlight Muon | 10 / 10 | 0 | 0.012191 | 0.011224 ± 0.005704 | 0.011183 |

At these frozen recipes, Adam, AdamW, and Muon stayed finite on all seeds;
SGD still reproduced numerical instability. Muon had the lowest observed
mean and median test error on both functions. AdamW's decay did not improve
mean error over Adam here. These are descriptive results of an untuned
exploration; they do not establish an optimizer ranking after equal-budget
tuning or a headline Muon/MM result under the main study gates.

### Failed scientific seeds

| Function | Method | Seed | Reason | Full epochs completed | W&B run |
| --- | --- | ---: | --- | ---: | --- |
| f1 | SGD | 0 | nonfinite gradient | 0 | `9dghineq` |
| f4 | SGD | 0 | nonfinite gradient | 0 | `jwh8b53t` |
| f4 | SGD | 1 | nonfinite loss | 0 | `hvid98wr` |
| f4 | SGD | 2 | nonfinite gradient | 0 | `vi2z3p97` |
| f4 | SGD | 4 | nonfinite loss | 0 | `o9eb85ya` |
| f4 | SGD | 6 | nonfinite loss | 75 | `1i35skiu` |

### Verification and launch recovery

The locked local and Oscar environments passed 114 tests (one deselected),
including forward/gradient checks, reference Moonlight updates, SGD matching,
and pairing rejection tests. Shell syntax, representative dry runs, and Slurm
test-only validation passed. Initial disposable smoke array `6721382` failed
before training because plan decoding reached the full home UV cache. Moving
cache defaults before the first UV call fixed this; repeated smoke array
`6721411` produced all eight expected records (six finite, two recorded SGD
numerical failures), with correct clean-source/Slurm metadata and pairing.
The disposable Oscar W&B group was verified and deleted before the scientific
array was submitted. Ignored scheduler logs are retained on Oscar for recovery.
