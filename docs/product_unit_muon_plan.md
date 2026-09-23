# Plan: Muon on product unit networks

Status: research plan, not an executable or frozen experiment config. Register new
configs and thresholds before scientific runs. Keep this study separate from the
existing MLP/RNN and historical Latch groups.

## Question and historical anchor

Test whether Muon or Stiefel Manifold Muon makes gradient training of classical
product unit neural networks (PUNNs) reliable. The primary historical target is
Engelbrecht and Gouldie (2024), which reports that its SGD implementation failed
on every tested PUNN problem while PSO and DE could train them. The authors
attribute the failure to large, highly variable gradients and deep ravines;
larger exponent bounds and oversized networks also worsened searchability or
generalization. These are published findings to reproduce, not outcomes already
observed in this repository.

Primary source: https://www.mdpi.com/1999-4893/17/6/241

The classical unit uses a hidden exponent matrix `V` and a linear output:
`h_j(x) = product_i x_i ** V[j,i]`. For signed inputs, follow the paper's
real-valued complex-extension convention, equivalent away from zero to
`exp(V @ log(abs(x))) * cos(pi * V @ 1[x < 0])`. Record an explicit policy for
exact zeros and validate it against the paper's equations. Do not silently
replace this with `abs(x)**V`, a positive shift, or a clipped exponential; each
changes the model. A positive-input variant may be a separately labeled control.

## Reuse before implementation

1. Reuse the local `optim/muon.py`, `optim/manifold_muon.py`, and
   `optim/param_groups.py` with a PUNN exponent matrix as the rank-two compared
   parameter. For the matched modern comparisons, train output weights and bias
   with the same auxiliary optimizer and budget in all conditions. Compare
   optimizer steps against the original Muon implementation, PyTorch Muon, and Thinking Machines Lab's Manifold Muon
   reference before deciding whether any local optimizer changes are needed.
2. Reuse the published PUNN implementations in the public KEEL Java repository
   as a model and baseline reference. Its NNEP code includes product-unit
   regression/classification networks, exponential hidden layers, and an
   IRPropPlus regression wrapper that selects `Product_Unit`. Inspect its
   preprocessing, forward rule, initialization, and runnable baselines before
   writing the PyTorch adapter. Also seek author code for the 2019 Keras
   multi-label product-unit classifier, the 2024 PyTorch hybrid/complex study,
   and the 2026 PyTorch complex product-unit preprint. Those papers describe
   Python implementations, but no public release has yet been verified;
   the 2019 paper's other methods and the 2024 study's basic real-valued
   PUN were implemented in MATLAB. Their positive-input and complex models
   change the historical 2024 landscape setup, so evaluate them as separate
   controls if used. Inspect the public PyTorch `ComplexSigmaProductUnit` for
   its learned-exponent forward path, while keeping its custom zero and
   overflow rules out of the primary historical reproduction.
   KEEL is reference/validation code rather than a drop-in optimizer
   integration: it is Java/GPLv3, while Muon here is PyTorch. Consider
   RKEEL's `NNEP-C` wrapper as a way to run the KEEL classifier baseline.
   The similarly named `product-nets` recommender model is a different
   architecture. Check separately whether the 2024 landscape paper authors
   provide their exact model/data code; their data statement offers datasets on
   request. If that source is unavailable, build only the small classical PUNN
   layer and test its positive-input forward function against KEEL; validate
   the 2024 paper's signed-input extension and all gradients with scalar
   calculations and finite differences.
3. Add a separate PUNN train/plan path and registered `configs/product_unit/`
   configs. Existing `experiment.py` handles MLP/RNN explicitly, so YAML alone
   cannot launch a PUNN. Reuse W&B provenance, condition IDs, paired seeds,
   online logging, completeness checks, and Oscar plan expansion.

Implementation references: https://github.com/KellerJordan/Muon ;
https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html ;
https://github.com/thinking-machines-lab/manifolds/

Published PUNN reference implementation: https://github.com/SCI2SUGR/KEEL ;
official algorithm catalogue: https://sci2s.ugr.es/keel/algorithms.php ;
KEEL class reference:
https://sci2s.ugr.es/keel/javadoc/keel/Algorithms/Neural_Networks/NNEP_Regr/neuralnet/NeuralNetRegressor.html
The associated published regression and classifier studies are
https://sci2s.ugr.es/keel/pdf/keel/articulo/2006-NN-Evolutionary%20product%20unit%20based%20NN.pdf
and
https://www.sciencedirect.com/science/article/pii/S0925231207003803 ;
the authors' research-materials page provides its Diabetes and Australian
benchmark partitions: https://www.uco.edu.es/grupos/ayrna/materials/
RKEEL wrapper: https://rdrr.io/cran/RKEEL/man/NNEP-C.html

Newer Python implementation descriptions (source release unverified):
2019 Keras study: https://www.iccs-meeting.org/archive/iccs2019/papers/115370168.pdf ;
2024 PyTorch hybrid/complex study: https://arxiv.org/pdf/2305.04675 ;
2026 PyTorch complex preprint: https://arxiv.org/pdf/2605.27158

Related public PyTorch code: https://github.com/PureTearsDropped/ComplexSigmaProductUnit
implements a complex learned-exponent product in `torch.nn.Module`. It uses
special zero/overflow arithmetic and four-channel exponent parameters, so it
is an implementation reference, not the classical real-valued baseline for
the 2024 landscape experiment or a drop-in Manifold Muon parameterization.

## Stages

1. **Reproduction and numerical checks.** Start with the paper's shallow
   PUNN, linear output, published hidden sizes, `[-1,1]` initialization, 75/25
   split, 500 epochs, and 30 seeds. Reproduce its SGD setup (LR 0.1, momentum
   0.9) on a two-input regression benchmark such as the published surface
   function, then on one classification benchmark. Keep the paper's MSE metric.
   Keep this historical SGD run's output weights on SGD as in the paper. Use an
   established DE implementation as a small-model trainability control,
   clearly labeled if its settings differ from the paper. Verify signed/near-zero
   forward values, finite-difference gradients, and CPU one-step behavior.
   Record when historical SGD failure is not reproduced; do not count a Muon
   gain against an unverified failure as a rescue.
2. **Expressivity gate for Manifold Muon.** A Stiefel constraint on `V` is an
   architectural restriction, not merely an optimizer change. A one-input,
   one-hidden-unit `1x1` exponent is forced to `+scale` or `-scale`, so exclude
   such tasks from the primary Manifold Muon comparison. For each chosen
   multi-input task and prespecified scale, measure the best attainable fit
   under the constraint and confirm that a small sample can be fit. Include
   an AdamW control with the same projected initialization, plus projected-step
   AdamW and Riemannian SGD at the *same scale*, to isolate initialization,
   constraint, and update effects. Report any
   task/scale failing this gate as capacity-limited.
3. **Matched optimizer pilot.** Compare momentum SGD, AdamW,
   unconstrained Muon, and Manifold Muon on the same PUNN/data/order/seed pairs.
   Use the exponent matrix for Muon; keep the linear head/bias treatment
   identical. Run five paired seeds on one two-input regression and one
   multi-input classification task. Include Muon's SVD backend on a small case
   as a numerical reference for Newton-Schulz, without making it a separate
   scientific condition. Pilot only determines feasibility, ranges, and
   numeric failures. Muon orthogonalizes its *updates* and does not bound the
   exponent matrix or prevent an exponential forward overflow by itself.
4. **Equal-budget calibration and frozen evaluation.** Register calibration
   tasks/seeds separately from evaluation tasks/seeds. Give each compared
   optimizer the same 16 learning-rate trials at the calibration anchors;
   include manifold scale in a prespecified, explicitly budgeted secondary
   study. Freeze one recipe per optimizer before the evaluation sweep. Evaluate
   30 paired seeds on the paper's regression and classification tasks, with at
   least one harder high-input task and both small/oversized hidden widths.
   Evaluate the published wider exponent-bound case separately. Count failed,
   nonfinite, and incomplete runs in the denominator. Report optimization steps,
   model evaluations, and wall time because DE and gradient methods have
   different costs.
5. **Mechanism and decision.** Primary outcomes are held-out MSE and the
   fraction of seeds reaching a training-loss target fixed in the registered
   config before evaluation; also report classification accuracy. For the
   historical comparison, report paper-style train/test MSE and its spread.
   Also log gradient norm quantiles and coefficient of variation, update norm,
   gradient/update alignment, loss change after a step, exponent magnitude,
   product-unit output range, nonfinite events, flat/low-gradient steps, and
   Manifold Muon tangent/Stiefel residuals. On small problems, perform local
   loss-line probes at identical parameter points to test whether orthogonal
   updates cross or overshoot ravines. Use paired seed differences with
   bootstrap confidence intervals. Interpret a general rescue only if the
   historical failure reproduces and the paired improvement in held-out MSE
   and success rate is positive on both a regression and classification task,
   with 95% bootstrap intervals excluding no improvement and no higher
   numerical-failure rate. Report narrower task-specific gains as such.
   Distinguish an update
   benefit from a capacity or initialization effect using the projected controls.

## Project execution rules

Run unit tests and CPU smokes first, then reduced FP32 local GPU pilots. For
long sweeps, use durable Oscar array scripts, L40S, at most two concurrent
single-GPU jobs, dry-run and scheduler validation, and retained Slurm streams.
Keep the primary comparison at constant LR, FP32, zero weight decay, no AMP,
and no clipping. W&B online is the run store; scientific runs require a clean
committed revision and complete provenance. Expand every registered config
through `uv run --frozen --no-sync python -m optimizer_resurrection.experiment
plan`. Do not make a headline Muon/Manifold Muon claim before the existing
Gates A-C report `headline_ready=true`.
