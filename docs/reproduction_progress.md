# Historical reproduction progress

## Authorized scope and stopping point

Implement the shared historical protocol infrastructure and the 1994 Section 5.5
Latch benchmark, validate, and submit the initial 12 Oscar jobs. Stop after
scheduler acceptance and this handoff; do not wait for scientific completion.
MLP, Two-Sequence, Parity, Muon comparisons, and later budget stages are deferred.

## Fidelity record

Source: Bengio, Simard, and Frasconi (1994), Section 5.5:
https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Bengio_94.pdf
(author preprint; DOI https://doi.org/10.1109/72.279181).

- Documented: one recurrent tanh unit; three adaptive parameters (recurrent
  weight and positive/negative initial input); parameter initialization uniform
  in [-0.5, 0.5]; fixed generated samples per trial; sequence lengths uniform
  between T/2 and T; no input noise or uniform noise in [-0.2, 0.2].
- Recurrence: x[0]=0; x[t]=tanh(w*x[t-1]+input[t]); the first input is the
  trainable value for the sequence class. The remaining inputs are fixed noise.
  Final squared error uses targets -0.8/+0.8; classification uses output sign.
- Reconstruction assumptions: 100 balanced training sequences and 1,000 balanced
  validation sequences; independent validation stream; fixed shuffled cyclic
  sample order; one sequence per plain SGD update; constant LR, FP32, no
  momentum, clipping, decay, or early termination. Integer lengths include both
  ceil(T/2) and T. These details are explicit rather than claims of exact replay.
- Use independent random streams for labels/order, lengths, and noise so noise
  conditions share labels and lengths. Model and data seeds are recorded.
- Do not mix this benchmark with Section 3's Gaussian-noise experiment, fixed
  recurrent weight, or multiple trainable early inputs.
- No verified historical plain-SGD learning rate was recovered. Register a
  short-task scout: T=10, seed/data_seed=1000, both noise settings, LR in
  {0.001, 0.01, 0.1}, each 5,000 presentations. Select one LR by mean validation
  loss across noise settings; ties choose the smaller LR. Require the selected
  recipe to learn both easy conditions before releasing the main array.
  Operational easy-control thresholds are train and validation accuracy >=0.95
  and both mean half-squared losses <=0.10. These are new screening thresholds,
  not reported historical thresholds; sign accuracy alone is insufficient when
  outputs are almost zero.

## Run plan

- Scout: 6 runs, separate from evaluation seeds.
- Evaluation: lengths 10/100 x noise 0/0.2 x seeds 0/1/2 = 12 runs.
- Each evaluation run: exactly 5,000 sequence presentations (batch size one).
- Diagnostics: whole-dataset train/validation loss and accuracy; full-span
  gradients back to the initial input; float64 diagnostic references distinguish
  FP32 underflow from small nonzero gradients. Diagnostic work is counted
  separately and must not advance the training random stream.
- Scientific records, metrics, and artifacts live only in online W&B. This
  document stores protocol decisions, provenance, status, and commands.

## Current progress

- [x] Audited original protocol against current code; current generic Latch is
  retained as a different, earlier experiment.
- [x] Verified Oscar campus SSH using the existing RSA key; remote checkout was
  clean at dfe0166 and no jobs were present at initial inspection.
- [x] Implement and independently test the historical protocol.
- [x] Review correctness, planner identity, and launch mapping. Fixed initial
  snapshot test and diagnostic accounting identified by the independent review.
- [x] CPU and online local GPU smoke validation. Full suite: 61 tests passed.
  Smoke histories contain steps 0,1,5,10, full-span gradients, dataset digests,
  terminal outcome, source digest, git metadata, and hardware metadata. Verified
  CPU run xpjf4qa5 and RTX 4060 Ti run drnak59m, then deleted both disposable
  W&B groups and confirmed zero remaining records with fresh API queries.
  Initial attempts pyn1usks/j5xychih lacked git revision because of Windows Git
  ownership checking; discarded and rerun using process-local safe.directory.
- [ ] Commit and synchronize clean code; recreate pinned Oscar environment.
- [ ] Validate and clean disposable Oscar smoke.
- [ ] Finish LR scout and freeze selected recipe.
- [ ] Submit 12-run evaluation array and record scheduler acceptance.

## Handoff (populate before stopping)

Launch revision, config paths, W&B groups, Slurm IDs, validation evidence, exact
inspection commands, and remaining actions will be recorded here as work proceeds.

Next agent: inspect every expected Latch condition in W&B and the retained Slurm
streams. Separate interrupted jobs from completed training that missed its
success criterion. Do not claim historical reproduction solely from a finished
run or an accuracy difference. Review the trajectories and easy-task controls
before proposing the 20,000/100,000-presentation stages. Existing Gates A-C and
headline readiness remain independent and unchanged.
