# Personal Multi-Agent Workflow

## Agent routing

- Keep the primary agent as the orchestrator, decision-maker, and final reviewer. The primary agent owns requirements, decomposition, integration, verification, and the user-facing response.
- Treat ambiguous, multi-step, cross-cutting, architectural, security-sensitive, research-heavy, or difficult debugging work as complex. Keep that work on the primary agent or delegate an independent complex subtask to `complex_worker`.
- Delegate clear, narrow, low-ambiguity, independently verifiable work to `simple_worker`. Good candidates include targeted exploration, mechanical edits, focused test additions, formatting, summarization, and repetitive checks.
- Use `reviewer` for an independent correctness review when a change is risky, complex, or explicitly requested for review.
- Do not give `simple_worker` architectural decisions, unresolved requirements, security-critical judgment, final review ownership, or broad multi-file tasks.
- Do not delegate work whose handoff and integration cost exceeds doing it directly.

## Coordination

- Give each subagent one bounded objective, explicit scope, relevant constraints, and a concrete return format.
- Parallelize only independent work. Avoid overlapping write scopes; prefer parallel reads and checks.
- Wait for delegated work that the result depends on. Inspect the evidence and diff yourself rather than accepting a subagent conclusion uncritically.
- The primary agent makes final decisions, resolves conflicts between agents, performs or confirms appropriate verification, and produces the final response.

## Proactive multi-agent delegation

- The user explicitly requests proactive multi-agent delegation for all tasks governed by this file.
- For every non-trivial task containing at least one independent, bounded workstream, the primary agent must spawn an appropriate subagent.
- Delegate narrow exploration, mechanical implementation, focused tests, and repetitive checks to `simple_worker`.
- Use `complex_worker` for an independent complex workstream and `reviewer` for risky changes.
- The primary agent retains requirements, integration, verification, and final-response ownership.
- A task may remain entirely on the primary agent only when it is short or cannot be divided into independent work. Briefly state that reason.

## Experiment workflow

1. Treat `configs/` as the pre-registered experiment record. Expand and run configs through `uv run --frozen --no-sync python -m optimizer_resurrection.experiment plan`; never silently change a frozen evaluation recipe or bypass its recorded overrides. Python is pinned by `.python-version`, dependencies by `uv.lock`, and the required UV version by `pyproject.toml`. On Oscar, create a fresh `.venv` with the pinned interpreter and `uv sync --frozen`; batch jobs must use `uv run --frozen --no-sync` so concurrent jobs cannot mutate the environment.
2. Run unit tests and CPU smoke tests before using a GPU. Run reduced experiments locally on the RTX 4060 Ti 16 GB; use FP32 and lower batch size before changing scientific hyperparameters if memory is tight.
3. Use Oscar for long sweeps. After pulling the clean revision, run `bash scripts/oscar_setup_uv.sh` to recreate `.venv` from `uv.lock`. Keep each durable job in `scripts/*.sbatch`, use direct array-task-to-condition/seed mapping, semantic W&B names, and retain ignored per-task `.slurm-*.out`/`.slurm-*.err` scheduler logs in the Oscar project root for failure recovery. Test with `DRY_RUN=1` plus `bash -n`, then run `sbatch --test-only` after connecting to Oscar and before submission. Request L40S explicitly and cap all GPU arrays at two simultaneous one-GPU tasks. Submit disposable smoke tasks separately; verify and delete their W&B group before releasing exploratory jobs.
4. Assume this device is connected to Brown's secure campus Wi-Fi. Connect directly to `sshcampus.ccv.brown.edu`; no VPN is required for this campus endpoint. Prefer non-interactive SSH-key authentication. Never store or use a password or Duo token. If key authentication fails and Oscar requires an interactive Duo login, ask Luke to start the authenticated session, then continue with sync, validation, and submission.
5. Load only the allowlisted W&B and Oscar credentials from `.secrets/env`, and only when the corresponding environment variable is unset. Never print secret values, commit `.secrets/`, or place credentials in Slurm logs.
6. Pair optimizer conditions by initialization seed, dataset seed, and minibatch order. Use equal tuning budgets. Tune only at the calibration depths/lengths, freeze the selected recipes, then evaluate the boundary sweep without retuning.
7. Do not report headline Muon/MM results until Gates A–C in the study spec pass. W&B is the authoritative run store and must operate online. W&B-managed local logs and cache are allowed, but project code must not create custom config, metric, result, checkpoint, or `outputs/` artifacts. Do not use W&B offline mode. Report failed jobs and excluded seeds explicitly.
8. Use constant learning rate, FP32, no AMP, no clipping, and zero weight decay for primary compared matrices unless a config is explicitly marked secondary.
9. After a run, preserve the exact sanitized command, resolved config, model and data seeds, source-tree digest, git revision and dirty state, dependency/environment metadata, Slurm identifiers, W&B run ID, and terminal outcome in the W&B run. Permit dirty or uncommitted source only for runs explicitly marked `engineering-smoke` or `test`; all scientific stages must fail before training without a clean committed revision. Analysis must query completed W&B records and reject duplicate condition IDs or incomplete cells.
10. Run `experiment select-recipes` after the equal-budget calibration and pass the resulting recipe file when expanding `full_study.yaml`. Run `experiment check-gates` and require `headline_ready=true` before any headline sweep or claim. Treat `oracle.yaml` as a separately labelled per-task tuning study.
