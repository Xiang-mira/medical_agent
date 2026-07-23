# Scheduler Final Report

## Implemented

- New `scheduler/` package with config loading, preflight, manifest validation, resource tier selection helpers, SLURM rendering, state files, controlled task execution, planning, dry-run submission, retry, cancel, and CLI commands.
- Local and HPC scheduler config examples.
- Resource profiles for CPU, T4, A100, official 4xH100 LabelCritic, validated alternative LabelCritic, and smaller triage LabelCritic.
- `abdomenatlaspro_pilot338` DAG with parallel Teacher test branch and final metrics dependency after Teacher and Student test predictions.
- 373-target mapping bootstrap and 338-target pilot config/allowlist.
- Same-job LabelCritic vLLM SLURM template with `NO_PROXY`, `curl --noproxy "*"`, and `trap cleanup EXIT`.
- Unit tests for config/preflight, manifest safety, mapping validation, SLURM directives, LabelCritic script generation, array execution, GT environment scrubbing, and dry-run planning.

## Verified Locally

- `python -m pytest tests/test_scheduler_*.py` passed.
- `python -m scheduler.cli validate-config --config configs/scheduler.local.example.yaml` passed.
- `python -m scheduler.cli preflight --pipeline abdomenatlaspro_pilot338 --config configs/scheduler.local.example.yaml` passed with expected local warnings for missing HPC data roots/manifests.
- SLURM dry-run planning generated all DAG task scripts locally without calling `sbatch`.
- Generated array scripts call `scheduler.cli run-task`, which maps `SLURM_ARRAY_TASK_ID` to exactly one case manifest.
- Non-evaluation tasks scrub GT-like environment variables before launching child processes.

## Requires HPC Verification

- Presence and readability of AbdomenAtlasPro image/mask roots.
- Presence of fixed 20/20/10 pilot manifests.
- True per-case mask filename normalization and collision audit.
- T4 Teacher inference smoke.
- T4/A100 Student inference smoke.
- 1-case, 10-step Student train smoke.
- Official 72B 4xH100 vLLM health check and one LabelCritic comparison.
- Any validated alternative LabelCritic tier smoke.

## Important Caveat

`configs/abdomenatlaspro_target_mapping_373.json` is a bootstrap contract generated from the current 373 target config. It enforces that 338 targets participate in pilot338 and 35 are skipped, but the exact 338/35 membership must be tightened with the real AbdomenAtlasPro mask audit before publishing results as a fully audited mapping.

## Main Commands

```bash
python -m scheduler.cli validate-config --config configs/scheduler.local.example.yaml
python -m scheduler.cli preflight --pipeline abdomenatlaspro_pilot338 --config configs/scheduler.hpc.yaml --require-hpc-paths
python -m scheduler.cli plan --pipeline abdomenatlaspro_pilot338 --backend slurm --config configs/scheduler.hpc.yaml
python -m scheduler.cli submit --pipeline abdomenatlaspro_pilot338 --config configs/scheduler.hpc.yaml --dry-run
python -m scheduler.cli submit --pipeline abdomenatlaspro_pilot338 --config configs/scheduler.hpc.yaml
python -m scheduler.cli status --run-id RUN_ID
python -m scheduler.cli retry-failed --run-id RUN_ID
python -m scheduler.cli cancel --run-id RUN_ID
```
