# Scheduler Implementation Plan

## Implementation Shape

- Add a lightweight `scheduler` Python package that wraps existing project entrypoints rather than reimplementing model algorithms.
- Add YAML configs for local/HPC paths, resource profiles, and the `abdomenatlaspro_pilot338` DAG.
- Add target mapping and pilot338 target configs so full 373 and pilot338 are distinct, auditable contracts.
- Add SLURM rendering with CPU/GPU separation, job arrays, dependency-compatible scripts, same-job vLLM LabelCritic template, and controlled `run-task` execution.
- Add local tests for config validation, target/mapping validation, GT isolation, SLURM safety, array execution, and dry-run planning.

## Resource Strategy

- Teacher inference: prefer T4 if single-case smoke succeeds; fallback A100.
- Student inference: prefer T4 if single-case smoke succeeds; fallback A100.
- Student training: single GPU only; GPU type selected by smoke; no DDP by default.
- LabelCritic: official tier is 72B on 4xH100 with vLLM TP=4; validated alternatives and smaller triage tiers are allowed only with explicit tier labeling.

## Safety Strategy

- Preflight checks 373 target config, pilot338 config, target mapping, split manifests, and GT leakage.
- Strict manifests are rejected if they contain GT-like columns or `mask_only` / `segmentations` values.
- Original data roots are never used as output roots.
- Dry-run submit generates scripts and `jobs.json` but does not call `sbatch`.

## Rollout

1. Run local `validate-config`, `preflight`, `plan`, and tests.
2. Sync code/configs to HPC.
3. Fill any cluster-specific account/qos/constraint/walltime values.
4. Run HPC preflight with `--require-hpc-paths`.
5. Generate and inspect smoke scripts.
6. Run smoke jobs for T4/A100/H100 tiers.
7. Select resources from smoke results.
8. Submit the full pipeline.
