# Repository Handoff

Last reviewed: 2026-09-28

This repository is the GitHub handoff point for the MedAI Agent Loop: a
registry-driven system for multi-teacher pseudo-label construction, 373-target
abdomen/CT label governance, LabelCritic audit support, and VoxTell-style
student training.

## What This Project Is

- **Goal:** build reliable 3D medical image pseudo-labels by routing CT cases
  through multiple teacher models, selecting only labels that pass strict
  anatomy/QC/evidence gates, and training a prompt-conditioned student model.
- **Formal target space:** 373 exact organ/structure prompts in
  `configs/student_3d_prompt_target_organs.json`.
- **Mainline workflow:** hierarchical teacher E-step, geometric consensus,
  audit-only LabelCritic support unless calibrated gates pass, project
  prompt-distillation M-step initialized from official VoxTell assets, then
  student evaluation and optional next EM round.
- **Current reporting rule:** do not describe the M-step as official VoxTell
  fine-tuning. It is project training initialized from official VoxTell assets.

## Current Git State

- Canonical remote: `git@github.com:Xiang-mira/medical_agent.git`
- Canonical branch for handoff: `main`
- Code/data/docs are arranged for GitHub. Large checkpoints, CT volumes,
  runtime outputs, logs, temporary caches, and private patient metadata are
  intentionally excluded by `.gitignore`.
- Useful historical branches with unique commits should be kept until their
  contents are reviewed. Branches with zero unique commits relative to `main`
  can be safely deleted locally or remotely after confirming no open PR depends
  on them.

## High-Level Repository Map

| Path | Handoff meaning |
| --- | --- |
| `README.md` | primary entry point, formal contract, setup, HPC migration, and EM workflow |
| `agent-harness/` | installable Python package and core runtime modules |
| `scheduler/` | resource-aware local/Slurm scheduling layer |
| `scripts/` | execution, audit, repair, migration, and dataset-delivery commands |
| `tools/dataset_delivery/` | Task 1/Task 2 dataset packaging, validation, and delivery utilities |
| `configs/` | model registry, taxonomy, target space, routing, scoring, and scheduler configs |
| `data/` | small demo assets only |
| `data_manifest/` | tracked CSV case manifests and templates |
| `generated_labels_100cases/` | final overlay destination and manifest placeholders for validated generated labels |
| `docs/` | architecture, operations, migration, status, and source-material notes |
| `reports/` | checked-in audit/status reports that summarize past runs |
| `wrappers/` | compatibility wrappers for model-specific integrations |
| `third_party/` | vendored or lightweight external project code and docs |
| `checkpoints/`, `outputs/`, `logs/`, `.runtime/` | local-only runtime assets; do not commit |

## Data and Asset Policy

Keep GitHub focused on reproducible code, configs, manifests, and small audit
artifacts.

| Asset type | Where it belongs |
| --- | --- |
| Source code, configs, tests, docs | GitHub repository |
| Small demo metadata and templates | `data/`, `data_manifest/`, `configs/` |
| CT volumes, NIfTI masks, PanTS/PAINTS data | local/HPC storage, not Git |
| Teacher/student model weights | Hugging Face or institutional checkpoint storage, not Git |
| Formal migration checkpoint bundle | `Xiang-mira/MedIA-Agentic-AI-Private-HPC` or current private asset store |
| Smoke outputs, logs, Slurm files, temporary caches | local-only runtime directories |
| PHI or institution-specific metadata | protected institutional storage only |

Before handing a machine to a new user, verify that `git status --short` is
clean and that no excluded private data has been forced into Git.

## New Maintainer Quick Start

```bash
git clone https://github.com/Xiang-mira/medical_agent.git
cd medical_agent

python -m venv .venv
source .venv/bin/activate
pip install -e agent-harness

python run_medai_cli.py --json doctor
python run_medai_cli.py --json model-inventory
PYTHONPATH=agent-harness pytest -q agent-harness/tests
```

For real inference or training, install GPU/runtime dependencies from
`agent-harness/requirements-real-inference.txt`, restore checkpoints according
to `README.md`, and update `data_manifest/*.csv` paths for the target machine.

## Core Files to Read First

1. `README.md` for formal guardrails and execution flow.
2. `docs/ARCHITECTURE_multimodel_routing.md` for routing architecture.
3. `docs/HIERARCHICAL_ROI_AND_ORGAN_IDENTITY.md` for parent/child organ rules.
4. `docs/HUGGINGFACE_MODEL_RELEASE.md` and `docs/migration/HPC_MIGRATION_AUDIT.md`
   for asset restoration.
5. `configs/README.md` for registry, taxonomy, and target-space files.
6. `scripts/README.md` for command categories and primary entry points.
7. `generated_labels_100cases/README.md` for final generated-label overlay rules.

## Formal Workflow Checklist

1. Confirm local environment and GPU resources:
   `python run_medai_cli.py --json doctor` and
   `PYTHONPATH=agent-harness python scripts/check_gpu_resources.py`.
2. Restore checkpoints and model assets to paths expected by
   `configs/model_registry.yaml`.
3. Rewrite `data_manifest/*.csv` for the target storage paths and validate with
   `python scripts/validate_case_list_50.py --case-list <manifest.csv>`.
4. Audit taxonomy and target identity:
   `PYTHONPATH=agent-harness python scripts/audit_organ_identity.py`.
5. Run the formal EM entry point through `scripts/run_em_training.py` or the
   scheduler configs in `configs/scheduler.*.example.yaml`.
6. Treat smoke/debug outputs as plumbing only. Only formal manifests that pass
   the documented gates should be used for claims, training, or delivery.

## Branch Handoff Rule

Use `main` for the handoff state. Keep branches only when they have unique
commits not present on `main`, an open PR, or a specific experiment owner.
Branches that are fully merged into `main` should be deleted to reduce
confusion.

Local branch cleanup performed during this handoff:

| Branch category | Branches |
| --- | --- |
| Deleted locally because they were fully merged into `main` | `codex/abdomenatlaspro-smart-resource-launcher`, `codex/adaptive-scheduler-push`, `codex/cads15-hpc-orchestration`, `codex/cads15-smoke-delivery`, `codex/complete-abdomenatlaspro-373-clean`, `codex/dataset-delivery-373`, `codex/fix-cads15-airrc-hpc-routing`, `codex/task1-rename-standalone-refactor`, `codex/task2-teacher-smoke-runtime-state` |
| Kept locally because they still have unique commits | `backup/local-main-diverged`, `codex/pre-push-backup-20260624`, `consolidate-orchestration-entry` |

Remote branch deletion was intentionally not performed in this pass; confirm
open pull requests or collaborator ownership before deleting remote refs.

Suggested checks:

```bash
git fetch --prune origin
git branch --merged main
git branch --no-merged main
git for-each-ref --format='%(refname:short) %(upstream:short) %(committerdate:short) %(subject)' refs/heads refs/remotes/origin
```

## Safety Notes

- Do not use `mock_seg`, LabelCritic `stub`, smoke overrides, dry-run outputs,
  or project projection fallback as scientific evidence.
- Do not let evidence families vote, choose winners, change training weight, or
  enter VLM prompts.
- Do not replace child-organ masks with parent masks. Parent masks are ROI
  supports only.
- Do not publish private checkpoint paths, PHI-bearing metadata, or institutional
  data snapshots in GitHub issues, README examples, or docs.
