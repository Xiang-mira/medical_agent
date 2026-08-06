# Dataset Delivery Config Inputs

Task 1 anatomical rename is maintained only under `configs/dataset_delivery/task1/`.
The files in this directory ending with `.example` are templates and are not formal
inputs.

- `task1/organ_rename_mapping_373.csv`: canonical Task 1 mapping.
- `task1/task_boundary_classification.csv`: row-level evidence and Task 1/Task 2 boundary.
- `task1/task1_alias_groups.csv`: explicit many-to-one alias declarations.
- `task1/task2_generate_targets_23.csv`: fixed Task 2 targets that must never execute as Task 1 rename.
- `task1/non_rename_decisions.csv`: documented coarse-to-fine or pending-review decisions.
- `task1/taxonomy_semantic_evidence.csv`: manually reviewed semantic relationship evidence for proposal audits.
- `organ_gap_resolution.csv.example`: Task 2 template only; Teacher inference reads confirmed generate rows there when populated.
- `cases_100_manifest.csv.example`: manifest schema example.

The canonical 373 target source remains `configs/student_3d_prompt_target_organs.json`.
The `main` branch is the only maintenance branch for Task 1.
The read-only proposal audit entrypoint is `tools/dataset_delivery/propose_task1_finalization.py`;
the HPC wrapper is `scripts/run_task1_proposal_audit_hpc.sh`.

## Task 2 CADS15 Smoke

CADS Task 2 delivery is now maintained as a 15-target canonical contract. The
historical completed-7/remaining-8 split remains only as a legacy audit record;
it is not a delivery success criterion.

The CADS15 contract source of truth is:

```bash
configs/cads15_target_contract.json
```

Static route audit:

```bash
python tools/dataset_delivery/cads15_contract_audit.py \
  --output-root reports/cads15_route_audit
```

Positive smoke panel generation:

```bash
python tools/dataset_delivery/cads15_smoke_panel.py \
  --case-manifest /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv \
  --output-root reports/cads15_smoke_panel
```

HPC smoke wrapper commands:

```bash
bash scripts/task2/submit_cads15_smoke.sh
bash scripts/task2/check_cads15_smoke.sh
```

The smoke does not launch the 100-case formal array. It selects a deterministic
minimal set of fixed-manifest cases in a CPU Slurm panel job, then starts the
GPU CADS smoke through an `afterok` dependency. Pending/running Slurm jobs are
reported as pending/running, not failed validation. A target passes only when the final
publication layer contains a non-empty, binary, CT-aligned mask with final status
`delivered` or `delivered_for_review` and no strict-delivery failure.

The formal 100-case CADS15 launcher is gated on a passed smoke and defaults to a
dry run:

```bash
DRY_RUN=1 bash scripts/task2/submit_cads15_formal_100cases.sh
```

More detail: `docs/CADS15_TASK2_SMOKE_DELIVERY.md`.
