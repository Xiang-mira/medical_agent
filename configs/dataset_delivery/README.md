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

## Task 2 CADS Remaining-target Smoke

CADS Task 2 status is recomputed from the canonical 23 generate-target CSV,
the 373 taxonomy, the model registry, the class checkpoint map, and existing
CADS reports. The read-only audit entrypoint is:

```bash
python tools/dataset_delivery/audit_cads_remaining_targets.py audit \
  --output-root reports/cads_remaining8_status
```

The current canonical CADS scope is 15 targets. The historical completed set is
7 targets, and the remaining strict-delivery smoke scope is the canonical
difference:

```text
blood
common_iliac_artery_left
common_iliac_artery_right
common_iliac_vein_left
common_iliac_vein_right
compact_bone
gland_structure
spongy_bone
```

The older candidate names `brain`, `trachea`, `brainstem`, `oral_cavity`, and
`larynx` are not in the current Task 2 23-target generate list and must not be
reintroduced by the CADS smoke workflow.

The single-case HPC smoke wrappers are:

```bash
bash scripts/submit_cads_remaining8_strict_delivery_smoke.sh
bash scripts/verify_cads_remaining8_strict_delivery_smoke.sh
```

The smoke runs CADS only, enables strict delivery, writes to a timestamped output
directory, records the selected case and complete command, and validates each
requested target NIfTI for existence, non-empty foreground, binary labels, and
CT shape/spacing/affine agreement. It does not launch the 100-case formal array,
does not modify original NIfTI files, and does not overwrite completed CADS
outputs.
