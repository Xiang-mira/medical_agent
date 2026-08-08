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
- `task2_append_cases.csv`: append-only Task 2 FOV-extension cases. The current fixed working cohort is base 100 plus `BDMAP_00000424` at index 100 and `BDMAP_00078156` at index 101.
- `task2_qualification_registry.json`: current model/target qualification status from real smoke evidence and known blockers.

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

FOV-compatible smoke panel generation:

```bash
python tools/dataset_delivery/cads15_smoke_panel.py \
  --case-manifest /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_102_manifest.csv \
  --output-root reports/cads15_smoke_panel
```

Canonical target reference masks are selection evidence only. Missing canonical
GT is not a smoke-panel failure for Task 2 generation targets. Current
target-specific FOV evidence is authoritative; historical Teacher outputs are
selection evidence only and are rejected when current FOV is out of scan.

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

## Task 2 Working Cohort

Do not reselect, reorder, or delete the original fixed 100 cases. Build the
working manifest deterministically from the fixed base manifest plus configured
append cases:

```bash
python tools/dataset_delivery/task2_working_cohort.py \
  --base-manifest /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv \
  --append-cases configs/dataset_delivery/task2_append_cases.csv \
  --output-manifest /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_102_manifest.csv \
  --audit-json /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_102_manifest.audit.json \
  --audit-csv /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_102_manifest.audit.csv
```

Manifest membership is not target eligibility. `BDMAP_00000424` and
`BDMAP_00078156` are thorax/pulmonary candidates, not head anchors and not
central-airway anchors.

## Task 2 FOV Search And Append

CPU-only FOV candidate search:

```bash
bash scripts/task2/submit_fov_candidate_search.sh \
  --python-executable /home/xhan74/envs/medical_agent/bin/python \
  --image-root /projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro \
  --mask-root /projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro \
  --output-root /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/fov_candidate_search_$(date +%Y%m%d_%H%M%S) \
  --progress-every 50
```

The launcher runs a login-node dependency preflight before `sbatch`, writes the
resolved Python into the generated Slurm script, and repeats the nibabel/numpy
preflight on the compute node before scanning any case.

Append newly qualified cases without disturbing existing indices:

```bash
python tools/dataset_delivery/task2_append_qualified_cases.py \
  --base-manifest /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_102_manifest.csv \
  --candidate-report /path/to/fov_candidate_search/head_candidates.csv \
  --target-group head \
  --max-cases 1 \
  --output-manifest /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_103_manifest.csv \
  --audit-json /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_103_manifest.audit.json
```

## TotalSegmentator Brain Ventricle Offline

Task 2 canonical `brain_ventricle` is produced by TotalSegmentator
`brain_structures`, task id `409`, source output `ventricle`, source label id
`10`. This task is license-gated. Do not commit the license value or model
weights to normal Git.

Prepare assets once on a licensed machine:

```bash
export MEDAI_TOTALSEG_HOME=/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/totalsegmentator/.totalsegmentator
export TOTALSEG_HOME_DIR="$MEDAI_TOTALSEG_HOME"
mkdir -p "$MEDAI_TOTALSEG_HOME"
totalseg_set_license -l "$TOTALSEG_LICENSE"
totalseg_download_weights -t brain_structures
python tools/dataset_delivery/totalseg_brain_ventricle_offline.py \
  --home "$MEDAI_TOTALSEG_HOME" \
  --manifest /projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/totalsegmentator/brain_ventricle/totalsegmentator_brain_ventricle_offline_manifest.json
```

Verify before GPU smoke:

```bash
export MEDAI_TOTALSEG_OFFLINE=1
export MEDAI_TOTALSEG_HOME=/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/totalsegmentator/.totalsegmentator
python tools/dataset_delivery/totalseg_brain_ventricle_offline.py \
  --home "$MEDAI_TOTALSEG_HOME" \
  --manifest /projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/totalsegmentator/brain_ventricle/totalsegmentator_brain_ventricle_offline_manifest.json \
  --verify
```

If task409 or its crop dependency is absent, production fails fast with
`TOTALSEG_OFFLINE_ASSET_MISSING`. If the license config is absent, it fails with
`TOTALSEG_LICENSE_MISSING`. In offline mode the runner sets
`TOTALSEG_HOME_DIR=$MEDAI_TOTALSEG_HOME` and does not rely on implicit
`~/.totalsegmentator`.
