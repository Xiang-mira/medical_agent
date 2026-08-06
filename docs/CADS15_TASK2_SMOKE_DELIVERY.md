# CADS15 Task 2 Smoke Delivery

CADS Task 2 delivery is now evaluated as 15 canonical targets, not as the
historical completed-7/remaining-8 bookkeeping split. The only code-level status
before real HPC inference is `READY_FOR_HPC_SMOKE`.

## Canonical Scope

The CADS15 targets are:

```text
blood
cerebrospinal_fluid
common_iliac_artery_left
common_iliac_artery_right
common_iliac_vein_left
common_iliac_vein_right
compact_bone
eyeball
face
gland_structure
gray_matter
muscle_of_head
scalp
spongy_bone
white_matter
```

The machine-readable contract is `configs/cads15_target_contract.json`. It
records the canonical target, primary CADS model, dataset id, source label,
source label id, checkpoint, output filename, FOV policy, delivery policy, and
Student training policy.

## Static Route Audit

Run this before submitting smoke jobs:

```bash
python tools/dataset_delivery/cads15_contract_audit.py \
  --output-root reports/cads15_route_audit \
  --checkpoint-root /projects/bodymaps/users/xhan74/medical_agent/models/checkpoints \
  --nnunet-predict-executable /home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict
```

The audit fail-fast checks registry routing, alias mapping, dataset.json label
existence and ids, plans, checkpoint files, predictor executability, left/right
iliac vessel identity, artery/vein identity, and compact/spongy bone identity.

## Delivery Policy

Task 2 Teacher delivery and Student training are separate decisions.

- A hard-valid Teacher mask is delivered when it is non-empty, binary, readable,
  CT-aligned, canonical-id valid, and in FOV.
- Weak evidence flags such as `many_connected_components`, single teacher,
  unavailable family consensus, or Student training weight zero produce
  `delivered_for_review`, not a hard reject.
- Student training may still set `training_weight=0` and
  `distillation_eligible=false`.
- Hard failures only include missing output, empty expected-present mask,
  unreadable NIfTI, geometry mismatch, illegal labels, canonical identity
  mismatch, wrong laterality/type, model failure, or out-of-FOV/unknown evidence.

Final run summaries include `final_delivery_status.json` and
`final_delivery_counts`. `total_updated` is rebuilt from final
`delivered + delivered_for_review` masks, not from intermediate candidates.

## Positive Smoke Panel

CADS15 smoke does not assume one case proves all 15 targets. The panel builder
reads the fixed 100-case manifest, uses read-only FOV/reference evidence only for
case qualification, and selects a deterministic minimal set of positive in-FOV
cases:

```bash
python tools/dataset_delivery/cads15_smoke_panel.py \
  --case-manifest /projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv \
  --output-root reports/cads15_smoke_panel
```

Outputs include:

- `cads15_smoke_case_panel.json`
- `cads15_smoke_case_panel_rows.csv`
- `cads15_selected_case_targets.csv`
- `cads15_smoke_case_panel.md`

If any target has no in-FOV positive smoke case, the status is
`BLOCKED_NO_POSITIVE_SMOKE_CASE`.

The production submit wrapper does not build this panel on the login node. It
generates a CPU Slurm panel-preparation job and a dependent GPU smoke job. Panel
state is recorded in `panel_state.json` and `panel_progress.json`; GPU state is
recorded in `gpu_state.json` and `gpu_progress.json`. The panel uses
`preflight/panel_context/<case_id>/presence_context.json` as a resumable cache,
and invalidates only the affected case cache when the CT or reference mask
fingerprint changes.

## HPC Commands

Submit CADS15 smoke only:

```bash
source "$HOME/.bodymaps_env"
cd /projects/bodymaps/users/xhan74/medical_agent/code/medical_agent
git pull --ff-only origin main
bash scripts/task2/submit_cads15_smoke.sh
```

Check the latest CADS15 smoke:

```bash
source "$HOME/.bodymaps_env"
cd /projects/bodymaps/users/xhan74/medical_agent/code/medical_agent
bash scripts/task2/check_cads15_smoke.sh
```

Both scripts use timestamped output roots and do not launch the 100-case formal
array. The submit script performs only lightweight repository and runtime checks
on the login node, writes the Slurm scripts, submits the CPU panel job, then
submits the GPU smoke with `afterok:<panel_job_id>`.

The check script is read-only. It reports the workflow state machine without
turning pending or running jobs into failed validation:

```text
NOT_SUBMITTED
PANEL_PENDING
PANEL_RUNNING
PANEL_FAILED
PANEL_COMPLETED
GPU_SMOKE_PENDING
GPU_SMOKE_RUNNING
GPU_SMOKE_FAILED
PASSED
VALIDATION_FAILED
```

The final validator runs only after the GPU smoke is terminally completed.

Expected output layout:

```text
cads15_teacher_smoke_<timestamp>/
├── preflight/
│   ├── route_audit/
│   ├── panel_context/
│   │   └── <case_id>/presence_context.json
│   ├── panel_progress.json
│   ├── panel_state.json
│   ├── cads15_smoke_case_panel.json
│   ├── cads15_smoke_case_panel_rows.csv
│   └── cads15_selected_case_targets.csv
├── cads15/
│   └── <case_id>/
│       ├── selected_case_manifest.csv
│       ├── command.txt
│       ├── git_commit.txt
│       └── run_loop/
├── slurm/
│   ├── cads15_panel_prepare.sbatch
│   └── cads15_gpu_smoke.sbatch
├── submission_manifest.json
├── cads15_smoke_jobs.csv
├── panel_job_id.txt
├── gpu_job_ids.txt
├── gpu_progress.json
├── gpu_state.json
├── task2_smoke_verdict.json
└── task2_smoke_verdict.md
```

The validator only reports `CADS15_SMOKE_STATUS=PASSED` after all 15 target
names have at least one positive in-FOV case with final status `delivered` or
`delivered_for_review`, valid non-empty NIfTI output, successful model inference,
and no strict-delivery failure.

## Formal 100-Case Launcher

The 100-case CADS15 launcher is present only as a gated, case-by-model launcher.
It does not run by default:

```bash
source "$HOME/.bodymaps_env"
cd /projects/bodymaps/users/xhan74/medical_agent/code/medical_agent
git pull --ff-only origin main
DRY_RUN=1 bash scripts/task2/submit_cads15_formal_100cases.sh
```

The launcher refuses real submission unless a CADS15 smoke root has a
machine-readable `PASSED` verdict. It creates one task per case and CADS model,
not one task per target, so each CADS family runs once per case and emits all of
its configured targets. To submit after a real smoke pass:

```bash
DRY_RUN=0 SMOKE_ROOT=/path/to/passed/cads15_teacher_smoke_<timestamp> \
  bash scripts/task2/submit_cads15_formal_100cases.sh
```

Resume controls are explicit:

```bash
RESUME=1 DRY_RUN=0 bash scripts/task2/submit_cads15_formal_100cases.sh
RETRY_FAILED=1 RESUME=1 DRY_RUN=0 bash scripts/task2/submit_cads15_formal_100cases.sh
CASE_ID=BDMAP_00000120 MODELS=cads557 DRY_RUN=0 bash scripts/task2/submit_cads15_formal_100cases.sh
```

Check formal outputs with:

```bash
bash scripts/task2/check_cads15_formal_100cases.sh
```

The formal validator writes `cads15_100case_status_matrix.csv` and
`cads15_100case_status_matrix.json`, with one row per fixed-manifest
case/target pair. Zero masks for expected-present targets are failures, and
`out_of_fov` or `confirmed_absent` rows do not substitute for positive smoke
evidence.
