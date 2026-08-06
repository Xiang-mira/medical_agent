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
array. The submit script creates one Slurm job per selected smoke-panel case.

Expected output layout:

```text
cads15_teacher_smoke_<timestamp>/
├── preflight/
│   ├── route_audit/
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
├── cads15_sbatch_manifest.csv
├── cads15_smoke_jobs.csv
├── slurm_status.csv
├── task2_smoke_verdict.json
└── task2_smoke_verdict.md
```

The validator only reports `CADS15_SMOKE_STATUS=PASSED` after all 15 target
names have at least one positive in-FOV case with final status `delivered` or
`delivered_for_review`, valid non-empty NIfTI output, successful model inference,
and no strict-delivery failure.
