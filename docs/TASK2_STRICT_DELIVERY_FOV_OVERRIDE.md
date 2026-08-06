# Task 2 Strict-delivery FOV Override

This workflow keeps the normal FOV gate enabled. It only adds an explicit
strict-delivery override for organs that were requested by name and are
classified by the existing FOV logic as `partially_visible`.

## Override Rules

`--strict-delivery-fov-override-organs` is valid only with
`--strict-delivery-targets`. The value is a comma-separated allowlist of formal
taxonomy organs.

An organ is restored after FOV pruning only when all conditions hold:

- strict delivery is enabled;
- the organ was explicitly listed in `--organs`;
- the organ is listed in `--strict-delivery-fov-override-organs`;
- the existing visibility function returns `partially_visible`;
- visibility is not `out_of_fov` or `unknown`;
- the requested model list contains a teacher that can route to that organ;
- the organ is present in the formal taxonomy.

The override does not restore neighboring organs in the same body region.
Partial visibility is recorded as partial coverage; it is not evidence that the
complete anatomical structure was covered.

## Audit Outputs

Each `case_execution_plan.json` records:

- requested override organs;
- initial FOV organ list;
- initially pruned requested organs;
- applied and rejected override organs;
- per-organ decision rows with visibility, reason, requested models, eligible
  teachers, and coverage evidence.

`run_summary.json` records:

- `strict_delivery_fov_override_enabled`;
- `strict_delivery_fov_override_organs_requested`;
- `strict_delivery_fov_override_organs`;
- `fov_override_applied`;
- applied and rejected counts;
- all override decision rows.

Opening or closing the override changes the hierarchical plan cache key, so a
previous "ATM not scheduled" plan cannot be reused as an override run.

## ATM Smoke

Run the single-case partial-thorax ATM strict-delivery smoke in an HPC
interactive allocation or batch step:

```bash
source "$HOME/.bodymaps_env"
export CODE_ROOT=/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent
export PYTHON=/home/xhan74/envs/medical_agent/bin/python
export CASE_MANIFEST=/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv
bash scripts/run_atm_strict_delivery_partial_thorax_smoke_hpc.sh
```

The wrapper uses `teacher-inference-mode=hierarchical_roi`, writes a timestamped
output directory, saves `command.txt`, `git_commit.txt`, `case_execution_plan.json`,
`run_summary.json`, `strict_delivery_failures.csv`, `inference_summary.json`,
`airway_tree_validation.json`, `smoke_verdict.json`, and `smoke_verdict.md`, and
returns `0` only when all smoke gates pass.

There is also a Slurm submit/verify pair when queue submission is preferred:

```bash
source "$HOME/.bodymaps_env"
export PYTHON=/home/xhan74/envs/medical_agent/bin/python
bash scripts/submit_atm_strict_delivery_smoke.sh
source /path/to/atm_smoke_submission.env
bash scripts/verify_atm_strict_delivery_smoke.sh
```

The verifier writes `ATM_STRICT_SMOKE_PASS.json` and
`ATM_STRICT_SMOKE_PASS.md`. The pass gate requires Slurm `COMPLETED`, exit code
`0:0`, `run_summary.status=success`, zero strict-delivery failures,
`total_updated=1`, `atm` in `teacher_run_list`, `airway_tree` in applied
override organs, successful ATM inference, and a non-empty binary
`airway_tree.nii.gz` with CT-matching shape, spacing, and affine.

## 100-case Preflight

Run the read-only coverage preflight:

```bash
source "$HOME/.bodymaps_env"
export CODE_ROOT=/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent
export PYTHON=/home/xhan74/envs/medical_agent/bin/python
export CASE_MANIFEST=/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv
export DATA_ROOT=/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro
export OUT_ROOT=/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/task2_100case_preflight_$(date +%Y%m%d_%H%M%S)

cd "$CODE_ROOT"
git pull --ff-only origin main
mkdir -p "$OUT_ROOT"
"$PYTHON" tools/dataset_delivery/audit_task2_100case_launch.py \
  --case-manifest "$CASE_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --taxonomy configs/student_3d_prompt_target_organs.json \
  --output-root "$OUT_ROOT" \
  --strict-delivery-fov-override-organs airway_tree
```

Outputs:

- `task2_100case_preflight_rows.csv`;
- `task2_100case_preflight_summary.json`;
- `task2_100case_preflight_report.md`;
- `eligible_case_manifests/{cads,atm,airrc,unest}_eligible_cases.csv`.

ATM cases are eligible when fully visible, or partially visible with the
explicit `airway_tree` override. ATM `out_of_fov` and `unknown` cases are not
submitted.

## Formal 100-case Launcher

The launcher refuses to submit arrays unless all smoke/readiness pass JSON files
are supplied and passing:

```bash
source "$HOME/.bodymaps_env"
export CODE_ROOT=/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent
export PYTHON=/home/xhan74/envs/medical_agent/bin/python
export CASE_MANIFEST=/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv
export DATA_ROOT=/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro
export ATM_SMOKE_PASS_JSON=/path/to/ATM_STRICT_SMOKE_PASS.json
export AIRRC_SMOKE_PASS_JSON=/path/to/AIRRC_STRICT_SMOKE_PASS.json
export UNEST_SMOKE_PASS_JSON=/path/to/UNEST_STRICT_SMOKE_PASS.json
export CADS_SMOKE_PASS_JSON=/path/to/CADS_SMOKE_PASS.json
bash scripts/submit_task2_100case_model_groups.sh
```

To resume only specific failed cases:

```bash
export RESUME_CASE_IDS=BDMAP_00000120,BDMAP_00000121
bash scripts/submit_task2_100case_model_groups.sh
```

Output layout:

```text
formal_task2_22targets_100cases_<timestamp>/
├── preflight/
├── readiness/
├── cads/
├── atm/
├── airrc/
├── unest/
├── slurm/
├── registry.json
├── launch_manifest.csv
└── launch_summary.json
```

The launcher submits CADS, ATM, AirRC, and UNEST arrays only. It does not submit
the blocked `brain_ventricle` target and does not submit TotalSegmentator.
