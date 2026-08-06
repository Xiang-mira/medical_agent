# Task 2 Teacher Runtime Repair

This document records the permanent runtime contract for Task 2 Teacher smoke
runs on JHU HPC. It does not start or certify the 100-case formal array.

## Canonical Roots

- Code root: `/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent`
- Checkpoint root: `/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints`
- Outer run-loop Python: `/home/xhan74/envs/medical_agent/bin/python`
- UNEST inner model Python: `/home/xhan74/envs/medical_agent_train_py311/bin/python`
- nnUNet predictor: `/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict`

Registry checkpoint paths such as `checkpoints/CADS_series/...` are resolved
against the explicit checkpoint root, not against the current working directory
or a copied registry inside an output folder.

## Runtime Parameters

`run-loop` accepts explicit runtime paths:

```bash
--checkpoint-root /projects/bodymaps/users/xhan74/medical_agent/models/checkpoints
--nnunet-predict-executable /home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict
--unest-python-executable /home/xhan74/envs/medical_agent_train_py311/bin/python
```

The values are recorded in `case_execution_plan.json`, `run_summary.json`, and
the hierarchical ROI cache key. Changing them invalidates stale hierarchical
plans.

## Preflight

Run the read-only preflight before submitting smoke jobs:

```bash
python tools/dataset_delivery/task2_preflight.py \
  --models atm,cads553,cads557,cads559,unest \
  --case-list "$CASE_MANIFEST" \
  --output-root "$OUT_ROOT/preflight_only" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --python-executable "$PYTHON" \
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE" \
  --unest-python-executable "$UNEST_PYTHON_EXECUTABLE" \
  --json-output "$OUT_ROOT/task2_preflight.json" \
  --formal-mode
```

The only top-level statuses are `READY` and `BLOCKED`. A blocked preflight must
stop submission before `sbatch`.

## Smoke Submission

Use the versioned wrapper; it generates one `.sbatch` file per model group and
does not use `sbatch --wrap`.

```bash
bash scripts/task2/submit_teacher_smokes.sh
```

Default groups are `atm,cads,unest`. Override with `GROUPS=cads` for a CADS-only
smoke. Each job writes command, git commit, preflight report, runtime manifest,
task markers, and run-loop artifacts under:

```text
task2_teacher_smokes_<timestamp>/
├── atm/
├── cads/
├── unest/
├── slurm/
├── prepare_summary.json
└── smoke_jobs.csv
```

The UNEST job checks CUDA on the GPU compute node before invoking MONAI bundle.
UNEST does not inherit nnUNet diagnostic arguments or predictor environment.

## Smoke Validation

After Slurm jobs finish:

```bash
SMOKE_ROOT=/path/to/task2_teacher_smokes_<timestamp> \
bash scripts/task2/check_teacher_smokes.sh
```

The validator treats `RUNNING`, `PENDING`, unknown Slurm state, non-zero Slurm
exit code, missing `run_summary.json`, strict-delivery failures, missing teacher
runs, and missing final masks as failures.

Masks are accepted only from the formal delivery layer:

```text
run_loop/annotation_versions/<case_id>/updated/<target>.nii.gz
```

Raw predictions, temporary exports, copied registries, and symlinks are not
accepted as smoke success.
