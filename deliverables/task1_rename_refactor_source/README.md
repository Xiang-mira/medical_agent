# Task 1 Rename Standalone Export

This directory is generated from canonical source in `configs/dataset_delivery/task1/` and `tools/dataset_delivery/`.
Do not hand-edit files here; regenerate with `tools/dataset_delivery/export_task1_standalone.py`.

Task 1 only renames existing masks when anatomy, laterality, and granularity are identical.
Task 2 targets are listed separately and are never executed as Task 1 renames.

Validate this export from the repository root:

```bash
python tools/dataset_delivery/validate_rename_mapping.py \
  --mapping configs/dataset_delivery/task1/organ_rename_mapping_373.csv \
  --taxonomy configs/student_3d_prompt_target_organs.json \
  --task2-targets configs/dataset_delivery/task1/task2_generate_targets_23.csv \
  --alias-groups configs/dataset_delivery/task1/task1_alias_groups.csv
```
