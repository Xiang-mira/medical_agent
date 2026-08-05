# Task 1 Rename Workflow

Task 1 is only for existing mask files whose source and target names refer to the same anatomy, same laterality, and same granularity. It performs no voxel edits, no merges, no splits, and no generated masks. Task 2 covers missing masks, coarse-to-fine targets, vessel segment refinement, tissue subclass generation, and all fixed 23 generate targets in `task2_generate_targets_23.csv`.

Canonical source is `configs/dataset_delivery/task1/organ_rename_mapping_373.csv`; boundary evidence is `task_boundary_classification.csv`; many-to-one aliases are explicit in `task1_alias_groups.csv`. The `main` branch is the only maintenance branch for this workflow. The standalone directory under `deliverables/task1_rename_refactor_source/` is generated with `tools/dataset_delivery/export_task1_standalone.py`; do not hand-maintain a second mapping.

Static validation:

```bash
python tools/dataset_delivery/validate_rename_mapping.py \
  --mapping configs/dataset_delivery/task1/organ_rename_mapping_373.csv \
  --taxonomy configs/student_3d_prompt_target_organs.json \
  --task2-targets configs/dataset_delivery/task1/task2_generate_targets_23.csv \
  --alias-groups configs/dataset_delivery/task1/task1_alias_groups.csv \
  --output-json reports/task1_mapping_validation.json \
  --output-md reports/task1_mapping_validation.md
```

HPC 100-case dry-run is read-only by default and writes only reports under `--output-root`. `would_rename` means exactly one confirmed source exists and target is absent. `already_normalized` means target exists and source is absent. `source_missing` means neither source nor target exists in that case. `conflict` means target exists beside source or multiple alias sources coexist, so nothing is chosen automatically.

Output-root apply is optional and must use `--apply --output-data-root <different-root>`; the source data root is refused when equal to the output root and original masks are not modified.
