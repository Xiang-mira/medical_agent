# Rename Delivery

Run from the repository root so `tools/dataset_delivery` is importable.

```bash
python rename_delivery/rename_anatomical_labels.py \
  --data-root /official/AbdomenAtlasPro \
  --mapping-file rename_delivery/organ_rename_mapping.csv \
  --taxonomy configs/student_3d_prompt_target_organs.json \
  --report rename_delivery/dry_run_report.csv \
  --dry-run
```

Apply only after reviewing the dry-run:

```bash
python rename_delivery/rename_anatomical_labels.py \
  --data-root /official/AbdomenAtlasPro \
  --mapping-file rename_delivery/organ_rename_mapping.csv \
  --taxonomy configs/student_3d_prompt_target_organs.json \
  --report rename_delivery/apply_report.csv \
  --apply
```

`organ_rename_mapping.csv` is empty until manually confirmed rows are added.

