# 373 Anatomical Label Rename Delivery

This package standardizes confirmed anatomical mask aliases to the canonical 373-class taxonomy.
Only the same anatomy, same side, and same granularity are renamed.
The package does not split, merge, generate, or modify NIfTI voxel values.

Confirmed mappings: configs/organ_rename_mapping_373.csv
Canonical taxonomy: configs/student_3d_prompt_target_organs.json

Run from the package root.

Dry run:
python tools/dataset_delivery/rename_anatomical_labels.py --data-root /PATH/TO/MASK_DATA --mapping-file configs/organ_rename_mapping_373.csv --taxonomy configs/student_3d_prompt_target_organs.json --report rename_dry_run_report.csv --dry-run

Apply only after reviewing every conflict in the dry-run report:
python tools/dataset_delivery/rename_anatomical_labels.py --data-root /PATH/TO/WRITABLE_MASK_DATA --mapping-file configs/organ_rename_mapping_373.csv --taxonomy configs/student_3d_prompt_target_organs.json --report rename_apply_report.csv --apply

Existing target files are never overwritten.
Multiple sources competing for one target are skipped and reported.
Only status=confirmed mappings are executed.
Missing or finer-grained structures are handled separately by the 100-case generation workflow.

Confirmed rename rules: 107.
Canonical taxonomy targets covered by rename rules: 94.
The four ambiguous carotid/subclavian aliases are excluded from automatic rename because they coexist with two independent canonical taxonomy targets; see non_rename_decisions.csv.
Targets not solvable by filename normalization are listed in unresolved_unmatched_targets.txt for the separate label-generation or derivation workflow.
