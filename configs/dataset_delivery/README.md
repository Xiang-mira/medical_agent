# Dataset Delivery Config Inputs

Task 1 anatomical rename is maintained only under `configs/dataset_delivery/task1/`.
The files in this directory ending with `.example` are templates and are not formal
inputs.

- `task1/organ_rename_mapping_373.csv`: canonical Task 1 mapping.
- `task1/task_boundary_classification.csv`: row-level evidence and Task 1/Task 2 boundary.
- `task1/task1_alias_groups.csv`: explicit many-to-one alias declarations.
- `task1/task2_generate_targets_23.csv`: fixed Task 2 targets that must never execute as Task 1 rename.
- `task1/non_rename_decisions.csv`: documented coarse-to-fine or pending-review decisions.
- `organ_gap_resolution.csv.example`: Task 2 template only; Teacher inference reads confirmed generate rows there when populated.
- `cases_100_manifest.csv.example`: manifest schema example.

The canonical 373 target source remains `configs/student_3d_prompt_target_organs.json`.
The `main` branch is the only maintenance branch for Task 1.
