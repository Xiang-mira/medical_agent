# Dataset Delivery Config Inputs

These files are examples only. Do not use them for formal inference until a human has filled confirmed rows.

- `organ_rename_mapping.csv.example`: rename-only mappings. Only `status=confirmed` rows execute.
- `organ_gap_resolution.csv.example`: rename/generate/manual-review control table. Teacher inference only reads `resolution=generate,status=confirmed`.
- `cases_100_manifest.csv.example`: required schema for the fixed 100 cases. The repo does not contain a frozen 100-case list.

The canonical 373 target source for this workflow is `configs/student_3d_prompt_target_organs.json`.

