# CADS Task 2 Target Status

- Read only: `True`
- CADS all targets: `15`
- Completed targets: `7`
- Remaining targets: `8`
- Existing CADS root: `/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/formal_teacher_22_full_volume_20260730/results/cads`

## Canonical CADS Targets

`blood`, `cerebrospinal_fluid`, `common_iliac_artery_left`, `common_iliac_artery_right`, `common_iliac_vein_left`, `common_iliac_vein_right`, `compact_bone`, `eyeball`, `face`, `gland_structure`, `gray_matter`, `muscle_of_head`, `scalp`, `spongy_bone`, `white_matter`

## Completed

`cerebrospinal_fluid`, `eyeball`, `face`, `gray_matter`, `muscle_of_head`, `scalp`, `white_matter`

## Remaining

`blood`, `common_iliac_artery_left`, `common_iliac_artery_right`, `common_iliac_vein_left`, `common_iliac_vein_right`, `compact_bone`, `gland_structure`, `spongy_bone`

## Stale Candidate List Check

The old CADS remaining list is compared against the current canonical Task 2 target CSV.

- In current Task 2: `blood, compact_bone, spongy_bone`
- Outside current Task 2: `brain, brainstem, larynx, oral_cavity, trachea`

## Per-target Status

| target | model | source label | status | masks | valid | root cause | next action |
|---|---|---|---|---:|---:|---|---|
| blood | cads557 | blood | remaining | 0 | 0 | post_export_missing | run_cads_remaining8_smoke |
| cerebrospinal_fluid | cads557 | csf | completed | 0 | 0 | completed_existing_100case | no_action_completed |
| common_iliac_artery_left | cads553 | iliac_artery_left | remaining | 0 | 0 | export_name_mismatch | run_cads_remaining8_smoke_after_alias_export_fix |
| common_iliac_artery_right | cads553 | iliac_artery_right | remaining | 0 | 0 | export_name_mismatch | run_cads_remaining8_smoke_after_alias_export_fix |
| common_iliac_vein_left | cads553 | iliac_vena_left | remaining | 0 | 0 | export_name_mismatch | run_cads_remaining8_smoke_after_alias_export_fix |
| common_iliac_vein_right | cads553 | iliac_vena_right | remaining | 0 | 0 | export_name_mismatch | run_cads_remaining8_smoke_after_alias_export_fix |
| compact_bone | cads557 | compact bone | remaining | 0 | 0 | export_name_mismatch | run_cads_remaining8_smoke_after_alias_export_fix |
| eyeball | cads557 | eye balls | completed | 0 | 0 | completed_existing_100case | no_action_completed |
| face | cads553 | face | completed | 0 | 0 | completed_existing_100case | no_action_completed |
| gland_structure | cads559 | glands | remaining | 0 | 0 | export_name_mismatch | run_cads_remaining8_smoke_after_alias_export_fix |
| gray_matter | cads557 | gray matter | completed | 0 | 0 | completed_existing_100case | no_action_completed |
| muscle_of_head | cads557 | head muscles | completed | 0 | 0 | completed_existing_100case | no_action_completed |
| scalp | cads557 | scalp | completed | 0 | 0 | completed_existing_100case | no_action_completed |
| spongy_bone | cads557 | spongy bone | remaining | 0 | 0 | export_name_mismatch | run_cads_remaining8_smoke_after_alias_export_fix |
| white_matter | cads557 | white matter | completed | 0 | 0 | completed_existing_100case | no_action_completed |
