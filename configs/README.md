# Configuration Index

## Registry and routing

| File | Purpose |
| --- | --- |
| `model_registry.yaml` | model definitions, checkpoint metadata, organ coverage, and M-step capabilities |
| `teacher_branch_map.yaml` | teacher branch routing |
| `routing_token_to_model.json` | routing-token normalization |
| `organ_routing_from_xlsx.json` | workbook-derived organ routing |
| `checkpoint_dataset_catalog.json` | checkpoint-to-dataset catalog |
| `checkpoint_drive_manifest.json` | checkpoint inventory |

## Target space and taxonomy

| File | Purpose |
| --- | --- |
| `student_3d_prompt_target_organs.json` | 373-target student label space |
| `abdomenatlaspro_target_mapping_373.json` | AbdomenAtlasPro mapping over the 373-target space; unresolved entries must not be treated as negative masks |
| `abdomenatlaspro_pilot_338_target_config.json` | Direct-match pilot subset only; not the full target space |
| `pipelines/abdomenatlaspro_373.yaml` | Full 373-target AbdomenAtlasPro pipeline using `student_3d_prompt_target_organs.json` |
| `pipelines/abdomenatlaspro_pilot338.yaml` | 338-target pilot pipeline for direct-match smoke/formal pilot runs |
| `organ_taxonomy.json` | canonical organ hierarchy |
| `global_label_space.json` | consolidated label space |
| `all_organs.json` | teacher coverage list |
| `organ_identity_mapping_audit.json` | generated identity audit |

## Scoring and label maps

| File | Purpose |
| --- | --- |
| `autolabel_core.yaml` | AutoLabelCore evidence weights and grading policy |
| `model_label_aliases.json` | model-label aliases |
| `atlasnet_label_map.json` | ATLAS-Net label map |
| `epai_20250421_dataset1017_labels.json` | ePAI label metadata |
| `totalseg_subtask_organs.json` | TotalSegmentator subtask map |
| `moose_gap_organs.json` | MOOSE coverage gaps |
| `moose_replacement_table.csv` | MOOSE replacement mapping |

## Source workbooks

`class_checkpoint_map.xlsx`, `class_checkpoint_map_updates.xlsx`, and
`class_checkpoint_map.parsed.csv` are the source and parsed checkpoint maps
used to generate registry and taxonomy artifacts.
