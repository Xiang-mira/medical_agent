# Final v3 deliverable status

## What is now included

This v3 package integrates all teacher-provided materials that were available as uploaded files or as lightweight Google Drive exports:

- `class_checkpoint_map.xlsx` parsed into `configs/model_registry.yaml` and `configs/class_checkpoint_map.parsed.csv`.
- Google Drive lightweight checkpoint metadata/scripts under `docs/checkpoint_drive_export/`.
- CADS/MOOSE/nnUNet_private/VSmTrans/UNEST source/config metadata from the Drive export.
- `third_party/ePAI-main/` with an export-label preservation patch. `qchen76_2025_0421.tar.gz` is verified as Dataset1017 with 25 foreground abdominal organ/duct/tumor labels.
- `third_party/LabelCritic-main/` and the `critic` CLI wrapper.
- `third_party/ShapeKit-main/` and core post-processing integration.
- `third_party/TotalSegmentator-master/` source plus installed-CLI runner support.
- `third_party/VISTA3D-Inference-Pipeline-master/` and `scripts/vista3d_predict_and_split.py`.
- ATLAS-Net model card under `docs/ATLAS-Net_model_card_from_teacher.docx`, plus `configs/atlasnet_label_map.json` and `scripts/atlasnet_predict_and_split.py`.
- `third_party/PanTS-main/` plus PanTS 50-case selection/download planning utilities.
- `scripts/r_super_pseudo_masks.py` for report-supervised pseudo-mask refinement support.
- RadThinking schema support through `radthinking.py`, `trace-build`, and `patient_traces.jsonl` output.

## What is intentionally not bundled

The package does not include private model weights/checkpoints or PanTS NIfTI CT volumes/labels. Those are too large and may be private. Expected local layout:

```text
checkpoints/
  CADS_series/
  MOOSE_series/
  nnUNet_private/
  ATLAS-Net/
third_party/PanTS-main/data/
  ImageTr/
  LabelTr/
```

## Final pipeline

```text
PanTS 50 tumor-case selection
→ multi-model inference through model_registry
→ standardized segmentations/*.nii.gz
→ ShapeKit anatomy-aware post-processing
→ DICE quality gate
→ LabelCritic/VLM comparison for low-quality/conflicting masks
→ annotation_versions/ updated masks
→ RadThinking-style reasoning traces
→ M-step training_manifest.json
```

## Remaining execution-only work

To produce real numbers for the teacher, run the pipeline on a machine with PanTS data and checkpoint folders mounted. The code is included; the heavy data/weights are not.
