# Task completion matrix — final v4

This matrix checks the project against the teacher's meeting tasks and the written task list.

| Teacher requirement | v4 implementation | Status |
|---|---|---|
| Batch-wrap multiple models, not only TotalSegmentator | `model_registry.yaml`, `registered_infer.py`, wrappers for `totalsegmentator`, `epai_20250421`, `cads`, `moose3_0`, `vsmtrans`, `atlasnet`, `vista3d`, `unest`, `nnunet_private`; all outputs normalize to `segmentations/*.nii.gz`. | Implemented at engineering level; real weights must be mounted in `checkpoints/`. |
| Use class checkpoint map for organ-to-model routing | `configs/class_checkpoint_map.xlsx`, `configs/class_checkpoint_map.parsed.csv`, `configs/model_registry.yaml`, and `registry-candidates` CLI. | Implemented. |
| ePAI should output more than the pancreas/tumor-only labels | `third_party/ePAI-main` is included; `scripts/patch_epai_enable_all_organs.py` adds `--output_label_mode all_organs/pancreas_only`; default is all-organ mode. | Implemented; verified by `verify_final_v4_integrity.py`. |
| Use 50 PanTS/PAINTS cases with tumor annotations, not arbitrary healthy cases | `scripts/select_pants50_cases.py` and `pants-select-50` check CT, reference labels, non-empty `pancreatic_lesion`, and required organ masks. | Implemented; real data still needs to be downloaded/mounted. |
| DICE as first quality gate | `label_verifier.py`, `label-verify`, `run-loop`; thresholds: `>=0.80 accept`, `0.50-0.80 uncertain`, `<0.50 LabelCritic/VLM`, `0 severe/auto-replace route`. | Implemented. |
| Replace naive 3D-to-2D projection with LabelCritic | `labelcritic_projection_runner.py` calls `ProjectDatasetFlex_single.py`; `projection_builder.py` defaults to `projection_backend=labelcritic`; local slice projection is fallback only. | Implemented in v4. |
| ShapeKit is core post-processing, not merely optional | `run-loop --enable-shapekit` places ShapeKit after inference and before DICE/LabelCritic. ShapeKit source is in `third_party/ShapeKit-main`. | Implemented; real before/after DICE requires real predictions. |
| LabelCritic/VLM should compare candidate A/B masks | `labelcritic_wrapper.py` wraps `CompareOrgan.py`; `critic` command outputs JSON; `run-loop` calls it for low-DICE cases when enabled. | Implemented; real VLM requires `--critic-backend labelcritic` and a VLM server. |
| Annotation update and version management | `annotation_versions/<case_id>/updated/`, `decisions.jsonl`, `review_queue.jsonl`, `vlm_decisions.jsonl`. | Implemented. |
| M-step should not remain a blank stub | `mstep_runner.py` writes `training_manifest.json` and `mstep_config.json`; this is the short-term interface for nnUNet/ePAI/private fine-tuning. | Implemented interface; real retraining not run because 50 cases are for debug and full training needs GPU/data. |
| ITK-SNAP sanity check for uncertain cases | `review_queue.jsonl` and `docs/ITKSNAP_REVIEW_GUIDE.md` identify cases to inspect manually. | Workflow support implemented; screenshots must be manually generated. |
| Combine mini annotation tool with RadThinking-style trace | `radthinking.py`, `reasoning_trace.py`, `patient_traces.jsonl`; fields include observations, temporal comparison, clinical context, conclusion, reasoning complexity. | Implemented; true clinical content depends on reports/metadata availability. |
| Use report supervision instead of old ROC sensitivity-specificity idea | `scripts/r_super_pseudo_masks.py` included and documented as report-supervised pseudo-mask/tumor bundle support. | Included and connected as an auxiliary module. |
| Preserve private checkpoint confidentiality | Real weights are not packaged; instructions use Google Drive/Colab symlinks under `checkpoints/`. | Implemented. |

## Remaining execution-only items

The source code and workflow are implemented. These items cannot be completed inside a source zip without the real runtime resources:

1. Download/mount real PanTS NIfTI CTs and labels.
2. Mount teacher private checkpoint folders in `checkpoints/`.
3. Run real inference for the selected 50 tumor cases.
4. Generate real `dice_metrics.csv`, `round_metrics.csv`, `vlm_decisions.jsonl`, `patient_traces.jsonl`.
5. Take 1–2 ITK-SNAP screenshots from `review_queue.jsonl` cases.
