# Final v6 Task Completion Matrix

This document maps the teacher's meeting requirements to the current source implementation.

| Teacher requirement | Source implementation | Status | Remaining non-code dependency |
|---|---|---:|---|
| Batch-wrap multiple models, not only TotalSegmentator | `model_registry.yaml`, `registered_infer.py`, wrappers for ePAI, CADS, MOOSE, VSmTrans, nnUNet_private, ATLAS-Net, VISTA3D, UNEST, TotalSegmentator | Implemented | Real private checkpoints must be mounted under `checkpoints/`. |
| Use 50 PanTS/PAINTS tumor cases for debugging | `select_pants50_cases.py`, `pants-select-50`, `validate_case_list_50.py` | Code implemented | Real PanTS CT/labels must be downloaded before generating final `case_list_50_tumor.csv`. |
| Improve organ masks; tumor masks are context | `run-loop` organs default to pancreas/liver/spleen/kidneys/colon/duodenum/stomach/aorta/postcava; tumor mask used in report supervision/context | Implemented | Needs real data to measure improvement. |
| DICE as first quality gate | `label_verifier.py`, CLI `verify`, run-loop `dice_metrics.csv` | Implemented | Needs predictions/reference masks for real metrics. |
| Replace naive average 3D→2D VLM projection | `labelcritic_projection_runner.py`, `projection_builder.py` default to LabelCritic `ProjectDatasetFlex_single.py + projection.py` | Implemented | LabelCritic dependencies and VLM server needed for real A/B decisions. |
| LabelCritic A/B mask comparison | `labelcritic_wrapper.py`, CLI `critic`, run-loop `vlm_decisions.jsonl` | Implemented | Start a vLLM/OpenAI-compatible VLM server; use host-only `--base-url`. |
| ShapeKit is core post-processing, not optional | `run-loop --enable-shapekit` default true; `em-loop --postprocess shapekit` default | Implemented | Needs model predictions. |
| M-step cannot stay blank stub | `mstep_runner.py`, CLI `mstep-train`, `em-loop --enable-mstep-training`; prepares nnUNet dataset and train command | Implemented as smoke-training interface | Real GPU and real CT/labels needed; 50 cases are only debug and will overfit. |
| ITK-SNAP sanity check | `itksnap_helper.py`, CLI `itksnap-review` creates load commands | Implemented helper | Actual screenshots must be taken manually. |
| Merge mini annotation tool with RadThinking-style trace | `radthinking.py`, `case_sample_builder.py`, CLI `build-samples` | Implemented structure | Natural-language clinical reasoning depends on report/clinical/pathology inputs; the code does not fabricate diagnosis. |
| Use report supervision instead of old ROC analysis for tumor checks | `report_supervision.py`, CLI `report-supervision`, `report-supervision-batch`, run-loop writes `report_supervision.jsonl` when reports exist | Implemented | Needs report files in case list. |
| Do not expose private checkpoints | Checkpoints are not bundled; only lightweight metadata and command templates included | Satisfied | Mount private folders in Colab/server locally. |

## Honest boundary

The source now satisfies the engineering requirements. The following are not possible to truthfully include in a source zip without running the real environment: final selected 50 validated cases, real multi-model predictions, real DICE improvement numbers, real VLM decisions, and ITK-SNAP screenshots.
