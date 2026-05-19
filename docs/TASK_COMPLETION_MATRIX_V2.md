# Task Completion Matrix after Integrating Latest Drive Export and PanTS Repo

| Teacher task | Current implementation status | Evidence in package | Remaining external requirement |
|---|---|---|---|
| Multi-model registry, not only TotalSegmentator | Implemented as registry-driven framework with real templates for CADS, ePAI, VSmTrans, private nnUNet, MOOSE, TotalSegmentator, VISTA3D templates. | `configs/model_registry.yaml`, `configs/checkpoint_dataset_catalog.json`, `scripts/nnunetv2_predict_and_split.py` | Actual checkpoint folders must be mounted under `checkpoints/` on your server/Colab. |
| Select 50 PanTS tumor cases, not random | Implemented as deterministic validated selector. It requires non-empty pancreatic lesion mask and required organ masks. | `scripts/select_pants50_cases.py`, CLI `pants-select-50` | Real PanTS CT/label data must be downloaded first. |
| DICE first-stage quality gate | Implemented with teacher thresholds: >=0.8 accept, 0.5-0.8 uncertain, <0.5 LabelCritic/VLM. | `core/label_verifier.py`, CLI `verify`, `run-loop` outputs `dice_metrics.csv`. | Needs real predictions/reference masks for real metrics. |
| ShapeKit core post-processing | Integrated in `run-loop` after model inference. | `core/multimodel_loop.py`, `core/shapekit_runner.py`, `third_party/ShapeKit-main`. | Needs real masks and ShapeKit dependencies. |
| LabelCritic/VLM comparison | Wrapped through `critic`; stub and real backend are available. | `core/labelcritic_wrapper.py`, CLI `critic`, `third_party/LabelCritic-main`. | Real VLM server/base URL needed for non-stub use. |
| Annotation update/versioning | Implemented with `annotation_versions/<case_id>/updated/*.nii.gz`. | `run-loop` output structure. | Needs real predictions to update. |
| ITK-SNAP sanity check | Not automatable in code; workflow now explicitly produces review queue and updated masks for manual checking. | `review_queue.jsonl`, `docs/TASK2_PANTS50_REAL_WORKFLOW.md`. | User must open CT/masks in ITK-SNAP and save screenshots. |
| RadThinking-style trace dataset | Integrated in loop with `patient_traces.jsonl`. | `core/radthinking.py`, `run-loop`. | True clinical/pathology text requires source reports/metadata; package does not fabricate diagnosis. |
| M-step interface | Implemented as manifest and config, not claiming real retraining from 50 cases. | `training_manifest.json`, `mstep_config.json`, `core/mstep_runner.py`. | Real training script/checkpoint fine-tuning must be provided by lab. |
