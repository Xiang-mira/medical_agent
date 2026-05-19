# Final Audit and Usage Guide (v8)

This document is the final delivery note for the medical multi-model annotation refinement workflow. It records what the teacher asked for, what has been implemented in this repository, how each uploaded material is used, which models can be trained, and what still requires real data/checkpoints/GPU execution.

## 0. One-sentence project positioning

This project is not a single TotalSegmentator demo. It is a **selected-model-aware EM-style medical annotation refinement workflow**:

```text
Task / organ routing
→ selected primary model + auxiliary model inference
→ normalized `segmentations/*.nii.gz`
→ ShapeKit anatomy-aware post-processing
→ DICE quality gate
→ LabelCritic/VLM candidate-mask comparison
→ annotation update + versioning
→ RadThinking-style reasoning/VQA sample construction
→ selected-model-aware M-step fine-tuning plan
→ next-round inference with updated model candidate
```

## 1. Task-by-task completion matrix

| Teacher task | Status in v8 | How it is implemented | What still requires real execution |
|---|---:|---|---|
| Batch-wrap multiple models, not only TotalSegmentator | Implemented at source level | `configs/model_registry.yaml`; `run_medai_cli.py infer --model ...`; wrappers/scripts for TotalSegmentator, ePAI, CADS, MOOSE, VSmTrans, ATLAS-Net, VISTA3D, UNEST, nnUNet-private families. | Real inference needs mounted checkpoints and model-specific Linux/GPU dependencies. |
| Use 50 PanTS/PAINTS tumor cases for debug | Selector implemented; final 50-case run still pending | `pants-select-50`, `scripts/select_pants50_cases.py`, `scripts/validate_case_list_50.py`, `data_manifest/case_list_50_tumor_template.csv`. Selection requires CT path, annotation folder, non-empty tumor/lesion mask, and key organ masks. | Must download/mount PanTS NIfTI data and run the selector to create `data_manifest/case_list_50_tumor.csv`. |
| DICE as first quality gate | Implemented | `label_verifier.py`, `pants-eval-case`, `verify`, and `run-loop` output `dice_metrics.csv`; thresholds: `>=0.80 accept`, `0.50–0.80 uncertain`, `<0.50 LabelCritic`, `0 severe error`. | Real DICE values require real predictions and labels. |
| Replace naive 3D-to-2D average projection with LabelCritic | Implemented | `labelcritic_projection_runner.py`, `labelcritic_wrapper.py`, `critic` command. Default projection backend is LabelCritic/ProjectDatasetFlex/CompareOrgan logic, not average projection. | Real VLM decision requires a running vLLM/OpenAI-compatible server. |
| ShapeKit must be core post-processing, not optional decoration | Implemented | `run-loop` has ShapeKit enabled by default; `postprocess shapekit`; `shapekit_runner.py`; outputs can be evaluated before/after. | Real before/after improvement needs real masks. |
| M-step must not remain a stub | Implemented as selected-model-aware interface | `mstep-update --target-model ...`; `mstep_runner.py`; model inventory carries `trainable`, `mstep_backend`, `mstep_role`; TotalSegmentator is explicitly baseline-only; ePAI/CADS/VSmTrans/MOOSE/ATLAS-Net/private nnUNet are conditional trainable targets when checkpoint/training state is available. | Real fine-tuning requires mounted checkpoint training state, PanTS data, GPU, and enough cases. 50 cases are for smoke/debug only. |
| ITK-SNAP sanity check | Workflow support implemented | `itksnap-review` command generates loading commands from `review_queue.jsonl`; `docs/ITKSNAP_REVIEW_GUIDE.md`. | Actual screenshots must be made manually in ITK-SNAP after real run. |
| Combine annotation tool with RadThinking-style reasoning trace / VQA sample | Implemented structurally | `radthinking.py`, `reasoning_trace.py`, `build-samples`; output fields include candidate masks, dice scores, VLM decision, final annotation, and 4-step reasoning trace schema. | Real trace text quality depends on real reports/images and optional VLM generation. |
| Report supervision replacing old ROC-style tumor check | Implemented | `report-supervision`, `report-supervision-batch`, `report_supervision.py`, and integrated output `report_supervision.jsonl`. | Needs real PanTS reports/metadata and predicted lesion masks. |
| Loop convergence table | Implemented | `convergence-table` command builds cross-round summary: checked masks, low-DICE masks, VLM-reviewed, updated, remaining uncertain. | Meaningful convergence requires multiple real rounds. |
| Preserve private checkpoints | Implemented | Private weights are not packed; use symlinks from Drive/server `checkpoints/`. | User must mount/link teacher checkpoint folders at runtime. |

## 2. Uploaded materials and how they are used

| Material | What it is | Where it is used in project | Role |
|---|---|---|---|
| `meeting2.docx` | Meeting transcript | `docs/raw_materials/meeting2.docx`, `docs/meeting2_transcript.docx`, requirements reflected in docs and CLI design | Source of teacher requirements: multi-model CLI, LabelCritic projection, ShapeKit as core, M-step training, 50 tumor cases, RadThinking/VQA dataset, ITK-SNAP review. |
| `task_zhou.docx` | Consolidated task analysis | `docs/raw_materials/task_zhou.docx`, `docs/task_zhou_requirements.docx`, task matrices | Source of task-by-task checklist. |
| Google Drive screenshot / lightweight export | Checkpoint pool view and folder structure | `docs/raw_materials/local_materials_screenshot.png`; `docs/checkpoint_drive_export/*`; `configs/checkpoint_dataset_catalog.json` | Confirms CADS, MOOSE, nnUNet_private, UNEST, VSmTrans, and class map checkpoint pool. |
| `class_checkpoint_map.xlsx` | Anatomical structure → checkpoint mapping | `configs/class_checkpoint_map.xlsx`, `configs/class_checkpoint_map.parsed.csv`, `configs/model_registry.yaml`, `route-models`, `model-inventory` | Core organ-to-model routing source. |
| `Another version of ShapeKit.docx` | Actually ATLAS-Net model card | `docs/ATLAS-Net_model_card_from_teacher.docx`, `configs/atlasnet_label_map.json`, `scripts/atlasnet_predict_and_split.py`, registry entry `atlasnet` | 25-class abdominal nnUNet-style candidate model. |
| `ePAI-main.zip` + `qchen76_2025_0421.tar.gz` | ePAI 25-class abdomen/pancreas checkpoint and source | `third_party/ePAI-main`, `scripts/patch_epai_enable_all_organs.py`, registry `epai_20250421`, wrappers | Teacher-specified 2025-04-21 checkpoint; verified Dataset1017 label map has 25 foreground organ/duct/tumor labels. |
| `TotalSegmentator-master.zip` | Public segmentation baseline | `third_party/TotalSegmentator-master`, `totalseg_runner.py`, registry `totalsegmentator` | Public baseline / E-step candidate; explicitly not the default M-step target. |
| `VISTA3D-Inference-Pipeline.zip` | VISTA3D inference pipeline | `third_party/VISTA3D-Inference-Pipeline-master`, `scripts/vista3d_predict_and_split.py`, registry `vista3d` | Foundation segmentation inference candidate; high GPU demand; not trained by current wrapper. |
| `ShapeKit-main.zip` | Anatomy-aware post-processing toolkit | `third_party/ShapeKit-main`, `shapekit_runner.py`, `postprocess`, `run-loop` | Core post-processing stage for organ mask correction. |
| `LabelCritic-main.zip` | VLM/LVLM mask comparison tool | `third_party/LabelCritic-main`, `labelcritic_projection_runner.py`, `labelcritic_wrapper.py`, `critic` | Replaces naive 3D-to-2D average projection; compares candidate masks. |
| `li2026radthinking.pdf` | RadThinking/VQA reasoning paper | `docs/RadThinking_li2026.pdf`, `radthinking.py`, `build-samples` | Defines 4-step reasoning trace / VQA-style sample schema. |
| `r_super_pseudo_masks.py` | Report-supervised pseudo-mask script | `scripts/r_super_pseudo_masks.py`, `report-supervision`, `report-supervision-batch` | Report supervision and tumor pseudo-mask validation/refinement support. |
| `PanTS-main.zip` | PanTS repo / download scripts | `third_party/PanTS-main`, `pants-info`, `pants-download-50-plan`, `pants-select-50` | Data source for 50 tumor-case debug run; actual NIfTI data not packed. |
| `drive_lightweight_sources.zip` | Lightweight checkpoint source/config export | `docs/checkpoint_drive_export/*`, selected `third_party/*_lightweight` folders | Provides run scripts/configs without private weights. |

## 3. Model inventory and trainability

The project recognizes 18 teacher-provided or teacher-referenced model families, excluding the internal `mock_seg` testing backend.

| Model key | Source | Inference status | Trainability / M-step status | Why |
|---|---|---|---|---|
| `epai_20250421` | ePAI qchen76 2025-04-21 | Ready if checkpoint folder present | Conditional trainable; preferred PanTS pancreas M-step target and 25-class organ candidate | Dataset1017, nnUNet-style, 25 foreground labels; MedAI keeps all labels by default. |
| `cads` | Drive + class map | Ready if checkpoint present | Conditional trainable | Broad anatomical coverage; nnUNet-style if full state exists. |
| `moose`, `moose3_0` | Drive + class map | Partial ready if checkpoint present | Conditional trainable | Clinical CT organ families; require full training state. |
| `vsmtrans` | Drive lightweight + class map | Ready if checkpoint present | Conditional trainable | Abdominal organ candidate; fine-tuning possible only if nnUNet-compatible state exists. |
| `nnunet_private` | Drive | Ready if folder present | Conditional trainable | Private abdominal nnUNet-style backend. |
| `saros_nnunet` | nnUNet_private/class map | Ready if folder present | Conditional trainable | Private body-region backend; not first-choice PanTS target. |
| `atlasnet` | ATLAS-Net model card | Ready if weights present | Conditional trainable | Described as nnUNet v2 25-class abdominal model. |
| `totalsegmentator` | Public source/class map | Ready if installed | Not trained in this project | Public baseline and E-step candidate; retraining released package requires upstream training recipe. |
| `vista3d` | VISTA3D pipeline | Ready if VISTA3D env/checkpoint present | Not with current wrapper | Inference pipeline only; fine-tuning requires MONAI/VISTA3D training recipe. |
| `unest` | Drive lightweight run script | Ready if checkpoint present | External training required | Only inference/run script is available for renal substructures. |
| `airrc`, `atm`, `dap`, `duke`, `goacc`, `pedro`, `vsnet` | `class_checkpoint_map.xlsx` | Template only | Not trainable now | Names appear in class map, but no complete folder/script/training state is available. |

## 4. Correct M-step interpretation

The M-step is no longer described as “training a random new nnUNet.” It is now **selected-model-aware**:

```text
If organ/task primary model is trainable:
    update/fine-tune that selected model family or compatible checkpoint.
If primary model is not trainable in this repo:
    keep it as E-step candidate/baseline and use another trainable backend only as fallback.
```

Examples:

```text
pancreas / pancreatic_duct / pancreatic tumor → primary: ePAI_20250421 → M-step target: ePAI_20250421
liver / spleen / stomach / colon / duodenum → primary: VSmTrans or CADS/MOOSE → M-step target: selected trainable family
aorta / vessels → primary: CADS/VISTA3D depending availability → if VISTA3D not trainable, CADS/MOOSE/ATLAS-Net trainable backend is used
kidney_cortex / kidney_medulla → primary: UNEST → no built-in M-step until UNEST training script is provided
TotalSegmentator → baseline only, not default retraining target
```

## 5. What is complete vs. what must still be run

### Complete in source code

- Multi-model CLI and registry.
- Organ-to-model routing from class map.
- ePAI 2025-04-21 model-folder wrapper and all-label export patch for Dataset1017 25-class output.
- ATLAS-Net, VISTA3D, UNEST, TotalSegmentator, CADS, MOOSE, VSmTrans entries/wrappers.
- ShapeKit as default/core post-processing stage.
- DICE quality gate.
- LabelCritic projection and VLM comparison wrapper.
- Annotation versioning.
- Report supervision CLI.
- RadThinking/VQA-style per-case sample schema.
- selected-model-aware M-step update command.
- ITK-SNAP review command generator.
- 50-case selector/validator.
- Loop convergence table builder.

### Not complete until real run

- Actual 50 validated PanTS tumor-case CSV.
- Real ePAI/CADS/MOOSE/VSmTrans/ATLAS-Net/VISTA3D inference results.
- Real ShapeKit before/after DICE improvement.
- Real LabelCritic/VLM decisions.
- Real M-step fine-tuned checkpoint.
- Real ITK-SNAP screenshots.
- Real multi-round convergence table.

These are not safe to fabricate. They require real PanTS data, mounted checkpoints, GPU, and a VLM server.

## 6. Minimal Colab/server run sequence

```bash
# 0. Unzip project and enter it
unzip medai_agent_loop_task2_final_v8_ready.zip -d /content/
cd /content/medai_agent_loop_task2_final_v8

# 1. Link teacher checkpoints from Google Drive
python scripts/prepare_colab_checkpoint_links.py \
  --drive-checkpoints /content/drive/MyDrive/checkpoints \
  --project-root . \
  --overwrite

# 2. Verify source-level completeness
python scripts/verify_final_v8_integrity.py
python run_medai_cli.py --json model-inventory
python run_medai_cli.py --json route-models --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex

# 3. Prepare/validate PanTS 50-case list
python run_medai_cli.py --json pants-download-50-plan
python run_medai_cli.py --json pants-select-50 \
  --pants-root third_party/PanTS-main \
  --output data_manifest/case_list_50_tumor.csv \
  --split train \
  --num-cases 50
python scripts/validate_case_list_50.py --case-list data_manifest/case_list_50_tumor.csv

# 4. Run one real case first, then scale to 50
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models epai_20250421,cads,vsmtrans,totalsegmentator \
  --organs pancreas,pancreatic_duct,liver,spleen,duodenum,colon,stomach,aorta,postcava \
  --output outputs/run_pants50_round1 \
  --enable-shapekit \
  --enable-critic \
  --critic-backend labelcritic \
  --critic-base-url http://localhost \
  --critic-port 8000

# 5. Build samples and convergence table
python run_medai_cli.py --json build-samples \
  --run-output outputs/run_pants50_round1 \
  --output-jsonl outputs/run_pants50_round1/per_case_samples.jsonl \
  --organs pancreas,pancreatic_duct,liver,spleen,duodenum,colon,stomach,aorta,postcava

python run_medai_cli.py --json convergence-table \
  --round-csvs outputs/run_pants50_round1/round_metrics.csv

# 6. Run selected-model-aware M-step, example: pancreas/ePAI
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_round1/training_manifest.json \
  --output-folder outputs/mstep_epai_round1 \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --max-epochs 5
```

## 7. Safe presentation wording

Use:

> I implemented a selected-model-aware EM loop. The E-step routes each organ/task to a primary model and auxiliary candidates using the checkpoint map. After inference, ShapeKit, DICE, LabelCritic, and limited human review select or update the annotation. The M-step then targets the selected trainable model family, such as ePAI for pancreas tasks or CADS/VSmTrans/MOOSE for abdominal organ tasks. TotalSegmentator is kept as a public baseline and E-step candidate, not as the default retraining target.

Avoid:

> We retrained TotalSegmentator.

Avoid unless actually run:

> We completed 50-case quantitative improvement.
