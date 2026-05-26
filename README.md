# MedAI Agent Loop Task 2 – Multi-Model Medical Annotation Refinement

This repository implements a step-by-step medical image annotation refinement pipeline for abdominal CT segmentation. It starts from selected PanTS/PAINTS tumor cases, runs multiple segmentation teacher models, refines masks with anatomy-aware post-processing and LabelCritic/VLM review, and prepares a selected-model-aware M-step so the improved annotations can update the next E-step model.

## What this project does, step by step

```text
1. Prepare local checkpoints and PanTS/PAINTS CT data
2. Build or inspect the model registry
3. Select 50 tumor-positive PanTS/PAINTS cases
4. Run E-step Round 1 with multiple segmentation models
5. Normalize model outputs into segmentations/*.nii.gz
6. Run ShapeKit anatomy-aware post-processing
7. Compute DICE/DSC quality scores against current annotations
8. Send low-confidence or conflicting masks to LabelCritic/VLM review
9. Save updated annotation versions and a training_manifest.json
10. Run selected-model-aware M-step to update the chosen trainable model
11. Run E-step Round 2 using the updated model/checkpoint
12. Compare Round 1 vs Round 2 metrics and export review artifacts
```

The intended full loop is:

```text
PanTS/PAINTS 50 tumor cases
        |
        v
Model registry chooses candidate teachers per organ
        |
        v
E-step: ePAI / VSmTrans / TotalSegmentator / VISTA3D / private templates
        |
        v
ShapeKit post-processing + mask format normalization
        |
        v
DICE/DSC gate: accept, uncertain, or send to LabelCritic/VLM
        |
        v
Annotation update + review queue + RadThinking-style traces
        |
        v
M-step: update the selected trainable model family when training state is available
        |
        v
Updated checkpoint returns to the next E-step candidate pool
```

## 1. Prerequisites

Use a Linux machine or GPU server with:

- Python environment that can run this repository's CLI.
- NVIDIA GPU for real model inference/training.
- CUDA-compatible PyTorch / nnUNet environment for ePAI and nnUNet-style M-step training.
- MONAI environment for VISTA3D fine-tuning if VISTA3D is used as an M-step backend.
- Local PanTS/PAINTS CT data in NIfTI format (`.nii.gz`).
- Local checkpoints for any private or large models you want to run.
- Optional LabelCritic/VLM server on `localhost:8000` for real visual review.

For command-generation or smoke tests, use `--dry-run` or `mock_seg`. Do not report dry-run or stub results as real segmentation results.

## 2. Required local files

Large/private files are not bundled. Place them locally like this:

```text
checkpoints/
  qchen76_2025_0421/
    nnUNetTrainer__nnUNetPlans__3d_fullres/
      dataset.json
      plans.json
      fold_all/checkpoint_final.pth
  CADS_series/
  MOOSE_series/
  nnUNet_private/
  UNEST/
  VSmTrans/
  ATLAS-Net/
third_party/PanTS-main/data/
  ImageTr/
  LabelTr/
```

The main 50-case manifest used by the current workflow is:

```text
data_manifest/case_list_50_tumor.csv
```

Each row should point to a CT file and an annotation folder. The CLI expects PanTS/ShapeKit-style masks under `segmentations/*.nii.gz`.

## 3. Install / prepare the CLI

From the repository root:

```bash
cd /path/to/medical_agent
pip install -e agent-harness
```

Then verify that the CLI can load:

```bash
python run_medai_cli.py --json doctor
```

## 4. Check the model registry

The workflow is registry-driven. The registry maps organs to candidate model families and records whether each model can be used for M-step training.

Inspect available models:

```bash
python run_medai_cli.py --json model-inventory
```

Check candidate models for specific organs:

```bash
python run_medai_cli.py --json registry-candidates \
  --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex
```

Ask the router which model should be primary/auxiliary for each organ:

```bash
python run_medai_cli.py --json route-models \
  --organs pancreas,liver,aorta,kidney_cortex
```

## 5. Validate or select the 50 tumor cases

If `data_manifest/case_list_50_tumor.csv` already exists, validate it:

```bash
python scripts/validate_case_list_50.py \
  --case-list data_manifest/case_list_50_tumor.csv
```

If you need to regenerate the manifest from a downloaded PanTS folder:

```bash
python run_medai_cli.py --json pants-select-50 \
  --pants-root third_party/PanTS-main \
  --output data_manifest/case_list_50_tumor.csv \
  --split train \
  --num-cases 50
```

The selector is not random for the formal run: it checks that the selected cases have usable CT paths, label folders, required organ masks, and non-empty pancreatic lesion/tumor masks unless smoke-test options are explicitly used.

## 6. Run E-step Round 1

The E-step runs multiple teacher models on each selected CT case, writes normalized masks, applies ShapeKit if enabled, computes DICE/DSC scores, and queues uncertain cases for LabelCritic/VLM review.

Example real run:

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models epai_20250421,vsmtrans \
  --organs pancreas,liver,spleen,kidney_left,kidney_right,aorta,postcava,duodenum,stomach \
  --output outputs/run_pants50_round1 \
  --enable-shapekit \
  --enable-critic \
  --critic-backend labelcritic \
  --critic-base-url http://localhost \
  --critic-port 8000
```

For an offline command check only:

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models mock_seg \
  --organs pancreas,liver,aorta \
  --output outputs/dry_run_multimodel \
  --dry-run
```

## 7. What happens inside E-step

For every case and every requested model, the loop does the following:

1. Loads the CT path and current annotation folder from the case-list CSV.
2. Runs the selected registry model wrapper.
3. Converts each model's output into a normalized `segmentations/*.nii.gz` folder.
4. Optionally runs ShapeKit post-processing to enforce anatomy-aware consistency.
5. Computes DICE/DSC between prediction and current annotation for each organ.
6. Applies the quality gate:
   - `DICE >= 0.8`: accept.
   - `0.5 <= DICE < 0.8`: uncertain, keep for review.
   - `DICE < 0.5`: send to LabelCritic/VLM if enabled.
7. Uses LabelCritic projection and comparison for conflicting candidate masks.
8. Writes annotation versions, review queues, VLM decisions, and a training manifest.

## 8. Expected E-step output

A full `run-loop` output folder contains:

```text
outputs/run_pants50_round1/
  dice_metrics.csv              # per-case/per-organ/per-model DICE decisions
  round_metrics.csv             # aggregated round metrics
  review_queue.jsonl            # uncertain masks for ITK-SNAP/manual review
  vlm_decisions.jsonl           # LabelCritic/VLM decisions when enabled
  annotation_versions/          # accepted or updated annotation versions
  patient_traces.jsonl          # RadThinking-style structured traces
  training_manifest.json        # cases/masks used by the M-step
  mstep_config.json             # selected M-step configuration
  run_summary.json              # machine-readable summary
```

The most important file for the next stage is:

```text
outputs/run_pants50_round1/training_manifest.json
```

## 9. Run selected-model-aware M-step

The M-step does not blindly train a generic model. It updates the selected trainable model family when the needed training state exists.

Example ePAI update:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_round1/training_manifest.json \
  --output-folder outputs/mstep_round1 \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --pretrained-weights checkpoints/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth
```

Dry-run the same command before spending GPU time:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_round1/training_manifest.json \
  --output-folder outputs/mstep_round1_dryrun \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --dry-run
```

M-step outputs include a prepared training dataset, training logs, a training result JSON, and a new checkpoint such as `checkpoint_best.pth` or `checkpoint_final.pth` when training succeeds. The exact training format depends on the selected backend: ePAI and other nnUNet-compatible models use nnUNet-style training data, while VISTA3D requires its MONAI bundle configuration and resources.

## 10. Run E-step Round 2 with the updated model

After M-step produces a checkpoint, run another E-step with the same 50 cases and compare metrics:

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models epai_20250421,vsmtrans \
  --organs pancreas,liver,spleen,kidney_left,kidney_right,aorta,postcava,duodenum,stomach \
  --output outputs/run_pants50_round2 \
  --enable-shapekit \
  --enable-critic \
  --critic-backend labelcritic \
  --critic-base-url http://localhost \
  --critic-port 8000
```

Then build a convergence table:

```bash
python run_medai_cli.py --json convergence-table \
  --round-csvs outputs/run_pants50_round1/round_metrics.csv,outputs/run_pants50_round2/round_metrics.csv
```

## 11. One-command EM loop scripts

For a GPU server, the repository also includes shell scripts that run the long workflow in order.

Full E-step → M-step → E-step:

```bash
bash run_em_loop.sh
```

Continuation script when Round 1 already exists:

```bash
bash run_m_e2_loop.sh
```

These scripts perform pre-flight checks for VLM server, GPU memory, disk space, Round 1 manifest existence, and then write logs under `outputs/logs_<timestamp>/`.

## 12. Manual review with ITK-SNAP

After `run-loop`, generate review commands for uncertain cases:

```bash
python run_medai_cli.py --json itksnap-review \
  --review-queue outputs/run_pants50_round1/review_queue.jsonl \
  --output-script outputs/run_pants50_round1/open_itksnap_review.sh \
  --max-cases 10
```

Run the generated script locally on a workstation with ITK-SNAP installed to inspect CT, current annotation, and candidate masks.

## 13. Build teacher-facing samples

To export per-case samples with candidate masks, DICE scores, VLM decision, final annotation, and RadThinking-style traces:

```bash
python run_medai_cli.py --json build-samples \
  --run-output outputs/run_pants50_round1 \
  --output-jsonl outputs/run_pants50_round1/case_samples.jsonl
```

## 14. Optional M-step backends: TotalSegmentator and VISTA3D

This implementation treats **TotalSegmentator** and **VISTA3D** as selectable model families in the registry, not just inference-only references. The distinction is precise:

- `totalsegmentator` remains a public E-step baseline, but it also exposes a **conditional TotalSegmentator-style public nnUNet training backend** through the bundled `third_party/TotalSegmentator-master/resources/train_nnunet.md`, `train_nnunet.sh`, and `convert_dataset_to_nnunet.py`. This does **not** claim to reproduce the official released TotalSegmentator model, because the official training used additional non-public data.
- `vista3d` remains a high-resource foundation candidate, but it also exposes a **conditional MONAI bundle fine-tuning backend** through `third_party/VISTA3D-Inference-Pipeline-master/configs/train.json`, `train_continual.json`, `multi_gpu_train.json`, and `scripts/trainer.py`. This supports fine-tuning/continual learning when a VISTA3D checkpoint, datalist, MONAI environment, and sufficient GPU memory are available.

Example dry-run checks:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_round1/training_manifest.json \
  --output-folder outputs/mstep_totalseg \
  --target-model totalsegmentator \
  --dry-run

python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_round1/training_manifest.json \
  --output-folder outputs/mstep_vista3d \
  --target-model vista3d \
  --dry-run
```

### Selected-model-aware M-step semantics

This project uses **selected-model-aware M-step** rather than training a generic new nnUNet regardless of the E-step model.

```text
choose task/organ-specific primary model
→ E-step inference and annotation refinement
→ update that selected model family if it is trainable
→ return the updated checkpoint to the next E-step candidate pool
```

Important clarification:

- `TotalSegmentator` is a **public baseline / E-step candidate** and an optional conditional M-step backend, but it is not the default retraining target.
- `ePAI_20250421` is the preferred trainable target for pancreas / pancreatic duct / pancreatic tumor tasks when the teacher's nnUNet-compatible checkpoint is mounted.
- CADS, MOOSE, VSmTrans, private nnUNet, SAROS, and ATLAS-Net are conditional M-step targets when their full nnUNet-compatible training state exists.
- VISTA3D is a high-resource foundation candidate. It can be used as a conditional MONAI fine-tuning backend only when the required checkpoint, datalist, MONAI configuration, and GPU resources are available.
- UNEST and template-only families remain inference/external-training candidates unless their original training recipes are provided.

Useful commands:

```bash
python run_medai_cli.py --json model-inventory

python run_medai_cli.py --json route-models \
  --organs pancreas,liver,aorta,kidney_cortex

python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_round1/training_manifest.json \
  --output-folder outputs/mstep_epai_round1 \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --dry-run
```

See:

- `docs/SELECTED_MODEL_AWARE_MSTEP_V7.md`
- `docs/MODEL_INVENTORY_AND_TRAINABILITY_V7.md`

## 15. Implementation status and verified source coverage

This project is a registry-driven, multi-model medical annotation refinement workflow for the teacher's Task 2. It is designed for PanTS/PAINTS 50 tumor-case debugging first, then scalable extension to larger cohorts.

### Core loop

```text
class_checkpoint_map.xlsx -> model_registry
                       |
                       v
Select 50 PanTS/PAINTS cases with tumor annotation
                       |
                       v
E-step: multi-model inference -> normalized segmentations/*.nii.gz
                       |
                       v
ShapeKit anatomy-aware post-processing -> before/after DICE gate
                       |
                       v
LabelCritic/VLM review for low-DICE or conflicting candidate masks
                       |
                       v
Annotation update/versioning + report supervision + ITK-SNAP queue
                       |
                       v
RadThinking-style structured + narrative reasoning sample builder
                       |
                       v
M-step: selected-model-aware fine-tuning plan/training manifest
                       |
                       +---- updated model/checkpoint -> next E-step
```

### What is implemented in source

| Module | Status | Notes |
|---|---:|---|
| Multi-model registry | Implemented | `configs/model_registry.yaml`, `configs/class_checkpoint_map.parsed.csv`; formal candidate lists exclude `mock_seg` unless `--include-mock` is used. |
| ePAI 2025-04-21 25-class output | Implemented | `qchen76_2025_0421.tar.gz` was verified as `Dataset1017_ePAI_3MM` with 25 foreground labels. MedAI calls it through `nnUNetv2_predict_from_modelfolder` and keeps all labels by default. |
| ATLAS-Net | Implemented wrapper | `configs/atlasnet_label_map.json`, `scripts/atlasnet_predict_and_split.py`. Real weights still required. |
| VISTA3D | Implemented wrapper + conditional fine-tuning backend | Bundled source and splitting wrapper. Real bundle/checkpoint, MONAI environment, datalist, and sufficient GPU memory are required for fine-tuning. |
| UNEST | Implemented wrapper | Bundled teacher Drive lightweight script/config and wrapper. Real checkpoint environment still required. |
| TotalSegmentator | Bundled source + CLI runner + conditional nnUNet-style backend | Source in `third_party/TotalSegmentator-master`; real use requires installation/weights. The training backend is conditional and does not claim to reproduce the official released TotalSegmentator model. |
| ShapeKit | Core post-processing | Enabled by default in `run-loop` and `em-loop`. |
| DICE/DSC quality gate | Implemented | `>=0.8 accept`, `0.5–0.8 uncertain`, `<0.5 LabelCritic/review`. |
| LabelCritic projection | Implemented | Uses teacher-provided `ProjectDatasetFlex_single.py` + `projection.py`, not naive average projection. |
| LabelCritic/VLM comparison | Implemented wrapper | Default real backend is `labelcritic`; use `--critic-backend stub` only for offline dry-run. |
| Report supervision | Implemented | Single and batch CLI compare tumor mask presence/size with report text. |
| RadThinking-style sample | Implemented structure + template prose | Builds observation/temporal/context/conclusion objects and deterministic natural-language trace fields; VLM-generated clinical prose remains an optional future backend. |
| M-step | Implemented selected-model-aware interface | Prepares the selected backend's training data/command when the required training state exists. Real training must run on GPU; 50 cases are smoke-test/debugging scale only. |
| ITK-SNAP review | Implemented helper | Generates commands from `review_queue.jsonl`; screenshots must be captured manually. |

### What is not bundled

The zip does **not** contain private checkpoints or real PanTS NIfTI images. Put them locally/Colab as:

```text
checkpoints/
  qchen76_2025_0421/
    nnUNetTrainer__nnUNetPlans__3d_fullres/
      dataset.json
      plans.json
      fold_all/checkpoint_final.pth
  CADS_series/
  MOOSE_series/
  nnUNet_private/
  UNEST/
  VSmTrans/
  ATLAS-Net/          # if available
third_party/PanTS-main/data/
  ImageTr/
  LabelTr/
```

### Minimal verification

```bash
python scripts/verify_final_v4_integrity.py
python run_medai_cli.py --json registry-candidates \
  --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex
```

Expected: `status: success`, ePAI 2025-04-21 appears for its verified 25-class labels, and no `mock_seg` in formal candidate lists. Use `--include-mock` only for smoke tests.

### Real run sequence

```bash
python scripts/validate_case_list_50.py \
  --case-list data_manifest/case_list_50_tumor.csv

python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models totalsegmentator,epai_20250421,cads,vsmtrans,nnunet_private \
  --organs pancreas,pancreatic_lesion,liver,spleen,kidney_left,kidney_right,colon,duodenum,stomach,aorta,postcava \
  --output outputs/run_pants50_real \
  --enable-shapekit \
  --enable-critic \
  --critic-backend labelcritic \
  --critic-base-url http://localhost \
  --critic-port 8000
```

For offline command checks only, add `--dry-run` or use `--critic-backend stub`. Do not report stub results as real LabelCritic/VLM results.
