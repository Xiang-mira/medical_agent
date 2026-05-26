# MedAI Agent Loop – Multi-Model Medical Annotation Refinement

This repository implements a registry-driven medical image annotation refinement workflow for abdominal CT segmentation. It starts from selected PanTS/PAINTS tumor cases, runs one or more segmentation teacher models, normalizes model outputs into `segmentations/*.nii.gz`, evaluates predictions against available annotations, optionally sends uncertain masks to LabelCritic/VLM review, writes updated annotation artifacts, and prepares a selected-model-aware M-step for the next E-step round.

The code is designed for PanTS/PAINTS 50-case debugging first, then extension to larger cohorts when real checkpoints, data, and GPU environments are available.

## What this project does

```text
1. Prepare local checkpoints and PanTS/PAINTS CT data
2. Inspect or rebuild the model registry
3. Select or validate 50 tumor-positive PanTS/PAINTS cases
4. Run E-step Round 1 with selected segmentation models
5. Normalize model outputs into segmentations/*.nii.gz
6. Evaluate organs that have current annotations with true DICE/DSC
7. Keep organs without reference annotations as teacher-derived pseudo-label candidates
8. Queue low-confidence or conflicting masks for LabelCritic/VLM/manual review when enabled
9. Save metrics, annotation versions, pseudo-label training data, and review artifacts
10. Run selected-model-aware M-step when the target model has a usable training backend
11. Run a later E-step round and compare metrics with convergence-table
```

Intended loop:

```text
PanTS/PAINTS selected cases
        |
        v
Model registry chooses candidate teachers per organ
        |
        v
E-step: ePAI / VSmTrans / TotalSegmentator / VISTA3D / private nnUNet-style entries
        |
        v
Mask normalization + optional post-processing/review
        |
        v
GT organs: true DICE/DSC gate
Non-GT organs: teacher-derived pseudo-label candidates and surrogate checks
        |
        v
Annotation update + review queue + RadThinking-style traces + training_manifest.json
        |
        v
M-step: prepare or run selected-model-aware update for a trainable target model
        |
        v
Updated checkpoint can be returned to the next E-step candidate pool if training succeeds
```

## 1. Prerequisites

Use a Linux machine or GPU server with:

- Python environment that can run this repository's CLI.
- NVIDIA GPU for real model inference/training.
- CUDA-compatible PyTorch / nnUNet v2 environment for nnUNet-style models such as ePAI.
- MONAI/VISTA3D environment if using VISTA3D fine-tuning.
- Local PanTS/PAINTS CT data in NIfTI format (`.nii.gz`).
- Local checkpoints for private or large models you want to run.
- Optional LabelCritic/VLM server on `localhost:8000` for real visual review.

`--dry-run`, `mock_seg`, and `--critic-backend stub` remain in the code only for developer smoke tests and offline command validation. They are disabled for formal experiments: do not use or report dry-run, mock, or stub outputs as project results.

## 2. Required local files

Large/private files are not bundled. Place them locally as needed:

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

## 3. Install and verify the CLI

From the repository root:

```bash
cd /path/to/medical_agent
pip install -e agent-harness
```

Verify that the CLI can load:

```bash
python run_medai_cli.py --json doctor
```

## 4. Inspect the model registry

The workflow is registry-driven. The registry maps organs to candidate model families and records conditional M-step trainability metadata.

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

The current registry includes model keys such as:

```text
airrc, atlasnet, atm, cads, dap, duke, epai_20250421, epai_finetuned,
goacc, mock_seg, moose, moose3_0, nnunet_private, pedro, saros_nnunet,
totalsegmentator, unest, vista3d, vsmtrans, vsnet
```

`mock_seg` is a developer-only smoke-test entry. It is not part of formal experiments and must not be reported as a real segmentation teacher.

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

The selector/validator checks that selected cases have usable CT paths, label folders, required organ masks, and non-empty pancreatic lesion/tumor masks unless smoke-test options are explicitly used.

## 6. Run E-step Round 1

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

For a developer-only offline command check, use a separate test output folder and do not include the results in formal metrics:

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models mock_seg \
  --organs pancreas,liver,aorta \
  --output outputs/dry_run_multimodel \
  --dry-run
```

### Current E-step behavior

For every case and every requested model, the loop:

1. Loads the CT path and current annotation folder from the case-list CSV.
2. Runs the selected registry model wrapper.
3. Converts each model's output into a normalized `segmentations/*.nii.gz` folder.
4. Computes DICE/DSC for organs that have reference annotation masks.
5. Applies the quality gate for annotated organs:
   - `DICE >= 0.8`: accept.
   - `0.5 <= DICE < 0.8`: uncertain, keep for review.
   - `DICE < 0.5`: send to LabelCritic/VLM if enabled.
6. Keeps candidate masks for requested organs without usable reference masks as pseudo-label candidates rather than true GT DSC results.
7. Writes annotation versions, review queues, VLM decisions, routing metadata, and a training manifest.

Important implementation note: the standalone `postprocess` command supports ShapeKit through `shapekit_runner.py`, and `run-loop` exposes `--enable-shapekit`. In the current `run_multimodel_annotation_loop` implementation, the ShapeKit call inside the loop should be verified or corrected before claiming that every successful prediction has been ShapeKit-postprocessed. Treat ShapeKit-in-loop as intended/conditional unless the run artifacts confirm it.

## 7. Ground-truth and non-ground-truth organ handling

The workflow distinguishes between organs with current reference annotations and organs without current reference annotations.

For organs with reference masks, predictions are compared against the current annotation using true DICE/DSC. These organs can be used for real Round 1 / Round 2 / later-round quality comparison.

For organs without reference masks, the pipeline cannot compute true ground-truth DSC. It can still retain teacher outputs as pseudo-label candidates for distillation. Downstream checks for those organs should be interpreted as teacher-reference or sanity metrics, such as teacher/student overlap and volume consistency, not as human-verified ground-truth DSC.

Do not report teacher-derived pseudo-label performance as human-verified ground-truth performance unless the masks have been manually reviewed, externally labeled, or otherwise validated.

## 8. Expected E-step output

A full `run-loop` output folder can contain:

```text
outputs/run_pants50_round1/
  dice_metrics.csv              # per-case/per-organ/per-model DICE decisions where reference masks exist
  round_metrics.csv             # aggregated round metrics
  inference_results.json        # raw inference status/result records
  review_queue.jsonl            # uncertain masks for ITK-SNAP/manual review
  vlm_decisions.jsonl           # LabelCritic/VLM decisions when enabled
  report_supervision.jsonl      # report supervision records when available
  annotation_versions/          # accepted or updated annotation versions
  cases/                        # per-case working artifacts
  critic/                       # LabelCritic artifacts when the real critic runs
  patient_traces.jsonl          # RadThinking-style structured/template traces when produced
  training_manifest.json        # cases/masks used by the M-step
  mstep_config.json             # selected M-step configuration
  mstep_model_routing.json      # selected primary/auxiliary model routing for M-step decisions
  run_summary.json              # machine-readable summary
```

The most important file for the next stage is:

```text
outputs/run_pants50_round1/training_manifest.json
```

## 9. Run selected-model-aware M-step

The M-step does not blindly train a generic model. It prepares or runs an update for the selected target model family when the required training state and environment exist.

Example ePAI update:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_round1/training_manifest.json \
  --output-folder outputs/mstep_round1 \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --pretrained-weights checkpoints/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth
```

Dry-run before spending GPU time:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_round1/training_manifest.json \
  --output-folder outputs/mstep_round1_dryrun \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --dry-run
```

M-step dry-run writes a training plan such as `mstep_training_plan.json`. Real training writes `mstep_training_result.json` and only reports an updated checkpoint if the backend actually creates a `checkpoint_*.pth` file under the expected result folder.

## 10. Conditional M-step backends

The registry includes selected-model-aware trainability metadata:

- `epai_20250421` is the preferred trainable target for pancreas / pancreatic duct / pancreatic tumor tasks when its nnUNet-compatible checkpoint and training state are mounted.
- `cads`, `moose`, `moose3_0`, `vsmtrans`, `nnunet_private`, `saros_nnunet`, and `atlasnet` are conditional M-step targets when their compatible training state exists.
- `totalsegmentator` is a public E-step baseline and has a conditional TotalSegmentator-style public nnUNet recipe entry. This does not claim to reproduce the official released TotalSegmentator model, because official training used additional non-public data.
- `vista3d` is a high-resource foundation candidate with a conditional MONAI bundle fine-tuning backend when the VISTA3D checkpoint, datalist, MONAI environment, and GPU resources are available.
- `unest` and template/private families remain inference or external-training candidates unless their original training recipes are provided.
- `mock_seg` is not trainable and is developer-only smoke-test infrastructure, not a formal teacher model.

Example developer-only backend dry-runs:

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

## 11. Run E-step Round 2 and compare metrics

After M-step produces a usable checkpoint, run another E-step with the same case list and compare metrics:

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

Build a convergence table:

```bash
python run_medai_cli.py --json convergence-table \
  --round-csvs outputs/run_pants50_round1/round_metrics.csv,outputs/run_pants50_round2/round_metrics.csv
```

For annotated organs, compare true DSC across rounds. For non-GT organs, use teacher-reference or volume sanity metrics and clearly label them as surrogate metrics.

## 12. One-command EM loop scripts

The repository includes orchestration scripts for longer GPU-server workflows:

```bash
bash run_em_loop.sh
bash run_m_e2_loop.sh
```

Check each script before running to confirm the exact pre-flight checks, model list, output paths, and log locations for your server.

## 13. Manual review with ITK-SNAP

After `run-loop`, generate review commands for uncertain cases:

```bash
python run_medai_cli.py --json itksnap-review \
  --review-queue outputs/run_pants50_round1/review_queue.jsonl \
  --output-script outputs/run_pants50_round1/open_itksnap_review.sh \
  --max-cases 10
```

Run the generated script locally on a workstation with ITK-SNAP installed to inspect CT, current annotation, and candidate masks.

## 14. Build teacher-facing samples

To export per-case samples with candidate masks, DICE scores, VLM decision, final annotation, and RadThinking-style traces:

```bash
python run_medai_cli.py --json build-samples \
  --run-output outputs/run_pants50_round1 \
  --output-jsonl outputs/run_pants50_round1/case_samples.jsonl
```

`patient_traces.jsonl` and sample traces are rule-based structured traces with deterministic template narrative fields. They are not validated VLM-authored clinical reasoning unless a future real VLM prose backend is explicitly added and documented.

## 15. Minimal verification

```bash
python scripts/verify_final_v4_integrity.py
python run_medai_cli.py --json registry-candidates \
  --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex
```

Expected behavior: the CLI returns JSON, registry lookups succeed, ePAI appears for its supported 25-class labels when configured, and `mock_seg` is excluded from formal candidate lists.

## 16. Real run sequence

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

For formal runs, do not use `--dry-run`, `mock_seg`, or `--critic-backend stub`. Those paths remain developer-only and must not be included in reported metrics or LabelCritic/VLM results.

## 17. Documentation status notes

The README intentionally avoids fixed claims such as exact GT/non-GT organ counts or total teacher capability counts unless those numbers are generated by a current script or stored in a current config. If a future protocol requires fixed numbers such as GT organ count, non-GT organ count, teacher count, or capability count, add the authoritative data source or statistic script first, then cite that source here.

See also:

- `docs/SELECTED_MODEL_AWARE_MSTEP_V7.md`
- `docs/MODEL_INVENTORY_AND_TRAINABILITY_V7.md`
