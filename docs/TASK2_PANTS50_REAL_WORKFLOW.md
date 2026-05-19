# Task 2 PanTS-50 Real Workflow Guide

This package continues the previous CLI prototype and integrates the latest Google Drive lightweight checkpoint export plus the PanTS repository files.

## What is now implemented

1. Registry-driven multi-model inference is implemented.
   - `configs/model_registry.yaml` is generated from `configs/class_checkpoint_map.xlsx`.
   - The registry now contains real command templates for the checkpoint families exposed by `drive_lightweight_sources.zip`:
     - `cads` -> CADS_series Dataset551 nnUNet v2 abdomen model
     - `epai_20250421` -> qchen76_2025_0421 Dataset1017 ePAI 25-class model
     - `vsmtrans` -> VSmTrans Dataset001 BDMAP model
     - `nnunet_private` -> nnUNet_private Dataset224 AbdomenAtlas model
     - `moose` / `moose3_0` -> available MOOSE Dataset888 cardiac/vascular model
     - `saros_nnunet` -> nnUNet_private Dataset1345 SAROS model
   - The wrapper `scripts/nnunetv2_predict_and_split.py` prepares `_0000.nii.gz` input, runs `nnUNetv2_predict`, and splits the combined label map into `segmentations/*.nii.gz`.

2. The PanTS case-selection logic is no longer random.
   - `scripts/select_pants50_cases.py` selects cases only after validating:
     - CT exists;
     - reference `segmentations/` exists;
     - required organ masks exist;
     - `pancreatic_lesion.nii.gz` is non-empty unless explicitly disabled for smoke testing.
   - Output: `data_manifest/case_list_50_tumor.csv`.

3. The DICE verifier now matches the teacher's thresholds.
   - `DSC >= 0.80` -> accept.
   - `0.50 <= DSC < 0.80` -> uncertain / manual sanity check.
   - `DSC < 0.50` -> LabelCritic / VLM review.
   - `DSC = 0` with empty reference and non-empty prediction -> replacement candidate.

4. ShapeKit remains a core stage in `run-loop`.
   - Raw model output -> normalized `segmentations/*.nii.gz` -> ShapeKit -> DICE before/after -> LabelCritic if needed.

5. LabelCritic remains wrapped through the `critic` command.
   - Stub backend works for dry-run.
   - Real backend calls the bundled `third_party/LabelCritic-main` structure when a VLM server is available.

6. The M-step interface is present.
   - `training_manifest.json` and `mstep_config.json` are produced after `run-loop`.
   - This is intentionally an interface, not a claimed scientific retraining result from 50 cases.

7. RadThinking-style trace generation is integrated.
   - `patient_traces.jsonl` is created during the loop when real case paths and masks are available.

## What cannot be completed inside this chat sandbox

The actual PanTS CT dataset and labels are not inside the uploaded `PanTS-main.zip`. That zip contains the repository, README, and download scripts only. The official PanTSMini image archive is large: image tar blocks are around 28-34GB each and the total dataset is around 300GB+.

Therefore, this package cannot include real CT files, real labels, or true numeric DICE results. You must run the download/selection commands below in Colab/Linux with enough storage.

## Step 1: Prepare checkpoints

Put the teacher Drive folders under the project root:

```text
checkpoints/
  CADS_series/
  MOOSE_series/
  nnUNet_private/
  UNEST/
  VSmTrans/
  class_checkpoint_map.xlsx
```

Do not commit this folder to public GitHub.

If the checkpoints are too large to copy locally, keep the original teacher
folder untouched and link local shortcuts into this project instead. Put the
Windows `.lnk` shortcuts in one folder, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/link_checkpoint_shortcuts.ps1 `
  -ShortcutRoot "D:\path\to\teacher_checkpoint_shortcuts" `
  -LinkRoot checkpoints
```

This creates junctions only under this project's `checkpoints/` folder and
writes `configs/checkpoint_links_manifest.json`. It does not rename, delete,
move, or edit the source checkpoint folders. Keep all inference outputs under
`outputs/`; the registry runner refuses to write outputs inside a checkpoint
path.

## Step 2: Download only the needed PanTS data blocks

Because the image data is distributed in 1000-case tar blocks, the practical minimal plan is to start with the first 50 training candidates, which all sit in the first image block. They are not treated as final selected cases until labels are checked.

```bash
python scripts/download_pants50_selective.py \
  --pants-root third_party/PanTS-main \
  --metadata \
  --download-images \
  --download-labels \
  --yes
```

This will use the planned candidate list `PanTS_00000001` to `PanTS_00000050`. It minimizes the number of image blocks, but it still downloads the first large image tar and the label archive.

If you already have a preferred candidate list:

```bash
python scripts/download_pants50_selective.py \
  --pants-root third_party/PanTS-main \
  --case-list data_manifest/case_list_50_planned_minimal_block.csv \
  --metadata \
  --download-images \
  --download-labels \
  --yes
```

## Step 3: Select the validated 50 tumor cases

After the CT and labels are extracted, run:

```bash
python run_medai_cli.py --json pants-select-50 \
  --pants-root third_party/PanTS-main \
  --output data_manifest/case_list_50_tumor.csv \
  --split train \
  --num-cases 50
```

This is the formal selection step. It will not randomly pick cases. It validates non-empty `pancreatic_lesion.nii.gz` and required organ masks.

If fewer than 50 are found, download/extract more training blocks, then rerun the selector.

## Step 4: Run one-case smoke test first

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models epai_20250421,cads,vsmtrans,nnunet_private,totalsegmentator \
  --organs pancreas,pancreatic_lesion,liver,spleen,kidney_left,kidney_right,colon,duodenum,stomach,aorta,postcava \
  --output outputs/run_pants50_smoke \
  --enable-shapekit \
  --enable-critic \
  --critic-backend stub \
  --timeout-sec 1800
```

For the first run, keep `--critic-backend stub`. After the segmentation and DICE stages work, switch to:

```bash
--critic-backend labelcritic
```

only after your VLM server is running.

## Step 5: Run the full 50-case debug workflow

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models epai_20250421,cads,vsmtrans,nnunet_private,totalsegmentator \
  --organs pancreas,pancreatic_lesion,liver,spleen,kidney_left,kidney_right,colon,duodenum,stomach,aorta,postcava \
  --output outputs/run_pants50_real \
  --enable-shapekit \
  --enable-critic \
  --critic-backend stub \
  --timeout-sec 3600
```

Expected outputs:

```text
outputs/run_pants50_real/
  dice_metrics.csv
  round_metrics.csv
  review_queue.jsonl
  vlm_decisions.jsonl
  annotation_versions/
  patient_traces.jsonl
  training_manifest.json
  mstep_config.json
  inference_results.json
  run_summary.json
```

## Step 6: ITK-SNAP sanity check

Open a few cases from `review_queue.jsonl` in ITK-SNAP:

- Image: `ct_path` from `case_list_50_tumor.csv`.
- Segmentation: corresponding mask under `outputs/run_pants50_real/annotation_versions/<case_id>/updated/`.
- Compare with candidate masks under `outputs/run_pants50_real/cases/<case_id>/raw_predictions/` and `refined_predictions/`.

Save 2-3 screenshots into:

```text
outputs/run_pants50_real/itk_snap_examples/
```

## What to report to the teacher

You should report:

1. `case_list_50_tumor.csv`: 50 selected cases are validated by non-empty pancreatic lesion masks and required organ labels.
2. `model_registry.yaml`: organ-to-candidate model routing.
3. `dice_metrics.csv`: raw and ShapeKit-refined DICE per case/organ/model.
4. `round_metrics.csv`: low-quality masks and uncertain masks after the loop.
5. `review_queue.jsonl` and `vlm_decisions.jsonl`: LabelCritic/VLM routing decisions.
6. `annotation_versions/`: updated organ masks.
7. `patient_traces.jsonl`: RadThinking-style reasoning trace samples.
8. ITK-SNAP screenshots for 2-3 uncertain cases.
