# Task 2: Multi-Model Medical Annotation Refinement Workflow

This package upgrades the previous single-model CLI prototype into a registry-driven, multi-model medical annotation workflow.

```text
50 tumor cases -> multi-model E-step -> ShapeKit -> DICE gate
      ^                                      |
      |                                      v
      |                          LabelCritic/VLM + ITK-SNAP queue
      |                                      |
      |                                      v
      └──── selected-model-aware M-step <- updated annotations
```

## What is implemented

1. **Model registry** from `class_checkpoint_map.xlsx`
   - `configs/model_registry.yaml`
   - `configs/class_checkpoint_map.parsed.csv`
   - organ → candidate model routing, for example pancreas → VSmTrans / CADS / ePAI / MOOSE3.0 / ATLAS-Net / VISTA3D. `mock_seg` is hidden from formal candidate lists unless `--include-mock` is used.

2. **Unified registry-based inference**
   - One CLI entry can call different model wrappers through `--model`.
   - Public/local models can run directly when installed.
   - Private models are represented as editable command templates because private checkpoint folders are not bundled.

3. **ShapeKit as a core post-processing stage**
   - `postprocess` supports ShapeKit directly.
   - `run-loop` can apply ShapeKit after each successful model prediction.

4. **DICE/DSC verification**
   - `verify` checks prediction vs. reference masks.
   - `run-loop` writes `dice_metrics.csv` and `round_metrics.csv`.

5. **LabelCritic wrapper**
   - `third_party/LabelCritic-main` is bundled.
   - `critic` wraps LabelCritic-style A/B mask comparison.
   - Default backend is `labelcritic` for real runs; use `--backend stub` only for offline/dry-run command checks.

6. **Annotation update + M-step interface**
   - `run-loop` writes `annotation_versions/`, `training_manifest.json`, and `mstep_config.json`.
   - This makes the EM-style loop explicit, even before the lab training script is available.

7. **RadThinking-style trace output**
   - `run-loop` writes `patient_traces.jsonl` when real files are available.
   - The trace schema follows observation → temporal comparison → clinical context → conclusion.
   - The current generator also emits deterministic natural-language trace text; VLM-authored prose can be added later.

8. **ePAI 2025-04-21 registry scope**
   - `qchen76_2025_0421.tar.gz` was verified as `Dataset1017_ePAI_3MM` with background + 25 foreground labels.
   - ePAI is now registered for its full 25-class output: abdominal organs, vessels, pancreas, pancreatic duct, CBD/stent, and PDAC/cyst/PNET.
   - The wrapper uses `nnUNetv2_predict_from_modelfolder -m checkpoints/qchen76_2025_0421/...` and keeps `--output-label-mode all_organs` by default.

9. **Report-supervision utility preserved**
   - `scripts/r_super_pseudo_masks.py` is included for future report-supervised pseudo-mask construction.

## Key commands

Build or rebuild the registry:

```bash
python run_medai_cli.py --json build-registry \
  --checkpoint-map configs/class_checkpoint_map.xlsx \
  --output configs/model_registry.yaml \
  --parsed-csv configs/class_checkpoint_map.parsed.csv
```

Check model candidates for organs:

```bash
python run_medai_cli.py --json registry-candidates \
  --organs pancreas,liver,aorta
```

Run one model through the registry:

```bash
python run_medai_cli.py --json infer \
  --input data/case_001/ct.nii.gz \
  --output outputs/case_001/epai \
  --model epai_20250421 \
  --dry-run
```

Run ShapeKit post-processing:

```bash
python run_medai_cli.py --json postprocess \
  --input outputs/case_001/epai \
  --output outputs/case_001/epai_shapekit
```

Run DICE/DSC verification:

```bash
python run_medai_cli.py --json verify \
  --prediction outputs/case_001/epai_shapekit/segmentations \
  --reference data/case_001/reference/segmentations \
  --organs pancreas,liver,aorta \
  --output outputs/case_001/verify.json
```

Run LabelCritic wrapper:

```bash
python run_medai_cli.py --json critic \
  --ct data/case_001/ct.nii.gz \
  --mask-a outputs/case_001/model_a/segmentations/pancreas.nii.gz \
  --mask-b outputs/case_001/model_b/segmentations/pancreas.nii.gz \
  --organ pancreas \
  --output outputs/case_001/critic/pancreas.json \
  --backend labelcritic
```

Run the full 50-case loop:

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor_template.csv \
  --models totalsegmentator,epai_20250421,vista3d \
  --organs pancreas,liver,spleen,kidney_left,kidney_right,colon,duodenum,stomach,aorta,postcava \
  --output outputs/run_50_cases \
  --critic-backend labelcritic
```

For a safe command-generation test without real CT data or checkpoints:

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor_template.csv \
  --models mock_seg \
  --organs pancreas,liver,aorta \
  --output outputs/dry_run_multimodel \
  --dry-run
```

## What you still need to add locally

The following files are intentionally not bundled because they are large/private or were only shown as a Google Drive screenshot:

- `checkpoints/CADS_series/`
- `checkpoints/MOOSE_series/`
- `checkpoints/nnUNet_private/`
- `checkpoints/UNEST/`
- `checkpoints/VSmTrans/`
- ePAI 2025-04-21 checkpoint
- the real 50 PanTS/PAINTS CT cases and reference annotations

After placing those files locally, edit the relevant `command_template` fields in `configs/model_registry.yaml` to match each original repo's inference command.

## Expected run outputs

`run-loop` creates:

```text
outputs/run_50_cases/
  dice_metrics.csv
  round_metrics.csv
  review_queue.jsonl
  vlm_decisions.jsonl
  annotation_versions/
  patient_traces.jsonl
  training_manifest.json
  mstep_config.json
  run_summary.json
```

## Important limitation

This zip provides a runnable engineering workflow and command interface. It does not include private checkpoints or real PanTS/PAINTS CT volumes. Therefore, private models can be dry-run or command-template tested until the checkpoint folders are placed under `checkpoints/` and their command templates are edited.


## Final v4 notes

This package includes the v4 correction for the teacher meeting: the VLM/LabelCritic projection now defaults to LabelCritic `ProjectDatasetFlex_single.py` + `projection.py` rather than a naive average/slice projection. See `docs/LABELCRITIC_PROJECTION_UPDATE.md`, `docs/TASK_COMPLETION_MATRIX_V4.md`, and `docs/FINAL_DELIVERABLE_V4_STATUS.md`.


## Final v6 corrections

- Formal `registry-candidates` output no longer includes `mock_seg` unless `--include-mock` is explicitly passed.
- LabelCritic `--base-url` is normalized for the teacher-provided `RunAPI_single.py`; pass host only, for example `http://localhost`, because the script appends `:<port>/v1`.
- `critic` and `run-loop` default to `labelcritic`; stub mode is only for offline smoke tests.
- `report-supervision-batch` is available for the selected 50-case list.
- `build-samples` creates the teacher-expected per-case JSONL structure with candidate masks, dice scores, VLM decision, final annotation, and RadThinking-style trace.
- `scripts/validate_case_list_50.py` checks that the final 50-case manifest has CT paths, label folders, non-empty tumor masks, and required organ masks.
