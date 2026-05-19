# MedAI Agent Loop Task 2 – Final v7 Selected-Model-Aware Delivery


## V9 update: TotalSegmentator and VISTA3D are now selectable M-step backends

This version treats **TotalSegmentator** and **VISTA3D** as selectable model families in the registry, not just inference-only references. The distinction is precise:

- `totalsegmentator` remains a public E-step baseline, but it also exposes a **conditional TotalSegmentator-style public nnUNet training backend** through the bundled `third_party/TotalSegmentator-master/resources/train_nnunet.md`, `train_nnunet.sh`, and `convert_dataset_to_nnunet.py`. This does **not** claim to reproduce released TotalSegmentator v2, because the official v2 training used additional non-public data.
- `vista3d` remains a high-resource foundation candidate, but it also exposes a **conditional MONAI bundle fine-tuning backend** through `third_party/VISTA3D-Inference-Pipeline-master/configs/train.json`, `train_continual.json`, `multi_gpu_train.json`, and `scripts/trainer.py`. This supports fine-tuning/continual learning when a VISTA3D checkpoint, datalist, MONAI environment, and sufficient GPU memory are available.

Example dry-run checks:

```bash
python run_medai_cli.py --json mstep-update --training-manifest outputs/run_pants50_real/training_manifest.json --output-folder outputs/mstep_totalseg --target-model totalsegmentator --dry-run
python run_medai_cli.py --json mstep-update --training-manifest outputs/run_pants50_real/training_manifest.json --output-folder outputs/mstep_vista3d --target-model vista3d --dry-run
```


This v7 revision fixes the M-step semantics. The project no longer presents M-step as "training a generic new nnUNet regardless of the E-step model." Instead, it implements **selected-model-aware M-step**:

```text
choose task/organ-specific primary model
→ E-step inference and annotation refinement
→ update that selected model family if it is trainable
→ return the updated checkpoint to the next E-step candidate pool
```

Important clarification:

- `TotalSegmentator` is a **public baseline / E-step candidate**, not the default retraining target in this implementation.
- `ePAI_20250421` is the preferred trainable target for pancreas / pancreatic duct / pancreatic tumor tasks when the teacher's nnUNet-compatible checkpoint is mounted.
- CADS, MOOSE, VSmTrans, private nnUNet, SAROS, and ATLAS-Net are conditional M-step targets when their full nnUNet-compatible training state exists.
- VISTA3D, UNEST, and template-only families remain inference/external-training candidates unless their original training recipes are provided.

New commands:

```bash
python run_medai_cli.py --json model-inventory
python run_medai_cli.py --json route-models --organs pancreas,liver,aorta,kidney_cortex
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_real/training_manifest.json \
  --output-folder outputs/mstep_epai_round1 \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --dry-run
```

See:

- `docs/SELECTED_MODEL_AWARE_MSTEP_V7.md`
- `docs/MODEL_INVENTORY_AND_TRAINABILITY_V7.md`

---

# MedAI Agent Loop CLI — Final v6 Verified Delivery

This project is a registry-driven, multi-model medical annotation refinement workflow for the teacher's Task 2. It is designed for PanTS/PAINTS 50 tumor-case debugging first, then scalable extension to larger cohorts.

## Core loop

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
## What is implemented in source

| Module | Status | Notes |
|---|---:|---|
| Multi-model registry | Implemented | `configs/model_registry.yaml`, `configs/class_checkpoint_map.parsed.csv`; formal candidate lists exclude `mock_seg` unless `--include-mock` is used. |
| ePAI 2025-04-21 25-class output | Implemented | `qchen76_2025_0421.tar.gz` was verified as `Dataset1017_ePAI_3MM` with 25 foreground labels. MedAI calls it through `nnUNetv2_predict_from_modelfolder` and keeps all labels by default. |
| ATLAS-Net | Implemented wrapper | `configs/atlasnet_label_map.json`, `scripts/atlasnet_predict_and_split.py`. Real weights still required. |
| VISTA3D | Implemented wrapper | Bundled source and splitting wrapper. Real bundle/checkpoint and GPU environment still required. |
| UNEST | Implemented wrapper | Bundled teacher Drive lightweight script/config and wrapper. Real checkpoint environment still required. |
| TotalSegmentator | Bundled source + CLI runner | Source in `third_party/TotalSegmentator-master`; real use requires installation/weights. |
| ShapeKit | Core post-processing | Enabled by default in `run-loop` and `em-loop`. |
| DICE/DSC quality gate | Implemented | `>=0.8 accept`, `0.5–0.8 uncertain`, `<0.5 LabelCritic/review`. |
| LabelCritic projection | Implemented | Uses teacher-provided `ProjectDatasetFlex_single.py` + `projection.py`, not naive average projection. |
| LabelCritic/VLM comparison | Implemented wrapper | Default real backend is `labelcritic`; use `--critic-backend stub` only for offline dry-run. |
| Report supervision | Implemented | Single and batch CLI compare tumor mask presence/size with report text. |
| RadThinking-style sample | Implemented structure + template prose | Builds observation/temporal/context/conclusion objects and deterministic natural-language trace fields; VLM-generated clinical prose remains an optional future backend. |
| M-step | Implemented interface | Prepares nnUNet v2 raw dataset and `nnUNetv2_train` command. Real training must run on GPU; 50 cases are smoke-test only. |
| ITK-SNAP review | Implemented helper | Generates commands from `review_queue.jsonl`; screenshots must be captured manually. |

## What is not bundled

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

## Minimal verification

```bash
python scripts/verify_final_v4_integrity.py
python run_medai_cli.py --json registry-candidates --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex
```

Expected: `status: success`, ePAI 2025-04-21 appears for its verified 25-class labels, and no `mock_seg` in formal candidate lists. Use `--include-mock` only for smoke tests.

## Real run sequence

```bash
python scripts/validate_case_list_50.py --case-list data_manifest/case_list_50_tumor.csv
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models totalsegmentator,epai_20250421,cads,vsmtrans,nnunet_private \
  --organs pancreas,pancreatic_lesion,liver,spleen,kidney_left,kidney_right,colon,duodenum,stomach,aorta,postcava \
  --output outputs/run_pants50_real \
  --enable-shapekit \
  --enable-critic \
  --critic-backend labelcritic \
  --base-url http://localhost \
  --port 8000
```

For offline command checks only, add `--dry-run` or use `--critic-backend stub`. Do not report stub results as real LabelCritic/VLM results.

