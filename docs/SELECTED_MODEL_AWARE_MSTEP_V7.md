# Selected-Model-Aware M-step (v7)

This version corrects the earlier ambiguity in the EM-loop design.

## Correct interpretation

The M-step should not be described as "we retrain TotalSegmentator" by default. In this project:

- **TotalSegmentator** is a public baseline and E-step candidate.
- The **selected trainable model family** should be the M-step target.
- If the selected model is not trainable with the currently available files, it remains an E-step candidate only.

The loop is therefore:

```text
Task/organ routing
→ choose primary E-step model and auxiliary candidates
→ model inference
→ ShapeKit + DICE + LabelCritic + optional human review
→ updated annotations
→ update the selected model family if trainable
→ add the updated checkpoint to the next E-step candidate pool
```

## New commands

### Inspect all models and trainability

```bash
python run_medai_cli.py --json model-inventory
```

### Route tasks/organs to primary models

```bash
python run_medai_cli.py --json route-models \
  --organs pancreas,liver,aorta,kidney_cortex
```

### Selected-model-aware M-step

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_real/training_manifest.json \
  --output-folder outputs/mstep_epai_round1 \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --max-epochs 5 \
  --dry-run
```

For true fine-tuning from a specific checkpoint file:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_real/training_manifest.json \
  --output-folder outputs/mstep_epai_round1 \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --pretrained-weights checkpoints/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth
```

## M-step policies

| Model | M-step policy |
|---|---|
| ePAI_20250421 | Preferred trainable target for pancreas / pancreatic duct / pancreatic tumor tasks if the nnUNet checkpoint is mounted. |
| CADS | Conditional trainable target for broad abdominal organs if full nnUNet training state exists. |
| MOOSE / MOOSE3.0 | Conditional trainable target if the relevant nnUNet-compatible checkpoint/training state exists. |
| VSmTrans | Conditional trainable target for abdominal organ tasks if full nnUNet state exists. |
| nnUNet_private / SAROS | Conditional private trainable backend. |
| ATLAS-Net | Conditional nnUNet v2 trainable backend if the downloaded ATLAS-Net checkpoint/training state is available. |
| TotalSegmentator | Public baseline and E-step candidate only in this project; not the default M-step target. |
| VISTA3D | Foundation inference candidate; training requires external MONAI/VISTA3D training recipe. |
| UNEST | Kidney substructure inference candidate; training wrapper not included. |
| AirRC / ATM / DAP / Duke / GOACC / Pedro / VSNet | Template-only entries from class_checkpoint_map.xlsx; no runnable training wrapper until teacher provides scripts/checkpoints. |

## Why this is different from the earlier generic nnUNet M-step

The earlier `mstep-train` command still exists as a low-level fallback, but the project-level command is now `mstep-update --target-model`. This makes the M-step semantically tied to the E-step model selection instead of silently training an unrelated generic model.
