# M-step Formal 10,000-Step Status and Repair — 2026-06-21

## Final status

- The original 10,000-step run completed in 3,401.647 seconds with 10,000
  finite loss entries, but **failed the holdout gate**: all 16 core-organ
  predictions across four validation cases were empty.
- The pretrained VoxTell baseline produced 16/16 nonempty masks on the same
  inputs, excluding CT data, prompt encoding, and inference I/O as the cause.
- Root cause: foreground-sparse binary targets made voxel-mean BCE background
  dominated; 10,000 updates with SGD momentum 0.99 and LR 1e-4 also moved the
  full spatial decoder far enough to cause catastrophic forgetting.
- Fix: foreground-balanced BCE, `prompt_path`-only adaptation, LR 1e-6, and an
  early 100-step holdout gate. The selected repair uses
  `--bce-pos-weight-cap 20`.
- Selected repaired inference model:
  `outputs/round1_373_hierarchical_repair_20260620/mstep/repair_balanced20_100steps/voxtell_finetuned_model`
- Four-case core holdout: 16/16 nonempty; mean pseudo-consistency Dice improved
  from 0.789187 (base) to 0.873903; 15/16 case-organ comparisons improved.
  The only regression was pancreas on PanTS_00000451, 0.847556 to 0.843440.
- These Dice values are consistency against repaired machine labels, not
  expert-ground-truth accuracy. The student remains a candidate and must not
  automatically replace teacher/fusion outputs.

## Original job (completed, rejected checkpoint)

- PID at launch verification: `935792`
- Started: `2026-06-21 08:33 UTC`
- GPU: active, approximately 10.3 GB allocated; encoder frozen.
- Output: `outputs/round1_373_hierarchical_repair_20260620/mstep/formal_10000steps`
- Train/validation split: 16/4 cases, case-disjoint.
- Training rows before sampler expansion: 19,499.
- Optimizer: SGD; LR: `1e-4`; target steps: 10,000.
- CT encoder frozen; Qwen text encoder frozen; prompt projection/decoder trainable.
- Recovery checkpoint: rotating `checkpoint_latest.pth` every 1,000 steps.
- Final artifacts: `model_finetune.pth`, VoxTell-compatible
  `voxtell_finetuned_model/fold_0/checkpoint_final.pth`, and
  `loss_history.json`.

Command:

```bash
python scripts/train_voxtell_prompt_student.py \
  --manifest outputs/round1_373_hierarchical_repair_20260620/mstep/splits/train_manifest.json \
  --model-dir checkpoints/VoxTell/voxtell_v1.1 \
  --text-encoding-model checkpoints/Qwen/Qwen3-Embedding-4B \
  --embedding-cache outputs/round1_373_hierarchical_repair_20260620/mstep/smoke_10steps/prompt_embeddings.pt \
  --output-dir outputs/round1_373_hierarchical_repair_20260620/mstep/formal_10000steps \
  --epochs 1 --max-steps 10000 --learning-rate 1e-4 --optimizer sgd \
  --freeze-encoder --save-every 1000 --device cuda
```

## Completed gates before formal launch

- 16/4 case split: no overlap; prompt variants case-locked.
- Positive A/B/C weights: exactly 1.0/0.5/0.1.
- 10-step smoke: finite loss, no OOM/NaN, checkpoint written and reloaded.
- Holdout core inference: liver, pancreas, kidney_left, kidney_right all
  successful and nonzero.
- Training I/O optimized with image/mask LRU cache and mask-locality shuffle.
- Disk protection: one rotating recovery checkpoint rather than accumulating
  1.7-GB files every 1,000 steps.

The first 1,000 steps of this formal run are also the pilot milestone. At that
checkpoint, inspect loss trend and checkpoint integrity; do not treat an
unfinished or failed milestone as a final model.

## Remaining validation before broad Round2 use

1. Report geometry, volume, parent containment, A/B/C strata,
   core organs/sub-organs, prompt variants, and allowed negative prompts.
2. Add student as a Round2 candidate only; it must not automatically replace
   teacher or fusion outputs.
