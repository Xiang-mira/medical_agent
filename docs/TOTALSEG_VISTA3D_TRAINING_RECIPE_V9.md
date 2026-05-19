# V9: TotalSegmentator and VISTA3D training/fine-tuning recipe integration

## Why this update was needed

The previous inventory was too conservative: it marked TotalSegmentator and VISTA3D as inference-only in the current project. After inspecting the two uploaded source packages, both have training-related recipes and should be represented as selectable, conditional M-step backends.

## TotalSegmentator

Bundled path:

```text
third_party/TotalSegmentator-master/
```

Training-recipe files verified in the package:

```text
third_party/TotalSegmentator-master/resources/train_nnunet.md
third_party/TotalSegmentator-master/resources/train_nnunet.sh
third_party/TotalSegmentator-master/resources/convert_dataset_to_nnunet.py
third_party/TotalSegmentator-master/totalsegmentator/custom_trainers.py
```

How it is exposed in this project:

```yaml
totalsegmentator:
  trainable: conditional_public_nnunet_recipe
  mstep_backend: totalseg_public_nnunet
```

Important limitation: this supports a TotalSegmentator-style public nnUNet training/fine-tuning plan. It does not claim to fully reproduce released TotalSegmentator v2, because the official v2 model used additional non-public training data.

Dry-run command:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_real/training_manifest.json \
  --output-folder outputs/mstep_totalseg_round1 \
  --target-model totalsegmentator \
  --dry-run
```

## VISTA3D

Bundled path:

```text
third_party/VISTA3D-Inference-Pipeline-master/
```

Training/fine-tuning recipe files verified in the package:

```text
third_party/VISTA3D-Inference-Pipeline-master/configs/train.json
third_party/VISTA3D-Inference-Pipeline-master/configs/train_continual.json
third_party/VISTA3D-Inference-Pipeline-master/configs/multi_gpu_train.json
third_party/VISTA3D-Inference-Pipeline-master/scripts/trainer.py
```

How it is exposed in this project:

```yaml
vista3d:
  trainable: conditional_monai_bundle_finetune
  mstep_backend: monai_bundle_finetune
```

Important limitation: this supports MONAI bundle fine-tuning/continual learning when a VISTA3D checkpoint, MONAI environment, datalist, and GPU resources are available. It does not claim to reproduce the original VISTA3D foundation model from scratch.

Dry-run command:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_real/training_manifest.json \
  --output-folder outputs/mstep_vista3d_round1 \
  --target-model vista3d \
  --pretrained-weights /path/to/vista3d_checkpoint.pt \
  --dry-run
```

## Selected-model-aware loop policy

- Use `route-models` to select the primary model for each organ/task.
- Use `mstep-update --target-model <model>` to update the selected model family whenever it is trainable or conditionally trainable.
- For PanTS pancreas-related tasks, `epai_20250421` remains the preferred primary M-step target.
- TotalSegmentator and VISTA3D are now available as optional selected targets when the user intentionally chooses those families and has the required environment.
