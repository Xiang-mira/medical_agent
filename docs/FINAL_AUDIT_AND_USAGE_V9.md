# Final audit and usage notes — V9

## Status

V9 keeps the selected-model-aware EM design and updates model trainability after inspecting the uploaded TotalSegmentator and VISTA3D source packages.

## What changed from V8

1. `totalsegmentator` is no longer classified as `not_trainable_in_current_project`. It is now `conditional_public_nnunet_recipe` with backend `totalseg_public_nnunet`.
2. `vista3d` is no longer classified as `external_training_required`. It is now `conditional_monai_bundle_finetune` with backend `monai_bundle_finetune`.
3. `mstep-update --target-model totalsegmentator` now creates a TotalSegmentator-style nnUNet M-step plan.
4. `mstep-update --target-model vista3d` now creates a MONAI bundle fine-tuning plan and datalist.
5. Verification now checks both recipe integrations.

## Still not claimed

- We do not claim to reproduce released TotalSegmentator v2.
- We do not claim to reproduce VISTA3D foundation-model pretraining.
- We do not claim real model improvement until the 50 PanTS tumor cases, checkpoints, VLM server, and GPU environment are used to run the real loop.

## Key commands

```bash
python scripts/verify_final_v9_integrity.py
python run_medai_cli.py --json model-inventory
python run_medai_cli.py --json route-models --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex
python run_medai_cli.py --json mstep-update --training-manifest outputs/run_pants50_real/training_manifest.json --output-folder outputs/mstep_totalseg --target-model totalsegmentator --dry-run
python run_medai_cli.py --json mstep-update --training-manifest outputs/run_pants50_real/training_manifest.json --output-folder outputs/mstep_vista3d --target-model vista3d --dry-run
```

On Windows, keep `--output-folder` short for M-step dry-runs and training plans,
especially for TotalSegmentator-style nnUNet plans. Deep project paths can hit
the legacy MAX_PATH limit. A short path such as
`C:\Users\<you>\Documents\Codex\mstep_ts` is safer than a deeply nested
`outputs/.../totalseg...` path.
