# Validation report v3

This report records what was validated inside the final v3 package.

## Integrity checks

```bash
python scripts/verify_final_v3_integrity.py
```

Expected output:

```json
{
  "status": "success",
  "missing": [],
  "epai_label_scope_is_conservative": true,
  "registry_models_ok": true
}
```

## Registry checks

```bash
python run_medai_cli.py --json registry-candidates \
  --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex
```

Validated behavior:

- `pancreas` includes `vsmtrans`, `cads`, `epai_20250421`, `moose3_0`, `atlasnet`, and `vista3d`.
- `liver` includes `vsmtrans`, `cads`, `moose3_0`, `atlasnet`, and `vista3d`.
- `aorta` includes `vista3d`, `cads`, `moose3_0`, and `atlasnet`.
- `pancreatic_duct` includes `epai_20250421` and `atlasnet`.
- `kidney_cortex` includes `unest`.

## Dry-run command rendering

The following models render valid medai registry commands in dry-run mode:

```bash
python run_medai_cli.py --json infer --model epai_20250421 --input data/fake/ct.nii.gz --output outputs/test_epai --dry-run
python run_medai_cli.py --json infer --model atlasnet --input data/fake/ct.nii.gz --output outputs/test_atlas --dry-run
python run_medai_cli.py --json infer --model vista3d --input data/fake/ct.nii.gz --output outputs/test_vista --dry-run
python run_medai_cli.py --json infer --model unest --input data/fake/ct.nii.gz --output outputs/test_unest --dry-run
```

The ePAI command now includes:

```bash
--output-label-mode all_organs
```

## What cannot be validated in this chat environment

Real inference and real quality improvement require heavy external assets that are intentionally not bundled:

- PanTS NIfTI CT volumes and label folders.
- Private teacher checkpoints under `checkpoints/`.
- GPU/MONAI/nnUNet runtime for VISTA3D/ePAI/CADS/VSmTrans/UNEST.
- ITK-SNAP manual screenshots.

The code path is ready for those assets; the final numerical results must be produced on the user's Colab/server.
