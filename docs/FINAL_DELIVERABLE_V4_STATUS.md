# Final deliverable v4 status

## What v4 fixes compared with v3

v3 already had the multi-model registry, ePAI export-label preservation patch, ATLAS-Net, VISTA3D, UNEST, TotalSegmentator source, ShapeKit, LabelCritic, PanTS utilities, RadThinking trace, and M-step manifest interface.

v4 adds the key missing correction from the meeting:

> The VLM label expert no longer relies on a naive 3D-to-2D average/slice projection. It now uses the teacher-provided LabelCritic projection path by default.

## Main v4 changes

- Added `labelcritic_projection_runner.py`.
- Updated `projection_builder.py` so the default backend is `labelcritic`.
- Updated `projection-build` CLI with `--projection-backend labelcritic/auto/slice`.
- Updated `labelcritic_wrapper.py` to build LabelCritic-style projections before VLM comparison.
- Added `LABELCRITIC_PROJECTION_UPDATE.md`.
- Added `TASK_COMPLETION_MATRIX_V4.md`.
- Added `verify_final_v4_integrity.py`.
- Adjusted `agent-harness/requirements.txt` to avoid automatically installing large/unstable real inference dependencies.

## Validation command

```bash
python scripts/verify_final_v4_integrity.py
```

Expected output:

```json
{
  "status": "success",
  "missing": [],
  "epai_label_scope_is_conservative": true,
  "labelcritic_projection_present": true,
  "registry_models_ok": true
}
```

## Real experiment command sketch

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models epai_20250421,cads,vsmtrans,nnunet_private,totalsegmentator \
  --organs pancreas,pancreatic_lesion,liver,spleen,kidney_left,kidney_right,colon,duodenum,stomach,aorta,postcava \
  --output outputs/run_pants50_real \
  --enable-shapekit \
  --enable-critic \
  --critic-backend labelcritic
```

For early smoke tests, use `--critic-backend stub`, then switch to `labelcritic` once the VLM server is available.
