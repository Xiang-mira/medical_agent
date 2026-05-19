# LabelCritic Projection Update: replacing naive 3D-to-2D projection

## Why this change was required

In the meeting, the VLM label expert was flagged because the previous 3D-to-2D conversion was too simple: it effectively relied on a simple averaged/slice-like view. The teacher specifically said that LabelCritic already implements a more appropriate projection and that the projection should treat bone and soft-tissue/organs differently so the rendered 2D images are clearer for VLM comparison.

## What changed in this version

The project now defaults to LabelCritic's projection pipeline rather than the earlier local projection fallback.

Implemented files:

- `agent-harness/cli_anything/medai/core/labelcritic_projection_runner.py`
- `agent-harness/cli_anything/medai/core/projection_builder.py`
- `agent-harness/cli_anything/medai/core/labelcritic_wrapper.py`
- `agent-harness/cli_anything/medai/medai_cli.py`

Teacher-provided LabelCritic files used:

- `third_party/LabelCritic-main/ProjectDatasetFlex_single.py`
- `third_party/LabelCritic-main/projection.py`
- `third_party/LabelCritic-main/CompareOrgan.py`
- `third_party/LabelCritic-main/RunAPI_single.py`

## New projection behavior

The default command:

```bash
python run_medai_cli.py --json projection-build \
  --ct-image case/ct.nii.gz \
  --annotation-a outputs/model_a/segmentations/pancreas.nii.gz \
  --annotation-b outputs/model_b/segmentations/pancreas.nii.gz \
  --organ pancreas \
  --output-folder outputs/projections/pancreas
```

now uses:

```text
LabelCritic ProjectDatasetFlex_single.py
  -> projection.py
  -> CT windows for organs / bone / skeleton
  -> candidate A/B comparison projections
```

The local mask-centered slice code is kept only as a fallback:

```bash
python run_medai_cli.py --json projection-build ... --projection-backend slice
```

For normal work, keep the default:

```text
--projection-backend labelcritic
```

or use:

```text
--projection-backend auto
```

which tries LabelCritic first and falls back only if the LabelCritic projection command cannot run in the current environment.

## How LabelCritic is used in the workflow

The full review path is:

```text
DICE quality gate
  -> low-quality or conflicting masks
  -> LabelCritic projection
  -> VLM A/B comparison
  -> JSON decision
  -> annotation update
```

This directly addresses the teacher's request: the project no longer depends on a naive average projection for VLM review.
