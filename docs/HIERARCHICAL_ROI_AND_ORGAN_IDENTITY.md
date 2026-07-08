# Hierarchical ROI and strict organ identity

The updated workbook is stored at `configs/class_checkpoint_map_updates.xlsx`.
`scripts/build_organ_taxonomy.py` converts its `Major Organ` and
`Primary organ involved` columns into `configs/organ_taxonomy.json`.

The hierarchy is used only for inference order and ROI construction. It never
defines label equivalence. Every Dice, fusion, LabelCritic, dashboard, and
M-step record uses an exact canonical organ identity. For example,
`liver_segment_1`, `liver`, `pancreas_head`, and `pancreas` are four distinct
targets. Parent masks cannot substitute for child masks.

## CPU-safe checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=agent-harness \
  python scripts/build_organ_taxonomy.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=agent-harness \
  python scripts/audit_organ_mappings.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=agent-harness \
  python scripts/audit_organ_identity.py
```

The legacy Round 1 audit never edits existing outputs. It writes a contamination
report and a separate sanitized manifest. Records without exact source-label
provenance are `legacy_unverified` and are excluded from strict M-step manifests.

## New Round 1 mode

```bash
python -m cli_anything.medai.medai_cli run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --output outputs/new_round1 \
  --teacher-inference-mode hierarchical_roi \
  --roi-margin-mm 20
```

Major organs run on the full CT. Child structures run on parent-mask ROIs and
are restored to the original CT geometry. Primary teachers run first; backups
run only when the prior result is unavailable. A missing parent blocks its
children instead of silently falling back to full-volume child inference.

When one model serves children from nearby but different parents, the scheduler
may merge the crops to avoid loading the same checkpoint twice. Each child keeps
its own 20 mm parent-support box. At least 95% of a merged-crop prediction must
remain inside that child's support box; otherwise the prediction is rejected and
only that child is rerun on its independent parent ROI. This prevents a pancreas
child from being accepted in the liver region while preserving the speed benefit
when the merged prediction remains anatomically scoped. Hard QC and exact
canonical identity checks still apply after restoration.

Round 1 teacher caches remain bootstrap/debug artifacts. In the repaired formal
Round2+ EM path, the previous selected pseudo label is carried as the immutable
pseudo reference, and the previous round cleaned student prediction enters only
as a verifier-gated candidate. Student inference still runs after each M-step
because the student checkpoint changes.

## GPU resource gate

Before any GPU smoke test, real inference, or benchmark:

```bash
PYTHONPATH=agent-harness python scripts/check_gpu_resources.py
```

The gate returns `skipped_resource_busy` when another compute process is
present or GPU utilization is above the configured limit. It never kills,
pauses, or changes the priority of another process.
