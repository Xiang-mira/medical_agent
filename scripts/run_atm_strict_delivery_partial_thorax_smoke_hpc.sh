#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
CASE_ID=${CASE_ID:-BDMAP_00000120}
OUT_ROOT=${OUT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/atm_partial_thorax_strict_smoke_$(date +%Y%m%d_%H%M%S)}
REGISTRY=${REGISTRY:-configs/model_registry.yaml}
TIMEOUT_SEC=${TIMEOUT_SEC:-14400}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints}
NNUNETV2_PREDICT_EXECUTABLE=${NNUNETV2_PREDICT_EXECUTABLE:-/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict}

cd "$CODE_ROOT"
if [ "${SKIP_GIT_PULL:-0}" != "1" ]; then
  git pull --ff-only origin main
fi

RUN_OUT=$OUT_ROOT/run_loop
CASE_CSV=$OUT_ROOT/case_${CASE_ID}.csv
COMMAND_TXT=$OUT_ROOT/command.txt
mkdir -p "$RUN_OUT"
git rev-parse HEAD > "$OUT_ROOT/git_commit.txt"

"$PYTHON" - "$CASE_MANIFEST" "$CASE_CSV" "$CASE_ID" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out = Path(sys.argv[2])
case_id = sys.argv[3]
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
row = next((item for item in rows if (item.get("case_id") or item.get("id")) == case_id), None)
if row is None:
    raise SystemExit(f"case not found in manifest: {case_id}")
ct_path = row.get("ct_path") or row.get("image_path")
ref_dir = row.get("annotation_folder") or row.get("reference_mask_dir") or row.get("mask_dir")
if not ct_path:
    raise SystemExit(f"case {case_id} missing ct_path/image_path")
if not ref_dir:
    raise SystemExit(f"case {case_id} missing annotation_folder/reference_mask_dir")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as dst:
    writer = csv.DictWriter(dst, fieldnames=["case_id", "ct_path", "annotation_folder"])
    writer.writeheader()
    writer.writerow({"case_id": case_id, "ct_path": ct_path, "annotation_folder": ref_dir})
PY

cat > "$COMMAND_TXT" <<EOF
"$PYTHON" run_medai_cli.py --json run-loop \\
  --case-list "$CASE_CSV" \\
  --models atm \\
  --organs airway_tree \\
  --registry "$REGISTRY" \\
  --output "$RUN_OUT" \\
  --checkpoint-root "$CHECKPOINT_ROOT" \\
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE" \\
  --timeout-sec "$TIMEOUT_SEC" \\
  --teacher-inference-mode hierarchical_roi \\
  --no-enable-shapekit \\
  --debug-allow-no-shapekit \\
  --no-enable-critic \\
  --strict-delivery-targets \\
  --strict-delivery-fov-override-organs airway_tree \\
  --log-file "$RUN_OUT/run_loop.log"
EOF

set +e
"$PYTHON" run_medai_cli.py --json run-loop \
  --case-list "$CASE_CSV" \
  --models atm \
  --organs airway_tree \
  --registry "$REGISTRY" \
  --output "$RUN_OUT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --nnunet-predict-executable "$NNUNETV2_PREDICT_EXECUTABLE" \
  --timeout-sec "$TIMEOUT_SEC" \
  --teacher-inference-mode hierarchical_roi \
  --no-enable-shapekit \
  --debug-allow-no-shapekit \
  --no-enable-critic \
  --strict-delivery-targets \
  --strict-delivery-fov-override-organs airway_tree \
  --log-file "$RUN_OUT/run_loop.log"
RUN_RC=$?
set -e

[ -f "$RUN_OUT/run_summary.json" ] && cp "$RUN_OUT/run_summary.json" "$OUT_ROOT/run_summary.json"
[ -f "$RUN_OUT/annotation_versions/$CASE_ID/case_execution_plan.json" ] && cp "$RUN_OUT/annotation_versions/$CASE_ID/case_execution_plan.json" "$OUT_ROOT/case_execution_plan.json"
[ -f "$RUN_OUT/strict_delivery_failures.csv" ] && cp "$RUN_OUT/strict_delivery_failures.csv" "$OUT_ROOT/strict_delivery_failures.csv"
INFER_SUMMARY=$(find "$RUN_OUT/cases/$CASE_ID/raw_predictions" -path "*/atm/$CASE_ID/inference_summary.json" -print -quit 2>/dev/null || true)
if [ -n "$INFER_SUMMARY" ]; then
  cp "$INFER_SUMMARY" "$OUT_ROOT/inference_summary.json"
fi

export OUT_ROOT RUN_OUT CASE_ID RUN_RC
"$PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


out_root = Path(os.environ["OUT_ROOT"])
run_out = Path(os.environ["RUN_OUT"])
case_id = os.environ["CASE_ID"]
run_rc = int(os.environ["RUN_RC"])
summary = read_json(run_out / "run_summary.json")
plan = read_json(run_out / "annotation_versions" / case_id / "case_execution_plan.json")
infer = read_json(out_root / "inference_summary.json")
failures: list[str] = []

if run_rc != 0:
    failures.append(f"run_loop_return_code:{run_rc}")
if summary.get("status") != "success":
    failures.append(f"run_summary_status:{summary.get('status')}")
if int(summary.get("strict_delivery_failure_count") or 0) != 0:
    failures.append(f"strict_delivery_failure_count:{summary.get('strict_delivery_failure_count')}")
if int(summary.get("total_updated") or 0) != 1:
    failures.append(f"total_updated:{summary.get('total_updated')}")

teacher_run_list = set(plan.get("teacher_run_list") or [])
if "atm" not in teacher_run_list:
    failures.append("teacher_run_list_missing_atm")

override_rows = (plan.get("fov_override") or {}).get("rows") or []
applied = [
    row for row in override_rows
    if row.get("organ") == "airway_tree" and row.get("decision") == "applied"
]
if not applied:
    failures.append("airway_tree_override_not_applied")
else:
    row = applied[0]
    if row.get("original_visibility") != "partially_visible":
        failures.append(f"original_visibility_not_partial:{row.get('original_visibility')}")
    if row.get("effective_action") != "allow_teacher_scheduling":
        failures.append(f"effective_action:{row.get('effective_action')}")
    if "atm" not in set(row.get("eligible_teachers") or []):
        failures.append("override_eligible_teachers_missing_atm")
    if row.get("model_key") != "atm":
        failures.append(f"override_model_key:{row.get('model_key')}")

bad_failure_reasons = {
    "requested_organs_pruned_by_fov",
    "requested_teacher_not_scheduled",
    "expected_mask_missing",
    "hierarchical_roi_manifest_missing",
}
for item in summary.get("strict_delivery_failures") or []:
    if item.get("reason") in bad_failure_reasons:
        failures.append(f"bad_strict_delivery_failure:{item.get('reason')}")

if infer.get("return_code") not in (0, None):
    failures.append(f"atm_return_code:{infer.get('return_code')}")
if infer.get("status") not in {"success", None}:
    failures.append(f"atm_inference_status:{infer.get('status')}")

mask_candidates = [
    run_out / "annotation_versions" / case_id / "updated" / "airway_tree.nii.gz",
    run_out / "cases" / case_id / "selected_after_candidate_shapekit" / case_id / "segmentations" / "airway_tree.nii.gz",
    run_out / "standard_dataset" / case_id / "segmentations" / "airway_tree.nii.gz",
    run_out / "cases" / case_id / "hierarchical_predictions" / "atm" / "segmentations" / "airway_tree.nii.gz",
]
mask_path = next((candidate for candidate in mask_candidates if candidate.exists()), mask_candidates[0])
ct_path = Path(str(plan.get("ct_path") or ""))
validation: dict[str, object] = {
    "mask_path": str(mask_path),
    "candidate_mask_paths": [str(candidate) for candidate in mask_candidates],
    "ct_path": str(ct_path),
}
try:
    mask = nib.load(str(mask_path))
    ct = nib.load(str(ct_path))
    arr = np.asanyarray(mask.dataobj)
    unique = set(np.unique(arr).astype(int).tolist())
    validation.update({
        "exists": mask_path.exists(),
        "foreground_voxels": int((arr > 0).sum()),
        "binary": unique.issubset({0, 1}),
        "shape_matches_ct": mask.shape[:3] == ct.shape[:3],
        "spacing_matches_ct": tuple(round(float(x), 6) for x in mask.header.get_zooms()[:3])
        == tuple(round(float(x), 6) for x in ct.header.get_zooms()[:3]),
        "affine_matches_ct": bool(np.allclose(mask.affine, ct.affine)),
    })
except Exception as exc:
    validation["error"] = f"{type(exc).__name__}: {exc}"
    failures.append(f"airway_tree_validation_error:{validation['error']}")
else:
    for key in ("exists", "binary", "shape_matches_ct", "spacing_matches_ct", "affine_matches_ct"):
        if not validation.get(key):
            failures.append(f"airway_tree_{key}_failed")
    if int(validation.get("foreground_voxels") or 0) <= 0:
        failures.append("airway_tree_foreground_voxels_zero")

(out_root / "airway_tree_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
verdict = {
    "status": "passed" if not failures else "failed",
    "case_id": case_id,
    "run_out": str(run_out),
    "run_return_code": run_rc,
    "checks": {
        "run_summary_status": summary.get("status"),
        "strict_delivery_failure_count": summary.get("strict_delivery_failure_count"),
        "total_updated": summary.get("total_updated"),
        "teacher_run_list": sorted(teacher_run_list),
        "fov_override_applied": applied,
        "atm_inference_summary": infer,
        "airway_tree_validation": validation,
    },
    "failures": failures,
}
(out_root / "smoke_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
md = [
    "# ATM Partial-thorax Strict-delivery Smoke",
    "",
    f"- Status: `{verdict['status']}`",
    f"- Case: `{case_id}`",
    f"- Run output: `{run_out}`",
    f"- Failures: `{';'.join(failures) if failures else 'none'}`",
]
(out_root / "smoke_verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(verdict, indent=2))
raise SystemExit(0 if not failures else 1)
PY
