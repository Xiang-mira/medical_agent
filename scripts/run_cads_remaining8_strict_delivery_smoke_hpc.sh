#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
PREFERRED_CASE_ID=${PREFERRED_CASE_ID:-BDMAP_00000120}
OUT_ROOT=${OUT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/cads_remaining8_strict_smoke_$(date +%Y%m%d_%H%M%S)}
REGISTRY=${REGISTRY:-configs/model_registry.yaml}
TIMEOUT_SEC=${TIMEOUT_SEC:-14400}

cd "$CODE_ROOT"
mkdir -p "$OUT_ROOT"
git rev-parse HEAD > "$OUT_ROOT/git_commit.txt"

AUDIT_ROOT=$OUT_ROOT/audit
RUN_OUT=$OUT_ROOT/run_loop
CASE_CSV=$OUT_ROOT/selected_case_manifest.csv
COMMAND_TXT=$OUT_ROOT/command.txt
TARGET_MODEL_MAP=$OUT_ROOT/target_model_map.json
CASE_SELECTION=$OUT_ROOT/case_selection.json
mkdir -p "$AUDIT_ROOT" "$RUN_OUT"

"$PYTHON" tools/dataset_delivery/audit_cads_remaining_targets.py audit \
  --output-root "$AUDIT_ROOT" \
  --case-manifest "$CASE_MANIFEST"

eval "$("$PYTHON" - "$AUDIT_ROOT/cads_remaining_targets.csv" "$TARGET_MODEL_MAP" <<'PY'
import csv
import json
import shlex
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open("r", encoding="utf-8-sig", newline="")))
targets = [row["target_name"] for row in rows]
models = sorted({row["model_key"] for row in rows if row.get("model_key")})
Path(sys.argv[2]).write_text(json.dumps({row["target_name"]: row["model_key"] for row in rows}, indent=2) + "\n", encoding="utf-8")
print("CADS_TARGETS=" + shlex.quote(",".join(targets)))
print("CADS_MODELS=" + shlex.quote(",".join(models)))
PY
)"

if [ -z "${CADS_TARGETS:-}" ]; then
  echo "No CADS remaining targets were found by the canonical audit." >&2
  exit 2
fi
if [ -z "${CADS_MODELS:-}" ]; then
  echo "No concrete CADS model keys were found for remaining targets." >&2
  exit 2
fi

"$PYTHON" - "$CASE_MANIFEST" "$CASE_CSV" "$CASE_SELECTION" "$PREFERRED_CASE_ID" "$CADS_TARGETS" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "agent-harness"))
from cli_anything.medai.core.multimodel_loop import _fov_status_for_organ, _load_case_presence_context

manifest = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
selection_json = Path(sys.argv[3])
preferred_case_id = sys.argv[4]
targets = [item for item in sys.argv[5].split(",") if item]

with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

def row_case_id(row, index):
    return row.get("case_id") or row.get("id") or f"case_{index:03d}"

def normalized_row(row, index):
    case_id = row_case_id(row, index)
    ct_path = row.get("ct_path") or row.get("image_path") or ""
    ref_dir = row.get("annotation_folder") or row.get("reference_mask_dir") or row.get("mask_dir") or ""
    return {"case_id": case_id, "ct_path": ct_path, "annotation_folder": ref_dir, "reference_mask_dir": ref_dir}

def statuses_for(row, index):
    normalized = normalized_row(row, index)
    if not normalized["ct_path"] or not normalized["annotation_folder"]:
        return normalized, {target: "unknown" for target in targets}
    context = _load_case_presence_context(
        normalized,
        Path(normalized["ct_path"]),
        normalized["case_id"],
        selection_json.parent / "case_selection_context" / normalized["case_id"],
    )
    return normalized, {target: _fov_status_for_organ(target, context) for target in targets}

candidates = []
for index, row in enumerate(rows):
    normalized, statuses = statuses_for(row, index)
    all_visible = all(status == "fully_visible" for status in statuses.values())
    usable = all(status not in {"out_of_fov", "unknown"} for status in statuses.values())
    candidates.append({"row": normalized, "statuses": statuses, "all_visible": all_visible, "usable": usable})

chosen = next(
    (item for item in candidates if item["row"]["case_id"] == preferred_case_id and item["all_visible"]),
    None,
)
if chosen is None:
    chosen = next((item for item in candidates if item["all_visible"]), None)
if chosen is None:
    chosen = next((item for item in candidates if item["row"]["case_id"] == preferred_case_id and item["usable"]), None)
if chosen is None:
    chosen = next((item for item in candidates if item["usable"]), None)
if chosen is None:
    payload = {
        "status": "failed",
        "reason": "no_case_with_all_cads_remaining_targets_visible",
        "preferred_case_id": preferred_case_id,
        "targets": targets,
        "sample_cases": [
            {"case_id": item["row"]["case_id"], "statuses": item["statuses"]}
            for item in candidates[:20]
        ],
    }
    selection_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(payload["reason"])

payload = {
    "status": "success",
    "preferred_case_id": preferred_case_id,
    "selected_case_id": chosen["row"]["case_id"],
    "selection_basis": "all_targets_fully_visible" if chosen["all_visible"] else "all_targets_not_out_of_fov_or_unknown",
    "targets": targets,
    "target_visibility": chosen["statuses"],
}
selection_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path", "annotation_folder"])
    writer.writeheader()
    writer.writerow({
        "case_id": chosen["row"]["case_id"],
        "ct_path": chosen["row"]["ct_path"],
        "annotation_folder": chosen["row"]["annotation_folder"],
    })
print(json.dumps(payload, indent=2))
PY

SELECTED_CASE_ID=$("$PYTHON" - "$CASE_SELECTION" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["selected_case_id"])
PY
)

cat > "$COMMAND_TXT" <<EOF
"$PYTHON" run_medai_cli.py --json run-loop \\
  --case-list "$CASE_CSV" \\
  --models "$CADS_MODELS" \\
  --organs "$CADS_TARGETS" \\
  --registry "$REGISTRY" \\
  --output "$RUN_OUT" \\
  --timeout-sec "$TIMEOUT_SEC" \\
  --teacher-inference-mode full_volume \\
  --no-enable-shapekit \\
  --debug-allow-no-shapekit \\
  --no-enable-critic \\
  --strict-delivery-targets \\
  --log-file "$RUN_OUT/run_loop.log"
EOF

set +e
"$PYTHON" run_medai_cli.py --json run-loop \
  --case-list "$CASE_CSV" \
  --models "$CADS_MODELS" \
  --organs "$CADS_TARGETS" \
  --registry "$REGISTRY" \
  --output "$RUN_OUT" \
  --timeout-sec "$TIMEOUT_SEC" \
  --teacher-inference-mode full_volume \
  --no-enable-shapekit \
  --debug-allow-no-shapekit \
  --no-enable-critic \
  --strict-delivery-targets \
  --log-file "$RUN_OUT/run_loop.log"
RUN_RC=$?
set -e

[ -f "$RUN_OUT/run_summary.json" ] && cp "$RUN_OUT/run_summary.json" "$OUT_ROOT/run_summary.json"
[ -f "$RUN_OUT/annotation_versions/$SELECTED_CASE_ID/case_execution_plan.json" ] && cp "$RUN_OUT/annotation_versions/$SELECTED_CASE_ID/case_execution_plan.json" "$OUT_ROOT/case_execution_plan.json"
[ -f "$RUN_OUT/strict_delivery_failures.csv" ] && cp "$RUN_OUT/strict_delivery_failures.csv" "$OUT_ROOT/strict_delivery_failures.csv"
find "$RUN_OUT" -path "*/inference_summary.json" -print > "$OUT_ROOT/inference_summary_paths.txt" 2>/dev/null || true

"$PYTHON" tools/dataset_delivery/audit_cads_remaining_targets.py validate-smoke \
  --run-out "$RUN_OUT" \
  --output-root "$OUT_ROOT" \
  --case-id "$SELECTED_CASE_ID" \
  --targets "$CADS_TARGETS" \
  --target-model-map "$TARGET_MODEL_MAP" \
  --run-return-code "$RUN_RC"
