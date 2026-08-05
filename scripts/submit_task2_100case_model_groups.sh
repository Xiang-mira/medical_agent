#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}
CASE_MANIFEST=${CASE_MANIFEST:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv}
DATA_ROOT=${DATA_ROOT:-/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro}
FORMAL_OUT_ROOT=${FORMAL_OUT_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/formal_task2_22targets_100cases_$(date +%Y%m%d_%H%M%S)}
REGISTRY=${REGISTRY:-configs/model_registry.yaml}
TARGET_CONFIG=${TARGET_CONFIG:-configs/student_3d_prompt_target_organs.json}
PARTITION=${PARTITION:-gpu}
GRES=${GRES:-gpu:t4:1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-48G}
TIME_LIMIT=${TIME_LIMIT:-12:00:00}
TIMEOUT_SEC=${TIMEOUT_SEC:-14400}
CADS_CONCURRENCY=${CADS_CONCURRENCY:-4}
ATM_CONCURRENCY=${ATM_CONCURRENCY:-2}
AIRRC_CONCURRENCY=${AIRRC_CONCURRENCY:-2}
UNEST_CONCURRENCY=${UNEST_CONCURRENCY:-2}
RESUME_CASE_IDS=${RESUME_CASE_IDS:-}
RUN_TESTS_BEFORE_LAUNCH=${RUN_TESTS_BEFORE_LAUNCH:-1}

cd "$CODE_ROOT"
git fetch origin
git checkout main
git pull --ff-only origin main
COMMIT=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
  echo "submit_task2_100case_model_groups.sh must run on main, got $BRANCH" >&2
  exit 2
fi
if [ -n "$(git status --short)" ]; then
  echo "Working tree is not clean; refusing to launch formal arrays." >&2
  git status --short >&2
  exit 2
fi

ATM_SMOKE_PASS_JSON=${ATM_SMOKE_PASS_JSON:-${ATM_SMOKE_ROOT:-}/ATM_STRICT_SMOKE_PASS.json}
: "${AIRRC_SMOKE_PASS_JSON:?Set AIRRC_SMOKE_PASS_JSON to the existing AirRC strict smoke pass JSON.}"
: "${UNEST_SMOKE_PASS_JSON:?Set UNEST_SMOKE_PASS_JSON to the existing UNEST strict smoke pass JSON.}"
: "${CADS_SMOKE_PASS_JSON:?Set CADS_SMOKE_PASS_JSON to the existing CADS readiness/smoke pass JSON.}"

mkdir -p "$FORMAL_OUT_ROOT"/{preflight,readiness,cads,atm,airrc,unest,slurm}

"$PYTHON" - "$ATM_SMOKE_PASS_JSON" "$AIRRC_SMOKE_PASS_JSON" "$UNEST_SMOKE_PASS_JSON" "$CADS_SMOKE_PASS_JSON" <<'PY'
import json
import sys
from pathlib import Path

labels = ["ATM", "AirRC", "UNEST", "CADS"]
ok_statuses = {"passed", "pass", "success", "completed"}
errors = []
for label, value in zip(labels, sys.argv[1:]):
    path = Path(value)
    if not path.exists():
        errors.append(f"{label} smoke/readiness pass file missing: {path}")
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{label} smoke/readiness pass file unreadable: {path}: {exc}")
        continue
    status = str(data.get("status") or "").lower()
    if status not in ok_statuses:
        errors.append(f"{label} smoke/readiness status is not passing: {status}")
    if int(data.get("strict_delivery_failure_count") or 0) != 0:
        errors.append(f"{label} strict_delivery_failure_count is nonzero")
if errors:
    raise SystemExit("\n".join(errors))
PY

"$PYTHON" -m py_compile \
  agent-harness/cli_anything/medai/medai_cli.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  tools/dataset_delivery/audit_task2_100case_launch.py

if [ "$RUN_TESTS_BEFORE_LAUNCH" = "1" ]; then
  "$PYTHON" -m pytest -q tests/dataset_delivery/test_task2_strict_delivery.py
fi

"$PYTHON" tools/dataset_delivery/task2_audit.py \
  --manifest "$CASE_MANIFEST" \
  --output-dir "$FORMAL_OUT_ROOT/readiness"

"$PYTHON" tools/dataset_delivery/audit_task2_100case_launch.py \
  --case-manifest "$CASE_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --taxonomy "$TARGET_CONFIG" \
  --output-root "$FORMAL_OUT_ROOT/preflight" \
  --strict-delivery-fov-override-organs airway_tree

EXTRACT_CASE_SCRIPT=$FORMAL_OUT_ROOT/slurm/extract_case_manifest.py
cat > "$EXTRACT_CASE_SCRIPT" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
index = int(sys.argv[2])
out = Path(sys.argv[3])
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if index < 0 or index >= len(rows):
    raise SystemExit(f"array index {index} out of range for {manifest}")
row = rows[index]
case_id = row.get("case_id") or row.get("id")
ct_path = row.get("ct_path") or row.get("image_path")
ref_dir = row.get("annotation_folder") or row.get("reference_mask_dir")
if not case_id or not ct_path or not ref_dir:
    raise SystemExit(f"missing case_id/ct_path/reference dir in row {index}: {row}")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as dst:
    writer = csv.DictWriter(dst, fieldnames=["case_id", "ct_path", "annotation_folder"])
    writer.writeheader()
    writer.writerow({"case_id": case_id, "ct_path": ct_path, "annotation_folder": ref_dir})
print(case_id)
PY

"$PYTHON" - "$FORMAL_OUT_ROOT/preflight/eligible_case_manifests" "$FORMAL_OUT_ROOT/launch_manifest.csv" "$RESUME_CASE_IDS" <<'PY'
import csv
import sys
from pathlib import Path

eligible_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
resume = {x.strip() for x in sys.argv[3].replace(";", ",").split(",") if x.strip()}
fields = ["model_group", "array_index", "case_id", "ct_path", "annotation_folder", "reference_mask_dir"]
rows_out = []
for group in ["cads", "atm", "airrc", "unest"]:
    path = eligible_dir / f"{group}_eligible_cases.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if resume:
        rows = [row for row in rows if row.get("case_id") in resume]
    group_manifest = out.parent / group / "eligible_cases_for_launch.csv"
    group_manifest.parent.mkdir(parents=True, exist_ok=True)
    with group_manifest.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["case_id", "ct_path", "annotation_folder", "reference_mask_dir"])
        writer.writeheader()
        writer.writerows(rows)
    for idx, row in enumerate(rows):
        rows_out.append({
            "model_group": group,
            "array_index": idx,
            "case_id": row.get("case_id"),
            "ct_path": row.get("ct_path"),
            "annotation_folder": row.get("annotation_folder"),
            "reference_mask_dir": row.get("reference_mask_dir"),
        })
with out.open("w", encoding="utf-8", newline="") as dst:
    writer = csv.DictWriter(dst, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows_out)
PY

submit_group() {
  local group=$1
  local model=$2
  local organs=$3
  local concurrency=$4
  local group_root=$FORMAL_OUT_ROOT/$group
  local group_manifest=$group_root/eligible_cases_for_launch.csv
  local count
  count=$("$PYTHON" - "$group_manifest" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open("r", encoding="utf-8-sig", newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
  if [ "$count" -eq 0 ]; then
    echo "${group}: no eligible cases, skipping array submission"
    return
  fi
  local max_index=$((count - 1))
  local sbatch_file=$FORMAL_OUT_ROOT/slurm/${group}_task2_array.sbatch
  local override_arg=""
  if [ "$group" = "atm" ]; then
    override_arg="--strict-delivery-fov-override-organs airway_tree"
  fi
  cat > "$sbatch_file" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=task2_${group}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --array=0-${max_index}%${concurrency}
#SBATCH --output=${FORMAL_OUT_ROOT}/slurm/${group}_%A_%a.out
#SBATCH --error=${FORMAL_OUT_ROOT}/slurm/${group}_%A_%a.err

set -euo pipefail
if [ -f "\$HOME/.bodymaps_env" ]; then
  source "\$HOME/.bodymaps_env"
fi
cd "$CODE_ROOT"
CASE_CSV="$group_root/case_lists/case_\${SLURM_ARRAY_TASK_ID}.csv"
CASE_ID=\$("$PYTHON" "$EXTRACT_CASE_SCRIPT" "$group_manifest" "\$SLURM_ARRAY_TASK_ID" "\$CASE_CSV")
CASE_OUT="$group_root/cases/\$CASE_ID"
mkdir -p "\$CASE_OUT/logs"

"$PYTHON" run_medai_cli.py --json run-loop \\
  --case-list "\$CASE_CSV" \\
  --models "$model" \\
  --organs "$organs" \\
  --target-config "$TARGET_CONFIG" \\
  --registry "$REGISTRY" \\
  --output "\$CASE_OUT/raw" \\
  --timeout-sec "$TIMEOUT_SEC" \\
  --teacher-inference-mode full_volume \\
  --no-enable-shapekit \\
  --debug-allow-no-shapekit \\
  --no-enable-critic \\
  --strict-delivery-targets \\
  ${override_arg} \\
  --log-file "\$CASE_OUT/logs/run_loop.log"
EOF
  local job_id
  job_id=$(sbatch --parsable "$sbatch_file")
  echo "$group,$job_id,$count,$sbatch_file" >> "$FORMAL_OUT_ROOT/slurm/submitted_jobs.csv"
  echo "${group}_JOB_ID=$job_id"
}

echo "model_group,job_id,case_count,sbatch_file" > "$FORMAL_OUT_ROOT/slurm/submitted_jobs.csv"
submit_group cads cads "blood,cerebrospinal_fluid,common_iliac_artery_left,common_iliac_artery_right,common_iliac_vein_left,common_iliac_vein_right,compact_bone,eyeball,face,gland_structure,gray_matter,muscle_of_head,scalp,spongy_bone,white_matter" "$CADS_CONCURRENCY"
submit_group atm atm "airway_tree" "$ATM_CONCURRENCY"
submit_group airrc airrc "airway_wall,lung_pulmonary_arteries,lung_pulmonary_veins" "$AIRRC_CONCURRENCY"
submit_group unest unest "kidney_cortex,kidney_medulla,kidney_pelvicalyceal_system" "$UNEST_CONCURRENCY"

"$PYTHON" - "$FORMAL_OUT_ROOT" "$COMMIT" "$CASE_MANIFEST" "$RESUME_CASE_IDS" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
jobs_path = root / "slurm" / "submitted_jobs.csv"
jobs = []
if jobs_path.exists():
    with jobs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        jobs = list(csv.DictReader(handle))
summary = {
    "status": "submitted",
    "commit": sys.argv[2],
    "case_manifest": sys.argv[3],
    "resume_case_ids": [x.strip() for x in sys.argv[4].replace(";", ",").split(",") if x.strip()],
    "strict_delivery_targets": True,
    "strict_delivery_fov_override_organs": ["airway_tree"],
    "blocked_targets_not_submitted": ["brain_ventricle"],
    "totalsegmentator_submitted": False,
    "jobs": jobs,
    "output_layout": {
        "preflight": str(root / "preflight"),
        "cads": str(root / "cads"),
        "atm": str(root / "atm"),
        "airrc": str(root / "airrc"),
        "unest": str(root / "unest"),
        "slurm": str(root / "slurm"),
    },
}
(root / "registry.json").write_text(json.dumps({
    "commit": sys.argv[2],
    "model_groups": {
        "cads": "15 CADS targets",
        "atm": "airway_tree with strict-delivery FOV override",
        "airrc": "3 AirRC targets",
        "unest": "3 UNEST targets",
    },
}, indent=2) + "\n", encoding="utf-8")
(root / "launch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "TASK2_FORMAL_OUT_ROOT=$FORMAL_OUT_ROOT"
