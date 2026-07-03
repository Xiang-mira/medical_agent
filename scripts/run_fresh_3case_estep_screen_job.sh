#!/usr/bin/env bash
set -euo pipefail

cd /home/teacher1/JHU-project1/medical_agent

export PYTHONPATH="agent-harness:${PYTHONPATH:-}"
export MEDAI_CASE_LIST="/home/teacher1/JHU-project1/medical_agent/outputs/labelcritic_373_repair_20260703/fresh_3case_case_list.csv"
export MEDAI_OUTPUT_ROOT="/home/teacher1/JHU-project1/medical_agent/outputs/fresh_3case_estep_20260703"
export MEDAI_STUDENT_BACKEND="voxtell_style_3d_prompt"
export MEDAI_EXPERIMENT_PROFILE="advisor_aligned_default"
export MEDAI_ENABLE_SHAPEKIT="1"
export MEDAI_ENABLE_CRITIC="1"
export MEDAI_TEACHER_INFERENCE_MODE="hierarchical_roi"
export MEDAI_CANDIDATE_MODE="route_pruned_with_competition"
export MEDAI_VLLM_BASE_URL="http://localhost:8000"
export MEDAI_DEBUG_ALLOW_NO_LABELCRITIC="0"

log_dir="${MEDAI_OUTPUT_ROOT}/logs"
log_file="${log_dir}/fresh_3case_estep.log"
mkdir -p "${log_dir}"
exec > >(tee -a "${log_file}") 2>&1

echo "==== Fresh 3-case E-step started: $(date -Is) ===="
echo "case_list=${MEDAI_CASE_LIST}"
echo "output_root=${MEDAI_OUTPUT_ROOT}"
echo "teacher_cache_policy=fresh_no_preseed"
echo "mstep=not_run"

python - <<'PY'
import json
from pathlib import Path

import scripts.run_em_training as em

em.ensure_current_student_backend_allowed()
em.ensure_formal_teacher_pool_registered()
em.ensure_formal_quality_gates()

result = em.run_estep(1)
dashboard = em.build_round_label_scoring_dashboard(1)
manifest = em.build_student_dataset(1)
gate = em.formal_estep_gate(1, result, manifest)

root = em.OUTPUT_ROOT / "round1"
gate_path = root / "estep" / "formal_gate.json"
gate_path.parent.mkdir(parents=True, exist_ok=True)
gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

summary = {
    "round": 1,
    "stage": "fresh_3case_estep_only",
    "status": "success" if result.get("status") == "success" else "failed",
    "estep_result": result,
    "formal_gate": gate,
    "label_scoring_dashboard": dashboard,
    "training_manifest": str(manifest),
    "mstep_status": "not_run_by_design",
}
summary_path = root / "round_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary["status"] != "success":
    raise SystemExit(1)
PY

echo "==== Fresh 3-case E-step finished: $(date -Is) ===="
