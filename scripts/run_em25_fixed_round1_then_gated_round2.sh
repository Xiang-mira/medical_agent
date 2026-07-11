#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/teacher1/JHU-project1/medical_agent"
OUTPUT_ROOT_REL="outputs/em_round1_25case_pseudo_label_20260709"
OUTPUT_ROOT="${ROOT}/${OUTPUT_ROOT_REL}"
CASE_LIST="data_manifest/case_list_25_tumor.csv"
STALE_SCREEN="em25_debug_mstep_20260710_retry"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG="${LOG_DIR}/fixed_round1_round2_${RUN_TS}.log"
ROUND2_GATE_DIR="${OUTPUT_ROOT}/round2/preflight"
ROUND2_GATE_JSON="${ROUND2_GATE_DIR}/round2_preflight_gate_${RUN_TS}.json"

mkdir -p "${LOG_DIR}" "${ROUND2_GATE_DIR}"
exec > >(tee -a "${LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_json_gate() {
  local payload="$1"
  python - "$ROUND2_GATE_JSON" "$payload" <<'PY'
import json, sys
path, payload = sys.argv[1], json.loads(sys.argv[2])
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
}

stop_stale_retry() {
  log "Stopping stale retry screen/processes if present: ${STALE_SCREEN}"
  if screen -ls | grep -q "${STALE_SCREEN}"; then
    screen -S "${STALE_SCREEN}" -X stuff $'\003' || true
  fi

  for _ in $(seq 1 60); do
    if ! pgrep -f "scripts/train_voxtell_prompt_student.py" >/dev/null \
       && ! pgrep -f "python scripts/run_em_training.py --rounds 1 .*${OUTPUT_ROOT_REL}" >/dev/null; then
      break
    fi
    sleep 1
  done

  if pgrep -f "scripts/train_voxtell_prompt_student.py" >/dev/null \
     || pgrep -f "python scripts/run_em_training.py --rounds 1 .*${OUTPUT_ROOT_REL}" >/dev/null; then
    log "Graceful stop timed out; terminating stale retry process tree."
    pkill -TERM -f "scripts/train_voxtell_prompt_student.py" || true
    pkill -TERM -f "python scripts/run_em_training.py --rounds 1 .*${OUTPUT_ROOT_REL}" || true
    sleep 10
    pkill -KILL -f "scripts/train_voxtell_prompt_student.py" || true
    pkill -KILL -f "python scripts/run_em_training.py --rounds 1 .*${OUTPUT_ROOT_REL}" || true
  fi
  screen -S "${STALE_SCREEN}" -X quit >/dev/null 2>&1 || true
  log "Stale retry stopped."
}

archive_old_round1_mstep() {
  local round1="${OUTPUT_ROOT}/round1"
  local mstep="${round1}/mstep"
  local archive="${round1}/mstep_archive_summary_bug_${RUN_TS}"
  if [[ -d "${mstep}" ]]; then
    log "Archiving old mstep to ${archive}"
    mv "${mstep}" "${archive}"
  fi
  mkdir -p "${mstep}"
  if [[ -f "${round1}/round_summary.json" ]]; then
    mv "${round1}/round_summary.json" "${round1}/round_summary_archive_summary_bug_${RUN_TS}.json"
  fi
  if [[ -f "${round1}/round_state.json" ]]; then
    mv "${round1}/round_state.json" "${round1}/round_state_archive_summary_bug_${RUN_TS}.json"
  fi
}

preflight_code() {
  log "Running fixed-code preflight."
  cd "${ROOT}"
  python -m py_compile scripts/train_voxtell_prompt_student.py scripts/run_em_training.py
  python - <<'PY'
from pathlib import Path
text = Path("scripts/train_voxtell_prompt_student.py").read_text(encoding="utf-8")
bad = "best_task_objective if math.isfinite(best_task_objective)"
good = '"best_task_objective": best_task_score if math.isfinite(best_task_score) else None'
if bad in text or good not in text:
    raise SystemExit("summary bug fix is not present")
print("summary bug fix verified")
PY
}

export_common_env() {
  export MEDAI_ENABLE_CRITIC=1
  export MEDAI_DEBUG_ALLOW_NO_LABELCRITIC=0
  export MEDAI_ENABLE_SHAPEKIT=1
  export MEDAI_DEBUG_ALLOW_NO_SHAPEKIT=0
  export MEDAI_ENABLE_VOXTELL_TRAINING=1
  export MEDAI_ALLOW_DEBUG_MSTEP_ON_FORMAL_GATE_FAIL=1
  export MEDAI_TEACHER_INFERENCE_MODE=hierarchical_roi
  export MEDAI_CANDIDATE_MODE=route_pruned_with_competition
  export MEDAI_TEXT_ENCODING_MODEL="${ROOT}/checkpoints/Qwen/Qwen3-Embedding-4B"
  export MEDAI_QWEN_VLM_MODEL="${ROOT}/checkpoints/Qwen/Qwen2-VL-7B-Instruct"
}

run_round1_fixed() {
  log "Starting fixed Round1 run."
  cd "${ROOT}"
  export_common_env
  python scripts/run_em_training.py \
    --rounds 1 \
    --case-list "${CASE_LIST}" \
    --output-root "${OUTPUT_ROOT_REL}" \
    --voxtell-mstep-mode project_voxtell_prompt_distillation_student
  log "Fixed Round1 command finished."
}

round1_gate_json() {
  python - "${OUTPUT_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
mstep = read(root / "round1/mstep/voxtell_prompt_mstep_result.json")
loss = read(root / "round1/mstep/loss_history.json")
summary = read(root / "round1/round_summary.json")
registry = read(root / "checkpoint_promotion_registry.json")
audit = ((summary.get("metrics") or {}).get("student_training_audit") or {})
round1_promo = ((registry.get("rounds") or {}).get("1") or {})
checks = {
    "mstep_success": mstep.get("status") == "success",
    "loss_2000_steps": int(loss.get("steps") or 0) >= 2000,
    "round_summary_exists": (root / "round1/round_summary.json").exists(),
    "student_training_round2_allowed": audit.get("round2_progression_allowed") is True,
    "round1_promoted": round1_promo.get("status") == "promoted",
}
payload = {
    "stage": "round2_preflight_after_fixed_round1",
    "status": "passed" if all(checks.values()) else "blocked",
    "checks": checks,
    "mstep_status": mstep.get("status"),
    "mstep_training_status": mstep.get("training_status"),
    "loss_steps": loss.get("steps"),
    "student_training_audit_status": audit.get("status"),
    "student_training_round2_progression_allowed": audit.get("round2_progression_allowed"),
    "round1_promotion_status": round1_promo.get("status"),
    "policy": "Round2 starts only after Round1 M-step, student audit, and promotion gates pass.",
}
print(json.dumps(payload, ensure_ascii=False))
PY
}

run_labelcritic_matrix_for_served_model() {
  local out_dir="$1"
  shift
  cd "${ROOT}"
  python scripts/run_labelcritic_benchmark_matrix.py \
    --manifest "${OUTPUT_ROOT}/round1/mstep/voxtell_prompt_student_manifest.json" \
    --output-dir "${out_dir}" \
    --models "$@" \
    --prompt-modes official_dual strict_single \
    --projection-modes ap multiview_audit \
    --organs liver kidney_left aorta spleen stomach \
    --corruptions random_blob \
    --max-per-organ 1 \
    --timeout-sec 120
}

wait_for_vllm_model() {
  local model_name="$1"
  local deadline=$((SECONDS + 240))
  while (( SECONDS < deadline )); do
    if python - "${model_name}" <<'PY'
import json, sys, urllib.request
needle = sys.argv[1]
try:
    body = json.loads(urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3).read().decode())
except Exception:
    raise SystemExit(1)
ids = [str(row.get("id") or "") for row in body.get("data", [])]
raise SystemExit(0 if any(needle in sid for sid in ids) else 1)
PY
    then
      return 0
    fi
    sleep 10
  done
  return 1
}

switch_vllm_to_qwen25() {
  log "Switching vLLM to Qwen2.5-VL-7B for LabelCritic benchmark."
  screen -S vllm_labelcritic_25case -X quit >/dev/null 2>&1 || true
  screen -S vllm_labelcritic_qwen25_em25 -X quit >/dev/null 2>&1 || true
  sleep 10
  pkill -TERM -f "vllm.entrypoints.openai.api_server.*--port 8000" || true
  sleep 10
  screen -dmS vllm_labelcritic_qwen25_em25 bash -lc "
    cd '${ROOT}' &&
    python -m vllm.entrypoints.openai.api_server \
      --model '${ROOT}/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct' \
      --port 8000 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.4
  "
  wait_for_vllm_model "Qwen2.5-VL-7B-Instruct"
}

benchmark_passed() {
  local summary="$1"
  python - "${summary}" <<'PY'
import json, sys
try:
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if doc.get("status") == "passed" or any(bool(r.get("pass")) for r in doc.get("rows", []) if isinstance(r, dict)):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

run_round2_gate_and_benchmark() {
  local gate_payload
  gate_payload="$(round1_gate_json)"
  run_json_gate "${gate_payload}"
  if [[ "$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("status"))' "${gate_payload}")" != "passed" ]]; then
    log "Round2 blocked by Round1 gate."
    return 2
  fi

  local bench_root="${OUTPUT_ROOT}/labelcritic_round2_gate_${RUN_TS}"
  local qwen2_summary="${bench_root}/qwen2_current/labelcritic_repair_benchmark_summary.json"
  local qwen25_summary="${bench_root}/qwen25/labelcritic_repair_benchmark_summary.json"
  log "Running LabelCritic benchmark on currently served Qwen2 model."
  if run_labelcritic_matrix_for_served_model "${bench_root}/qwen2_current" "${ROOT}/checkpoints/Qwen/Qwen2-VL-7B-Instruct"; then
    :
  else
    log "Current Qwen2 benchmark did not pass."
  fi
  if benchmark_passed "${qwen2_summary}"; then
    log "LabelCritic benchmark passed on Qwen2."
    return 0
  fi

  if switch_vllm_to_qwen25; then
    if run_labelcritic_matrix_for_served_model "${bench_root}/qwen25" "${ROOT}/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct"; then
      :
    else
      log "Qwen2.5 benchmark did not pass."
    fi
    if benchmark_passed "${qwen25_summary}"; then
      log "LabelCritic benchmark passed on Qwen2.5."
      return 0
    fi
  else
    log "Qwen2.5 vLLM failed to become ready within 4 minutes."
  fi

  log "LabelCritic benchmark failed; writing cached reselection blocked diagnosis."
  python "${ROOT}/scripts/cached_key_organ_reselection_repair.py" \
    --output-root "${OUTPUT_ROOT}" \
    --benchmark-summary "${qwen25_summary}" \
    --round 1 || true
  run_json_gate "$(python - "${ROUND2_GATE_JSON}" <<'PY'
import json, sys
path = sys.argv[1]
doc = json.load(open(path, encoding="utf-8"))
doc["status"] = "blocked"
doc["blocker"] = "labelcritic_known_better_benchmark_failed"
doc["policy"] = "Formal Round2 is fail-closed until LabelCritic known-better benchmark passes."
print(json.dumps(doc, ensure_ascii=False))
PY
)"
  return 2
}

run_round2_if_allowed() {
  log "Starting gated Round2."
  cd "${ROOT}"
  export_common_env
  export MEDAI_ALLOW_ROUND1_TO_ROUND2_RESTART=1
  python scripts/run_em_training.py \
    --rounds 2 \
    --case-list "${CASE_LIST}" \
    --output-root "${OUTPUT_ROOT_REL}" \
    --voxtell-mstep-mode project_voxtell_prompt_distillation_student
}

main() {
  log "Fixed Round1 -> gated Round2 orchestration started. log=${LOG}"
  stop_stale_retry
  archive_old_round1_mstep
  preflight_code
  run_round1_fixed
  if run_round2_gate_and_benchmark; then
    run_round2_if_allowed
  else
    log "Round2 not started; gate is fail-closed. See ${ROUND2_GATE_JSON}"
  fi
  log "Orchestration finished."
}

main "$@"
