#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/teacher1/JHU-project1/medical_agent"
OUTPUT_ROOT="${ROOT}/outputs/em_round1_25case_pseudo_label_20260709"
TRAINSET_AUDIT_ROOT="${OUTPUT_ROOT}/round1/trainset_pseudo_consistency_full_mstep_lr3e-5_negative_fix_reaudit"
MANIFEST="${OUTPUT_ROOT}/round1/mstep/voxtell_prompt_student_manifest.json"
RUN_DATE="$(date -u +%Y%m%d)"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
BENCH_ROOT="${OUTPUT_ROOT}/labelcritic_round2_gate_${RUN_DATE}"
LOG_DIR="${OUTPUT_ROOT}/logs"
MASTER_GATE="${OUTPUT_ROOT}/round2_formal_master_gate_after_trainset_and_labelcritic.json"
QWEN2_MODEL="${ROOT}/checkpoints/Qwen/Qwen2-VL-7B-Instruct"
QWEN25_MODEL="${ROOT}/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct"
AUTO_REPAIR_VLM_WEIGHTS="${MEDAI_AUTO_REPAIR_VLM_WEIGHTS:-1}"

mkdir -p "${LOG_DIR}" "${BENCH_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_vllm_any() {
  local deadline=$((SECONDS + 240))
  while (( SECONDS < deadline )); do
    if python - <<'PY'
import json
import urllib.request
try:
    body = json.loads(urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3).read().decode())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if body.get("data") else 1)
PY
    then
      return 0
    fi
    sleep 10
  done
  return 1
}

wait_for_vllm_model() {
  local model_name="$1"
  local deadline=$((SECONDS + 240))
  while (( SECONDS < deadline )); do
    if python - "${model_name}" <<'PY'
import json
import sys
import urllib.request
needle = sys.argv[1]
try:
    body = json.loads(urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3).read().decode())
except Exception:
    raise SystemExit(1)
ids = [str(row.get("id") or "") for row in body.get("data", []) if isinstance(row, dict)]
raise SystemExit(0 if any(needle in sid for sid in ids) else 1)
PY
    then
      return 0
    fi
    sleep 10
  done
  return 1
}

benchmark_passed() {
  local summary="$1"
  python - "${summary}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
doc = json.loads(path.read_text(encoding="utf-8"))
if doc.get("status") == "passed" or any(bool(row.get("pass")) for row in doc.get("rows", []) if isinstance(row, dict)):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

write_blocked_benchmark_summary() {
  local summary="$1"
  local blocker="$2"
  local detail="${3:-}"
  mkdir -p "$(dirname "${summary}")"
  python - "${summary}" "${blocker}" "${detail}" <<'PY'
import json
import sys
from pathlib import Path
summary, blocker, detail = sys.argv[1:4]
payload = {
    "stage": "labelcritic_benchmark_matrix",
    "status": "blocked",
    "blocker": blocker,
    "detail": detail,
    "rows": [],
    "policy": "Formal Round2 is fail-closed until a known-better benchmark passes.",
}
Path(summary).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

validate_safetensors_model() {
  local model="$1"
  python - "${model}" <<'PY'
import sys
from pathlib import Path
from safetensors import safe_open
model = Path(sys.argv[1])
files = sorted(model.glob("*.safetensors"))
if not files:
    print(f"no safetensors files found under {model}")
    raise SystemExit(1)
for path in files:
    try:
        with safe_open(path, framework="pt") as handle:
            list(handle.keys())
    except Exception as exc:
        print(f"{path.name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
print(f"validated_safetensors_files={len(files)}")
PY
}

repair_safetensors_model_from_hf() {
  local model="$1"
  local repo_id="$2"
  local repair_tag
  repair_tag="$(date -u +%Y%m%d_%H%M%S)"
  local repair_dir="${model}/.corrupt_${repair_tag}"
  local repair_log="${LOG_DIR}/vlm_weight_repair_${repo_id//\//_}_${repair_tag}.log"
  local repair_summary="${model}/weight_repair_${repair_tag}.json"

  if [[ "${AUTO_REPAIR_VLM_WEIGHTS}" != "1" ]]; then
    log "Auto-repair disabled by MEDAI_AUTO_REPAIR_VLM_WEIGHTS=${AUTO_REPAIR_VLM_WEIGHTS}."
    return 1
  fi

  log "Attempting automatic local weight repair for ${repo_id}; log=${repair_log}"
  mkdir -p "${repair_dir}" "${LOG_DIR}"
  python - "${model}" "${repair_dir}" "${repair_summary}" <<'PY'
import json
import shutil
import sys
from pathlib import Path
from safetensors import safe_open
model = Path(sys.argv[1])
repair_dir = Path(sys.argv[2])
summary = Path(sys.argv[3])
invalid = []
valid = []
for path in sorted(model.glob("*.safetensors")):
    try:
        with safe_open(path, framework="pt") as handle:
            list(handle.keys())
        valid.append(path.name)
    except Exception as exc:
        invalid.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
        shutil.move(str(path), str(repair_dir / path.name))
payload = {
    "stage": "vlm_local_weight_repair_pre_download",
    "status": "moved_invalid_shards" if invalid else "no_invalid_shards_found",
    "model_dir": str(model),
    "repair_dir": str(repair_dir),
    "invalid_shards": invalid,
    "valid_shards": valid,
}
summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
raise SystemExit(0 if invalid else 1)
PY

  log "Re-downloading missing/invalid files for ${repo_id} into ${model}."
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo_id}" \
      --local-dir "${model}" \
      --include "*.safetensors" "*.json" "*.txt" "*.model" "*.py" \
      2>&1 | tee -a "${repair_log}"
  else
    python - "${repo_id}" "${model}" <<'PY' 2>&1 | tee -a "${repair_log}"
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1:3]
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.py"],
    resume_download=True,
)
PY
  fi

  validate_safetensors_model "${model}"
}

validate_or_repair_model() {
  local model="$1"
  local repo_id="$2"
  if validate_safetensors_model "${model}"; then
    return 0
  fi
  log "Local model validation failed for ${model}; trying automatic repair from ${repo_id}."
  repair_safetensors_model_from_hf "${model}" "${repo_id}"
}

start_vllm_model() {
  local screen_name="$1"
  local model="$2"
  local model_name="$3"
  local vllm_log="${LOG_DIR}/${screen_name}_${RUN_TS}.log"
  log "Starting vLLM screen=${screen_name} model=${model_name}; log=${vllm_log}"
  screen -S vllm_labelcritic_25case -X quit >/dev/null 2>&1 || true
  screen -S vllm_labelcritic_qwen25_em25 -X quit >/dev/null 2>&1 || true
  screen -S "${screen_name}" -X quit >/dev/null 2>&1 || true
  pkill -TERM -f "vllm.entrypoints.openai.api_server.*--port 8000" || true
  sleep 8
  screen -dmS "${screen_name}" bash -lc "
    cd '${ROOT}' &&
    python -m vllm.entrypoints.openai.api_server \
      --model '${model}' \
      --port 8000 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.4 \
      2>&1 | tee '${vllm_log}'
  "
  wait_for_vllm_model "${model_name}"
}

run_labelcritic_matrix_for_model() {
  local out_dir="$1"
  local model="$2"
  cd "${ROOT}"
  python scripts/run_labelcritic_benchmark_matrix.py \
    --manifest "${MANIFEST}" \
    --output-dir "${out_dir}" \
    --models "${model}" \
    --prompt-modes official_dual strict_single \
    --projection-modes ap multiview_audit \
    --organs liver kidney_left aorta spleen stomach \
    --corruptions random_blob \
    --max-per-organ 1 \
    --timeout-sec 120
}

switch_vllm_to_qwen25() {
  log "Switching vLLM on port 8000 to Qwen2.5-VL-7B for bounded benchmark."
  start_vllm_model "vllm_labelcritic_qwen25_em25" "${QWEN25_MODEL}" "Qwen2.5-VL-7B-Instruct"
}

run_negative_manifest_audit() {
  log "Running negative_absent manifest source audit."
  cd "${ROOT}"
  python scripts/audit_negative_absent_manifest_source.py \
    --manifest "${MANIFEST}" \
    --negative-diagnosis "${TRAINSET_AUDIT_ROOT}/negative_absent_false_positive_diagnosis.csv" \
    --target-config "${ROOT}/configs/student_3d_prompt_target_organs.json" \
    --appearance-config "${ROOT}/configs/organ_ct_appearance_373.json" \
    --output-dir "${TRAINSET_AUDIT_ROOT}"
}

run_cached_reselection_gate() {
  local summary="$1"
  log "Writing cached key-organ reselection gate from benchmark summary: ${summary}"
  cd "${ROOT}"
  python scripts/cached_key_organ_reselection_repair.py \
    --output-root "${OUTPUT_ROOT}" \
    --benchmark-summary "${summary}" \
    --round 1 || true
}

build_master_gate() {
  local summary="$1"
  log "Building formal Round2 master gate with LabelCritic summary: ${summary}"
  cd "${ROOT}"
  python scripts/build_round2_formal_master_gate.py \
    --trainset-gate "${TRAINSET_AUDIT_ROOT}/round2_progression_gate_after_trainset_consistency.json" \
    --negative-safe-summary "${TRAINSET_AUDIT_ROOT}/negative_safe_postprocess_summary.json" \
    --negative-manifest-audit "${TRAINSET_AUDIT_ROOT}/negative_absent_manifest_source_audit.json" \
    --labelcritic-summary "${summary}" \
    --cached-reselection-summary "${OUTPUT_ROOT}/round1/estep/repair/key_organ_coverage_diagnosis_after_repair.json" \
    --output "${MASTER_GATE}" || true
}

main() {
  log "Round2 gate + LabelCritic repair orchestration started. ts=${RUN_TS}"
  cd "${ROOT}"
  python -m py_compile \
    scripts/audit_negative_absent_manifest_source.py \
    scripts/build_round2_formal_master_gate.py \
    scripts/run_labelcritic_benchmark_matrix.py \
    scripts/cached_key_organ_reselection_repair.py

  run_negative_manifest_audit

  local qwen2_summary="${BENCH_ROOT}/qwen2_current/labelcritic_repair_benchmark_summary.json"
  local qwen25_summary="${BENCH_ROOT}/qwen25/labelcritic_repair_benchmark_summary.json"
  local active_summary="${qwen2_summary}"

  log "Waiting up to 4 minutes for current vLLM."
  if wait_for_vllm_any; then
    log "Current vLLM is online; running Qwen2-VL-7B benchmark matrix."
    run_labelcritic_matrix_for_model "${BENCH_ROOT}/qwen2_current" "${QWEN2_MODEL}" || true
  else
    log "Current vLLM did not become ready within 4 minutes; attempting to start Qwen2-VL-7B locally."
    if validate_or_repair_model "${QWEN2_MODEL}" "Qwen/Qwen2-VL-7B-Instruct" && start_vllm_model "vllm_labelcritic_qwen2_em25" "${QWEN2_MODEL}" "Qwen2-VL-7B-Instruct"; then
      run_labelcritic_matrix_for_model "${BENCH_ROOT}/qwen2_current" "${QWEN2_MODEL}" || true
    else
      log "Qwen2-VL-7B local vLLM did not become ready within 4 minutes or local weights failed validation."
      write_blocked_benchmark_summary "${qwen2_summary}" "qwen2_vllm_start_or_weight_validation_failed"
    fi
  fi

  if benchmark_passed "${qwen2_summary}"; then
    log "LabelCritic benchmark passed on Qwen2-VL-7B."
    active_summary="${qwen2_summary}"
  else
    log "Qwen2-VL-7B benchmark failed or was not applicable; trying Qwen2.5-VL-7B."
    active_summary="${qwen25_summary}"
    if ! validate_or_repair_model "${QWEN25_MODEL}" "Qwen/Qwen2.5-VL-7B-Instruct"; then
      log "Qwen2.5-VL-7B local safetensors validation/repair failed."
      write_blocked_benchmark_summary "${qwen25_summary}" "qwen25_local_weights_validation_repair_failed"
    elif switch_vllm_to_qwen25; then
      run_labelcritic_matrix_for_model "${BENCH_ROOT}/qwen25" "${QWEN25_MODEL}" || true
    else
      log "Qwen2.5-VL-7B vLLM did not become ready within 4 minutes."
      write_blocked_benchmark_summary "${qwen25_summary}" "qwen25_vllm_start_timeout"
    fi
  fi

  run_cached_reselection_gate "${active_summary}"
  build_master_gate "${active_summary}"

  if python - "${MASTER_GATE}" <<'PY'
import json
import sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if doc.get("status") == "passed" else 1)
PY
  then
    log "Master gate passed. Cached reselection is ready; Round2 itself is not launched by this script."
  else
    log "Master gate blocked. Formal Round2 replacement remains fail-closed."
  fi
  log "Round2 gate + LabelCritic repair orchestration finished."
}

main "$@"
