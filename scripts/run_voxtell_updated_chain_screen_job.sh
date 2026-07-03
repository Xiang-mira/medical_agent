#!/usr/bin/env bash
set -euo pipefail

cd /home/teacher1/JHU-project1/medical_agent

SOURCE_ESTEP="${SOURCE_ESTEP:-outputs/formal_round1_final_20260627/round1/estep}"
EXP_ROOT="${EXP_ROOT:-outputs/voxtell_updated_chain_20260702}"
ROUND1_ROOT="${EXP_ROOT}/round1"
CASE_LIST="${EXP_ROOT}/case_list_from_old_round1_estep.csv"
LOG_DIR="${EXP_ROOT}/logs"
LOG_FILE="${LOG_DIR}/voxtell_updated_chain.log"
VLLM_SCREEN="${VLLM_SCREEN:-vllm_voxtell_updated_chain}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "==== VoxTell updated EM chain started: $(date -Is) ===="
echo "cwd=$(pwd)"
echo "source_estep=${SOURCE_ESTEP}"
echo "exp_root=${EXP_ROOT}"
echo "case_list=${CASE_LIST}"
echo "log_file=${LOG_FILE}"
echo

export PYTHONPATH="agent-harness:${PYTHONPATH:-}"
export SOURCE_ESTEP
export EXP_ROOT
export ROUND1_ROOT
export CASE_LIST
export VLLM_SCREEN
export MEDAI_STUDENT_BACKEND="voxtell_style_3d_prompt"
export MEDAI_VOXTELL_MSTEP_MODE="project_voxtell_prompt_distillation_student"
export MEDAI_CANONICAL_TRAINING_BACKEND="project_voxtell_prompt_distillation_student"
export MEDAI_EXPERIMENT_PROFILE="advisor_aligned_default"
export MEDAI_VOXTELL_TRAINING_PROFILE="quality_weighted_ablation"
export MEDAI_ENABLE_VOXTELL_TRAINING="1"
export MEDAI_VOXTELL_MODEL_DIR="/home/teacher1/JHU-project1/medical_agent/checkpoints/VoxTell/voxtell_v1.1"
export MEDAI_TEXT_ENCODING_MODEL="/home/teacher1/JHU-project1/medical_agent/checkpoints/Qwen/Qwen3-Embedding-4B"
export MEDAI_TRAINABLE_SCOPE="prompt_path"
export MEDAI_BCE_POS_WEIGHT_CAP="20"
export MEDAI_MSTEP_BATCH_SIZE="${MEDAI_MSTEP_BATCH_SIZE:-2}"
export MEDAI_FINETUNE_EPOCHS="${MEDAI_FINETUNE_EPOCHS:-1}"
export MEDAI_MAX_STEPS="${MEDAI_MAX_STEPS:-0}"
export MEDAI_ENABLE_SHAPEKIT="1"
export MEDAI_ENABLE_CRITIC="1"
export MEDAI_TEACHER_INFERENCE_MODE="hierarchical_roi"
export MEDAI_CANDIDATE_MODE="route_pruned_with_competition"
export MEDAI_OUTPUT_ROOT="${EXP_ROOT}"
export MEDAI_CASE_LIST="${CASE_LIST}"

if [[ "${MEDAI_STUDENT_BACKEND}" == "vista3d_legacy" ]]; then
  echo "ERROR: legacy VISTA3D student backend is forbidden for this updated VoxTell chain."
  exit 2
fi

echo "---- Step 0/5: derive case list and verify old Round1 E-step evidence ----"
python - <<'PY'
from pathlib import Path
import csv
import json
import os
import sys

source = Path(os.environ["SOURCE_ESTEP"]).resolve()
case_list = Path(os.environ["CASE_LIST"]).resolve()
meta_paths = sorted((source / "annotation_versions").glob("*/selection_metadata.json"))
if not meta_paths:
    raise SystemExit(f"No selection_metadata.json found under {source / 'annotation_versions'}")

rows = []
bad = []
for path in meta_paths:
    doc = json.loads(path.read_text(encoding="utf-8"))
    case_id = str(doc.get("case_id") or path.parent.name)
    ct_path = Path(str(doc.get("ct_path") or ""))
    if not ct_path.exists():
        bad.append({"case_id": case_id, "ct_path": str(ct_path)})
    rows.append({
        "case_id": case_id,
        "ct_path": str(ct_path),
        "annotation_folder": str(Path("data/PanTS/LabelTr") / case_id / "segmentations"),
    })

case_list.parent.mkdir(parents=True, exist_ok=True)
with case_list.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case_id", "ct_path", "annotation_folder"])
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "stage": "old_round1_estep_preflight",
    "status": "success" if not bad else "failed",
    "source_estep": str(source),
    "num_cases": len(rows),
    "case_list": str(case_list),
    "missing_ct_paths": bad,
}
(case_list.parent / "old_round1_estep_preflight.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
if bad:
    sys.exit(1)
PY

echo
echo "---- Step 1/5: metadata replay from old Round1 E-step; no teacher/model inference ----"
python scripts/rebuild_mstep_from_existing_estep.py \
  --source-estep "${SOURCE_ESTEP}" \
  --output-root "${ROUND1_ROOT}" \
  --case-list "${CASE_LIST}" \
  --target-config configs/student_3d_prompt_target_organs.json \
  --model-dir "${MEDAI_VOXTELL_MODEL_DIR}"

echo
echo "---- Step 2/5: updated VoxTell M-step + student inference/QC through current EM entry ----"
screen -S "${VLLM_SCREEN}" -X quit >/dev/null 2>&1 || true
python scripts/run_em_training.py \
  --rounds 1 \
  --case-list "${CASE_LIST}" \
  --output-root "${EXP_ROOT}" \
  --voxtell-mstep-mode "${MEDAI_VOXTELL_MSTEP_MODE}"

python - <<'PY'
from pathlib import Path
import json
import os

summary_path = Path(os.environ["EXP_ROOT"]) / "round1" / "round_summary.json"
mstep_path = Path(os.environ["EXP_ROOT"]) / "round1" / "mstep" / "voxtell_prompt_mstep_result.json"
if not summary_path.exists():
    raise SystemExit(f"Round1 summary missing after M-step stage: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
mstep = json.loads(mstep_path.read_text(encoding="utf-8")) if mstep_path.exists() else {}
eligible = bool(mstep.get("eligible_for_next_round_prompt_student", mstep.get("checkpoint_eligible_for_next_round", False)))
ok = summary.get("mstep_status") == "success" and eligible
print(json.dumps({
    "stage": "round1_mstep_required_before_round2",
    "status": "success" if ok else "failed",
    "round_summary": str(summary_path),
    "mstep_result": str(mstep_path),
    "mstep_status": summary.get("mstep_status"),
    "checkpoint_eligible_for_next_round": eligible,
    "training_status": mstep.get("training_status"),
}, indent=2, ensure_ascii=False))
if not ok:
    raise SystemExit("Round1 M-step did not produce an eligible updated VoxTell checkpoint; refusing to enter Round2 competition.")
PY

echo
echo "---- Step 3/5: ensure vLLM/LabelCritic is online for Round2 competition ----"
python - <<'PY'
import os
import subprocess
import time
import urllib.request
from pathlib import Path

root = Path("/home/teacher1/JHU-project1/medical_agent")
model = root / "checkpoints/Qwen/Qwen2-VL-7B-Instruct"
screen_name = os.environ.get("VLLM_SCREEN", "vllm_voxtell_updated_chain")

def online() -> bool:
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
        return True
    except Exception:
        return False

if online():
    print("vLLM already online at http://localhost:8000")
else:
    cmd = [
        "screen", "-dmS", screen_name, "bash", "-lc",
        (
            f"cd {root} && "
            "python -m vllm.entrypoints.openai.api_server "
            f"--model {model} "
            "--port 8000 --max-model-len 4096 --gpu-memory-utilization 0.4"
        ),
    ]
    subprocess.run(cmd, check=True)
    print(f"Started vLLM screen: {screen_name}")
    for idx in range(60):
        time.sleep(5)
        if online():
            print(f"vLLM online after {(idx + 1) * 5}s")
            break
    else:
        raise SystemExit("vLLM did not become ready within 300s")
PY

echo
echo "---- Step 4/5: Round2 E-step competition only; reuse Round1 teacher cache + previous selected + student_prev ----"
python - <<'PY'
import json
import os
from pathlib import Path

import scripts.run_em_training as em

em.CASE_LIST = Path(os.environ["CASE_LIST"]).resolve()
em.OUTPUT_ROOT = Path(os.environ["EXP_ROOT"]).resolve()
em.LOG_FILE = em.OUTPUT_ROOT / "training.log"

em.ensure_current_student_backend_allowed()
em.ensure_formal_teacher_pool_registered()
em.ensure_formal_quality_gates()

round_idx = 2
estep_result = em.run_estep(round_idx)
dashboard = em.build_round_label_scoring_dashboard(round_idx)
manifest_path = em.build_student_dataset(round_idx)
gate = em.formal_estep_gate(round_idx, estep_result, manifest_path)

gate_path = em.OUTPUT_ROOT / f"round{round_idx}" / "estep" / "formal_gate.json"
gate_path.parent.mkdir(parents=True, exist_ok=True)
gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

summary = {
    "round": round_idx,
    "stage": "round2_estep_competition_only",
    "status": "success" if estep_result.get("status") == "success" and gate.get("status") == "success" else "failed",
    "estep_status": estep_result.get("status"),
    "estep_result": estep_result,
    "estep_formal_gate": gate,
    "mstep_status": "not_run_by_design",
    "student_backend": em.STUDENT_BACKEND,
    "competition_policy": "Round2 reuses old Round1 teacher cache, injects round_prev_selected, and injects eligible student_prev as a competing candidate; student does not auto-overwrite prior pseudo labels.",
    "label_scoring_dashboard": dashboard,
    "round2_manifest": str(manifest_path),
}
summary_path = em.OUTPUT_ROOT / f"round{round_idx}" / "round_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary["status"] != "success":
    raise SystemExit(1)
PY

echo
echo "---- Step 5/5: final artifact summary ----"
python - <<'PY'
from pathlib import Path
import json
import os

root = Path(os.environ["EXP_ROOT"]).resolve()
paths = [
    root / "case_list_from_old_round1_estep.csv",
    root / "round1/mstep/voxtell_prompt_mstep_result.json",
    root / "round1/student_predictions/student_inference_summary.json",
    root / "round1/metrics/evaluation_chain/evaluation_chain_summary.json",
    root / "round2/estep/formal_gate.json",
    root / "round2/round_summary.json",
]
summary = {"stage": "voxtell_updated_chain_final_artifacts", "exp_root": str(root), "artifacts": []}
for path in paths:
    item = {"path": str(path), "exists": path.exists()}
    if path.suffix == ".json" and path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            item["status"] = doc.get("status") or doc.get("training_status") or doc.get("estep_status")
        except Exception as exc:
            item["read_error"] = str(exc)
    summary["artifacts"].append(item)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "==== VoxTell updated EM chain finished: $(date -Is) ===="
