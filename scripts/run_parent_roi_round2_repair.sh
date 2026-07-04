#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/teacher1/JHU-project1/medical_agent"
RUN="$ROOT/outputs/em_round_pure_cached_10case_formal_lite_20260703"
REFERENCE="$ROOT/outputs/formal_round1_final_20260627/round1/estep/annotation_versions"
CASES="$ROOT/outputs/formal_round1_final_20260627/case_list_10.csv"
TARGETS="liver_portal_vein,liver_hepatic_vein,pancreatic_duct,kidney_cortex,kidney_medulla,kidney_pelvicalyceal_system,lung_upper_left_lobe,lung_lower_left_lobe,lung_upper_right_lobe,lung_middle_right_lobe,lung_lower_right_lobe,lung_vessels,liver_segment_1,liver_segment_2,liver_segment_3,liver_segment_4,liver_segment_5,liver_segment_6,liver_segment_7,liver_segment_8"
LOG="$RUN/parent_roi_round2_repair.log"

cd "$ROOT"
export PYTHONPATH="$ROOT/agent-harness:${PYTHONPATH:-}"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%FT%TZ)] parent-ROI repair pipeline start"

python scripts/postprocess_student_containment.py \
  --input-root "$RUN/round1/student_predictions" \
  --output-root "$RUN/round1/student_predictions_containment_only" \
  --teacher-root "$REFERENCE" \
  --case-list "$CASES" \
  --organs "$TARGETS" \
  --overwrite

python scripts/run_student_infer_then_round2.py \
  --round 1 \
  --output-root "$RUN" \
  --case-list "$CASES" \
  --prompts "$TARGETS" \
  --cascade \
  --teacher-root "$REFERENCE" \
  --cascaded-output-root "$RUN/round1/student_predictions_cascaded"

rm -rf "$RUN/round1/student_predictions_postprocessed"
python scripts/build_student_round2_allowlist.py \
  --raw-root "$RUN/round1/student_predictions" \
  --containment-root "$RUN/round1/student_predictions_containment_only" \
  --cascade-root "$RUN/round1/student_predictions_cascaded" \
  --reference-root "$REFERENCE" \
  --case-list "$CASES" \
  --output-dir "$RUN/round1/metrics/student_round2_gate" \
  --postprocessed-root "$RUN/round1/student_predictions_postprocessed"

test ! -e "$RUN/round2"
export MEDAI_OUTPUT_ROOT="$RUN"
export MEDAI_CASE_LIST="$CASES"
export MEDAI_STUDENT_BACKEND="voxtell_style_3d_prompt"
export MEDAI_ENABLE_SHAPEKIT=1
export MEDAI_ENABLE_CRITIC=1
export MEDAI_DEBUG_ALLOW_NO_LABELCRITIC=0
export MEDAI_TEACHER_INFERENCE_MODE="hierarchical_roi"
export MEDAI_CANDIDATE_MODE="route_pruned_with_competition"

echo "[$(date -u +%FT%TZ)] quality gate passed; starting clean Round2"
python scripts/run_pure_cached_10case_round2_estep.py
