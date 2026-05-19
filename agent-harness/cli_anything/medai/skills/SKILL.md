# medai-cli skill

Use `medai-cli` to run a medical AI segmentation, label verification, and longitudinal reasoning pipeline.

## Tool roles

- **TotalSegmentator**: AI segmentation inference backend (104 organ classes).
- **ShapeKit**: anatomy-aware post-processing — corrects anatomically implausible masks.
- **Label Verifier**: computes DSC between two annotations and routes: accept / send_to_vlm / auto_replace.
- **VLM Label Expert**: sends CT projection images to a vision model (Ollama qwen2.5vl:7b) to compare two annotations; outputs winner A/B + confidence.
- **AnnotationManager**: versioned annotation storage — raw/, predictions/round_NNN/, updated/, decisions.jsonl.
- **EM Loop**: annotation-level refinement loop — E-step (Label Verifier + VLM) + M-step (annotation update; retraining is stub). Operates on a single case/organ pair.
- **RadThinking**: patient-level longitudinal reasoning trace — observation, temporal_comparison, clinical_context, diagnostic_conclusion.
- **Human Review Queue**: routes uncertain cases to review_queue.jsonl for human inspection.
- **Agent Loop**: patient-level controller — Observe → Infer → QC → Label Verify → VLM → EM → Review → Trace → Save State. Drives per-scan inference + ShapeKit + optional EM refinement for all scans of one patient.

## Important commands

```bash
# Environment check
medai-cli --json doctor

# AI inference
medai-cli --json infer --image ct.nii.gz --output-folder outputs/raw --backend totalseg --fast

# ShapeKit post-processing
medai-cli --json postprocess --input-folder outputs/raw --output-folder outputs/refined --shapekit-root third_party/ShapeKit-main

# Label Verifier (DSC routing)
medai-cli --json label-verify \
  --annotation-folder data/pants_real/CASE/reference_labels/CASE/segmentations \
  --prediction-folder outputs/raw/CASE/segmentations \
  --organs pancreas,liver,spleen

# Build 2D projection PNGs for manual inspection (no VLM needed)
medai-cli --json projection-build \
  --ct-image ct.nii.gz \
  --annotation-a ref/pancreas.nii.gz \
  --annotation-b pred/pancreas.nii.gz \
  --organ pancreas \
  --output-folder outputs/projections
  # --strict-alignment  # add this flag to fail if mask/CT affine or shape mismatch

# VLM Label Expert (requires Ollama + qwen2.5vl:7b)
medai-cli --json vlm-label-expert \
  --ct-image ct.nii.gz \
  --annotation-a ref/pancreas.nii.gz \
  --annotation-b pred/pancreas.nii.gz \
  --organ pancreas \
  --output-folder outputs/vlm_out \
  --vlm-backend ollama \
  --vlm-model qwen2.5vl:7b

# EM Loop (dry-run safe)
medai-cli --json em-loop \
  --case-id CASE_ID \
  --ct-image ct.nii.gz \
  --annotation-folder ref/segmentations \
  --output-folder outputs/em_out \
  --organs pancreas,liver \
  --vlm-backend stub \
  --dry-run

# RadThinking trace check
medai-cli --json radthinking-check --patient-folder data/radthinking_demo/patient_001

# RadThinking trace build
medai-cli --json trace-build --ct-image ct.nii.gz --current-mask liver.nii.gz --organ liver

# Full agent loop
medai-cli --json agent-loop \
  --patient-folder data/patient_001 \
  --output-folder outputs/agent_loop \
  --backend totalseg \
  --postprocess shapekit \
  --shapekit-root third_party/ShapeKit-main \
  --organ liver \
  --expected-organs liver,pancreas,aorta

# PanTS evaluation
medai-cli --json pants-eval-case \
  --pred-folder outputs/raw/PanTS_00000023_PanTS_00000023/segmentations \
  --reference-label-folder data/pants_real/PanTS_00000023/reference_labels/PanTS_00000023/segmentations \
  --organs pancreas,liver,spleen,aorta
```

Always prefer JSON output when the command is called by an agent:

```bash
medai-cli --json <command> ...
```

## Output artifacts

- `agent_state.json`: auditable state trajectory (one event per step).
- `final_summary.json`: final pipeline summary with `radthinking_style_reasoning`.
- `patient_traces.json`: full RadThinking-style traces (4-step per scan).
- `review_queue.jsonl`: cases needing VLM/human review.
- `annotation_versions/`: AnnotationManager versioned storage (raw/, predictions/, updated/, decisions.jsonl).
- `rounds_metrics.json`: EM loop per-round metrics.

## Key JSON fields in agent-loop output

```json
{
  "status": "success | warning | failed",
  "execution_status": "success | warning | failed",
  "agent_decision_status": "success | warning",
  "implemented_steps": [...],
  "radthinking_style_reasoning": [
    {
      "scan_id": "...",
      "organ": "...",
      "temporal_label": "STABLE | GROWING | SHRINKING | NEW | RESOLVED",
      "complexity": "PERCEPTUAL | TEMPORAL | INTEGRATIVE | AMBIGUOUS",
      "suspicious_terms": [...]
    }
  ],
  "scan_results": [...],
  "review_items": [...]
}
```

## Safety and scope

This tool does not infer clinical diagnosis. It only processes provided CT images, masks, reports, clinical JSON, and pathology/follow-up JSON. The M-step (model retraining) in the EM loop is a stub — it logs intent but does not execute without a GPU.
