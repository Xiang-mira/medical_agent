from __future__ import annotations


def decide_next_action(infer_result: dict | None, postprocess_result: dict | None, qc_result: dict | None, trace: dict | None = None) -> dict:
    if not infer_result or infer_result.get("status") == "failed":
        return {"next_action": "review_queue", "decision_status": "failed", "reason": "AI inference failed.", "suggested_followup": "Retry with --fast / smaller ROI, inspect command/stdout/stderr, or send to review."}
    if postprocess_result and postprocess_result.get("status") == "failed":
        return {"next_action": "build_trace_with_raw_prediction_and_review", "decision_status": "warning", "reason": "ShapeKit post-processing failed; raw prediction may still be usable.", "suggested_followup": "Inspect ShapeKit logs and review queue."}
    if qc_result:
        issues = qc_result.get("issues", []) or []
        high = [i for i in issues if i.get("severity") == "high"]
        med = [i for i in issues if i.get("severity") == "medium"]
        if high:
            return {"next_action": "review_queue", "decision_status": "warning", "reason": f"QC found {len(high)} high-severity issue(s).", "suggested_followup": "Send case to VLM Label Expert or human review before trusting the trace."}
        if med:
            return {"next_action": "build_trace_and_review_queue", "decision_status": "warning", "reason": f"QC found {len(med)} medium-severity issue(s).", "suggested_followup": "Keep trace but mark it for review."}
    return {"next_action": "accept_trace", "decision_status": "success", "reason": "Inference/postprocess/QC passed.", "suggested_followup": "Use the trace as an agent-readable intermediate result."}
