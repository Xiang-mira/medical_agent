from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapter import normalize_totalseg_to_shapekit
from .agent_state import AgentState
from .annotation_manager import AnnotationManager
from .decision_policy import decide_next_action
from .human_review import append_review_item, read_review_queue
from .json_utils import write_json
from .projection_vlm_stub import make_vlm_label_expert_stub
from .qc_checker import check_segmentation_quality, check_trace_quality, merge_qc_results
from .radthinking import build_reasoning_trace, check_radthinking_patient, discover_radthinking_scans
from .shapekit_runner import run_shapekit
from .totalseg_runner import run_custom_inference, run_totalsegmentator
from .vlm_label_expert import run_vlm_label_expert


def _find_case_seg_dir(root: Path, case_id: str) -> Path | None:
    direct = root / case_id / "segmentations"
    if direct.exists(): return direct
    cands = list(root.glob("*/segmentations")) if root.exists() else []
    return cands[0] if cands else None


def _find_mask(root: Path, case_id: str, organ: str) -> Path | None:
    direct = root / case_id / "segmentations" / f"{organ}.nii.gz"
    if direct.exists(): return direct
    cands = list(root.glob(f"*/segmentations/{organ}.nii.gz")) if root.exists() else []
    return cands[0] if cands else None


def run_agent_loop(patient_folder: str | Path, output_folder: str | Path, backend: str = "totalseg", model_command: str | None = None, postprocess: str = "shapekit", shapekit_root: str | Path = "third_party/ShapeKit-main", fast: bool = True, task: str | None = None, roi_preset: str = "shapekit_abdomen", roi_subset: str | None = None, device: str | None = None, cpu_count: int = 2, organ: str = "liver", expected_organs: list[str] | None = None, enable_trace: bool = True, enable_qc: bool = True, retry_failed: bool = True, max_iterations: int = 1, dry_run: bool = False, max_scans: int | None = None, shapekit_timeout_sec: int = 120, enable_vlm_refinement: bool = False, vlm_backend_refinement: str = "stub", vlm_model_refinement: str = "qwen2.5vl:7b") -> dict[str, Any]:
    """Lightweight deterministic medical agent loop.

    It is not the full ScaleMAI retraining loop. It is the controller layer:
    observe patient data -> call tools -> inspect outputs/QC -> decide -> trace/review -> save state.
    """
    patient_root = Path(patient_folder).resolve(); out_root = Path(output_folder).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    patient_id = patient_root.name
    review_queue_path = out_root / "review_queue.jsonl"; trace_json_path = out_root / "patient_traces.json"; final_summary_path = out_root / "final_summary.json"
    state = AgentState(patient_id, out_root, {"backend": backend, "postprocess": postprocess, "organ_for_trace": organ, "expected_organs_for_qc": expected_organs, "enable_trace": enable_trace, "enable_qc": enable_qc, "retry_failed": retry_failed, "max_iterations": max_iterations, "dry_run": dry_run, "max_scans": max_scans, "shapekit_timeout_sec": shapekit_timeout_sec, "enable_vlm_refinement": enable_vlm_refinement, "vlm_backend_refinement": vlm_backend_refinement, "vlm_model_refinement": vlm_model_refinement, "scope": "first executable agent-loop prototype; no model retraining or clinical diagnosis inference"})
    patient_check = check_radthinking_patient(patient_root); state.add_event("observe_patient", patient_check.get("status", "unknown"), patient_check)
    scans = discover_radthinking_scans(patient_root)
    if max_scans is not None and max_scans > 0:
        scans = scans[:max_scans]
    if not scans:
        final = {"status": "failed", "reason": "No scan folders were discovered.", "patient_folder": str(patient_root), "agent_state_json": str(state.state_path)}
        write_json(final_summary_path, final); final["final_summary_json"] = str(final_summary_path); state.set_summary(final, "failed"); return final
    traces: list[dict[str, Any]] = []; scan_results: list[dict[str, Any]] = []; previous_mask: Path | None = None
    for scan in scans:
        scan_id = scan.scan_id; scan_out = out_root / "scan_outputs" / scan_id; raw_root = scan_out / "raw_predictions"; refined_root = scan_out / "refined_predictions"; log_root = scan_out / "logs"; case_id = f"{patient_id}_{scan_id}"
        state.add_event("observe_scan", "success", scan.to_dict(), scan_id)
        if not scan.ct_image or not scan.ct_image.exists():
            issue = {"patient_id": patient_id, "scan_id": scan_id, "organ": organ, "reason": "CT image missing; cannot run inference.", "suggested_review": "Check patient folder layout."}
            append_review_item(review_queue_path, issue); state.add_event("decision", "warning", {"next_action": "review_queue", "reason": issue["reason"]}, scan_id); scan_results.append({"scan_id": scan_id, "status": "skipped", "reason": issue["reason"]}); continue
        if backend == "totalseg":
            infer_result = run_totalsegmentator(str(scan.ct_image), str(raw_root), case_id=case_id, fast=fast, task=task, roi_preset=roi_preset, roi_subset=roi_subset, device=device, dry_run=dry_run)
        else:
            infer_result = run_custom_inference(str(scan.ct_image), str(raw_root), model_command=model_command or "", case_id=case_id, dry_run=dry_run) if model_command else {"stage": "infer", "backend": "custom", "status": "failed", "reason": "model_command is required for custom backend"}
        state.add_event("act_infer", infer_result.get("status", "unknown"), infer_result, scan_id)
        if retry_failed and infer_result.get("status") == "failed" and backend == "totalseg" and not dry_run:
            retry_result = run_totalsegmentator(str(scan.ct_image), str(raw_root), case_id=case_id, fast=True, task=task, roi_preset=roi_preset, roi_subset=roi_subset, device=device, dry_run=False)
            state.add_event("act_infer_retry_fast", retry_result.get("status", "unknown"), retry_result, scan_id)
            if retry_result.get("status") == "success": infer_result = retry_result
        adapter_result = None
        if infer_result.get("status") == "success" and infer_result.get("segmentation_output") and not dry_run:
            adapter_result = normalize_totalseg_to_shapekit(infer_result["segmentation_output"])
        elif dry_run and infer_result.get("segmentation_output"):
            adapter_result = {"stage": "adapter", "status": "dry_run", "segmentation_folder": infer_result.get("segmentation_output")}
        if adapter_result: state.add_event("act_adapter", adapter_result.get("status", "unknown"), adapter_result, scan_id)
        raw_seg_dir = _find_case_seg_dir(raw_root, case_id)
        raw_qc = check_segmentation_quality(raw_seg_dir, expected_organs=expected_organs) if enable_qc and not dry_run else {"stage": "qc_segmentation", "status": "dry_run" if dry_run else "skipped", "issues": []}
        state.add_event("check_raw_segmentation", raw_qc.get("status", "unknown"), raw_qc, scan_id)
        post_result = {"stage": "postprocess", "status": "skipped", "reason": f"postprocess={postprocess}"}; refined_qc = None; final_root = raw_root
        if postprocess == "shapekit" and infer_result.get("status") == "success":
            post_result = run_shapekit(str(Path(shapekit_root).resolve()), str(raw_root), str(refined_root), str(log_root), cpu_count=cpu_count, dry_run=dry_run, timeout_sec=shapekit_timeout_sec)
            state.add_event("act_postprocess_shapekit", post_result.get("status", "unknown"), post_result, scan_id)
            if post_result.get("status") == "success": final_root = refined_root
            refined_seg_dir = _find_case_seg_dir(refined_root, case_id)
            refined_qc = check_segmentation_quality(refined_seg_dir, expected_organs=expected_organs) if enable_qc and not dry_run else {"stage": "qc_segmentation_refined", "status": "dry_run" if dry_run else "skipped", "issues": []}
            state.add_event("check_refined_segmentation", refined_qc.get("status", "unknown"), refined_qc, scan_id)
        else:
            state.add_event("act_postprocess_shapekit", "skipped", post_result, scan_id)
        current_mask = _find_mask(final_root, case_id, organ)
        trace = None; trace_qc = None
        if enable_trace and not dry_run:
            trace = build_reasoning_trace(patient_root, scan_id, scan.ct_image, current_mask, previous_mask, organ, scan.report_path, scan.clinical_path, patient_root / "pathology.json" if (patient_root / "pathology.json").exists() else None)
            traces.append(trace); state.add_event("act_build_radthinking_trace", trace.get("status", "unknown"), trace, scan_id)
            trace_qc = check_trace_quality(trace) if enable_qc else {"stage": "qc_trace", "status": "skipped", "issues": []}; state.add_event("check_trace", trace_qc.get("status", "unknown"), trace_qc, scan_id)
        elif enable_trace and dry_run:
            trace = {"status": "dry_run", "scan_id": scan_id, "organ": organ, "note": "Trace is not built during dry-run because masks do not exist."}; state.add_event("act_build_radthinking_trace", "dry_run", trace, scan_id)
        merged_qc = merge_qc_results(raw_qc, refined_qc, trace_qc) if enable_qc else {"stage": "qc_merged", "status": "skipped", "issues": []}
        decision = decide_next_action(infer_result, post_result, merged_qc, trace); state.add_event("decide_next_action", decision.get("decision_status", "unknown"), {"decision": decision, "qc": merged_qc}, scan_id)
        if decision.get("next_action") in {"review_queue", "build_trace_and_review_queue", "build_trace_with_raw_prediction_and_review"}:
            # ScaleMAI flow: ShapeKit → VLM Label Expert → human escalation.
            vlm_stub = make_vlm_label_expert_stub(
                case_id=case_id, organ=organ,
                reason=decision.get("reason", "QC/inference issue requires expert review"),
            )
            state.add_event("act_vlm_label_expert_stub", vlm_stub.get("status", "not_implemented"), vlm_stub, scan_id)
            item = {
                "patient_id": patient_id,
                "scan_id": scan_id,
                "organ": organ,
                "decision": decision,
                "qc_issues": merged_qc.get("issues", []),
                "ct_image": str(scan.ct_image),
                "current_mask": str(current_mask) if current_mask else None,
                "report_path": str(scan.report_path) if scan.report_path else None,
                "infer_backend": infer_result.get("backend"),
                "infer_command": infer_result.get("command"),
                "infer_return_code": infer_result.get("return_code"),
                "infer_stdout_tail": infer_result.get("stdout_tail"),
                "infer_stderr_tail": infer_result.get("stderr_tail"),
                "postprocess_stderr_tail": post_result.get("stderr_tail"),
                "vlm_label_expert": vlm_stub,
                "suggested_review": "ScaleMAI flow: VLM Label Expert stub recorded. Inspect segmentation mask, report/context, temporal comparison, and inference stderr/stdout before accepting this trace.",
            }
            queue_path = append_review_item(review_queue_path, item); state.add_event("escalate_review_queue", "success", {"review_queue": queue_path, "item": item}, scan_id)

        # ── Refinement iterations (max_iterations > 1) ─────────────────────
        # If QC found issues and we have more iterations budgeted, run a
        # Label Verifier + VLM pass comparing any available reference labels
        # against the model predictions, then update annotations and re-check.
        refinement_log: list[dict] = []
        _vlm_backend = vlm_backend_refinement if enable_vlm_refinement else "stub"
        _vlm_model   = vlm_model_refinement   if enable_vlm_refinement else "qwen2.5vl:7b"
        if max_iterations >= 2 and enable_qc and not dry_run and merged_qc.get("issues"):
            ref_seg_dir = scan.ct_image.parent / "segmentations" if scan.ct_image else None
            pred_seg_dir = raw_seg_dir  # raw TotalSegmentator predictions

            ann_mgr = AnnotationManager(scan_out / "annotation_versions", case_id)
            # Seed AnnotationManager with raw reference labels (if they exist)
            for o in (expected_organs or [organ]):
                if ref_seg_dir:
                    ann_mgr.save_raw(o, ref_seg_dir / f"{o}.nii.gz")
                pred_mask_path = pred_seg_dir / f"{o}.nii.gz" if pred_seg_dir else None
                if pred_mask_path and pred_mask_path.exists():
                    ann_mgr.save_prediction(o, pred_mask_path, 0)

            for iter_idx in range(1, max_iterations):
                iter_organs = list({iss.get("organ", organ) for iss in merged_qc.get("issues", []) if iss.get("organ")}) or [organ]
                iter_decisions: list[dict] = []
                for o in iter_organs:
                    ann_a = ann_mgr.get_current(o)    # currently accepted annotation (may differ from raw after prior iterations)
                    ann_b = ann_mgr.get_prediction(o, 0)  # model prediction round 0
                    if ann_a is None and ann_b is None:
                        continue
                    vlm_refine = run_vlm_label_expert(
                        ct_image=scan.ct_image,
                        annotation_a=ann_a,
                        annotation_b=ann_b,
                        organ=o,
                        output_folder=scan_out / "refinement" / f"iter_{iter_idx:03d}",
                        vlm_backend=_vlm_backend,
                        vlm_model=_vlm_model,
                        case_id=case_id,
                    )
                    winner = vlm_refine.get("winner", "A")
                    ann_mgr.log_decision(o, iter_idx, vlm_refine)
                    ann_mgr.apply_update(o, winner, 0, candidate_a=ann_a, candidate_b=ann_b)
                    iter_decisions.append({"organ": o, "winner": winner, "decision": vlm_refine.get("decision"), "reason": vlm_refine.get("reason", "")})
                    state.add_event(f"act_refine_iter{iter_idx}_{o}", vlm_refine.get("status", "unknown"), vlm_refine, scan_id)

                # Re-run QC after this refinement iteration — check updated annotation dir, not raw predictions
                iter_qc = check_segmentation_quality(ann_mgr.updated_dir, expected_organs=expected_organs)
                state.add_event(f"check_refined_iter{iter_idx}", iter_qc.get("status", "unknown"), iter_qc, scan_id)
                refinement_log.append({
                    "iteration": iter_idx,
                    "organs_refined": iter_organs,
                    "decisions": iter_decisions,
                    "qc_after": iter_qc,
                    "annotation_summary": ann_mgr.summary(),
                })
                # Stop early if QC no longer has issues
                if not iter_qc.get("issues"):
                    break

        if current_mask and current_mask.exists(): previous_mask = current_mask
        scan_results.append({
            "scan_id": scan_id,
            "case_id": case_id,
            "ct_image": str(scan.ct_image),
            "infer_status": infer_result.get("status"),
            "infer_backend": infer_result.get("backend"),
            "infer_command": infer_result.get("command"),
            "infer_return_code": infer_result.get("return_code"),
            "infer_num_masks": infer_result.get("num_masks"),
            "infer_stdout_tail": infer_result.get("stdout_tail"),
            "infer_stderr_tail": infer_result.get("stderr_tail"),
            "postprocess_status": post_result.get("status"),
            "postprocess_return_code": post_result.get("return_code"),
            "postprocess_stderr_tail": post_result.get("stderr_tail"),
            "decision": decision,
            "final_segmentation_root": str(final_root),
            "current_mask_for_trace": str(current_mask) if current_mask else None,
            "qc_status": merged_qc.get("status"),
            "refinement_iterations": refinement_log if refinement_log else None,
        })
    if traces: write_json(trace_json_path, {"patient_id": patient_id, "organ": organ, "num_traces": len(traces), "traces": traces})

    # Build radthinking_style_reasoning summary — one entry per scan, matching
    # the teacher's expected result.json field name exactly.
    radthinking_summary: list[dict] = []
    for tr in traces:
        rt = tr.get("radthinking_trace", {})
        temp = rt.get("temporal_comparison", {})
        ctx = rt.get("clinical_context", {})
        radthinking_summary.append({
            "scan_id": tr.get("scan_id"),
            "organ": tr.get("organ"),
            "temporal_label": temp.get("temporal_label"),
            "volume_ratio": temp.get("volume_ratio"),
            "complexity": tr.get("complexity_level_prototype"),
            "suspicious_terms": ctx.get("suspicious_terms_found", []),
            "observation_status": rt.get("observation", {}).get("status"),
            "volume_cm3": rt.get("observation", {}).get("volume_cm3"),
        })

    review_items = read_review_queue(review_queue_path)
    successful_scans = [r for r in scan_results if r.get("infer_status") == "success"]
    failed_scans = [r for r in scan_results if r.get("infer_status") == "failed"]
    # Execution status: reflects whether inference ran successfully.
    # Agent decision status: reflects whether all cases were accepted or some
    # were routed to review. These are separate concerns — inference can succeed
    # but still produce cases that the agent flags for expert review.
    if not successful_scans:
        execution_status = "failed"
    elif failed_scans:
        execution_status = "warning"
    else:
        execution_status = "success"
    agent_decision_status = "warning" if review_items else "success"
    # Top-level status is the stricter of the two.
    if execution_status == "failed":
        final_status = "failed"
    elif execution_status == "warning" or agent_decision_status == "warning":
        final_status = "warning"
    else:
        final_status = "success"
    # Build agent_loop_trace: compact per-scan step record for teacher's result.json format.
    agent_loop_trace = [
        {
            "scan_id": r.get("scan_id"),
            "steps": [
                {"step": "infer", "status": r.get("infer_status"), "backend": r.get("infer_backend")},
                {"step": "postprocess", "status": r.get("postprocess_status")},
                {"step": "qc", "status": r.get("qc_status")},
                {"step": "decide", "next_action": r.get("decision", {}).get("next_action")},
            ],
            "refinement_iterations": len(r.get("refinement_iterations") or []),
        }
        for r in scan_results
    ]

    final = {"status": final_status, "execution_status": execution_status, "agent_decision_status": agent_decision_status, "pipeline": "lightweight_medical_agent_loop", "patient_id": patient_id, "patient_folder": str(patient_root), "output_folder": str(out_root), "implemented_steps": ["observe patient/timepoints", "call AI segmentation backend", "adapt masks to ShapeKit/PanTS style", "call ShapeKit postprocess", "QC check", "build RadThinking trace", "route uncertain cases to review_queue", "save agent_state", "AnnotationManager versioned storage", "Label Verifier + VLM Label Expert refinement (max_iterations)", "mini EM loop (em-loop command)"], "not_implemented_yet": ["ScaleMAI GPU retraining/continual tuning", "clinical diagnosis inference"], "max_iterations_used": max_iterations, "num_scans": len(scans), "num_successful_scans": len(successful_scans), "num_failed_scans": len(failed_scans), "num_traces": len(traces), "num_review_items": len(review_items), "agent_state_json": str(state.state_path), "patient_traces_json": str(trace_json_path) if traces else None, "review_queue_jsonl": str(review_queue_path) if review_items else None, "agent_loop_trace": agent_loop_trace, "radthinking_style_reasoning": radthinking_summary, "scan_results": scan_results, "review_items": review_items}
    write_json(final_summary_path, final); final["final_summary_json"] = str(final_summary_path); state.set_summary(final, final["status"]); return final
