#!/usr/bin/env python3
"""Audit the P0/P2/P3 repair contracts without launching training.

This script is intentionally lightweight: it validates target-space coverage,
optional E-step/student manifests, metric-contract CSV/JSON outputs, and the
GPU/vLLM readiness signals that gate heavier replay/training jobs.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent-harness"
sys.path.insert(0, str(AGENT_ROOT))

from cli_anything.medai.core.target_space import canonical_target_name


METRIC_CONTRACT_FIELDS = {
    "metric_target",
    "metric_subject",
    "metric_comparison",
    "metric_interpretation",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit repair-plan contracts and write a JSON report.")
    ap.add_argument("--target-config", default=str(ROOT / "configs/student_3d_prompt_target_organs.json"))
    ap.add_argument("--appearance", default=str(ROOT / "configs/organ_ct_appearance_373.json"))
    ap.add_argument("--estep-summary", default=None, help="Optional E-step summary JSON containing expected_targets/manifest_targets.")
    ap.add_argument("--student-manifest", default=None, help="Optional voxtell_prompt_student_manifest.json.")
    ap.add_argument("--selection-manifest", default=None, help="Full case×373 E-step selection JSON/JSONL.")
    ap.add_argument("--sampling-audit", default=None, help="Training sampling_audit.json.")
    ap.add_argument("--postprocess-csv", default=None, help="Containment postprocess per-mask CSV.")
    ap.add_argument("--metric-artifact", action="append", default=[], help="CSV or JSON artifact that must carry metric contract fields.")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--output", default=str(ROOT / "outputs/repair_plan_audit.json"))
    return ap.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_target_space(target_config: Path, appearance_path: Path) -> dict[str, Any]:
    target_doc = read_json(target_config)
    targets = [str(x).strip() for x in target_doc.get("target_organs", []) if str(x).strip()]
    target_set = set(targets)
    by_canonical: dict[str, list[str]] = defaultdict(list)
    for organ in targets:
        by_canonical[canonical_target_name(organ)].append(organ)

    appearance = read_json(appearance_path)
    entries = appearance.get("entries", [])
    appearance_map = appearance.get("organ_ct_appearance", {})
    entry_organs = {
        str(item.get("canonical_organ") or item.get("canonical_id") or "").strip()
        for item in entries
        if isinstance(item, dict)
    }
    required_entry_fields = {
        "canonical_organ", "display_name", "aliases", "expected_region",
        "ct_appearance", "shape_and_continuity_prior", "anatomical_constraints",
        "common_failure_modes", "rejection_rules", "labelcritic_instruction",
        "source", "requires_manual_review",
    }
    incomplete_entries = []
    for item in entries:
        if not isinstance(item, dict):
            incomplete_entries.append({"organ": "", "missing": sorted(required_entry_fields)})
            continue
        missing = sorted(
            k for k in required_entry_fields
            if k not in item or (k != "requires_manual_review" and not item.get(k))
        )
        if missing:
            incomplete_entries.append({
                "organ": item.get("canonical_organ") or item.get("canonical_id"),
                "missing": missing,
            })

    appearance_diversity = len({str(item.get("ct_appearance")) for item in entries if isinstance(item, dict)})
    return {
        "target_config": str(target_config),
        "appearance": str(appearance_path),
        "exact_target_count": len(targets),
        "unique_exact_target_count": len(target_set),
        "canonical_unique_count": len(by_canonical),
        "canonical_collisions": {k: v for k, v in sorted(by_canonical.items()) if len(v) > 1},
        "appearance_entry_count": len(entries),
        "appearance_map_count": len(appearance_map) if isinstance(appearance_map, dict) else None,
        "appearance_missing_targets": sorted(target_set - entry_organs),
        "appearance_extra_targets": sorted(entry_organs - target_set),
        "appearance_incomplete_entries": incomplete_entries[:50],
        "appearance_ct_text_unique_count": appearance_diversity,
        "status": "success" if (
            len(targets) == 373
            and len(target_set) == 373
            and len(entries) == 373
            and isinstance(appearance_map, dict)
            and len(appearance_map) == 373
            and not (target_set - entry_organs)
            and not incomplete_entries
            and appearance_diversity >= 100
        ) else "failed",
    }


def _manifest_items(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, dict) and isinstance(doc.get("items"), list):
        return [x for x in doc["items"] if isinstance(x, dict)]
    if isinstance(doc, list):
        return [x for x in doc if isinstance(x, dict)]
    return []


def audit_student_manifest(path: Path, target_count: int) -> dict[str, Any]:
    doc = read_json(path)
    items = _manifest_items(doc)
    cases = {str(x.get("case_id")) for x in items if x.get("case_id")}
    expected = len(cases) * target_count
    duplicate_training_keys = [
        key for key, count in Counter(
            (str(x.get("case_id")), str(x.get("organ")), str(x.get("prompt"))) for x in items
        ).items() if count > 1
    ]
    unresolved_in_training = [
        {"case_id": x.get("case_id"), "organ": x.get("organ")}
        for x in items if str(x.get("target_type")) == "unresolved_review"
    ]
    target_types = Counter(str(x.get("target_type") or "") for x in items)
    absent = [
        x for x in items
        if str(x.get("target_type") or "") == "absent_negative"
        or str(x.get("negative_source") or "") == "case_373_expected_absent"
        or str(x.get("zero_mask_role") or "") == "absent_negative_target_mask"
    ]
    bad_absent = [
        {
            "case_id": x.get("case_id"),
            "organ": x.get("organ"),
            "target_type": x.get("target_type"),
            "supervision_type": x.get("supervision_type"),
            "grade": x.get("grade"),
            "grade_scope": x.get("grade_scope"),
            "training_weight": x.get("training_weight"),
        }
        for x in absent
        if not (
            str(x.get("target_type")) == "absent_negative"
            and str(x.get("supervision_type")) == "negative"
            and str(x.get("grade")) == "A"
            and str(x.get("grade_scope")) == "absence_target"
        )
    ]
    organs_by_case: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.get("case_id") and item.get("organ"):
            organs_by_case[str(item["case_id"])].add(str(item["organ"]))
    case_target_gaps = {
        case: target_count - len(organs)
        for case, organs in sorted(organs_by_case.items())
        if len(organs) != target_count
    }
    return {
        "path": str(path),
        "num_items": len(items),
        "num_cases": len(cases),
        "expected_targets": expected,
        "manifest_targets": len(items),
        "target_type_counts": dict(target_types),
        "absent_negative_targets": len(absent),
        "bad_absent_negative_examples": bad_absent[:50],
        "case_target_gaps": case_target_gaps,
        "duplicate_training_keys": duplicate_training_keys[:50],
        "unresolved_review_items": unresolved_in_training[:50],
        "note": "Training manifest is an eligible subset; case×373 completeness is enforced on the full E-step manifest.",
        "status": "success" if items and not bad_absent and not duplicate_training_keys and not unresolved_in_training else "failed",
    }


def audit_estep_summary(path: Path) -> dict[str, Any]:
    doc = read_json(path)
    expected = int(doc.get("expected_targets") or 0)
    manifest = int(doc.get("manifest_targets") or doc.get("num_selected_organs") or 0)
    absent = int(doc.get("absent_negative_targets") or 0)
    candidates = int(doc.get("candidate_pseudo_targets") or 0)
    return {
        "path": str(path),
        "expected_targets": expected,
        "manifest_targets": manifest,
        "candidate_pseudo_targets": candidates,
        "absent_negative_targets": absent,
        "all_zero_masks": doc.get("all_zero_masks"),
        "target_type_counts": doc.get("target_type_counts"),
        "status": "success" if expected and manifest == expected and absent + candidates == manifest else "failed",
    }


def audit_metric_artifact(path: Path) -> dict[str, Any]:
    missing: list[str] = []
    rows_checked = 0
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = set(reader.fieldnames or [])
            missing = sorted(METRIC_CONTRACT_FIELDS - header)
            for _row in reader:
                rows_checked += 1
                if rows_checked >= 25:
                    break
    else:
        doc = read_json(path)
        if isinstance(doc, dict):
            missing = sorted(METRIC_CONTRACT_FIELDS - set(doc))
            rows_checked = 1
        elif isinstance(doc, list):
            for row in doc[:25]:
                rows_checked += 1
                if isinstance(row, dict):
                    missing.extend(sorted(METRIC_CONTRACT_FIELDS - set(row)))
            missing = sorted(set(missing))
        else:
            missing = sorted(METRIC_CONTRACT_FIELDS)
    return {
        "path": str(path),
        "rows_checked": rows_checked,
        "missing_metric_contract_fields": missing,
        "status": "success" if not missing else "failed",
    }


def audit_mainline_source() -> dict[str, Any]:
    loop = (ROOT / "agent-harness/cli_anything/medai/core/multimodel_loop.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "agent-harness/cli_anything/medai/core/labelcritic_wrapper.py").read_text(encoding="utf-8")
    failures = []
    if "enable_fusion: bool = False" not in loop:
        failures.append("formal fusion is not disabled by default")
    if "strict_labelcritic_selection: bool = True" not in loop:
        failures.append("strict LabelCritic selection is not the default")
    if 'selected = None if strict_labelcritic_selection' not in loop:
        failures.append("inconclusive LabelCritic can silently fall back")
    if "fusion_ablation_only" not in loop:
        failures.append("fusion is not isolated as an ablation artifact")
    if "--organ_description_json" not in wrapper or "rendered_organ_prompt_hash" not in wrapper:
        failures.append("373-organ description is not persisted and injected")
    return {"failures": failures, "status": "success" if not failures else "failed"}


def audit_selection_manifest(path: Path, target_count: int) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        items = _manifest_items(read_json(path))
    cases = {str(row.get("case_id")) for row in items if row.get("case_id")}
    required = {
        "case_id", "organ", "target_type", "candidate_count", "candidate_ids",
        "teacher_names", "teacher_families", "labelcritic_called",
        "selected_candidate_id", "selected_reason", "rejected_reasons",
        "failure_modes", "training_weight", "should_enter_student_training",
    }
    missing_fields = sorted({field for row in items for field in required if field not in row})
    duplicate_keys = [
        key for key, count in Counter((str(x.get("case_id")), str(x.get("organ"))) for x in items).items()
        if count != 1
    ]
    bad_selected = [
        {"case_id": x.get("case_id"), "organ": x.get("organ"), "selected_candidate_id": x.get("selected_candidate_id")}
        for x in items
        if x.get("selected_candidate_id") and x.get("selected_candidate_id") not in (x.get("candidate_ids") or [])
    ]
    silent_fallback = [
        {"case_id": x.get("case_id"), "organ": x.get("organ"), "method": x.get("selection_method")}
        for x in items
        if x.get("selection_status") in {"fallback", "selected"}
        and x.get("selection_method") in {"label_critic_inconclusive", "critic_disabled_fallback", "autolabel_core_evidence"}
    ]
    expected = len(cases) * target_count
    ok = len(items) == expected and not missing_fields and not duplicate_keys and not bad_selected and not silent_fallback
    return {
        "path": str(path), "num_cases": len(cases), "expected_targets": expected,
        "actual_targets": len(items), "missing_required_fields": missing_fields,
        "duplicate_case_organ_keys": duplicate_keys[:50],
        "invalid_selected_candidate_ids": bad_selected[:50],
        "silent_fallbacks": silent_fallback[:50],
        "status": "success" if ok else "failed",
    }


def audit_sampling(path: Path) -> dict[str, Any]:
    doc = read_json(path)
    history = doc.get("sampling_history", [])
    required = {"batch_positive_count", "batch_negative_count", "foreground_voxel_ratio", "all_zero_target_count", "negative_reason_counts"}
    missing = sorted({key for row in history for key in required if key not in row})
    return {
        "path": str(path), "windows": len(history), "missing_fields": missing,
        "status": "success" if history and not missing else "failed",
    }


def audit_postprocess(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {
        "containment_source", "parents", "roi_margin_mm", "voxels_before",
        "voxels_after", "false_positive_voxels_removed",
        "connected_component_count_before", "connected_component_count_after",
    }
    header = set(rows[0]) if rows else set()
    missing = sorted(required - header)
    return {
        "path": str(path), "rows": len(rows), "missing_fields": missing,
        "status": "success" if rows and not missing else "failed",
    }


def audit_gpu() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            rows.append({
                "index": parts[0],
                "name": parts[1],
                "memory_used_mib": int(float(parts[2])),
                "memory_total_mib": int(float(parts[3])),
                "utilization_gpu_pct": int(float(parts[4])),
            })
    return {
        "status": "success" if proc.returncode == 0 else "failed",
        "gpus": rows,
        "stderr": proc.stderr[-1000:] if proc.stderr else "",
    }


def audit_vllm(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models", timeout=5) as resp:
            return {"url": url, "online": True, "status_code": resp.status}
    except Exception as exc:
        return {"url": url, "online": False, "reason": str(exc)}


def main() -> int:
    args = parse_args()
    target_audit = audit_target_space(Path(args.target_config), Path(args.appearance))
    report: dict[str, Any] = {
        "stage": "repair_plan_audit",
        "target_space": target_audit,
        "estep_summary": audit_estep_summary(Path(args.estep_summary)) if args.estep_summary else None,
        "student_manifest": audit_student_manifest(Path(args.student_manifest), int(target_audit["exact_target_count"])) if args.student_manifest else None,
        "selection_manifest": audit_selection_manifest(Path(args.selection_manifest), int(target_audit["exact_target_count"])) if args.selection_manifest else None,
        "sampling_audit": audit_sampling(Path(args.sampling_audit)) if args.sampling_audit else None,
        "postprocess": audit_postprocess(Path(args.postprocess_csv)) if args.postprocess_csv else None,
        "metric_artifacts": [audit_metric_artifact(Path(p)) for p in args.metric_artifact],
        "mainline_source": audit_mainline_source(),
        "gpu": audit_gpu(),
        "vllm": audit_vllm(args.vllm_url),
    }
    checks = [target_audit, report["mainline_source"]]
    if report["estep_summary"]:
        checks.append(report["estep_summary"])
    if report["student_manifest"]:
        checks.append(report["student_manifest"])
    for name in ("selection_manifest", "sampling_audit", "postprocess"):
        if report[name]:
            checks.append(report[name])
    checks.extend(report["metric_artifacts"])
    report["status"] = "success" if all(c.get("status") == "success" for c in checks) else "failed"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
