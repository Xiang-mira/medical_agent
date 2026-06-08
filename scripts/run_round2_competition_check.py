#!/usr/bin/env python3
"""Run a small Round2 competition check with preseeded pseudo/student masks.

This validates the teacher-meeting rule that a trained student is only a new
candidate source. It must compete against the previous selected pseudo label
and must not automatically overwrite it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop


def parse_args() -> argparse.Namespace:
    base = ROOT / "outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit"
    ap = argparse.ArgumentParser(description="Validate Round2 student-vs-Round1 competition wiring.")
    ap.add_argument("--case-list", default=str(base / "case_list_subset.csv"))
    ap.add_argument("--round-prev-selected", default=str(base / "estep/annotation_versions"))
    ap.add_argument("--student-prev", default=str(base / "student_real_infer_ministep/round1/student_predictions"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/audit_21_models/round2_competition_check"))
    ap.add_argument("--models", default="mock_seg", help="Additional comma-separated comparator models.")
    ap.add_argument("--organs", default="liver")
    ap.add_argument("--critic-backend", default="stub", choices=["stub", "labelcritic"])
    ap.add_argument("--critic-base-url", default="http://localhost")
    ap.add_argument("--critic-port", type=int, default=8000)
    ap.add_argument("--timeout-sec", type=int, default=600)
    ap.add_argument("--no-shapekit", action="store_true", help="Debug only; formal checks should keep ShapeKit on.")
    return ap.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    estep = out / "estep"
    models = [x.strip() for x in args.models.replace(";", ",").split(",") if x.strip()]
    organs = [x.strip() for x in args.organs.replace(";", ",").split(",") if x.strip()]

    result = run_multimodel_annotation_loop(
        case_list=Path(args.case_list).resolve(),
        output_folder=estep,
        models=models,
        organs=organs,
        registry_path=ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=False,
        shapekit_root=ROOT / "third_party/ShapeKit-main",
        enable_shapekit=not args.no_shapekit,
        enable_critic=True,
        critic_backend=args.critic_backend,
        critic_base_url=args.critic_base_url,
        critic_port=args.critic_port,
        dry_run=False,
        timeout_sec=args.timeout_sec,
        device="cuda",
        resume=False,
        preseeded_model_dirs={
            "round_prev_selected": Path(args.round_prev_selected).resolve(),
            "student_prev": Path(args.student_prev).resolve(),
        },
    )

    manifest_path = estep / "training_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else []
    competition = result.get("round2_competition_audit") or {}
    per_source = competition.get("per_source") or {}
    round_prev_entries = int((per_source.get("round_prev_selected") or {}).get("candidate_entries") or 0)
    student_entries = int((per_source.get("student_prev") or {}).get("candidate_entries") or 0)
    selected_student = int((per_source.get("student_prev") or {}).get("selected_entries") or 0)
    selected_round_prev = int((per_source.get("round_prev_selected") or {}).get("selected_entries") or 0)
    manifest_with_both = [
        item for item in manifest
        if {"round_prev_selected", "student_prev"}.issubset(set(item.get("candidate_models") or []))
    ]

    status = "success"
    failures: list[str] = []
    if result.get("status") != "success":
        failures.append(f"E-step status is {result.get('status')}")
    if not manifest:
        failures.append("training_manifest.json is empty or missing")
    if not competition:
        failures.append("round2_competition_audit missing")
    if round_prev_entries <= 0:
        failures.append("round_prev_selected never appeared as a candidate")
    if student_entries <= 0:
        failures.append("student_prev never appeared as a candidate")
    if not manifest_with_both:
        failures.append("no manifest item contains both round_prev_selected and student_prev")
    if failures:
        status = "failed"

    summary = {
        "stage": "round2_competition_check",
        "status": status,
        "failures": failures,
        "output_dir": str(out),
        "estep_status": result.get("status"),
        "total_updated": result.get("total_updated"),
        "total_labelcritic_decisions": result.get("total_labelcritic_decisions"),
        "critic_backend": args.critic_backend,
        "enable_shapekit": not args.no_shapekit,
        "manifest_items": len(manifest),
        "manifest_items_with_both_preseeded_sources": len(manifest_with_both),
        "round_prev_candidate_entries": round_prev_entries,
        "student_candidate_entries": student_entries,
        "round_prev_selected_entries": selected_round_prev,
        "student_selected_entries": selected_student,
        "round2_competition_audit": competition,
        "vlm_decision_lines": _line_count(estep / "vlm_decisions.jsonl"),
        "review_queue_lines": _line_count(estep / "review_queue.jsonl"),
        "accuracy_warning": "This validates competition wiring and pseudo-label consistency, not true expert-label accuracy.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "round2_competition_check_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
