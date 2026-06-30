#!/usr/bin/env python3
"""CPU-only report that reverse-engineers the implemented AutoLabelCore C formula.

The report is based on configs/autolabel_core.yaml and the implemented scoring
contract in cli_anything.medai.core.auto_label_core. It does not run inference,
training, Qwen, LabelCritic, torch, CUDA, or embedding models.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.auto_label_core import score_evidence_record

DEFAULT_CONFIG = ROOT / "configs/autolabel_core.yaml"

CODE_FORMULA = {
    "scope": "per_case_per_organ_mask_candidate_or_selected_pseudo_label",
    "implemented_formula": "C = clip01(normalized_evidence_confidence - conflict_penalty - correlation_penalty + labelcritic_tiebreak_adjustment)",
    "normalization": "If missing_evidence_policy=normalize_available, normalized_evidence_confidence = raw_weighted_score / available_weight_sum over available or failed evidence; failed evidence contributes score 0. Otherwise it is raw_weighted_score.",
    "raw_weighted_score": "sum_i w_i * S_i over evidence terms with status available or failed",
    "penalty_formula": {
        "conflict_penalty": "min(0.15, family_conflict_penalty + longtail_corruption_penalty)",
        "family_conflict_penalty": "if family_consensus exists and independent_family_count>=2: min(0.10, max(0, high_family_agreement_dice - S_family) * 0.125), else 0",
        "longtail_corruption_penalty": "min(longtail_confidence_penalty_cap, longtail_confidence_penalty_cap * structural_corruption_probability)",
        "correlation_penalty": "min(0.05, 0.01 * max(0, len(independent_eligible) - independent_family_count))",
        "labelcritic_tiebreak_adjustment": "only if labelcritic_supported=True; clipped to +/- labelcritic_tiebreak_cap",
    },
    "hard_gates": [
        "missing_candidate", "missing_final_mask", "all_candidates_failed_qc", "candidate_qc_fail",
        "missing_file", "unreadable_mask", "geometry_mismatch", "shape_mismatch_ct",
        "identity_mismatch", "left_right_mismatch", "zero_volume_mask",
        "selected_candidate_qc_status == fail", "identity_status not in {valid, empty, None}",
    ],
    "qwen_usage": "C formula itself does not require Qwen. LabelCritic/Qwen can only contribute a bounded tiebreak adjustment if explicitly enabled and supported.",
}

EVIDENCE_DESCRIPTIONS = {
    "family_consensus": "Independent teacher-family agreement; computed from representative masks after collapsing correlated checkpoints.",
    "ct_support": "Generic CT-mask support from boundary gradient, inside/outside contrast, and connected-component compactness.",
    "anatomy_plausibility": "Anatomy/parent containment and structural identity plausibility.",
    "perturbation_stability": "Cached TTA/perturbation stability against inverse-mapped variants, if available.",
    "cross_round_stability": "Agreement with prior-round pseudo-label, if available.",
    "loo_model_reliability": "Leave-one-family-out estimated model reliability, if available.",
    "teacher_student_oof": "Verified out-of-fold student-teacher agreement; only valid when provenance proves held-out status.",
}


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to read AutoLabelCore config")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evidence_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "term": term,
            "symbol": f"S_{{{term}}}",
            "weight": float(weight),
            "description": EVIDENCE_DESCRIPTIONS.get(term, "Configured evidence component."),
            "input_field": f"{term}_score" if term != "loo_model_reliability" else "loo_model_reliability_score | estimated_model_reliability",
        }
        for term, weight in (config.get("evidence_weights") or {}).items()
    ]


def grade_policy(config: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = config.get("thresholds") or {}
    weights = config.get("training_weights") or {}
    soft = config.get("soft_labels") or {}
    return [
        {"grade": "A", "condition": f"C >= {thresholds.get('grade_a')} and independent_family_count >= 2, unless expert_verified", "target_type": "hard", "training_weight": weights.get("A")},
        {"grade": "B", "condition": f"C >= {thresholds.get('grade_b')} or single-teacher OOF cap permits B", "target_type": "hard", "training_weight": weights.get("B")},
        {"grade": "C", "condition": f"{thresholds.get('grade_c')} <= C < {thresholds.get('grade_b')} or severe conflict with enough C", "target_type": "soft only if probability mask exists; otherwise provisional", "training_weight": f"hard={weights.get('C')}, soft={soft.get('c_grade_training_weight') if soft.get('enabled') else 0.0}"},
        {"grade": "D", "condition": "hard failure or C below grade_c", "target_type": "rejected", "training_weight": weights.get("D")},
    ]


def load_example_selection(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    candidate_lists = []
    if isinstance(doc, dict):
        for key in ("selection_rows", "selected_organs", "items"):
            if isinstance(doc.get(key), list):
                candidate_lists.append(doc[key])
    elif isinstance(doc, list):
        candidate_lists.append(doc)
    for rows in candidate_lists:
        for row in rows:
            if isinstance(row, dict) and row.get("organ") == "liver":
                return row
        for row in rows:
            if isinstance(row, dict):
                return row
    if isinstance(doc, dict):
        if doc.get("organ") == "liver":
            return doc
        return {**doc, "_example_warning": "no real liver row found; using provided top-level record"}
    return None


def example_breakdown(selection: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, Any]:
    if not selection:
        return {
            "status": "no_example_selection_provided",
            "message": "No real liver/example selection was provided. Pass --example-selection-json to compute a real case-organ C breakdown. Do not fabricate C=0.95 in slides.",
        }
    weights = config.get("evidence_weights") or {}
    scores = selection.get("evidence_scores") or {}
    details = selection.get("evidence_details") or {}
    terms = []
    active_weight = 0.0
    raw = 0.0
    for term, weight in weights.items():
        detail = details.get(term) if isinstance(details, dict) else None
        score = scores.get(term)
        if isinstance(detail, dict) and detail.get("score") is not None:
            score = detail.get("score")
        status = (detail or {}).get("status", "available" if score is not None else "unavailable") if isinstance(detail, dict) else ("available" if score is not None else "unavailable")
        contribution = None
        if status in {"available", "failed"}:
            active_weight += float(weight)
            contribution = float(weight) * float(score or 0.0)
            raw += contribution
        terms.append({"term": term, "weight": weight, "score": score, "status": status, "weighted_contribution": contribution})
    normalized = raw / active_weight if str(config.get("missing_evidence_policy")) == "normalize_available" and active_weight > 0 else raw
    penalties = selection.get("penalties") if isinstance(selection.get("penalties"), dict) else {}
    conflict_penalty = float(selection.get("conflict_penalty") or penalties.get("family_conflict", 0.0) + penalties.get("longtail_corruption", 0.0) or 0.0)
    correlation_penalty = float(selection.get("correlation_penalty") or penalties.get("correlation", 0.0) or 0.0)
    scored = score_evidence_record({**selection, "conflict_penalty": conflict_penalty, "correlation_penalty": correlation_penalty}, config)
    return {
        "status": "computed_from_selection_record",
        "case_id": selection.get("case_id"),
        "organ": selection.get("organ"),
        "reported_evidence_confidence": selection.get("evidence_confidence"),
        "recomputed_evidence_confidence": scored.get("evidence_confidence"),
        "reported_grade": selection.get("grade"),
        "recomputed_grade": scored.get("grade"),
        "reported_training_weight": selection.get("training_weight"),
        "recomputed_training_weight": scored.get("training_weight"),
        "target_type": scored.get("target_type"),
        "decision_status": scored.get("decision_status"),
        "terms": terms,
        "raw_weighted_score_from_available_terms": round(raw, 6),
        "available_weight_sum": round(active_weight, 6),
        "normalized_before_penalties": round(normalized, 6),
        "conflict_penalty": round(conflict_penalty, 6),
        "correlation_penalty": round(correlation_penalty, 6),
        "labelcritic_tiebreak_adjustment_applied": scored.get("labelcritic_tiebreak_adjustment_applied"),
        "final_formula_instance": f"C = clip01({round(normalized, 6)} - {round(conflict_penalty, 6)} - {round(correlation_penalty, 6)} + {scored.get('labelcritic_tiebreak_adjustment_applied')}) = {scored.get('evidence_confidence')}",
        "warning": "Use recomputed fields for slides when present. Penalty fields may be absent from legacy selection JSON; prefer label passport or autolabel_core_v2/v3 records for exact reproduction.",
    }


def build_report(config: dict[str, Any], example: dict[str, Any] | None) -> dict[str, Any]:
    thresholds = config.get("thresholds") or {}
    return {
        "stage": "confidence_formula_code_report",
        "status": "success",
        "config_schema_version": config.get("schema_version"),
        "c_is_dice": False,
        "scope": CODE_FORMULA["scope"],
        "code_formula": CODE_FORMULA,
        "missing_evidence_policy": config.get("missing_evidence_policy"),
        "evidence_terms": evidence_rows(config),
        "grade_policy": grade_policy(config),
        "thresholds": thresholds,
        "training_weights": config.get("training_weights"),
        "soft_label_policy": config.get("soft_labels"),
        "single_teacher_caps": {
            "single_teacher_grade_cap": config.get("single_teacher_grade_cap"),
            "single_teacher_with_oof_grade_cap": config.get("single_teacher_with_oof_grade_cap"),
        },
        "example_breakdown": example_breakdown(example, config),
        "ppt_formula": "C = clip01((sum_i w_i S_i / sum_i w_i_available) - P_conflict - P_correlation + Delta_LabelCritic)",
        "teacher_warning": "C is computed per CT-organ selected pseudo-label from evidence available for that pair; aggregate organ reliability is a separate dataset-level analysis.",
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AutoLabelCore Confidence / C Formula", "",
        "C is not Dice. C is a per CT-organ pseudo-label reliability score.", "",
        f"Formula: `{report['ppt_formula']}`", "",
        "## Evidence Terms",
    ]
    for row in report["evidence_terms"]:
        lines.append(f"- `{row['term']}` weight={row['weight']}: {row['description']}")
    lines.extend(["", "## Penalties And Adjustment"])
    for name, formula in report["code_formula"]["penalty_formula"].items():
        lines.append(f"- `{name}`: {formula}")
    lines.extend(["", "## Grade / Training Weight Policy"])
    for row in report["grade_policy"]:
        lines.append(f"- {row['grade']}: {row['condition']}; target={row['target_type']}; weight={row['training_weight']}")
    lines.extend(["", "## Qwen / LabelCritic"])
    lines.append(report["code_formula"]["qwen_usage"])
    lines.extend(["", "## Example"])
    ex = report["example_breakdown"]
    lines.append(f"Status: `{ex.get('status')}`")
    if ex.get("message"):
        lines.append(ex["message"])
    if ex.get("final_formula_instance"):
        lines.append(ex["final_formula_instance"])
        lines.append(f"Grade: reported `{ex.get('reported_grade')}`, recomputed `{ex.get('recomputed_grade')}`; training weight `{ex.get('recomputed_training_weight')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build CPU-only report of the implemented AutoLabelCore confidence formula.")
    ap.add_argument("--autolabel-config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--example-selection-json", default="", help="Optional pseudo_label_selection.json or passport JSON for a real C breakdown.")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/audits/confidence_formula"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(Path(args.autolabel_config).resolve())
    example = load_example_selection(args.example_selection_json or None)
    report = build_report(config, example)
    out = Path(args.output_dir).resolve()
    write_json(out / "confidence_formula_code_report.json", report)
    write_csv(out / "confidence_evidence_terms.csv", report["evidence_terms"])
    write_csv(out / "grade_training_policy.csv", report["grade_policy"])
    write_markdown(out / "confidence_formula_code_report.md", report)
    print(json.dumps({"status": "success", "output_dir": str(out), "c_is_dice": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
