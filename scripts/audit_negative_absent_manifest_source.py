#!/usr/bin/env python3
"""Audit suspicious confirmed-negative student false positives before Round2.

This is a no-expert-reference audit. It explains why confirmed
negative_absent/out-of-FOV rows produced non-empty student candidates before
the safety suppression step, and marks rows that should be withheld instead of
used as zero-mask supervision.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/em_round1_25case_pseudo_label_20260709"
DEFAULT_TRAINSET_AUDIT_ROOT = (
    DEFAULT_OUTPUT_ROOT / "round1/trainset_pseudo_consistency_full_mstep_lr3e-5_negative_fix_reaudit"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "round1/mstep/voxtell_prompt_student_manifest.json"
DEFAULT_NEGATIVE_DIAGNOSIS = DEFAULT_TRAINSET_AUDIT_ROOT / "negative_absent_false_positive_diagnosis.csv"
DEFAULT_TARGET_CONFIG = ROOT / "configs/student_3d_prompt_target_organs.json"
DEFAULT_APPEARANCE_CONFIG = ROOT / "configs/organ_ct_appearance_373.json"

PANCREAS_PARTS = {"pancreas_head", "pancreas_body", "pancreas_tail"}
ABDOMINAL_KEYWORDS = (
    "abdomen",
    "abdominal",
    "retroperitone",
    "pancreas",
    "liver",
    "spleen",
    "kidney",
    "adrenal",
    "stomach",
    "duodenum",
    "bowel",
    "colon",
    "intestine",
    "gall_bladder",
    "bladder",
    "aorta",
    "inferior_vena_cava",
    "portal_vein",
)
THORAX_KEYWORDS = (
    "thorax",
    "thoracic",
    "lung",
    "heart",
    "mediast",
    "rib",
    "costal",
    "sternum",
    "vertebrae_t",
)
HEAD_NECK_KEYWORDS = (
    "head_neck",
    "cranio",
    "brain",
    "cerebrospinal",
    "eye",
    "eyeball",
    "optic",
    "carotid",
    "brainstem",
    "neck",
    "skull",
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return [text]
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        return [str(parsed)]
    return [str(value)]


def norm_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_manifest_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    doc = read_json(path, {})
    items = doc.get("items") if isinstance(doc, dict) else doc
    if not isinstance(items, list):
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or "")
        organ = str(row.get("organ") or row.get("resolved_canonical_id") or row.get("requested_canonical_id") or "")
        if case_id and organ:
            out[(case_id, organ)] = row
    return out


def load_appearance(path: Path) -> dict[str, dict[str, Any]]:
    doc = read_json(path, {})
    rows = doc.get("entries") or doc.get("rows") if isinstance(doc, dict) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        organ = str(row.get("canonical_id") or row.get("canonical_name") or row.get("organ") or "")
        if organ:
            out[organ] = row
    return out


def text_blob(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value)
        elif isinstance(value, dict):
            parts.extend(f"{k}:{v}" for k, v in value.items())
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def has_any(blob: str, needles: tuple[str, ...]) -> bool:
    return any(needle in blob for needle in needles)


def is_abdominal_target(organ: str, appearance: dict[str, Any]) -> bool:
    regions = parse_list(appearance.get("expected_body_regions"))
    blob = text_blob(organ, regions, appearance.get("ct_location"), appearance.get("expected_region"))
    return organ in PANCREAS_PARTS or has_any(blob, ABDOMINAL_KEYWORDS)


def is_thorax_boundary_target(organ: str, appearance: dict[str, Any]) -> bool:
    regions = parse_list(appearance.get("expected_body_regions"))
    blob = text_blob(organ, regions, appearance.get("ct_location"), appearance.get("expected_region"))
    return has_any(blob, THORAX_KEYWORDS)


def is_head_neck_target(organ: str, appearance: dict[str, Any]) -> bool:
    regions = parse_list(appearance.get("expected_body_regions"))
    blob = text_blob(organ, regions, appearance.get("ct_location"), appearance.get("expected_region"))
    return has_any(blob, HEAD_NECK_KEYWORDS)


def classify_row(row: dict[str, Any], manifest_row: dict[str, Any], appearance: dict[str, Any]) -> dict[str, Any]:
    organ = str(row.get("organ") or manifest_row.get("organ") or "")
    fov_status = str(row.get("fov_status") or manifest_row.get("fov_status") or "")
    fov_evidence = parse_list(row.get("fov_evidence") or manifest_row.get("fov_evidence"))
    coverage_evidence = parse_list(row.get("coverage_evidence") or manifest_row.get("coverage_evidence"))
    negative_source = str(row.get("negative_source") or manifest_row.get("negative_source") or "")
    postprocess_status = str(row.get("postprocess_status") or "")
    roi_status = str(row.get("roi_status") or "")
    containment_enabled = norm_bool(row.get("containment_enabled"))
    evidence_blob = text_blob(fov_evidence, coverage_evidence, negative_source, row.get("negative_reason"))
    source_blob = text_blob(evidence_blob, fov_status)
    abdomen_prior = "pants_abdomen_only" in source_blob or "abdomen_only" in source_blob

    postprocess_flags: list[str] = []
    if "copied_no_containment_rule" in postprocess_status or not containment_enabled:
        postprocess_flags.append("postprocess_no_containment_rule")
    if "parent_roi" in postprocess_status or "parent_roi" in roi_status:
        postprocess_flags.append("postprocess_parent_roi_unavailable")

    abdominal = is_abdominal_target(organ, appearance)
    thorax = is_thorax_boundary_target(organ, appearance)
    head_neck = is_head_neck_target(organ, appearance)

    withheld_required = False
    if abdominal and fov_status == "out_of_fov" and abdomen_prior:
        primary = "suspicious_abdominal_negative_should_withhold"
        recommendation = "convert_to_withheld_uncertain_before_formal_round2"
        withheld_required = True
    elif thorax and fov_status == "out_of_fov" and abdomen_prior:
        primary = "thorax_boundary_requires_case_fov_review"
        recommendation = "case_fov_review_or_keep_withheld_uncertain"
        withheld_required = True
    elif head_neck and fov_status == "out_of_fov":
        primary = "true_out_of_fov_negative"
        recommendation = "retain_confirmed_negative_absent"
    elif postprocess_flags:
        primary = postprocess_flags[0]
        recommendation = "keep_replacement_eligible_false_and_fix_postprocess_policy_if_needed"
    elif fov_status == "out_of_fov":
        primary = "true_out_of_fov_negative"
        recommendation = "retain_confirmed_negative_absent"
    else:
        primary = "postprocess_no_containment_rule"
        recommendation = "review_negative_source_before_reuse"
        withheld_required = True

    return {
        "source_classification": primary,
        "postprocess_issue_flags": postprocess_flags,
        "withheld_required": withheld_required,
        "recommended_action": recommendation,
        "is_abdominal_target": abdominal,
        "is_thorax_boundary_target": thorax,
        "is_head_neck_target": head_neck,
        "pancreas_part_force_rule_applied": organ in PANCREAS_PARTS and primary == "suspicious_abdominal_negative_should_withhold",
    }


def build_audit(
    manifest_path: Path,
    diagnosis_csv: Path,
    appearance_config: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest_index(manifest_path)
    appearance = load_appearance(appearance_config)
    diagnosis_rows = read_csv(diagnosis_csv)
    out_rows: list[dict[str, Any]] = []
    for row in diagnosis_rows:
        case_id = str(row.get("case_id") or "")
        organ = str(row.get("organ") or "")
        manifest_row = manifest.get((case_id, organ), {})
        app = appearance.get(organ, {})
        cls = classify_row(row, manifest_row, app)
        out_rows.append(
            {
                **row,
                "manifest_target_type": manifest_row.get("target_type"),
                "manifest_training_weight": manifest_row.get("training_weight"),
                "expected_body_regions": parse_list(app.get("expected_body_regions")),
                "ct_location": app.get("ct_location"),
                "formal_selection_eligible": app.get("formal_selection_eligible"),
                **cls,
                "replacement_eligible": False,
                "metric_target": "selected_pseudo_label",
            }
        )

    class_counts = Counter(str(row["source_classification"]) for row in out_rows)
    withheld_rows = [row for row in out_rows if row.get("withheld_required")]
    suspicious = [row for row in out_rows if row.get("source_classification") == "suspicious_abdominal_negative_should_withhold"]
    payload = {
        "stage": "negative_absent_manifest_source_audit",
        "status": "passed",
        "metric_target": "selected_pseudo_label",
        "input_rows": len(diagnosis_rows),
        "audited_rows": len(out_rows),
        "classification_counts": dict(class_counts),
        "withheld_required_count": len(withheld_rows),
        "suspicious_abdominal_negative_count": len(suspicious),
        "suspicious_abdominal_all_withheld_required": all(bool(row.get("withheld_required")) for row in suspicious),
        "pancreas_force_rule": (
            "pancreas_head/body/tail cannot be confirmed out-of-FOV negatives from a PanTS abdomen-only prior; "
            "they must be withheld until reliable positive or absence evidence exists."
        ),
        "teacher_inference_rerun": False,
        "replacement_policy": "Audited confirmed negatives remain replacement_eligible=false.",
        "rows": out_rows,
    }
    return payload, out_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--negative-diagnosis", type=Path, default=DEFAULT_NEGATIVE_DIAGNOSIS)
    ap.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    ap.add_argument("--appearance-config", type=Path, default=DEFAULT_APPEARANCE_CONFIG)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINSET_AUDIT_ROOT)
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.negative_diagnosis.exists():
        raise SystemExit(f"negative diagnosis not found: {args.negative_diagnosis}")
    if not args.appearance_config.exists():
        raise SystemExit(f"appearance config not found: {args.appearance_config}")

    payload, rows = build_audit(args.manifest, args.negative_diagnosis, args.appearance_config)
    out_dir = args.output_dir
    write_json(out_dir / "negative_absent_manifest_source_audit.json", payload)
    write_csv(out_dir / "negative_absent_manifest_source_audit.csv", rows)
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
