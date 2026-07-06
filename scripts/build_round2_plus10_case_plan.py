#!/usr/bin/env python3
"""Build a Round2 +10 case expansion plan without running inference.

The script is intentionally file/CPU-only.  It uses the current Round1
manifest to find trainable teacher-positive organ coverage, then uses candidate
case metadata/annotation folders only as a case-selection signal.  It never
copies annotation masks into the teacher/student training manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT_MANIFEST = (
    ROOT
    / "outputs"
    / "em_round_pure_cached_10case_formal_lite_20260703"
    / "round1"
    / "mstep"
    / "voxtell_prompt_student_manifest.json"
)
DEFAULT_TARGET_CONFIG = ROOT / "configs" / "student_3d_prompt_target_organs.json"
DEFAULT_CANDIDATE_POOL = ROOT / "data_manifest" / "case_list_50_tumor.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "round2_plus10_case_plan"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_case_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
            if str(row.get("case_id") or "").strip()
        ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_name(value: str) -> str:
    raw = value.strip().lower()
    raw = raw.replace(".nii.gz", "").replace(".nii", "")
    raw = re.sub(r"\s*\(([^)]+)\)\s*", r"_\1", raw)
    raw = raw.replace("-", "_").replace(" ", "_")
    raw = re.sub(r"[^a-z0-9_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw


def organ_family(organ: str) -> str:
    organ_l = organ.lower()
    rules = [
        ("airway", ("airway", "bronch", "trachea", "glottis")),
        ("vessel_artery", ("artery", "aorta", "celiac", "trunk")),
        ("vessel_vein", ("vein", "vena", "portal", "postcava", "ivc")),
        ("heart", ("heart", "atrium", "ventricle", "myocardium", "pericard", "coronary")),
        ("lung", ("lung", "pulmonary")),
        ("bone_spine_rib", ("vertebra", "rib", "bone", "clavicula", "femur", "fibula", "humerus", "hip")),
        ("pancreas_biliary", ("pancreas", "pancreatic", "bile", "gall", "cbd")),
        ("kidney_adrenal", ("kidney", "renal", "adrenal")),
        ("gi", ("stomach", "colon", "bowel", "intestine", "duodenum", "esophagus")),
        ("head_neck_eye_brain", ("brain", "eye", "eyeball", "optic", "carotid", "thyroid", "parotid", "mandible", "skull")),
        ("muscle", ("muscle", "psoas", "rectus", "scalene", "gluteus", "oblique")),
        ("gland_reproductive", ("gland", "prostate", "uterus", "gonad", "breast")),
    ]
    for family, tokens in rules:
        if any(token in organ_l for token in tokens):
            return family
    return "other"


def trainable_positive_organs(manifest: dict[str, Any]) -> tuple[set[str], set[tuple[str, str]], list[dict[str, Any]]]:
    organs: set[str] = set()
    keys: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for row in manifest.get("items") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("supervision_type") or "positive").lower() != "positive":
            continue
        target_type = str(row.get("target_type") or "").lower()
        if target_type in {"negative_absent", "absent_negative", "rejected", "unresolved_visible", "partial_fov"}:
            continue
        grade = str(row.get("grade") or "D").upper()
        if grade not in {"A", "B", "C"}:
            continue
        if grade == "C" and target_type not in {"positive_soft", "soft"}:
            continue
        if row.get("distillation_eligible") is False:
            continue
        try:
            if float(row.get("training_weight") or 0.0) <= 0.0:
                continue
        except Exception:
            continue
        case_id = str(row.get("case_id") or "")
        organ = str(row.get("organ") or "")
        if case_id and organ:
            organs.add(organ)
            keys.add((case_id, organ))
            rows.append(row)
    return organs, keys, rows


def annotation_organs(row: dict[str, str], target_by_norm: dict[str, str]) -> set[str]:
    folder = Path(row.get("annotation_folder") or "")
    if not folder.is_dir():
        return set()
    out: set[str] = set()
    for path in folder.glob("*.nii.gz"):
        normalized = normalize_name(path.name)
        target = target_by_norm.get(normalized)
        if target:
            out.add(target)
    return out


def score_candidates(
    *,
    candidate_rows: list[dict[str, str]],
    original_case_ids: set[str],
    missing_organs: set[str],
    target_by_norm: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    scored: list[dict[str, Any]] = []
    selected_case_rows: list[dict[str, str]] = []
    uncovered = set(missing_organs)
    remaining = [
        {**row, "_pool_index": str(index)}
        for index, row in enumerate(candidate_rows)
        if row["case_id"] not in original_case_ids
    ]
    chosen: set[str] = set()

    def row_payload(row: dict[str, str], order: int, covered: set[str], marginal: set[str]) -> dict[str, Any]:
        families = sorted({organ_family(organ) for organ in covered})
        return {
            "rank": order,
            "case_id": row["case_id"],
            "ct_path": row.get("ct_path", ""),
            "annotation_folder": row.get("annotation_folder", ""),
            "candidate_known_organs": len(covered),
            "missing_organs_covered_by_annotation": len(covered & missing_organs),
            "marginal_missing_organs": len(marginal),
            "marginal_families": len({organ_family(organ) for organ in marginal}),
            "families": ";".join(families),
            "marginal_organs": ";".join(sorted(marginal)),
            "selection_signal": "annotation_folder_case_selection_only_not_training_label",
        }

    for rank in range(1, 11):
        best: tuple[tuple[int, int, int, str], dict[str, str], set[str], set[str]] | None = None
        for row in remaining:
            if row["case_id"] in chosen:
                continue
            covered = annotation_organs(row, target_by_norm)
            marginal = covered & uncovered
            # Maximize new coverage first.  For ties, prefer candidate-pool
            # order so the plan is stable and easy to compare with existing
            # case-list templates.
            pool_index = int(row.get("_pool_index") or 0)
            score = (
                len(marginal),
                len({organ_family(organ) for organ in marginal}),
                len(covered & missing_organs),
                -pool_index,
            )
            if best is None or score > best[0]:
                best = (score, row, covered, marginal)
        if best is None:
            break
        _, row, covered, marginal = best
        chosen.add(row["case_id"])
        uncovered -= marginal
        scored.append(row_payload(row, rank, covered, marginal))
        selected_case_rows.append(row)

    # If set-cover saturates early, fill to 10 by broad annotation coverage so
    # the output remains a concrete +10 expansion plan.
    if len(selected_case_rows) < 10:
        for row in remaining:
            if row["case_id"] in chosen:
                continue
            covered = annotation_organs(row, target_by_norm)
            marginal = covered & uncovered
            chosen.add(row["case_id"])
            scored.append(row_payload(row, len(scored) + 1, covered, marginal))
            selected_case_rows.append(row)
            if len(selected_case_rows) >= 10:
                break

    return scored, selected_case_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-manifest", type=Path, default=DEFAULT_CURRENT_MANIFEST)
    parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-new-cases", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.current_manifest.resolve()
    target_path = args.target_config.resolve()
    pool_path = args.candidate_pool.resolve()
    output_dir = args.output_dir.resolve()

    manifest = read_json(manifest_path, {})
    target_doc = read_json(target_path, {})
    target_organs = list(target_doc.get("target_organs") or [])
    target_set = set(target_organs)
    target_by_norm = {normalize_name(organ): organ for organ in target_organs}
    covered_organs, covered_keys, trainable_rows = trainable_positive_organs(manifest)
    missing_organs = target_set - covered_organs
    candidate_rows = read_case_rows(pool_path)
    original_case_ids = {case_id for case_id, _ in covered_keys}
    if not original_case_ids:
        original_case_ids = {
            str(row.get("case_id") or "")
            for row in manifest.get("items", [])
            if isinstance(row, dict) and row.get("case_id")
        }

    candidate_scores, selected_new_cases = score_candidates(
        candidate_rows=candidate_rows,
        original_case_ids=original_case_ids,
        missing_organs=missing_organs,
        target_by_norm=target_by_norm,
    )
    selected_new_cases = selected_new_cases[: max(0, args.num_new_cases)]
    candidate_scores = candidate_scores[: max(0, args.num_new_cases)]

    original_rows_by_case = {row["case_id"]: row for row in candidate_rows if row["case_id"] in original_case_ids}
    original_case_rows = [row for row in candidate_rows if row["case_id"] in original_rows_by_case]
    selected_20_rows = original_case_rows + selected_new_cases

    output_dir.mkdir(parents=True, exist_ok=True)
    gap_rows = []
    family_counts = Counter(organ_family(organ) for organ in missing_organs)
    for organ in sorted(target_set):
        gap_rows.append({
            "organ": organ,
            "family": organ_family(organ),
            "covered_by_current_trainable_positive": organ in covered_organs,
            "current_trainable_positive_count": sum(1 for row in trainable_rows if row.get("organ") == organ),
        })
    write_csv(
        output_dir / "organ_gap_report.csv",
        gap_rows,
        ["organ", "family", "covered_by_current_trainable_positive", "current_trainable_positive_count"],
    )
    write_csv(
        output_dir / "candidate_case_scores.csv",
        candidate_scores,
        [
            "rank",
            "case_id",
            "ct_path",
            "annotation_folder",
            "candidate_known_organs",
            "missing_organs_covered_by_annotation",
            "marginal_missing_organs",
            "marginal_families",
            "families",
            "marginal_organs",
            "selection_signal",
        ],
    )
    write_csv(
        output_dir / "case_list_round2_plus10.csv",
        selected_20_rows,
        ["case_id", "ct_path", "annotation_folder"],
    )

    asset_dir = output_dir / "teacher_assets" / "round2_plus10"
    cache_manifest = {
        "stage": "round2_plus10_teacher_asset_manifest",
        "status": "planned_not_run",
        "teacher_inference_rerun": False,
        "new_teacher_inference_required_for_selected_cases": True,
        "selected_new_case_ids": [row["case_id"] for row in selected_new_cases],
        "cache_root": str(asset_dir),
        "policy": "Build teacher cache once for selected new cases; original 10case Round1 cache is reused.",
    }
    write_json(asset_dir / "cache_manifest.json", cache_manifest)

    summary = {
        "stage": "round2_plus10_case_plan",
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_manifest": str(manifest_path),
        "target_config": str(target_path),
        "candidate_pool": str(pool_path),
        "output_dir": str(output_dir),
        "current_case_count": len(original_case_ids),
        "selected_new_case_count": len(selected_new_cases),
        "round2_case_count": len(selected_20_rows),
        "target_organ_count": len(target_organs),
        "covered_organ_count": len(covered_organs),
        "missing_organ_count": len(missing_organs),
        "missing_family_counts": dict(sorted(family_counts.items())),
        "selected_new_case_ids": [row["case_id"] for row in selected_new_cases],
        "gt_annotation_policy": "annotation_folder used only for case-selection coverage estimation; never copied into E-step/M-step labels",
        "outputs": {
            "organ_gap_report_csv": str(output_dir / "organ_gap_report.csv"),
            "organ_gap_report_json": str(output_dir / "organ_gap_report.json"),
            "candidate_case_scores_csv": str(output_dir / "candidate_case_scores.csv"),
            "case_list_round2_plus10_csv": str(output_dir / "case_list_round2_plus10.csv"),
            "teacher_asset_cache_manifest": str(asset_dir / "cache_manifest.json"),
        },
    }
    write_json(output_dir / "organ_gap_report.json", {"summary": summary, "rows": gap_rows})
    write_json(output_dir / "round2_plus10_case_plan_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
