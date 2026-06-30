#!/usr/bin/env python3
"""CPU-only audit for AutoLabelCore candidate selection.

This script is intentionally read-only with respect to E-step artifacts. It
recomputes Dice/ranking from existing NIfTI masks and writes audit tables under
an output directory. It never runs teacher inference, training, LabelCritic,
Qwen, ShapeKit, or CUDA code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ESTEP = ROOT / "outputs/round1_373_hierarchical_repair_20260620/estep"
DEFAULT_TARGET_CONFIG = ROOT / "configs/student_3d_prompt_target_organs.json"
DEFAULT_AUTOLABEL_CONFIG = ROOT / "configs/autolabel_core.yaml"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonish(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonish(row.get(k)) for k in fieldnames})


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_target_organs(target_config: Path, run_summary: dict[str, Any]) -> list[str]:
    doc = read_json(target_config, {})
    organs = [str(x) for x in doc.get("target_organs", []) if str(x).strip()]
    if organs:
        return organs
    return [str(x) for x in run_summary.get("organs", []) if str(x).strip()]


def selected_case_ids(estep: Path) -> list[str]:
    case_ids = sorted({p.parent.name for p in (estep / "cases").glob("*/pseudo_label_selection.json")})
    if case_ids:
        return case_ids
    return sorted({p.parent.name for p in (estep / "annotation_versions").glob("*/selection_metadata.json")})


def load_selection_rows(estep: Path, case_ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((estep / "cases").glob("*/pseudo_label_selection.json")):
        case_id = path.parent.name
        if case_ids and case_id not in case_ids:
            continue
        doc = read_json(path, {})
        for row in doc.get("selection_rows", []) or []:
            if isinstance(row, dict) and row.get("organ"):
                rows.append({"selection_path": str(path), **row})
    return rows


def load_selected_metadata(estep: Path, case_ids: set[str] | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((estep / "annotation_versions").glob("*/selection_metadata.json")):
        case_id = path.parent.name
        if case_ids and case_id not in case_ids:
            continue
        doc = read_json(path, {})
        for row in doc.get("selected_organs", []) or []:
            if isinstance(row, dict) and row.get("organ"):
                by_key[(case_id, str(row["organ"]))] = {"metadata_path": str(path), **row}
    return by_key


def load_manifest(estep: Path) -> dict[tuple[str, str], dict[str, Any]]:
    doc = read_json(estep / "training_manifest.json", [])
    items = doc.get("items", []) if isinstance(doc, dict) else doc
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in items or []:
        if isinstance(row, dict) and row.get("case_id") and row.get("organ"):
            out[(str(row["case_id"]), str(row["organ"]))] = row
    return out


def candidate_source_paths(estep: Path, case_id: str, model: str, organ: str) -> dict[str, list[Path]]:
    return {
        "raw_teacher_pool": [
            estep / "cases" / case_id / "hierarchical_predictions" / model / "segmentations" / f"{organ}.nii.gz",
            estep / "cases" / case_id / "raw_predictions" / model / case_id / "segmentations" / f"{organ}.nii.gz",
            estep / "cases" / case_id / "raw_predictions" / model / case_id / "per_model" / model / "segmentations" / f"{organ}.nii.gz",
        ],
        "post_shapekit_pool": [
            estep / "cases" / case_id / "refined_predictions" / "candidate_shapekit" / model / case_id / "segmentations" / f"{organ}.nii.gz",
        ],
        "updated_pool": [
            estep / "annotation_versions" / case_id / "updated" / f"{organ}.nii.gz",
            estep / "standard_dataset" / case_id / "segmentations" / f"{organ}.nii.gz",
        ],
    }


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def mask_path_candidates(estep: Path, case_id: str, model: str, organ: str) -> dict[str, Path | None]:
    sources = candidate_source_paths(estep, case_id, model, organ)
    return {
        "raw_teacher": first_existing(sources["raw_teacher_pool"]),
        "raw_teacher_primary": sources["raw_teacher_pool"][0],
        "post_shapekit": first_existing(sources["post_shapekit_pool"]),
        "post_shapekit_primary": sources["post_shapekit_pool"][0],
        "updated": first_existing(sources["updated_pool"]),
        "updated_primary": sources["updated_pool"][0],
    }


def reference_kind(path: str | Path | None) -> str:
    if not path:
        return "missing"
    text = str(path)
    if "/data/PanTS/LabelTr/" in text or "data/PanTS/LabelTr/" in text:
        return "pants_gt"
    if "annotation_versions" in text or "standard_dataset" in text or "outputs/" in text:
        return "pseudo_gt"
    return "unknown_reference"


def discover_teacher_models(estep: Path, case_ids: list[str], run_summary: dict[str, Any]) -> list[str]:
    """Union configured teachers with teacher directories found on disk."""
    models = {str(x) for x in run_summary.get("models_requested", []) if str(x).strip()}
    for case in case_ids:
        for root in [
            estep / "cases" / case / "hierarchical_predictions",
            estep / "cases" / case / "raw_predictions",
            estep / "cases" / case / "refined_predictions" / "candidate_shapekit",
        ]:
            if root.exists():
                models.update(p.name for p in root.iterdir() if p.is_dir())
    return sorted(models)


def mask_stats(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {"mask_status": "missing", "voxels": None}
    p = Path(path)
    if not p.exists():
        return {"mask_status": "missing", "voxels": None, "path": str(p)}
    try:
        img = nib.load(str(p))
        arr = np.asanyarray(img.dataobj) > 0
        voxels = int(arr.sum())
        return {
            "mask_status": "zero_volume" if voxels == 0 else "nonzero",
            "voxels": voxels,
            "shape": list(img.shape[:3]),
            "path": str(p),
        }
    except Exception as exc:
        return {"mask_status": "unreadable", "voxels": None, "path": str(p), "error": str(exc)}


def binary_dice_paths(mask_a: Path | str | None, mask_b: Path | str | None) -> float | None:
    if not mask_a or not mask_b:
        return None
    pa = Path(mask_a)
    pb = Path(mask_b)
    if not pa.exists() or not pb.exists():
        return None
    try:
        ia = nib.load(str(pa))
        ib = nib.load(str(pb))
        if ia.shape[:3] != ib.shape[:3]:
            return None
        a = np.asanyarray(ia.dataobj) > 0
        b = np.asanyarray(ib.dataobj) > 0
        total = int(a.sum()) + int(b.sum())
        return 1.0 if total == 0 else float(2 * np.logical_and(a, b).sum() / total)
    except Exception:
        return None


def rank_scores(scores: dict[str, float | None]) -> dict[str, int | None]:
    valid = sorted(
        ((model, score) for model, score in scores.items() if score is not None),
        key=lambda item: (-float(item[1]), item[0]),
    )
    ranks: dict[str, int | None] = {model: None for model in scores}
    previous_score: float | None = None
    previous_rank = 0
    for idx, (model, score) in enumerate(valid, start=1):
        if previous_score is not None and math.isclose(float(score), previous_score, abs_tol=1e-9):
            ranks[model] = previous_rank
        else:
            ranks[model] = idx
            previous_rank = idx
            previous_score = float(score)
    return ranks


def rank_of_score_against_pool(pool_scores: dict[str, float | None], score: float | None) -> int | None:
    if score is None:
        return None
    valid = sorted(float(v) for v in pool_scores.values() if v is not None)
    if not valid:
        return None
    return 1 + sum(1 for value in valid if value > float(score) + 1e-9)


def relative_score_by_model(selection: dict[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for item in selection.get("autolabel_candidate_scores", []) or []:
        model = item.get("model")
        if model:
            scores[str(model)] = float_or_none(item.get("relative_score"))
    return scores


def reference_metric_names(kind: str) -> tuple[str, str]:
    if kind == "pants_gt":
        return "dice_selected_vs_gt", "selected_rank_by_gt"
    if kind == "pseudo_gt":
        return "dice_selected_vs_pseudo_gt", "selected_rank_by_pseudo_gt"
    return "dice_selected_vs_unknown_reference", "selected_rank_by_unknown_reference"


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and math.isclose(indexed[j + 1][1], indexed[i][1], abs_tol=1e-12):
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(rank_values(xs), rank_values(ys))


def float_or_none(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def review_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    dice = float_or_none(row.get("selected_gt_dice"))
    if dice is not None and dice < 0.5:
        reasons.append("selected_gt_dice_below_0_5")
    rank = row.get("selected_rank_by_gt")
    if rank not in {None, "", 1}:
        reasons.append("selected_not_gt_rank1")
    if row.get("selected_reference_quality_bucket") == "empty_reference_nonempty_prediction":
        reasons.append("empty_reference_nonempty_prediction")
    if row.get("selected_reference_quality_bucket") == "critical_low_dice":
        reasons.append("critical_low_dice")
    if row.get("confidence_flag"):
        reasons.append(str(row["confidence_flag"]))
    if row.get("selected_model") and row.get("selected_model") not in set(row.get("full_raw_candidate_models") or []):
        reasons.append("selected_model_missing_from_full_raw_pool")
    return sorted(set(reasons))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CPU-only full-candidate audit for AutoLabelCore selection artifacts.")
    ap.add_argument("--estep", default=str(DEFAULT_ESTEP), help="Existing E-step output folder.")
    ap.add_argument("--target-config", default=str(DEFAULT_TARGET_CONFIG))
    ap.add_argument("--autolabel-config", default=str(DEFAULT_AUTOLABEL_CONFIG))
    ap.add_argument("--output-dir", default="", help="Default: <estep>/audits/autolabel_core_full_candidate_audit")
    ap.add_argument("--max-cases", type=int, default=0, help="Optional CPU smoke limit. 0 means all cases.")
    ap.add_argument("--organs", default="", help="Optional comma-separated organ subset for CPU smoke checks.")
    ap.add_argument("--skip-dice", action="store_true", help="Inventory only; do not read NIfTI arrays.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    estep = Path(args.estep).resolve()
    run_summary = read_json(estep / "run_summary.json", {})
    target_organs = load_target_organs(Path(args.target_config).resolve(), run_summary)
    if args.organs:
        requested = {x.strip() for x in args.organs.replace(";", ",").split(",") if x.strip()}
        target_organs = [organ for organ in target_organs if organ in requested]
    case_ids = selected_case_ids(estep)
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]
    case_id_set = set(case_ids)
    models = discover_teacher_models(estep, case_ids, run_summary)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else estep / "audits" / "autolabel_core_full_candidate_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = load_selection_rows(estep, case_id_set)
    selected_metadata = load_selected_metadata(estep, case_id_set)
    manifest_by_key = load_manifest(estep)
    selection_by_key = {(str(r["case_id"]), str(r["organ"])): r for r in selection_rows}

    inventory_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    selection_audit_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    exclusion_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case_id in case_ids:
        for organ in target_organs:
            key = (case_id, organ)
            selection = selection_by_key.get(key, {})
            selected_meta = selected_metadata.get(key, {})
            manifest = manifest_by_key.get(key, {})
            reference = selection.get("reference") or selected_meta.get("reference")
            reference_path = Path(str(reference)) if reference else None
            ref_kind = reference_kind(reference)
            gt_available = bool(reference_path and reference_path.exists() and ref_kind == "pants_gt")
            reference_available = bool(reference_path and reference_path.exists())
            pseudo_gt_available = bool(reference_available and ref_kind == "pseudo_gt")
            candidate_models = set(str(x) for x in selection.get("candidate_models", []) or [])
            selection_candidate_by_model = {
                str(c.get("model")): c
                for c in selection.get("candidate_predictions", []) or []
                if isinstance(c, dict) and c.get("model")
            }
            route_models = set(
                str(x)
                for x in [
                    selection.get("route_primary_teacher"),
                    *(selection.get("route_backup_teachers") or []),
                    *(selection.get("route_competition_teachers") or []),
                ]
                if x
            )
            raw_scores: dict[str, float | None] = {}
            post_scores: dict[str, float | None] = {}
            full_raw_models: list[str] = []
            post_models: list[str] = []
            for model in models:
                paths = mask_path_candidates(estep, case_id, model, organ)
                raw_exists = paths["raw_teacher"] is not None
                post_exists = paths["post_shapekit"] is not None
                if raw_exists:
                    full_raw_models.append(model)
                if post_exists:
                    post_models.append(model)
                raw_dice = None if args.skip_dice or not reference_available else binary_dice_paths(paths["raw_teacher"], reference_path)
                post_dice = None if args.skip_dice or not reference_available else binary_dice_paths(paths["post_shapekit"], reference_path)
                raw_scores[model] = raw_dice if raw_exists else None
                post_scores[model] = post_dice if post_exists else None
                selection_candidate = selection_candidate_by_model.get(model, {})
                excluded_reason = None
                if not raw_exists and not post_exists and model not in candidate_models:
                    excluded_reason = "missing_mask"
                elif model not in route_models and model not in candidate_models:
                    excluded_reason = "not_routed_or_not_in_selection_pool"
                elif model in route_models and model not in candidate_models:
                    excluded_reason = "routed_but_not_in_selection_pool"
                elif selection_candidate.get("candidate_qc_status") == "fail":
                    excluded_reason = "candidate_qc_fail"
                elif selection_candidate.get("identity_status") not in {None, "", "valid"}:
                    excluded_reason = "identity_mismatch"
                if excluded_reason:
                    exclusion_counts[excluded_reason] += 1
                    if len(exclusion_examples[excluded_reason]) < 20:
                        exclusion_examples[excluded_reason].append({
                            "case_id": case_id, "organ": organ, "model": model,
                            "raw_exists": raw_exists, "post_shapekit_exists": post_exists,
                        })
                inventory_rows.append({
                    "case_id": case_id,
                    "organ": organ,
                    "teacher": model,
                    "raw_mask_path": str(paths["raw_teacher"] or paths["raw_teacher_primary"]),
                    "raw_mask_exists": raw_exists,
                    "post_shapekit_mask_path": str(paths["post_shapekit"] or paths["post_shapekit_primary"]),
                    "post_shapekit_mask_exists": post_exists,
                    "is_route_primary": model == selection.get("route_primary_teacher"),
                    "is_route_backup": model in set(selection.get("route_backup_teachers") or []),
                    "is_route_competition": model in set(selection.get("route_competition_teachers") or []),
                    "is_routed_any": model in route_models,
                    "in_selection_candidate_pool": model in candidate_models,
                    "is_selected": model == selection.get("selected_model"),
                    "excluded_reason": excluded_reason,
                    "reference": str(reference_path) if reference_path else "",
                    "reference_kind": ref_kind,
                    "gt_available": gt_available,
                    "dice_raw_teacher_vs_reference": raw_dice,
                    "dice_post_shapekit_vs_reference": post_dice,
                    "raw_gt_dice": raw_dice if gt_available else None,
                    "post_shapekit_gt_dice": post_dice if gt_available else None,
                    "raw_pseudo_gt_dice": raw_dice if pseudo_gt_available else None,
                    "post_shapekit_pseudo_gt_dice": post_dice if pseudo_gt_available else None,
                    "candidate_qc_status": selection_candidate.get("candidate_qc_status"),
                    "candidate_qc_score": selection_candidate.get("candidate_qc_score"),
                    "candidate_qc_flags": selection_candidate.get("candidate_qc_flags", []),
                    "candidate_shapekit_status": selection_candidate.get("candidate_shapekit_status"),
                    "identity_status": selection_candidate.get("identity_status"),
                })

            selection_scores: dict[str, float | None] = {}
            for model, candidate in selection_candidate_by_model.items():
                selection_scores[model] = None if args.skip_dice or not reference_available else binary_dice_paths(candidate.get("prediction"), reference_path)
            raw_ranks = rank_scores(raw_scores)
            post_ranks = rank_scores(post_scores)
            selection_ranks = rank_scores(selection_scores)

            selected_model = str(selection.get("selected_model") or selected_meta.get("selected_model") or "")
            selected_artifact = (
                selection.get("selected_prediction")
                or selected_meta.get("mask_path")
                or selected_meta.get("final_mask")
                or selected_meta.get("mask")
            )
            selected_reference_dice = None if args.skip_dice or not reference_available else binary_dice_paths(selected_artifact, reference_path)
            not_rankable_reason = rankability_reason(args.skip_dice, reference_path, reference_available, selected_artifact, selected_reference_dice)
            best_raw_model, best_raw_dice = best_score(raw_scores)
            best_post_model, best_post_dice = best_score(post_scores)
            best_selection_model, best_selection_dice = best_score(selection_scores)
            selected_raw_rank = rank_of_score_against_pool(raw_scores, selected_reference_dice)
            selected_post_rank = rank_of_score_against_pool(post_scores, selected_reference_dice)
            selected_selection_rank = rank_of_score_against_pool(selection_scores, selected_reference_dice)
            selected_model_raw_rank = raw_ranks.get(selected_model)
            selected_model_post_rank = post_ranks.get(selected_model)
            selected_model_selection_rank = selection_ranks.get(selected_model)
            confidence = first_float(
                selection.get("evidence_confidence"),
                selection.get("auto_fine_label_reliability_score"),
                selected_meta.get("evidence_confidence"),
                selected_meta.get("auto_fine_label_reliability_score"),
                manifest.get("evidence_confidence"),
                manifest.get("auto_fine_label_reliability_score"),
                manifest.get("label_confidence"),
            )
            confidence_flag = None
            if gt_available and confidence is not None and selected_reference_dice is not None:
                if confidence >= 0.75 and selected_reference_dice < 0.5:
                    confidence_flag = "high_confidence_low_dice"
                elif confidence < 0.5 and selected_reference_dice >= 0.8:
                    confidence_flag = "low_confidence_high_dice"
            if gt_available and selected_selection_rank and selected_selection_rank > 1:
                confidence_flag = confidence_flag or "rank_mismatch"
            relative_scores = relative_score_by_model(selection)
            confidence_ranks = rank_scores(relative_scores)

            selection_audit = {
                "case_id": case_id,
                "organ": organ,
                "selected_model": selected_model or None,
                "selected_artifact": selected_artifact,
                "reference": str(reference_path) if reference_path else "",
                "reference_kind": ref_kind,
                "gt_available": gt_available,
                "pseudo_gt_available": pseudo_gt_available,
                "not_rankable_reason": not_rankable_reason,
                "selection_status": selection.get("selection_status"),
                "selection_method": selection.get("selection_method"),
                "candidate_mode": selection.get("candidate_mode"),
                "selection_candidate_count": len(candidate_models),
                "full_raw_candidate_count": len(full_raw_models),
                "post_shapekit_candidate_count": len(post_models),
                "rank_denominator": "full_raw_pool",
                "full_raw_candidate_models": sorted(full_raw_models),
                "selection_candidate_models": sorted(candidate_models),
                "dice_selected_vs_gt": selected_reference_dice if gt_available else None,
                "dice_selected_vs_pseudo_gt": selected_reference_dice if pseudo_gt_available else None,
                "dice_selected_vs_reference": selected_reference_dice,
                "selected_gt_dice": selected_reference_dice if gt_available else None,
                "selected_pseudo_gt_dice": selected_reference_dice if pseudo_gt_available else None,
                "selected_rank_by_gt": selected_raw_rank if gt_available else None,
                "selected_rank_by_pseudo_gt": selected_raw_rank if pseudo_gt_available else None,
                "selected_rank_in_raw_pool": selected_raw_rank,
                "selected_rank_in_post_shapekit_pool": selected_post_rank,
                "selected_rank_in_selection_pool": selected_selection_rank,
                "selected_model_rank_in_raw_pool": selected_model_raw_rank,
                "selected_model_rank_in_post_shapekit_pool": selected_model_post_rank,
                "selected_model_rank_in_selection_pool": selected_model_selection_rank,
                "best_gt_model": best_raw_model if gt_available else None,
                "best_gt_dice": best_raw_dice if gt_available else None,
                "gap_to_best_gt": (best_raw_dice - selected_reference_dice) if gt_available and best_raw_dice is not None and selected_reference_dice is not None else None,
                "best_post_shapekit_model": best_post_model,
                "best_post_shapekit_dice": best_post_dice,
                "best_selection_model": best_selection_model,
                "best_selection_dice": best_selection_dice,
                "legacy_selected_dice_field": selection.get("selected_dice"),
                "metric_warning": "selected_gt_dice is recomputed offline; old selected_dice is not treated as expert accuracy.",
                "confidence": confidence,
                "confidence_flag": confidence_flag,
                "grade": selection.get("grade") or selected_meta.get("grade") or manifest.get("grade"),
                "training_weight": selection.get("training_weight") or selected_meta.get("training_weight") or manifest.get("training_weight"),
                "selected_reference_quality_bucket": selection.get("selected_reference_quality_bucket"),
                "selected_candidate_qc_status": selection.get("selected_candidate_qc_status"),
                "selected_candidate_qc_flags": selection.get("selected_candidate_qc_flags", []),
                "selected_shapekit_status": selection.get("selected_candidate_shapekit_status") or selection.get("shapekit_status"),
                "labelcritic_compare_used": selection.get("labelcritic_compare_used"),
                "labelcritic_grade_used": selection.get("labelcritic_grade_used"),
                "missing_evidence": selection.get("missing_evidence") or selected_meta.get("missing_evidence") or manifest.get("missing_evidence") or [],
                "evidence_scores": selection.get("evidence_scores") or selected_meta.get("evidence_scores") or manifest.get("evidence_scores") or {},
                "evidence_details": selection.get("evidence_details") or selected_meta.get("evidence_details") or manifest.get("evidence_details") or {},
            }
            selection_audit_rows.append(selection_audit)
            reasons = review_reasons(selection_audit)
            if reasons:
                review_rows.append({**selection_audit, "review_reasons": reasons})

            for model, candidate in selection_candidate_by_model.items():
                relative_score = relative_scores.get(model)
                calibration_rows.append({
                    "case_id": case_id,
                    "organ": organ,
                    "model": model,
                    "prediction": candidate.get("prediction"),
                    "reference": str(reference_path) if reference_path else "",
                    "reference_kind": ref_kind,
                    "gt_available": gt_available,
                    "pseudo_gt_available": pseudo_gt_available,
                    "dice_candidate_vs_reference": selection_scores.get(model),
                    "dice_candidate_vs_gt": selection_scores.get(model) if gt_available else None,
                    "dice_candidate_vs_pseudo_gt": selection_scores.get(model) if pseudo_gt_available else None,
                    "gt_dice": selection_scores.get(model) if gt_available else None,
                    "reference_dice_rank": selection_ranks.get(model),
                    "gt_dice_rank": selection_ranks.get(model) if gt_available else None,
                    "dice_rank": selection_ranks.get(model),
                    "confidence": confidence if model == selected_model else None,
                    "autolabel_candidate_relative_score": relative_score,
                    "confidence_rank": confidence_ranks.get(model),
                    "is_selected": model == selected_model,
                    "candidate_qc_status": candidate.get("candidate_qc_status"),
                    "candidate_qc_score": candidate.get("candidate_qc_score"),
                    "candidate_qc_flags": candidate.get("candidate_qc_flags", []),
                    "candidate_shapekit_status": candidate.get("candidate_shapekit_status"),
                    "candidate_shapekit_reason": candidate.get("candidate_shapekit_reason"),
                    "ct_support_score": score_lookup(selection, model, "ct_support_score"),
                    "anatomy_plausibility_score": score_lookup(selection, model, "anatomy_plausibility_score"),
                    "family_consensus_score": selection.get("family_consensus_score"),
                    "identity_status": candidate.get("identity_status"),
                    "rank_denominator": "selection_pool",
                })

    confidence_summary = build_confidence_summary(selection_audit_rows, calibration_rows)
    grade_calibration = build_grade_calibration(selection_audit_rows)
    confidence_formula = build_confidence_formula_report(Path(args.autolabel_config).resolve(), selection_audit_rows)
    summary = build_summary(
        case_ids=case_ids,
        target_organs=target_organs,
        models=models,
        inventory_rows=inventory_rows,
        selection_rows=selection_audit_rows,
        calibration_rows=calibration_rows,
        review_rows=review_rows,
        exclusion_counts=exclusion_counts,
        confidence_summary=confidence_summary,
        skip_dice=args.skip_dice,
    )
    summary["grade_calibration_status"] = grade_calibration.get("status")
    write_outputs(
        output_dir=output_dir,
        inventory_rows=inventory_rows,
        selection_rows=selection_audit_rows,
        calibration_rows=calibration_rows,
        review_rows=review_rows,
        exclusion_counts=exclusion_counts,
        exclusion_examples=exclusion_examples,
        confidence_summary=confidence_summary,
        grade_calibration=grade_calibration,
        confidence_formula=confidence_formula,
        summary=summary,
    )
    print(json.dumps({k: summary[k] for k in ("status", "total_targets", "selection_candidate_targets", "gt_rankable_targets", "multi_candidate_targets", "review_queue_items")}, indent=2))
    return 0


def rankability_reason(skip_dice: bool, reference_path: Path | None, reference_available: bool, selected_artifact: str | None, selected_reference_dice: float | None) -> str | None:
    if skip_dice:
        return "dice_skipped"
    if not reference_path:
        return "reference_missing"
    if not reference_available:
        return "reference_path_missing"
    if not selected_artifact:
        return "selected_artifact_missing"
    if not Path(str(selected_artifact)).exists():
        return "selected_artifact_path_missing"
    if selected_reference_dice is None:
        return "dice_geometry_or_read_error"
    return None


def first_float(*values: Any) -> float | None:
    for value in values:
        out = float_or_none(value)
        if out is not None:
            return out
    return None


def best_score(scores: dict[str, float | None]) -> tuple[str | None, float | None]:
    valid = [(model, score) for model, score in scores.items() if score is not None]
    if not valid:
        return None, None
    return max(valid, key=lambda item: (float(item[1]), item[0]))


def score_lookup(selection: dict[str, Any], model: str, field: str) -> Any:
    for item in selection.get("autolabel_candidate_scores", []) or []:
        if str(item.get("model")) == model:
            return item.get(field)
    return None


def build_confidence_summary(selection_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_pairs = [
        (float(row["confidence"]), float(row["selected_gt_dice"]))
        for row in selection_rows
        if row.get("gt_available") and row.get("confidence") is not None and row.get("selected_gt_dice") is not None
    ]
    candidate_pairs = [
        (float(row["autolabel_candidate_relative_score"]), float(row["dice_candidate_vs_gt"]))
        for row in calibration_rows
        if row.get("gt_available") and row.get("autolabel_candidate_relative_score") is not None and row.get("dice_candidate_vs_gt") is not None
    ]
    selected_x = [x for x, _ in selected_pairs]
    selected_y = [y for _, y in selected_pairs]
    candidate_x = [x for x, _ in candidate_pairs]
    candidate_y = [y for _, y in candidate_pairs]
    return {
        "selected_confidence_gt_dice_n": len(selected_pairs),
        "selected_confidence_gt_dice_pearson": maybe_round(pearson(selected_x, selected_y)),
        "selected_confidence_gt_dice_spearman": maybe_round(spearman(selected_x, selected_y)),
        "candidate_relative_score_gt_dice_n": len(candidate_pairs),
        "candidate_relative_score_gt_dice_pearson": maybe_round(pearson(candidate_x, candidate_y)),
        "candidate_relative_score_gt_dice_spearman": maybe_round(spearman(candidate_x, candidate_y)),
        "high_confidence_low_dice": sum(1 for row in selection_rows if row.get("confidence_flag") == "high_confidence_low_dice"),
        "low_confidence_high_dice": sum(1 for row in selection_rows if row.get("confidence_flag") == "low_confidence_high_dice"),
        "rank_mismatch": sum(1 for row in selection_rows if row.get("confidence_flag") == "rank_mismatch"),
    }


def maybe_round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def build_grade_calibration(selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for grade in ["A", "B", "C", "D"]:
        items = [r for r in selection_rows if str(r.get("grade") or "") == grade]
        dices = [float(r["dice_selected_vs_gt"]) for r in items if r.get("dice_selected_vs_gt") is not None]
        rows.append({
            "grade": grade,
            "count": len(items),
            "gt_evaluable_count": len(dices),
            "status": "success" if dices else "insufficient_gt_for_calibration",
            "mean_dice_selected_vs_gt": maybe_round(statistics.mean(dices)) if dices else None,
            "median_dice_selected_vs_gt": maybe_round(statistics.median(dices)) if dices else None,
            "min_dice_selected_vs_gt": maybe_round(min(dices)) if dices else None,
            "low_dice_lt_0_5_count": sum(1 for x in dices if x < 0.5),
            "high_dice_ge_0_8_count": sum(1 for x in dices if x >= 0.8),
        })
    return {
        "status": "success" if any(row["gt_evaluable_count"] for row in rows) else "insufficient_gt_for_calibration",
        "rows": rows,
        "warning": "ABCD calibration uses only rows with real GT; pseudo-GT rows are excluded from real-quality calibration.",
    }


def build_confidence_formula_report(config_path: Path, selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if yaml is not None and config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        config = read_json(config_path, {})
    weights = config.get("evidence_weights") or {}
    thresholds = config.get("thresholds") or {}
    return {
        "status": "success" if weights else "missing_config_or_weights",
        "config_path": str(config_path),
        "confidence_is_dice": False,
        "definition": "C/confidence is an estimated pseudo-label reliability score, not a segmentation Dice metric and not expert accuracy.",
        "formula": {
            "positive_terms": weights,
            "penalties": ["family_conflict_penalty", "longtail_corruption_penalty", "correlation_penalty"],
            "bounded_adjustments": {"labelcritic_tiebreak_adjustment_cap": thresholds.get("labelcritic_tiebreak_cap")},
            "grade_thresholds": {k: thresholds.get(k) for k in ["grade_a", "grade_b", "grade_c"]},
        },
        "component_presence_counts": dict(Counter(
            key for row in selection_rows for key in (row.get("evidence_scores") or {}).keys()
        )),
        "component_missing_counts": dict(Counter(
            key for row in selection_rows for key in (row.get("missing_evidence") or [])
        )),
    }


def write_confidence_formula_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AutoLabelCore Confidence / C Score",
        "",
        "C is not Dice. It is an estimated pseudo-label reliability score used when GT is unavailable.",
        "",
        "## Evidence Weights",
    ]
    for key, value in (report.get("formula", {}).get("positive_terms") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## Penalties And Adjustments",
        "- penalties: family conflict, long-tail corruption, correlation",
        f"- LabelCritic bounded adjustment cap: {(report.get('formula', {}).get('bounded_adjustments') or {}).get('labelcritic_tiebreak_adjustment_cap')}",
        "",
        "## Grade Thresholds",
    ])
    for key, value in (report.get("formula", {}).get("grade_thresholds") or {}).items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(
    *,
    case_ids: list[str],
    target_organs: list[str],
    models: list[str],
    inventory_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    exclusion_counts: Counter[str],
    confidence_summary: dict[str, Any],
    skip_dice: bool,
) -> dict[str, Any]:
    total_targets = len(case_ids) * len(target_organs)
    raw_candidate_targets = {
        (r["case_id"], r["organ"]) for r in inventory_rows if r.get("raw_mask_exists")
    }
    selection_candidate_targets = {
        (r["case_id"], r["organ"]) for r in selection_rows if int(r.get("selection_candidate_count") or 0) > 0
    }
    gt_rankable = [
        r for r in selection_rows
        if r.get("gt_available") and r.get("selected_gt_dice") is not None
    ]
    selected_rank1 = sum(1 for r in gt_rankable if r.get("selected_rank_by_gt") == 1)
    return {
        "stage": "autolabel_core_full_candidate_audit",
        "status": "success",
        "case_count": len(case_ids),
        "organ_count": len(target_organs),
        "teacher_count": len(models),
        "total_targets": total_targets,
        "raw_candidate_targets": len(raw_candidate_targets),
        "selection_candidate_targets": len(selection_candidate_targets),
        "gt_rankable_targets": len(gt_rankable),
        "multi_candidate_targets": sum(1 for r in selection_rows if int(r.get("selection_candidate_count") or 0) > 1),
        "full_raw_multi_candidate_targets": sum(1 for r in selection_rows if int(r.get("full_raw_candidate_count") or 0) > 1),
        "selected_gt_rank1": selected_rank1,
        "selected_gt_rank1_rate": maybe_round(selected_rank1 / len(gt_rankable)) if gt_rankable else None,
        "within_0_01_best_gt": sum(1 for r in gt_rankable if float_or_none(r.get("gap_to_best_gt")) is not None and float(r["gap_to_best_gt"]) <= 0.01),
        "within_0_05_best_gt": sum(1 for r in gt_rankable if float_or_none(r.get("gap_to_best_gt")) is not None and float(r["gap_to_best_gt"]) <= 0.05),
        "inventory_rows": len(inventory_rows),
        "calibration_rows": len(calibration_rows),
        "review_queue_items": len(review_rows),
        "selection_candidate_count_counts": dict(Counter(str(r.get("selection_candidate_count")) for r in selection_rows)),
        "full_raw_candidate_count_counts": dict(Counter(str(r.get("full_raw_candidate_count")) for r in selection_rows)),
        "selection_method_counts": dict(Counter(str(r.get("selection_method")) for r in selection_rows)),
        "exclusion_reason_counts": dict(exclusion_counts),
        "confidence_summary": confidence_summary,
        "skip_dice": skip_dice,
        "accuracy_warning": "All GT Dice fields are recomputed offline. Legacy selected_dice/pseudo-consistency fields are not treated as expert accuracy.",
        "gpu_policy": "CPU-only audit; does not call inference, training, LabelCritic, Qwen, ShapeKit, or CUDA.",
    }


def write_outputs(
    *,
    output_dir: Path,
    inventory_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    exclusion_counts: Counter[str],
    exclusion_examples: dict[str, list[dict[str, Any]]],
    confidence_summary: dict[str, Any],
    grade_calibration: dict[str, Any],
    confidence_formula: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    inventory_fields = [
        "case_id", "organ", "teacher", "raw_mask_exists", "post_shapekit_mask_exists",
        "is_routed_any", "in_selection_candidate_pool", "is_selected", "excluded_reason",
        "reference_kind", "gt_available", "dice_raw_teacher_vs_reference",
        "dice_post_shapekit_vs_reference", "raw_gt_dice", "post_shapekit_gt_dice",
        "raw_pseudo_gt_dice", "post_shapekit_pseudo_gt_dice",
        "candidate_qc_status", "candidate_qc_score", "candidate_qc_flags",
        "candidate_shapekit_status", "identity_status", "raw_mask_path", "post_shapekit_mask_path",
    ]
    selection_fields = [
        "case_id", "organ", "selected_model", "selected_artifact", "reference", "reference_kind",
        "gt_available", "pseudo_gt_available", "not_rankable_reason", "selection_status", "selection_method",
        "candidate_mode", "selection_candidate_count", "full_raw_candidate_count",
        "post_shapekit_candidate_count", "rank_denominator", "dice_selected_vs_gt",
        "dice_selected_vs_pseudo_gt", "dice_selected_vs_reference", "selected_gt_dice",
        "selected_pseudo_gt_dice",
        "selected_rank_by_gt", "selected_rank_by_pseudo_gt", "selected_rank_in_raw_pool",
        "selected_rank_in_post_shapekit_pool", "selected_rank_in_selection_pool",
        "selected_model_rank_in_raw_pool", "selected_model_rank_in_post_shapekit_pool",
        "selected_model_rank_in_selection_pool",
        "best_gt_model", "best_gt_dice", "gap_to_best_gt", "best_post_shapekit_model",
        "best_post_shapekit_dice", "best_selection_model", "best_selection_dice",
        "legacy_selected_dice_field", "confidence", "confidence_flag", "grade",
        "training_weight", "selected_reference_quality_bucket", "selected_candidate_qc_status",
        "selected_candidate_qc_flags", "selected_shapekit_status", "labelcritic_compare_used",
        "labelcritic_grade_used", "missing_evidence", "evidence_scores", "evidence_details",
        "full_raw_candidate_models", "selection_candidate_models", "metric_warning",
    ]
    calibration_fields = [
        "case_id", "organ", "model", "prediction", "reference", "reference_kind",
        "gt_available", "pseudo_gt_available", "dice_candidate_vs_reference",
        "dice_candidate_vs_gt", "dice_candidate_vs_pseudo_gt", "gt_dice",
        "reference_dice_rank", "gt_dice_rank", "dice_rank", "confidence",
        "autolabel_candidate_relative_score", "confidence_rank", "is_selected",
        "candidate_qc_status", "candidate_qc_score", "candidate_qc_flags",
        "candidate_shapekit_status", "candidate_shapekit_reason", "ct_support_score",
        "anatomy_plausibility_score", "family_consensus_score", "identity_status",
        "rank_denominator",
    ]
    review_fields = [*selection_fields, "review_reasons"]
    write_csv(output_dir / "all_teacher_candidate_inventory.csv", inventory_rows, inventory_fields)
    write_jsonl(output_dir / "all_teacher_candidate_inventory.jsonl", inventory_rows)
    write_csv(output_dir / "selection_rank_audit.csv", selection_rows, selection_fields)
    write_jsonl(output_dir / "selection_rank_audit.jsonl", selection_rows)
    write_csv(output_dir / "candidate_score_calibration.csv", calibration_rows, calibration_fields)
    write_jsonl(output_dir / "candidate_score_calibration.jsonl", calibration_rows)
    write_csv(output_dir / "review_queue_for_teacher.csv", review_rows, review_fields)
    write_json(output_dir / "review_queue_for_teacher.json", review_rows)
    write_json(output_dir / "exclusion_reasons.json", {
        "counts": dict(exclusion_counts),
        "examples": exclusion_examples,
    })
    write_json(output_dir / "confidence_vs_gt_dice_summary.json", confidence_summary)
    write_json(output_dir / "grade_calibration.json", grade_calibration)
    write_csv(output_dir / "grade_calibration.csv", grade_calibration.get("rows", []))
    write_json(output_dir / "confidence_formula_report.json", confidence_formula)
    write_confidence_formula_md(output_dir / "confidence_formula_report.md", confidence_formula)
    write_teacher_report_summary(output_dir / "teacher_report_summary.md", summary, confidence_formula, grade_calibration)
    write_json(output_dir / "teacher_report_summary.json", {"summary": summary, "confidence_formula": confidence_formula, "grade_calibration": grade_calibration})
    write_json(output_dir / "summary.json", summary)

def write_teacher_report_summary(path: Path, summary: dict[str, Any], confidence_formula: dict[str, Any], grade_calibration: dict[str, Any]) -> None:
    lines = [
        "# Evidence Chain Audit Summary",
        "",
        "## What The Dice Means",
        "- `dice_selected_vs_gt` is selected pseudo-label vs real GT when GT exists.",
        "- `dice_selected_vs_pseudo_gt` is selected pseudo-label vs pseudo reference when real GT is unavailable.",
        "- Legacy `selected_dice` is not treated as real segmentation accuracy.",
        "",
        "## What C / Confidence Means",
        "- C/confidence is not Dice.",
        f"- Formula status: {confidence_formula.get('status')}",
        "- It estimates pseudo-label reliability from evidence components and penalties.",
        "",
        "## AutoLabelCore Selection",
        f"- Total targets: {summary.get('total_targets')}",
        f"- Raw candidate targets: {summary.get('raw_candidate_targets')}",
        f"- Selection candidate targets: {summary.get('selection_candidate_targets')}",
        f"- GT-rankable targets: {summary.get('gt_rankable_targets')}",
        f"- Selected rank-1 rate on GT-rankable targets: {summary.get('selected_gt_rank1_rate')}",
        "",
        "## ABCD Calibration",
        f"- Calibration status: {grade_calibration.get('status')}",
        "- Rows without GT are excluded from real-quality calibration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
