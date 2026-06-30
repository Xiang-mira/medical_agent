"""Ground-truth-free evidence scoring for the formal 373-organ target space.

The scores in this module are estimates of pseudo-label reliability.  They are
never expert accuracy estimates.  Structural QC is a gate, not positive
evidence, and correlated checkpoints are collapsed into evidence families.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


SCORING_SCHEMA_VERSION = "autolabel_core_v2"
ACCEPTED_SCORING_SCHEMA_VERSIONS = {SCORING_SCHEMA_VERSION, "autolabel_core_v3"}
HARD_FAILURE_FLAGS = {
    "missing_candidate", "missing_final_mask", "all_candidates_failed_qc",
    "candidate_qc_fail", "missing_file", "unreadable_mask", "geometry_mismatch",
    "shape_mismatch_ct", "identity_mismatch", "left_right_mismatch", "zero_volume_mask",
    "expected_present_zero_volume_mask", "expected_present_missing_candidate",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_autolabel_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else _project_root() / "configs" / "autolabel_core.yaml"
    if not config_path.is_absolute():
        config_path = _project_root() / config_path
    if yaml is None:
        raise ImportError("PyYAML is required for AutoLabelCore configuration")
    doc = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if str(doc.get("schema_version")) not in {"2.0", "3.0"}:
        raise ValueError(f"Unsupported AutoLabelCore schema: {doc.get('schema_version')}")
    weights = doc.get("evidence_weights") or {}
    if not math.isclose(sum(float(x) for x in weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError("AutoLabelCore evidence weights must sum to 1.0")
    return doc


def stable_case_fold(case_id: str, folds: int = 5) -> int:
    digest = hashlib.sha256(str(case_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(1, int(folds))


def verify_oof_student_provenance(
    case_id: str,
    provenance: dict[str, Any] | None,
    *,
    expected_manifest_hash: str | None = None,
    folds: int = 5,
) -> dict[str, Any]:
    """Verify that a student prediction is genuinely held out for case_id."""
    provenance = provenance or {}
    held_out = provenance.get("student_held_out_fold")
    training_cases = {str(x) for x in provenance.get("student_training_case_ids", [])}
    training_folds = {int(x) for x in provenance.get("student_training_folds", [])}
    manifest_hash = str(provenance.get("training_manifest_hash") or "")
    case_fold = stable_case_fold(case_id, folds)
    reasons: list[str] = []
    if held_out is None or int(held_out) != case_fold:
        reasons.append("held_out_fold_mismatch")
    if str(case_id) in training_cases:
        reasons.append("case_present_in_student_training_manifest")
    if held_out is not None and int(held_out) in training_folds:
        reasons.append("held_out_fold_present_in_training_folds")
    if expected_manifest_hash and manifest_hash != str(expected_manifest_hash):
        reasons.append("training_manifest_hash_mismatch")
    if not manifest_hash:
        reasons.append("training_manifest_hash_missing")
    return {
        "verified": not reasons,
        "case_fold": case_fold,
        "student_held_out_fold": held_out,
        "student_training_folds": sorted(training_folds),
        "training_manifest_hash": manifest_hash or None,
        "reasons": reasons,
    }


def _evidence_detail(name: str, score: Any, record: dict[str, Any]) -> dict[str, Any]:
    supplied = (record.get("evidence_details") or {}).get(name)
    if isinstance(supplied, dict):
        status = str(supplied.get("status") or "unavailable")
        clipped = _clip_score(supplied.get("score"))
        if status == "available" and clipped is None:
            status = "failed"
        return {
            "score": clipped,
            "status": status if status in {"available", "unavailable", "failed"} else "failed",
            "source": supplied.get("source") or name,
            "reason": supplied.get("reason"),
            "provenance": supplied.get("provenance"),
        }
    clipped = _clip_score(score)
    return {
        "score": clipped,
        "status": "available" if clipped is not None else "unavailable",
        "source": name,
        "reason": None if clipped is not None else "not_produced",
        "provenance": None,
    }


def derive_evidence_family(model_key: str, entry: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> str:
    entry = entry or {}
    explicit = str(entry.get("evidence_family") or "").strip()
    if explicit:
        return explicit
    key = str(model_key or "unknown").strip().lower()
    config = config or load_autolabel_config()
    for family, members in (config.get("evidence_family_rules") or {}).items():
        member_set = {str(x).lower() for x in members or []}
        if key in member_set or any(key.startswith(f"{member}_") for member in member_set):
            return str(family)
    if key.startswith("cads"):
        return "cads"
    if key in {"round_prev_selected", "previous_round_selected", "fusion_consensus"}:
        return "prior_pseudo_label"
    if "student" in key or "voxtell" in key:
        return "voxtell_student"
    return key or "unknown"


def enrich_registry_lineage(registry: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Populate the v2 lineage contract without mutating the registry on disk."""
    config = config or load_autolabel_config()
    for key, entry in (registry.get("models") or {}).items():
        had_explicit_family = bool(entry.get("evidence_family"))
        family = derive_evidence_family(key, entry, config)
        entry.setdefault("evidence_family", family)
        entry.setdefault("architecture_lineage", entry.get("recipe") or family)
        entry.setdefault("training_data_lineage", entry.get("dataset_id") or entry.get("source") or "unreviewed")
        entry.setdefault("is_student", "student" in key.lower() or family == "voxtell_student")
        entry.setdefault("lineage_contract_status", "explicit" if had_explicit_family else "derived")
    registry["autolabel_scoring_schema_version"] = SCORING_SCHEMA_VERSION
    return registry


def _clip_score(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return max(0.0, min(1.0, value))
    except Exception:
        return None


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else float(2 * np.logical_and(a, b).sum() / total)


def _load_mask(path: str | Path) -> tuple[Any, np.ndarray]:
    import nibabel as nib
    image = nib.load(str(path))
    return image, np.asanyarray(image.dataobj) > 0


def family_membership(candidates: Iterable[dict[str, Any]], registry: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, list[str]]:
    registry_models = (registry or {}).get("models", {})
    groups: dict[str, list[str]] = {}
    for candidate in candidates:
        model = str(candidate.get("model") or candidate.get("model_key") or "unknown")
        family = str(candidate.get("evidence_family") or derive_evidence_family(model, registry_models.get(model), config))
        candidate["evidence_family"] = family
        groups.setdefault(family, []).append(model)
    return groups


def pairwise_family_consensus(candidates: list[dict[str, Any]], registry: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    groups = family_membership(candidates, registry, config)
    representatives: dict[str, dict[str, Any]] = {}
    for family, models in groups.items():
        members = [c for c in candidates if c.get("evidence_family") == family and c.get("prediction") and Path(c["prediction"]).exists()]
        if not members:
            continue
        # Medoid within a correlated family. QC only breaks exact agreement ties;
        # it never contributes to the evidence confidence itself.
        if len(members) == 1:
            representatives[family] = members[0]
            continue
        arrays = []
        for member in members:
            try:
                arrays.append((member, _load_mask(member["prediction"])[1]))
            except Exception:
                pass
        if not arrays:
            continue
        scored = []
        for member, arr in arrays:
            ds = [binary_dice(arr, other) for other_member, other in arrays if other_member is not member]
            scored.append((statistics.mean(ds) if ds else 0.0, float(member.get("candidate_qc_score") or 0.0), member))
        representatives[family] = max(scored, key=lambda x: (x[0], x[1]))[2]

    pair_scores: list[float] = []
    rep_items = list(representatives.items())
    for idx, (_, a) in enumerate(rep_items):
        for _, b in rep_items[idx + 1:]:
            try:
                pair_scores.append(binary_dice(_load_mask(a["prediction"])[1], _load_mask(b["prediction"])[1]))
            except Exception:
                pass
    median = statistics.median(pair_scores) if pair_scores else None
    return {
        "score": round(float(median), 6) if median is not None else None,
        "pairwise_family_dice": [round(x, 6) for x in pair_scores],
        "independent_family_count": len(representatives),
        "family_membership": groups,
        "representatives": {family: row.get("model") for family, row in representatives.items()},
        "representative_candidates": list(representatives.values()),
    }


def write_family_probability_fusion(
    candidates: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    registry: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Write equal-family probability, uncertainty and hard masks."""
    consensus = pairwise_family_consensus(candidates, registry)
    reps = consensus.pop("representative_candidates")
    if not reps:
        return {**consensus, "status": "failed", "reason": "no readable family representatives"}
    images_arrays = []
    for rep in reps:
        try:
            images_arrays.append((*_load_mask(rep["prediction"]), rep))
        except Exception:
            pass
    if not images_arrays:
        return {**consensus, "status": "failed", "reason": "no readable family representatives"}
    ref_image, ref_array, _ = images_arrays[0]
    usable = [(image, arr, rep) for image, arr, rep in images_arrays if arr.shape == ref_array.shape and np.allclose(image.affine, ref_image.affine, atol=1e-3)]
    if not usable:
        return {**consensus, "status": "failed", "reason": "family representative geometry mismatch"}
    probability = np.mean([arr.astype(np.float32) for _, arr, _ in usable], axis=0, dtype=np.float32)
    uncertainty = (4.0 * probability * (1.0 - probability)).astype(np.float32)
    hard = (probability >= float(threshold)).astype(np.uint8)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    probability_path = out / "probability_mask.nii.gz"
    uncertainty_path = out / "voxel_uncertainty.nii.gz"
    hard_path = out / "hard_mask.nii.gz"
    import nibabel as nib
    probability_header = ref_image.header.copy(); probability_header.set_data_dtype(np.float32)
    uncertainty_header = ref_image.header.copy(); uncertainty_header.set_data_dtype(np.float32)
    hard_header = ref_image.header.copy(); hard_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(probability, ref_image.affine, probability_header), str(probability_path))
    nib.save(nib.Nifti1Image(uncertainty, ref_image.affine, uncertainty_header), str(uncertainty_path))
    nib.save(nib.Nifti1Image(hard, ref_image.affine, hard_header), str(hard_path))
    return {
        **consensus,
        "status": "success",
        "probability_mask_path": str(probability_path),
        "voxel_uncertainty_path": str(uncertainty_path),
        "hard_mask_path": str(hard_path),
        "family_fusion_count": len(usable),
    }


def ct_mask_support(ct_path: str | Path | None, mask_path: str | Path | None) -> dict[str, Any]:
    """Generic CT-mask support without organ-specific unvalidated HU cutoffs."""
    if not ct_path or not mask_path or not Path(ct_path).exists() or not Path(mask_path).exists():
        return {"score": None, "status": "missing"}
    try:
        import nibabel as nib
        from scipy.ndimage import binary_dilation, binary_erosion, label
        ct_img = nib.load(str(ct_path)); mask_img = nib.load(str(mask_path))
        ct = np.asanyarray(ct_img.dataobj).astype(np.float32)
        mask = np.asanyarray(mask_img.dataobj) > 0
        if ct.shape[:3] != mask.shape[:3] or not np.allclose(ct_img.affine, mask_img.affine, atol=1e-3) or not mask.any():
            return {"score": None, "status": "invalid_geometry_or_empty"}
        finite = np.isfinite(ct)
        lo, hi = np.percentile(ct[finite], [1, 99])
        scale = max(float(hi - lo), 1.0)
        grad = np.sqrt(sum(g.astype(np.float32) ** 2 for g in np.gradient(np.clip(ct, lo, hi)))) / scale
        boundary = np.logical_xor(binary_dilation(mask), binary_erosion(mask))
        ring = np.logical_and(binary_dilation(mask, iterations=2), ~mask)
        boundary_support = float(np.clip(np.median(grad[boundary]) / (np.median(grad[finite]) + 1e-6), 0, 2) / 2) if boundary.any() else 0.0
        inside = ct[mask & finite]; outside = ct[ring & finite]
        contrast = abs(float(np.median(inside)) - float(np.median(outside))) / scale if inside.size and outside.size else 0.0
        components = int(label(mask)[1])
        compact_component_score = 1.0 / (1.0 + max(0, components - 1) / 10.0)
        score = 0.45 * boundary_support + 0.35 * min(1.0, contrast * 4.0) + 0.20 * compact_component_score
        return {
            "score": round(max(0.0, min(1.0, score)), 6), "status": "success",
            "boundary_support": round(boundary_support, 6), "inside_outside_contrast": round(contrast, 6),
            "connected_components": components,
        }
    except Exception as exc:
        return {"score": None, "status": "failed", "reason": str(exc)}


def anatomy_support(candidate: dict[str, Any], parent_masks: dict[str, str | Path] | None = None, threshold: float = 0.95) -> dict[str, Any]:
    flags = set(candidate.get("candidate_qc_flags") or [])
    if flags & HARD_FAILURE_FLAGS or candidate.get("identity_status") not in {None, "", "valid"}:
        return {"score": 0.0, "status": "hard_fail", "flags": sorted(flags & HARD_FAILURE_FLAGS)}
    parents = [str(x) for x in candidate.get("parent_ids") or []]
    if not parents:
        return {"score": 1.0, "status": "no_parent_required"}
    parent_masks = parent_masks or {}
    available = [parent_masks[p] for p in parents if p in parent_masks and Path(parent_masks[p]).exists()]
    if not available:
        return {"score": None, "status": "missing_parent", "parents": parents}
    try:
        child = _load_mask(candidate["prediction"])[1]
        union = np.zeros_like(child, dtype=bool)
        for path in available:
            parent = _load_mask(path)[1]
            if parent.shape != child.shape:
                return {"score": 0.0, "status": "parent_geometry_mismatch"}
            union |= parent
        containment = float(np.logical_and(child, union).sum() / max(1, int(child.sum())))
        return {
            "score": round(min(1.0, containment / max(threshold, 1e-6)), 6),
            "status": "pass" if containment >= threshold else "review",
            "parent_containment": round(containment, 6), "parents": parents,
        }
    except Exception as exc:
        return {"score": None, "status": "failed", "reason": str(exc)}


def normalized_surface_dice(a: np.ndarray, b: np.ndarray, tolerance_voxels: float = 1.0) -> float:
    """CPU NSD approximation on a common grid."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    a = np.asarray(a) > 0; b = np.asarray(b) > 0
    if not a.any() and not b.any(): return 1.0
    if not a.any() or not b.any(): return 0.0
    sa = np.logical_xor(a, binary_erosion(a)); sb = np.logical_xor(b, binary_erosion(b))
    da = distance_transform_edt(~sa); db = distance_transform_edt(~sb)
    supported = int((db[sa] <= tolerance_voxels).sum()) + int((da[sb] <= tolerance_voxels).sum())
    return float(supported / max(1, int(sa.sum()) + int(sb.sum())))


def perturbation_stability_score(reference_mask: str | Path, inverse_mapped_variant_masks: Iterable[str | Path]) -> dict[str, Any]:
    """Score already inverse-mapped TTA predictions without treating them as independent teachers."""
    try:
        ref_image, ref = _load_mask(reference_mask)
        scores = []
        for path in inverse_mapped_variant_masks:
            image, arr = _load_mask(path)
            if arr.shape != ref.shape or not np.allclose(image.affine, ref_image.affine, atol=1e-3):
                continue
            scores.append(0.7 * binary_dice(ref, arr) + 0.3 * normalized_surface_dice(ref, arr))
        return {
            "score": round(statistics.median(scores), 6) if scores else None,
            "status": "success" if scores else "missing_variants",
            "variant_count": len(scores),
            "evidence_independence": "same_model_family_not_an_independent_teacher",
        }
    except Exception as exc:
        return {"score": None, "status": "failed", "reason": str(exc), "variant_count": 0}


def tta_plan_for_candidate(candidate: dict[str, Any], *, family_conflict: bool = False, confidence: float | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the risk-triggered TTA contract; execution stays model-runner specific."""
    config = config or load_autolabel_config()
    policy = config.get("tta") or {}
    organ = str(candidate.get("organ") or "").lower()
    high_value_small = any(token in organ for token in ("duct", "artery", "vein", "nerve", "node", "lesion", "tumor"))
    single = int(candidate.get("independent_family_count") or 1) <= 1
    boundary = confidence is not None and 0.55 <= float(confidence) <= 0.75
    triggered = single or family_conflict or boundary or high_value_small
    return {
        "run": bool(policy.get("enabled")) and (triggered or not policy.get("risk_triggered_only", True)),
        "triggered": triggered,
        "reasons": [name for name, flag in (("single_family", single), ("family_conflict", family_conflict), ("grade_boundary", boundary), ("high_value_small_structure", high_value_small)) if flag],
        "transforms": list(policy.get("transforms") or []),
        "allow_lr_flip_with_prompt_swap": bool(policy.get("allow_lr_flip_with_prompt_swap", False)),
        "evidence_independence": "tta_remains_in_original_evidence_family",
    }


def leave_one_family_out_observations(
    candidates: list[dict[str, Any]], ct_path: str | Path, *, registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build model observations against consensus from other families only."""
    family_membership(candidates, registry)
    candidates = [
        c for c in candidates
        if c.get("evidence_family") not in {"voxtell_student", "prior_pseudo_label"}
    ]
    families = sorted({str(c.get("evidence_family")) for c in candidates if c.get("prediction") and Path(c["prediction"]).exists()})
    observations: list[dict[str, Any]] = []
    for candidate in candidates:
        model = str(candidate.get("model") or "unknown")
        own_family = str(candidate.get("evidence_family") or "unknown")
        others = [c for c in candidates if c.get("evidence_family") != own_family and c.get("prediction") and Path(c["prediction"]).exists()]
        other_consensus = pairwise_family_consensus(others, registry)
        reps = other_consensus.pop("representative_candidates")
        other_family_count = int(other_consensus.get("independent_family_count") or 0)
        base = {"model": model, "evidence_family": own_family, "other_family_count": other_family_count}
        if other_family_count < 2 or not reps:
            observations.append({**base, "status": "insufficient_other_families"})
            continue
        try:
            target_image, target = _load_mask(candidate["prediction"])
            arrays = []
            for rep in reps:
                image, arr = _load_mask(rep["prediction"])
                if arr.shape == target.shape and np.allclose(image.affine, target_image.affine, atol=1e-3):
                    arrays.append(arr.astype(np.float32))
            if len(arrays) < 2:
                observations.append({**base, "status": "insufficient_compatible_families"})
                continue
            consensus = np.mean(arrays, axis=0) >= 0.5
            ct = ct_mask_support(ct_path, candidate["prediction"])
            anatomy = anatomy_support(candidate)
            if ct.get("score") is None or anatomy.get("score") is None:
                observations.append({**base, "status": "missing_observation_component"})
                continue
            observations.append({
                **base, "status": "success", "dice": binary_dice(target, consensus),
                "nsd": normalized_surface_dice(target, consensus), "ct_support": ct.get("score"),
                "anatomy_plausibility": anatomy.get("score"),
                "excluded_family": own_family, "consensus_models": [str(x.get("model")) for x in reps],
            })
        except Exception as exc:
            observations.append({**base, "status": "failed", "reason": str(exc)})
    return observations


@dataclass
class AutoLabelDecision:
    decision_status: str
    selected_model: str | None
    hard_mask_path: str | None
    probability_mask_path: str | None
    voxel_uncertainty_path: str | None
    grade: str
    training_weight: float
    evidence_confidence: float
    evidence_scores: dict[str, float | None]
    missing_evidence: list[str]
    independent_family_count: int
    family_membership: dict[str, list[str]]
    conflict_score: float
    winner_margin: float | None
    selection_reason: str
    target_type: str
    penalties: dict[str, float]
    scoring_schema_version: str = SCORING_SCHEMA_VERSION
    decision_reasons: list[str] = field(default_factory=list)
    evidence_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_evidence: list[str] = field(default_factory=list)
    available_weight_sum: float = 0.0
    raw_weighted_score: float = 0.0
    normalized_evidence_confidence: float = 0.0
    grade_cap: str | None = None
    grade_cap_reason: str | None = None
    out_of_fold_verified: bool = False
    oof_verification: dict[str, Any] = field(default_factory=dict)
    family_consensus_dice: float | None = None
    teacher_student_oof_dice: float | None = None
    cross_round_stability_dice: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_evidence_record(record: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score reliability using availability-aware evidence and explicit grade caps."""
    config = config or load_autolabel_config()
    weights = config["evidence_weights"]
    thresholds = config["thresholds"]
    existing = record.get("evidence_scores") or {}
    legacy_sources = {
        "family_consensus": record.get("family_consensus_score", record.get("teacher_consensus_score", existing.get("family_consensus"))),
        "ct_support": record.get("ct_support_score", existing.get("ct_support")),
        "anatomy_plausibility": record.get("anatomy_plausibility_score", existing.get("anatomy_plausibility")),
        "perturbation_stability": record.get("perturbation_stability_score", existing.get("perturbation_stability")),
        "cross_round_stability": record.get("cross_round_stability_score", existing.get("cross_round_stability")),
        "loo_model_reliability": record.get("loo_model_reliability_score", record.get("estimated_model_reliability", existing.get("loo_model_reliability"))),
        "teacher_student_oof": record.get("teacher_student_oof_score", existing.get("teacher_student_oof")),
    }
    details = {name: _evidence_detail(name, legacy_sources.get(name), record) for name in weights}
    evidence = {name: detail["score"] for name, detail in details.items()}
    active = {name: detail for name, detail in details.items() if detail["status"] in {"available", "failed"}}
    available_weight_sum = sum(float(weights[name]) for name in active)
    raw_weighted_score = sum(
        float(weights[name]) * float(detail["score"] or 0.0)
        for name, detail in active.items()
    )
    policy = str(config.get("missing_evidence_policy") or "zero_fill")
    normalized = raw_weighted_score / available_weight_sum if policy == "normalize_available" and available_weight_sum > 0 else raw_weighted_score
    normalized = max(0.0, min(1.0, normalized))

    conflict_penalty = _clip_score(record.get("conflict_penalty")) or 0.0
    correlation_penalty = _clip_score(record.get("correlation_penalty")) or 0.0
    lc_adjustment = 0.0
    if record.get("labelcritic_supported") is True and record.get("labelcritic_tiebreak_adjustment") is not None:
        cap = float(thresholds["labelcritic_tiebreak_cap"])
        lc_adjustment = max(-cap, min(cap, float(record["labelcritic_tiebreak_adjustment"])))
    confidence = max(0.0, min(1.0, normalized - conflict_penalty - correlation_penalty + lc_adjustment))

    family_count = int(record.get("independent_family_count") or 0)
    expected_presence = str(record.get("expected_presence") or "unknown")
    flags = set(record.get("review_flags") or []) | set(record.get("quality_flags") or []) | set(record.get("selected_candidate_qc_flags") or [])
    if expected_presence == "expected_present" and "zero_volume_mask" in flags:
        flags.add("expected_present_zero_volume_mask")
    if expected_presence == "expected_present" and record.get("identity_status") in {"missing", "missing_candidate"}:
        flags.add("expected_present_missing_candidate")
    hard_fail = bool(flags & HARD_FAILURE_FLAGS) or record.get("identity_status") not in {None, "", "valid"} or record.get("selected_candidate_qc_status") == "fail"
    severe_conflict = bool(record.get("severe_family_conflict")) or (
        evidence.get("family_consensus") is not None and family_count >= 2
        and float(evidence["family_consensus"]) < float(thresholds["severe_family_conflict_dice"])
    )
    expert_verified = record.get("ground_truth_status") == "expert_verified"

    oof_cfg = config.get("oof_promotion") or {}
    anatomy = evidence.get("anatomy_plausibility")
    oof_verified = record.get("out_of_fold_verified") is True
    parent_containment = record.get("parent_containment")
    labelcritic_calibrated_score = _clip_score(record.get("labelcritic_calibrated_score"))
    no_review_or_hard_flags = not flags
    oof_promotable = bool(
        oof_verified
        and float(evidence.get("teacher_student_oof") or 0.0) >= float(oof_cfg.get("teacher_student_dice", 0.85))
        and float(evidence.get("cross_round_stability") or 0.0) >= float(oof_cfg.get("cross_round_stability", 0.90))
        and float(evidence.get("perturbation_stability") or 0.0) >= float(oof_cfg.get("perturbation_stability", 0.85))
        and record.get("selected_candidate_qc_status") == "pass"
        and float(evidence.get("ct_support") or 0.0) >= float(oof_cfg.get("ct_support", 0.60))
        and anatomy is not None and float(anatomy) >= float(oof_cfg.get("anatomy_plausibility", 0.95))
        and normalized >= float(thresholds["grade_b"])
        and not hard_fail and not severe_conflict
    )
    single_teacher_b_eligible = bool(
        family_count <= 1
        and record.get("identity_status") in {None, "", "valid"}
        and record.get("selected_candidate_qc_status") == "pass"
        and expected_presence != "expected_absent"
        and float(evidence.get("ct_support") or 0.0) >= 0.70
        and float(evidence.get("perturbation_stability") or 0.0) >= 0.90
        and (parent_containment is None or float(parent_containment) >= 0.90)
        and float(labelcritic_calibrated_score or 0.0) >= 0.70
        and no_review_or_hard_flags
    )

    if hard_fail:
        grade = "D"
    elif expert_verified:
        grade = "A"
    elif severe_conflict:
        grade = "C" if confidence >= float(thresholds["grade_c"]) else "D"
    elif confidence >= float(thresholds["grade_a"]) and family_count >= 2:
        grade = "A"
    elif confidence >= float(thresholds["grade_b"]):
        grade = "B"
    elif confidence >= float(thresholds["grade_c"]):
        grade = "C"
    else:
        grade = "D"

    grade_cap = None
    grade_cap_reason = None
    grade_rank = {"D": 0, "C": 1, "B": 2, "A": 3}
    if not expert_verified and family_count <= 1:
        if oof_promotable or single_teacher_b_eligible:
            grade_cap = str(config.get("single_teacher_with_oof_grade_cap", "B"))
            grade_cap_reason = "verified_oof_student_corroboration" if oof_promotable else "single_teacher_b_evidence_gate"
        else:
            grade_cap = str(config.get("single_teacher_grade_cap", "C"))
            grade_cap_reason = "single_teacher_without_verified_oof"
        if grade_rank[grade] > grade_rank[grade_cap]:
            grade = grade_cap

    soft_enabled = bool(config.get("soft_labels", {}).get("enabled"))
    has_probability = bool(record.get("probability_mask_path"))
    if grade in {"A", "B"}:
        target_type, status = "hard", "accepted"
    elif grade == "C" and soft_enabled and has_probability:
        target_type, status = "soft", "provisional"
    elif grade == "C":
        target_type, status = "provisional", "provisional"
    else:
        target_type, status = "rejected", "rejected"
    training_weight = float(config["training_weights"].get(grade, 0.0))
    if target_type == "soft":
        training_weight = float(config["soft_labels"].get("c_grade_training_weight", 0.1))

    missing = [name for name, detail in details.items() if detail["status"] == "unavailable"]
    failed = [name for name, detail in details.items() if detail["status"] == "failed"]
    reasons = []
    if hard_fail: reasons.append("hard_structural_or_identity_failure")
    if family_count <= 1: reasons.append("single_independent_teacher_family")
    if severe_conflict: reasons.append("severe_family_conflict")
    if missing: reasons.append("unavailable_evidence:" + ",".join(missing))
    if failed: reasons.append("failed_evidence:" + ",".join(failed))
    if grade_cap_reason: reasons.append("grade_cap:" + grade_cap_reason)
    return {
        "evidence_confidence": round(confidence, 6),
        "normalized_evidence_confidence": round(normalized, 6),
        "auto_fine_label_reliability_score": round(confidence, 6),
        "raw_weighted_score": round(raw_weighted_score, 6),
        "available_weight_sum": round(available_weight_sum, 6),
        "evidence_scores": evidence,
        "evidence_details": details,
        "missing_evidence": missing,
        "failed_evidence": failed,
        "grade": grade,
        "grade_cap": grade_cap,
        "grade_cap_reason": grade_cap_reason,
        "oof_promotable": oof_promotable,
        "single_teacher_b_eligible": single_teacher_b_eligible,
        "training_weight": training_weight,
        "target_type": target_type,
        "decision_status": status,
        "decision_reasons": reasons,
        "scoring_schema_version": SCORING_SCHEMA_VERSION,
        "labelcritic_tiebreak_adjustment_applied": round(lc_adjustment, 6),
        "distillation_eligible": training_weight > 0.0,
    }


def score_candidate_set(
    case_id: str,
    organ: str,
    ct_path: str | Path,
    candidates: list[dict[str, Any]],
    prior_round: str | Path | None = None,
    labelcritic_result: dict[str, Any] | None = None,
    *,
    selected_model: str | None = None,
    output_dir: str | Path | None = None,
    registry: dict[str, Any] | None = None,
    parent_masks: dict[str, str | Path] | None = None,
) -> AutoLabelDecision:
    config = load_autolabel_config()
    eligible = [c for c in candidates if c.get("candidate_exists", True) and c.get("identity_status") in {None, "", "valid"} and c.get("candidate_qc_status") != "fail"]
    family_membership(eligible, registry, config)
    # Students and historical pseudo labels may corroborate a teacher, but
    # never increase the independent teacher-family count.
    independent_eligible = [
        c for c in eligible
        if c.get("evidence_family") not in {"voxtell_student", "prior_pseudo_label"}
    ]
    consensus = pairwise_family_consensus(independent_eligible, registry, config)
    ranking_eligible = [c for c in eligible if c.get("evidence_family") != "prior_pseudo_label"] or eligible
    incumbent = next((c for c in ranking_eligible if str(c.get("model")) == str(selected_model)), ranking_eligible[0] if ranking_eligible else None)
    candidate_scores: list[tuple[float, dict[str, Any]]] = []
    longtail_policy = config.get("longtail_critic") or {}
    longtail_model_path = Path(str(longtail_policy.get("model_path") or ""))
    if longtail_model_path and not longtail_model_path.is_absolute():
        longtail_model_path = _project_root() / longtail_model_path
    longtail_model = None
    if longtail_policy.get("enabled") and longtail_model_path.exists():
        try:
            longtail_model = json.loads(longtail_model_path.read_text(encoding="utf-8"))
        except Exception:
            longtail_model = None
    for candidate in ranking_eligible:
        agreement_scores = []
        own_family = candidate.get("evidence_family")
        try:
            own_array = _load_mask(candidate["prediction"])[1]
            for representative in consensus.get("representative_candidates", []):
                if representative.get("evidence_family") != own_family:
                    agreement_scores.append(binary_dice(own_array, _load_mask(representative["prediction"])[1]))
        except Exception:
            pass
        candidate_ct = ct_mask_support(ct_path, candidate.get("prediction"))
        candidate_anatomy = anatomy_support(candidate, parent_masks, float(config["thresholds"]["parent_containment"]))
        variant_masks = candidate.get("tta_variant_masks") or []
        stability = perturbation_stability_score(candidate["prediction"], variant_masks).get("score") if variant_masks else candidate.get("perturbation_stability_score")
        candidate["perturbation_stability_score"] = stability
        candidate["ct_support_score"] = candidate_ct.get("score")
        candidate["anatomy_plausibility_score"] = candidate_anatomy.get("score")
        candidate["parent_containment"] = candidate_anatomy.get("parent_containment")
        corruption_probability = _clip_score(candidate.get("structural_corruption_probability"))
        if corruption_probability is None and longtail_model is not None:
            try:
                from .longtail_critic import structural_corruption_probability
                corruption_probability = structural_corruption_probability(_load_mask(candidate["prediction"])[1], longtail_model)
            except Exception:
                corruption_probability = None
        candidate["structural_corruption_probability"] = corruption_probability
        candidate["tta_plan"] = tta_plan_for_candidate(
            candidate,
            family_conflict=(
                consensus.get("score") is not None
                and float(consensus["score"]) < float(config["thresholds"]["high_family_agreement_dice"])
            ),
            config=config,
        )
        relative_score = (
            0.35 * (statistics.mean(agreement_scores) if agreement_scores else 0.0)
            + 0.30 * float(candidate_ct.get("score") or 0.0)
            + 0.20 * float(candidate_anatomy.get("score") or 0.0)
            + 0.15 * float(stability or 0.0)
            - float(longtail_policy.get("candidate_ranking_penalty_weight", 0.10)) * float(corruption_probability or 0.0)
        )
        candidate["autolabel_candidate_relative_score"] = round(relative_score, 6)
        candidate_scores.append((relative_score, candidate))
    candidate_scores.sort(key=lambda item: item[0], reverse=True)
    winner_margin = candidate_scores[0][0] - candidate_scores[1][0] if len(candidate_scores) > 1 else None
    decisive = winner_margin is not None and winner_margin >= float(config["thresholds"]["decisive_candidate_margin"])
    chosen = candidate_scores[0][1] if decisive else incumbent
    fusion: dict[str, Any] = {}
    if output_dir and independent_eligible:
        fusion = write_family_probability_fusion(independent_eligible, output_dir, registry=registry, threshold=float(config["soft_labels"]["threshold"]))
    ct = ct_mask_support(ct_path, chosen.get("prediction") if chosen else None)
    anatomy = anatomy_support(chosen or {}, parent_masks, float(config["thresholds"]["parent_containment"])) if chosen else {"score": 0.0, "status": "missing_candidate"}
    cross_round = None
    if prior_round and chosen and Path(prior_round).exists():
        try:
            cross_round = binary_dice(_load_mask(prior_round)[1], _load_mask(chosen["prediction"])[1])
        except Exception:
            cross_round = None
    family_score = consensus.get("score")
    family_count = int(consensus.get("independent_family_count") or 0)
    oof_verification: dict[str, Any] = {}
    teacher_student_oof = None
    oof_students = [c for c in eligible if c.get("evidence_family") == "voxtell_student"]
    for student in oof_students:
        verification = verify_oof_student_provenance(
            case_id,
            student.get("oof_provenance"),
            expected_manifest_hash=student.get("expected_training_manifest_hash"),
            folds=int((config.get("oof_promotion") or {}).get("folds", 5)),
        )
        if not verification["verified"] or not chosen:
            continue
        try:
            teacher_image, teacher_mask = _load_mask(chosen["prediction"])
            student_image, student_mask = _load_mask(student["prediction"])
            if teacher_mask.shape != student_mask.shape or not np.allclose(teacher_image.affine, student_image.affine, atol=1e-3):
                verification = {**verification, "verified": False, "reasons": [*verification["reasons"], "student_teacher_geometry_mismatch"]}
                continue
            score = binary_dice(teacher_mask, student_mask)
            if teacher_student_oof is None or score > teacher_student_oof:
                teacher_student_oof = score
                oof_verification = {**verification, "student_model": student.get("model")}
        except Exception as exc:
            oof_verification = {**verification, "verified": False, "reasons": [*verification["reasons"], f"oof_mask_read_failed:{exc}"]}
    severe = family_score is not None and family_count >= 2 and family_score < float(config["thresholds"]["severe_family_conflict_dice"])
    family_conflict_penalty = (
        min(0.10, max(0.0, float(config["thresholds"]["high_family_agreement_dice"]) - float(family_score)) * 0.125)
        if family_score is not None and family_count >= 2 else 0.0
    )
    longtail_penalty = min(
        float(longtail_policy.get("confidence_penalty_cap", 0.05)),
        float(longtail_policy.get("confidence_penalty_cap", 0.05)) * float(chosen.get("structural_corruption_probability") or 0.0),
    ) if chosen else 0.0
    correlation_penalty = min(0.05, 0.01 * max(0, len(independent_eligible) - family_count))
    perturbation = chosen.get("perturbation_stability_score") if chosen else None
    loo = chosen.get("estimated_model_reliability") if chosen else None
    def detail(score: Any, status: str, source: str, reason: str | None = None, provenance: Any = None) -> dict[str, Any]:
        return {"score": score, "status": status, "source": source, "reason": reason, "provenance": provenance}
    evidence_details = {
        "family_consensus": detail(family_score, "available" if family_score is not None else "unavailable", "independent_teacher_families", None if family_score is not None else "single_teacher_family"),
        "ct_support": detail(ct.get("score"), "available" if ct.get("score") is not None else "failed", "ct_mask_support", ct.get("status")),
        "anatomy_plausibility": detail(anatomy.get("score"), "available" if anatomy.get("score") is not None else ("unavailable" if anatomy.get("status") == "missing_parent" else "failed"), "parent_containment", anatomy.get("status")),
        "perturbation_stability": detail(perturbation, "available" if perturbation is not None else "unavailable", "risk_triggered_tta", None if perturbation is not None else "policy_not_triggered_or_not_cached"),
        "cross_round_stability": detail(cross_round, "available" if cross_round is not None else "unavailable", "historical_pseudo_label", None if cross_round is not None else "historical_reference_unavailable"),
        "loo_model_reliability": detail(loo, "available" if loo is not None else "unavailable", "leave_one_family_out", None if loo is not None else "insufficient_other_families"),
        "teacher_student_oof": detail(teacher_student_oof, "available" if teacher_student_oof is not None else "unavailable", "verified_oof_student", None if teacher_student_oof is not None else "verified_oof_student_unavailable", oof_verification or None),
    }
    record = {
        "family_consensus_score": family_score,
        "ct_support_score": ct.get("score"),
        "anatomy_plausibility_score": anatomy.get("score"),
        "perturbation_stability_score": perturbation,
        "cross_round_stability_score": cross_round,
        "loo_model_reliability_score": loo,
        "teacher_student_oof_score": teacher_student_oof,
        "out_of_fold_verified": bool(oof_verification.get("verified")),
        "expected_presence": chosen.get("expected_presence") if chosen else "unknown",
        "parent_containment": chosen.get("parent_containment") if chosen else None,
        "labelcritic_calibrated_score": (labelcritic_result or {}).get("calibrated_score"),
        "evidence_details": evidence_details,
        "independent_family_count": family_count,
        "probability_mask_path": fusion.get("probability_mask_path"),
        "severe_family_conflict": severe,
        "identity_status": chosen.get("identity_status") if chosen else "missing",
        "selected_candidate_qc_status": chosen.get("candidate_qc_status") if chosen else "fail",
        "selected_candidate_qc_flags": chosen.get("candidate_qc_flags", []) if chosen else ["missing_candidate"],
        "labelcritic_supported": (labelcritic_result or {}).get("supported") is True,
        "labelcritic_tiebreak_adjustment": (labelcritic_result or {}).get("tiebreak_adjustment"),
        "correlation_penalty": correlation_penalty,
        "conflict_penalty": min(0.15, family_conflict_penalty + longtail_penalty),
    }
    scored = score_evidence_record(record, config)
    conflict_score = 1.0 - float(family_score) if family_score is not None else 1.0
    decisive_candidate_path = None
    if (
        decisive and chosen
        and float(chosen.get("ct_support_score") or 0.0) >= 0.6
        and float(chosen.get("anatomy_plausibility_score") or 0.0) >= 0.6
        and float(chosen.get("perturbation_stability_score") or 0.0) >= 0.7
    ):
        decisive_candidate_path = chosen.get("prediction")
    decision = AutoLabelDecision(
        decision_status=scored["decision_status"], selected_model=str(chosen.get("model")) if chosen else None,
        hard_mask_path=decisive_candidate_path or fusion.get("hard_mask_path") or (chosen.get("prediction") if chosen else None),
        probability_mask_path=fusion.get("probability_mask_path"), voxel_uncertainty_path=fusion.get("voxel_uncertainty_path"),
        grade=scored["grade"], training_weight=float(scored["training_weight"]), evidence_confidence=float(scored["evidence_confidence"]),
        evidence_scores=scored["evidence_scores"], missing_evidence=scored["missing_evidence"], independent_family_count=family_count,
        family_membership=consensus.get("family_membership", {}), conflict_score=round(conflict_score, 6), winner_margin=round(winner_margin, 6) if winner_margin is not None else None,
        selection_reason=("decisive_evidence_winner;" if decisive else "incumbent_retained_close_or_incomplete_evidence;") + (";".join(scored["decision_reasons"]) or "evidence_scored"),
        target_type=scored["target_type"],
        penalties={"family_conflict": round(family_conflict_penalty, 6), "longtail_corruption": round(longtail_penalty, 6), "correlation": round(correlation_penalty, 6)},
        decision_reasons=scored["decision_reasons"],
        evidence_details=scored["evidence_details"], failed_evidence=scored["failed_evidence"],
        available_weight_sum=float(scored["available_weight_sum"]), raw_weighted_score=float(scored["raw_weighted_score"]),
        normalized_evidence_confidence=float(scored["normalized_evidence_confidence"]),
        grade_cap=scored["grade_cap"], grade_cap_reason=scored["grade_cap_reason"],
        out_of_fold_verified=bool(oof_verification.get("verified")), oof_verification=oof_verification,
        family_consensus_dice=family_score, teacher_student_oof_dice=teacher_student_oof,
        cross_round_stability_dice=cross_round,
    )
    return decision


def read_json_record(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
