from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import time
import threading
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .json_utils import write_json
from .auto_fine_label import build_label_passport, passport_path_for_mask
from .auto_label_core import (
    leave_one_family_out_observations,
    score_candidate_set,
    stable_case_fold,
)
from .label_fusion import fuse_candidate_masks
from .label_verifier import verify_annotation
from .labelcritic_wrapper import run_labelcritic_compare, run_labelcritic_compare_batch, run_labelcritic_grade, run_labelcritic_grade_batch
from .mstep_runner import build_training_manifest, write_mstep_config
from .model_registry import candidate_models_for_organs, load_registry, recommend_primary_models_for_organs
from .organ_model_performance import OrganModelPerformance
from .radthinking import build_reasoning_trace
from .registered_infer import run_registered_model
from .organ_taxonomy import identity_contract, load_taxonomy, taxonomy_entry, topological_order_organs
from .hierarchical_roi import HIERARCHICAL_PIPELINE_VERSION, execute_roi_tasks, plan_roi_tasks, write_hierarchical_manifest
from .shapekit_runner import run_shapekit, _SHAPEKIT_TARGET_REQUIREMENTS
from .target_space import validate_formal_373_target_space
from .backend_capabilities import OFFICIAL_VOXTELL_PRETRAINED, profile_runtime_policy
from .organ_prompt_bank import _infer_region_and_landmarks
from .scan_coverage import infer_scan_coverage
from .case_quality_report import build_case_quality_report

QUALITY_CONTRACT_VERSION = "estep_quality_contract_v3"
FOV_POLICY_VERSION = "fov_appearance_regions_v4"

def _file_sha256(path: str | Path | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _historical_reference_provenance(path: str | Path | None) -> dict[str, Any] | None:
    if not path or not Path(path).exists():
        return None
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "source_type": "historical_pseudo_label",
        "version": "imported_annotation_v1",
        "sha256": _file_sha256(resolved),
    }


def _read_case_list(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not any((v or "").strip() for v in row.values()):
                continue
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return rows


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _merge_case_timing_rows(cases: list[dict[str, str]], updated_root: Path, in_memory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge resumed per-case timing files with rows collected in this process.

    During resume, completed cases are skipped before the in-memory timing row is
    appended. Without this merge, the final case_timing_breakdown.csv/json can be
    overwritten with only the newly processed cases, or even an empty list.
    """
    by_case: dict[str, dict[str, Any]] = {}
    for row in in_memory_rows:
        case_id = str(row.get("case_id") or "")
        if case_id:
            by_case[case_id] = row

    merged: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        case_id = case.get("case_id") or Path(case.get("ct_path", f"case_{idx}")).parent.name
        row = by_case.get(case_id)
        timing_path = updated_root / case_id / "case_timing_breakdown.json"
        if timing_path.exists():
            try:
                cached = json.loads(timing_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("case_id"):
                    row = cached
            except Exception:
                pass
        if row:
            merged.append(row)
    return merged


def _mask_path(seg_dir: Path, organ: str) -> Path:
    return seg_dir / f"{organ}.nii.gz"


def _load_model_label_aliases(root: Path) -> dict[str, Any]:
    path = root / "configs" / "model_label_aliases.json"
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_organ_taxonomy(root: Path) -> dict[str, Any]:
    path = root / "configs" / "organ_taxonomy.json"
    if not path.exists():
        raise FileNotFoundError(f"Strict organ taxonomy is required: {path}")
    taxonomy = load_taxonomy(path)
    validation = taxonomy.get("validation", {}) or {}
    if validation.get("status") != "success":
        raise ValueError(f"Organ taxonomy validation failed: {validation.get('errors', [])[:20]}")
    return taxonomy


def _load_target_space_policy(root: Path, requested_organs: list[str]) -> dict[str, Any]:
    """Load the accepted 373-target policy so every run summary explains exclusions."""
    path = root / "configs" / "student_3d_prompt_target_organs.json"
    if not path.exists():
        return {
            "status": "missing_target_config",
            "target_config": str(path),
            "requested_organs": len(requested_organs),
        }
    doc = json.loads(path.read_text(encoding="utf-8"))
    target_organs = [str(x) for x in doc.get("target_organs", [])]
    target_set = set(target_organs)
    requested_set = set(requested_organs)
    return {
        "status": "loaded",
        "target_config": str(path),
        "counts": doc.get("counts", {}),
        "accepted_current_exact_prompt_targets": len(target_organs),
        "requested_organs": len(requested_organs),
        "requested_is_full_accepted_target": requested_set == target_set,
        "requested_target_organs": sorted(requested_set & target_set),
        "requested_non_target_organs": sorted(requested_set - target_set),
        "target_organs_not_requested": sorted(target_set - requested_set),
        "policy_skipped_organs": doc.get("policy_skipped_organs", []),
        "no_enabled_route_organs": doc.get("no_enabled_route_organs", []),
        "ground_truth_status": "pseudo_label_candidate",
        "accuracy_warning": "Teacher outputs are pseudo-label candidates; this target policy does not prove true accuracy.",
    }


def _target_validation_for_run(root: Path, requested_organs: list[str]) -> dict[str, Any]:
    try:
        return validate_formal_373_target_space(
            root / "configs" / "student_3d_prompt_target_organs.json",
            requested_organs=requested_organs,
            require_full_target=len(requested_organs) == 373,
        )
    except Exception as exc:
        return {
            "stage": "formal_373_target_validation",
            "status": "failed",
            "reason": str(exc),
            "requested_organs": len(requested_organs),
        }


def _load_default_target_organs(root: Path) -> list[str]:
    """Load current accepted 373-organ target for mainline loop defaults."""
    path = root / "configs" / "student_3d_prompt_target_organs.json"
    if not path.exists():
        return ["pancreas", "liver", "spleen", "kidney_left", "kidney_right", "colon", "duodenum", "stomach", "aorta", "postcava"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    targets = [str(x) for x in doc.get("target_organs", [])]
    return targets or ["pancreas", "liver", "spleen", "kidney_left", "kidney_right", "colon", "duodenum", "stomach", "aorta", "postcava"]


def _load_organ_prompts(root: Path) -> dict[str, str]:
    path = root / "configs" / "student_3d_prompt_target_organs.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (doc.get("organ_to_prompt", {}) or {}).items()}


def _load_student_target_ids(root: Path) -> dict[str, Any]:
    path = root / "configs" / "student_3d_prompt_target_organs.json"
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(k): v for k, v in (doc.get("organ_to_student_id", {}) or {}).items()}


ABDOMEN_PELVIS_COVERAGE_TERMS = {
    "abdomen",
    "abdominal",
    "abdomen_pelvis",
    "abdomen-pelvis",
    "abdominopelvic",
    "pelvis",
    "pelvic",
}
ABDOMEN_ONLY_COVERAGE_TERMS = {
    "abdomen",
    "abdominal",
}
PELVIS_ONLY_COVERAGE_TERMS = {
    "pelvis",
    "pelvic",
}
HEAD_COVERAGE_TERMS = {"head", "brain", "cranial", "skull", "neck", "head_neck", "head-neck"}

HEAD_NECK_ORGAN_TOKENS = {
    "head", "neck", "brain", "brainstem", "cerebellum", "cerebrospinal", "cranial", "skull",
    "eyeball", "eye", "lens", "cochlear", "auditory", "nasal", "oral",
    "buccal", "lips", "cheek", "face", "carotid", "pharynx", "larynx",
    "glottis", "cricoid", "arytenoid", "scalene", "digastric",
}
PELVIS_ORGAN_TOKENS = {
    "bladder", "prostate", "uterus", "rectum", "pelvic", "gluteus",
    "gonad", "gonads", "seminal", "sacrum", "uterocervix",
}
EXTREMITY_ORGAN_TOKENS = {
    "humerus", "radius", "ulna", "carpal", "metacarpal", "fingers",
    "femur", "tibia", "fibula", "patella", "tarsal", "metatarsal",
}
THORAX_ORGAN_TOKENS = {
    "lung", "heart", "cardiac", "mediastinum", "mediastinal", "pulmonary",
    "bronch", "airway", "trachea", "sternum", "rib", "clavicula",
}
ABDOMEN_ORGAN_TOKENS = {
    "abdomen", "abdominal", "liver", "hepatic", "spleen", "pancreas",
    "pancreatic", "kidney", "renal", "adrenal", "aorta", "celiac",
    "duodenum", "colon", "bowel", "intestine", "stomach", "gall",
    "bile", "portal", "mesenteric", "postcava", "vena_cava", "ivc",
}


def _norm_organ_key(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _metadata_terms(doc: dict[str, Any], keys: tuple[str, ...]) -> set[str]:
    terms: set[str] = set()
    for key in keys:
        for value in _as_text_list(doc.get(key)):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized:
                terms.add(normalized)
            expanded = normalized
            for sep in (",", ";", "|", "/", "\\"):
                expanded = expanded.replace(sep, "_")
            for token in expanded.split("_"):
                if token:
                    terms.add(token)
    return terms


def _organ_region_hint(norm: str) -> str | None:
    tokens = set(norm.split("_"))
    haystack = f"_{norm}_"
    if tokens & HEAD_NECK_ORGAN_TOKENS or any(f"_{token}_" in haystack for token in HEAD_NECK_ORGAN_TOKENS):
        return "head_neck"
    if tokens & EXTREMITY_ORGAN_TOKENS or any(f"_{token}_" in haystack for token in EXTREMITY_ORGAN_TOKENS):
        return "extremity"
    if tokens & PELVIS_ORGAN_TOKENS or any(f"_{token}_" in haystack for token in PELVIS_ORGAN_TOKENS):
        return "pelvis"
    if tokens & THORAX_ORGAN_TOKENS or any(f"_{token}_" in haystack for token in THORAX_ORGAN_TOKENS):
        return "thorax"
    if tokens & ABDOMEN_ORGAN_TOKENS or any(f"_{token}_" in haystack for token in ABDOMEN_ORGAN_TOKENS):
        return "abdomen"
    return None


@lru_cache(maxsize=1)
def _appearance_region_index() -> dict[str, tuple[str, ...]]:
    """Load the structured 373-organ region map used by LabelCritic.

    Token matching misses anatomy names such as ``caudate_nucleus`` and
    ``lentiform_nucleus``. Those must not remain absence-unproven on a
    high-confidence abdomen-only scan when the maintained appearance contract
    already identifies them as head/neck structures.
    """
    project_root = Path(__file__).resolve().parents[4]
    path = project_root / "configs" / "organ_ct_appearance_373.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        entries = doc.get("organ_ct_appearance", {}) or {}
    except Exception:
        return {}
    index: dict[str, tuple[str, ...]] = {}
    for organ, entry in entries.items():
        regions = tuple(
            str(region).strip().lower()
            for region in (entry.get("expected_body_regions", []) or [])
            if str(region).strip() and str(region).strip().lower() != "unknown"
        )
        if regions:
            index[_norm_organ_key(organ)] = regions
    # TotalSegmentator's "autochthon" denotes paraspinal intrinsic back
    # muscles, which are visible on abdominal CT; the generated appearance
    # seed incorrectly classified them as appendicular/extremity anatomy.
    index["autochthon_left"] = ("abdomen",)
    index["autochthon_right"] = ("abdomen",)
    return index


def _presence_context_summary(presence_context: dict[str, Any] | None) -> dict[str, Any]:
    context = presence_context or {}
    inferred_regions = [
        name
        for name, key in (
            ("abdomen", "has_abdomen_coverage"),
            ("pelvis", "has_pelvis_coverage"),
            ("thorax", "has_thorax_coverage"),
            ("head_neck", "has_head_coverage"),
            ("extremity", "has_extremity_coverage"),
        )
        if context.get(key)
    ]
    return {
        "has_region_evidence": bool(context.get("has_region_evidence")),
        "inferred_coverage_regions": inferred_regions,
        "coverage_terms": context.get("coverage_terms", []),
        "coverage_evidence": context.get("coverage_evidence", []),
        "confirmed_absent_organs": context.get("confirmed_absent_organs", []),
    }


def _confirmed_absent_organs(doc: dict[str, Any]) -> list[str]:
    organs: list[str] = []
    for key in ("confirmed_absent_organs", "out_of_scan_organs", "negative_organs", "absent_organs"):
        raw_organs: list[str] = []
        for raw in _as_text_list(doc.get(key)):
            parts = [raw]
            for sep in (",", ";", "|"):
                parts = [piece for part in parts for piece in part.split(sep)]
            raw_organs.extend(parts)
        for organ in raw_organs:
            norm = _norm_organ_key(organ)
            if norm and norm not in organs:
                organs.append(norm)
    return organs


def _case_level_metadata_sources(case: dict[str, str], ct: Path, case_id: str, case_out: Path) -> list[Path]:
    candidates: list[Path] = []
    for key in ("case_metadata_path", "scan_coverage_path", "metadata_path"):
        raw = (case.get(key) or "").strip()
        if raw:
            candidates.append(Path(raw).resolve())
    ct_parent = ct.parent if ct else None
    if ct_parent is not None:
        candidates.extend([
            ct_parent / "scan_coverage.json",
            ct_parent / "case_metadata.json",
            ct_parent / f"{case_id}_scan_coverage.json",
            ct_parent / f"{case_id}_case_metadata.json",
        ])
    candidates.extend([
        case_out / "scan_coverage.json",
        case_out / "case_metadata.json",
    ])
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _load_case_presence_context(case: dict[str, str], ct: Path, case_id: str, case_out: Path) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    for key in (
        "scan_coverage",
        "coverage_regions",
        "body_region",
        "ct_region",
        "confirmed_absent_organs",
        "out_of_scan_organs",
        "negative_organs",
        "absent_organs",
    ):
        if case.get(key):
            doc[key] = case.get(key)
    for path in _case_level_metadata_sources(case, ct, case_id, case_out):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            doc.update(payload)
    coverage_terms = _metadata_terms(doc, ("scan_coverage", "coverage_regions", "body_region", "ct_region"))
    confirmed_absent = set(_confirmed_absent_organs(doc))
    auto_coverage: dict[str, Any] = {}
    if not coverage_terms:
        try:
            auto_coverage = infer_scan_coverage(ct)
            write_json(case_out / "scan_coverage.json", auto_coverage)
            doc.update(auto_coverage)
            coverage_terms = _metadata_terms(doc, ("scan_coverage", "coverage_regions", "body_region", "ct_region"))
        except Exception as exc:
            auto_coverage = {
                "source": "ct_intensity_geometry_v1",
                "uses_annotations": False,
                "scan_coverage": ["unknown"],
                "coverage_regions": [],
                "confidence": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            write_json(case_out / "scan_coverage.json", auto_coverage)
    has_abdomen = bool(coverage_terms & ABDOMEN_ONLY_COVERAGE_TERMS) or "abdomen_pelvis" in coverage_terms or "abdominopelvic" in coverage_terms
    has_pelvis = bool(coverage_terms & PELVIS_ONLY_COVERAGE_TERMS) or "abdomen_pelvis" in coverage_terms or "abdominopelvic" in coverage_terms
    has_head = bool(coverage_terms & HEAD_COVERAGE_TERMS)
    has_thorax = bool(coverage_terms & {"thorax", "chest", "lung", "cardiac"})
    has_extremities = bool(coverage_terms & {"extremity", "extremities", "arms", "legs"})
    coverage_evidence: list[str] = []
    dataset_prior = None
    dataset_text = " ".join([
        str(case.get("dataset") or ""),
        str(case.get("dataset_name") or ""),
        str(case_id),
        str(ct),
    ]).lower()
    if "pants" in dataset_text:
        dataset_prior = "PanTS_abdomen_only_not_373_whole_body"
        has_abdomen = True
        # PanTS may include lung bases, but the dataset prior cannot establish
        # complete thorax, head/neck, pelvis, or extremity coverage.
        has_thorax = False
        coverage_evidence.append(f"dataset_prior:{dataset_prior}")
    if auto_coverage:
        has_abdomen = bool(auto_coverage.get("has_abdomen_coverage", has_abdomen))
        has_pelvis = bool(auto_coverage.get("has_pelvis_coverage", has_pelvis))
        has_head = bool(auto_coverage.get("has_head_coverage", has_head))
        has_thorax = bool(auto_coverage.get("has_thorax_coverage", has_thorax))
        has_extremities = bool(auto_coverage.get("has_extremity_coverage", has_extremities))
        coverage_evidence.append(f"automatic_ct_coverage:{auto_coverage.get('confidence', 'unknown')}")
    informative_terms = coverage_terms - {"unknown", "multi", "region"}
    has_region_evidence = bool(informative_terms)
    excludes_head = has_region_evidence and not has_head
    return {
        "metadata": doc,
        "coverage_terms": sorted(coverage_terms),
        "confirmed_absent_organs": sorted(confirmed_absent),
        "has_abdomen_coverage": has_abdomen,
        "has_pelvis_coverage": has_pelvis,
        "has_head_coverage": has_head,
        "has_thorax_coverage": has_thorax,
        "has_partial_thorax_coverage": bool(
            auto_coverage.get("has_partial_thorax_coverage", False)
        ) if auto_coverage else False,
        "has_extremity_coverage": has_extremities,
        "has_region_evidence": has_region_evidence,
        "coverage_evidence": coverage_evidence,
        "dataset_prior": dataset_prior,
        "excludes_head": excludes_head,
    }


def _augment_presence_from_model_landmarks(
    context: dict[str, Any],
    model_seg_dirs: dict[str, Path],
    case_out: Path,
) -> dict[str, Any]:
    """Add region evidence from independently produced non-empty masks."""
    import nibabel as nib
    import numpy as np

    landmarks = {
        "abdomen": ("liver", "spleen", "pancreas", "kidney_left", "kidney_right"),
        "pelvis": ("bladder", "prostate", "uterus", "femur_left", "femur_right"),
        "thorax": ("lung_left", "lung_right", "heart", "sternum"),
        "head_neck": ("brain", "skull", "eyeball_left", "eyeball_right"),
        "extremity": ("humerus_left", "humerus_right", "tibia_left", "tibia_right"),
    }
    support: dict[str, set[str]] = {region: set() for region in landmarks}
    for model, seg_dir in model_seg_dirs.items():
        for region, organs in landmarks.items():
            for organ in organs:
                path = seg_dir / f"{organ}.nii.gz"
                if not path.exists():
                    continue
                try:
                    if np.any(np.asanyarray(nib.load(str(path)).dataobj) > 0):
                        support[region].add(model)
                        break
                except Exception:
                    continue
    updated = dict(context)
    key_map = {
        "abdomen": "has_abdomen_coverage", "pelvis": "has_pelvis_coverage",
        "thorax": "has_thorax_coverage", "head_neck": "has_head_coverage",
        "extremity": "has_extremity_coverage",
    }
    evidence = list(updated.get("coverage_evidence", []))
    for region, models in support.items():
        if len(models) >= 2:
            if (
                updated.get("dataset_prior") == "PanTS_abdomen_only_not_373_whole_body"
                and region != "abdomen"
            ):
                if region == "thorax":
                    updated["has_partial_thorax_coverage"] = True
                evidence.append(
                    f"independent_model_landmarks_partial_only:{region}:{','.join(sorted(models))}"
                )
                continue
            updated[key_map[region]] = True
            evidence.append(f"independent_model_landmarks:{region}:{','.join(sorted(models))}")
    updated["coverage_evidence"] = evidence
    updated["has_region_evidence"] = any(bool(updated.get(key)) for key in key_map.values())
    metadata = dict(updated.get("metadata") or {})
    regions = [
        region for region, key in key_map.items() if updated.get(key)
    ]
    metadata.update({
        "coverage_regions": regions,
        "scan_coverage": regions or ["unknown"],
        "body_region": "multi_region" if len(regions) > 1 else (regions[0] if regions else "unknown"),
        "ct_region": regions,
        "model_landmark_support": {k: sorted(v) for k, v in support.items()},
        "uses_annotations": False,
    })
    updated["metadata"] = metadata
    write_json(case_out / "scan_coverage.json", metadata)
    return updated


def _load_teacher_branch_map(root: Path) -> dict[str, Any]:
    path = root / "configs" / "teacher_branch_map.yaml"
    if not path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_routed_models(
    registry: dict[str, Any],
    organ: str,
    branch_entry: dict[str, Any],
    routed_candidates: list[str],
) -> dict[str, Any]:
    models = registry.get("models", {})
    primary = str(branch_entry.get("teacher_model") or "").strip()
    if primary not in models:
        primary = routed_candidates[0] if routed_candidates else None

    backup_teachers: list[str] = []
    seen: set[str] = set()
    for raw in branch_entry.get("fallback_teachers", []) or []:
        key = str(raw or "").strip()
        if key in models and key not in seen and key != primary:
            backup_teachers.append(key)
            seen.add(key)
    for model_key in routed_candidates:
        if model_key not in seen and model_key != primary:
            backup_teachers.append(model_key)
            seen.add(model_key)

    competition_teachers = [m for m in routed_candidates if m not in {primary, *backup_teachers}]
    route_confidence = "high" if primary else ("medium" if backup_teachers else "low")
    return {
        "organ": organ,
        "primary_teacher": primary,
        "backup_teachers": backup_teachers,
        "competition_teachers": competition_teachers,
        "route_confidence": route_confidence,
        "production_policy": "route_primary_with_backups",
        "critic_policy": "pairwise_compare_then_absolute_grade",
    }


def _build_case_execution_plan(
    *,
    registry: dict[str, Any],
    project_root: Path,
    organs: list[str],
    requested_models: list[str],
    preseeded_model_dirs: dict[str, Path] | None,
    candidate_mode: str,
) -> dict[str, Any]:
    branch_map = _load_teacher_branch_map(project_root)
    routed = candidate_models_for_organs(registry, organs)
    registry_models = registry.get("models", {}) or {}
    requested_set = {str(m).strip() for m in (requested_models or []) if str(m).strip()}
    runtime_profile = profile_runtime_policy(os.getenv("MEDAI_EXPERIMENT_PROFILE", "advisor_aligned_default"))
    official_voxtell_policy = runtime_profile.get("official_voxtell_pretrained") or {}
    special_candidate_keys = {
        OFFICIAL_VOXTELL_PRETRAINED,
    } if official_voxtell_policy.get("active_as_teacher_candidate") else set()
    requested_models_are_registry_keys = bool(requested_set) and all(m in registry_models for m in requested_set)
    per_organ: dict[str, dict[str, Any]] = {}
    teacher_run_list: list[str] = []
    seen_teachers: set[str] = set()

    for organ in organs:
        norm = _norm_organ_key(organ)
        branch_entry = branch_map.get(organ) or branch_map.get(norm) or {}
        raw_routed_candidates = [str(m).strip() for m in routed.get(norm, []) if str(m).strip()]
        if requested_set:
            routed_candidates = [m for m in raw_routed_candidates if m in requested_set]
        else:
            routed_candidates = [m for m in raw_routed_candidates if m in registry_models]
        route = _resolve_routed_models(
            registry,
            organ,
            branch_entry if isinstance(branch_entry, dict) else {},
            routed_candidates,
        )
        if official_voxtell_policy.get("active_as_teacher_candidate"):
            route.setdefault("competition_teachers", [])
            if OFFICIAL_VOXTELL_PRETRAINED not in route["competition_teachers"]:
                route["competition_teachers"].append(OFFICIAL_VOXTELL_PRETRAINED)
        if requested_set and route.get("primary_teacher") not in requested_set:
            route["primary_teacher"] = routed_candidates[0] if routed_candidates else None
        route["backup_teachers"] = [
            m for m in route.get("backup_teachers", [])
            if not requested_set or m in requested_set or m in special_candidate_keys
        ]
        route["competition_teachers"] = [
            m for m in route.get("competition_teachers", [])
            if not requested_set or m in requested_set or m in special_candidate_keys
        ]
        if candidate_mode == "route_pruned":
            route["competition_teachers"] = []
        family_count_hint = len({
            str((registry_models.get(model_key, {}) or {}).get("evidence_family") or model_key)
            for model_key in routed_candidates
        })
        if family_count_hint >= 2:
            backup_plus_competition = [
                m for m in [*route.get("backup_teachers", []), *route.get("competition_teachers", [])]
                if m
            ]
            if not backup_plus_competition:
                fallback = next(
                    (m for m in routed_candidates if m and m != route.get("primary_teacher")),
                    None,
                )
                if fallback:
                    route.setdefault("backup_teachers", []).append(fallback)
        deduped_backup: list[str] = []
        seen_backup: set[str] = set()
        for model_key in route.get("backup_teachers", []) or []:
            if (
                model_key
                and model_key != route.get("primary_teacher")
                and model_key not in seen_backup
            ):
                deduped_backup.append(model_key)
                seen_backup.add(model_key)
        route["backup_teachers"] = deduped_backup
        route["competition_teachers"] = [
            model_key for model_key in route.get("competition_teachers", []) or []
            if model_key
            and model_key != route.get("primary_teacher")
            and model_key not in set(route["backup_teachers"])
        ]
        eligible = [
            m for m in [
                route.get("primary_teacher"),
                *route.get("backup_teachers", []),
                *route.get("competition_teachers", []),
            ]
            if m
        ]
        if not eligible and requested_models and not requested_models_are_registry_keys:
            # Compatibility fallback for unit tests/ad-hoc custom model keys that
            # are not in the formal registry. Formal route-aware runs must leave
            # unmapped organs unresolved instead of expanding back to all teachers.
            eligible = list(requested_models)
        if candidate_mode == "formal_full_legacy":
            eligible = list(requested_models)
        route["eligible_teachers"] = eligible
        route["grade_required_for_training"] = True
        route["compare_required_when_conflict"] = True
        route["compare_bypass_when_single_or_high_agreement"] = True
        per_organ[organ] = route
        for model_key in eligible:
            if model_key not in seen_teachers:
                teacher_run_list.append(model_key)
                seen_teachers.add(model_key)

    if candidate_mode == "formal_full_legacy":
        teacher_run_list = list(requested_models)

    return {
        "candidate_mode": candidate_mode,
        "organs": organs,
        "teacher_run_list": teacher_run_list,
        "preseeded_models": sorted((preseeded_model_dirs or {}).keys()),
        "per_organ": per_organ,
    }


def _load_organ_task_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"organs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"organs": {}}
    if not isinstance(data, dict):
        return {"organs": {}}
    data.setdefault("organs", {})
    return data


def _record_organ_task_state(
    state: dict[str, Any],
    organ: str,
    *,
    status: str,
    candidate_fingerprint: list[str] | None = None,
    compare_used: bool | None = None,
    grade_used: bool | None = None,
) -> None:
    organs = state.setdefault("organs", {})
    row = dict(organs.get(organ, {}))
    row["status"] = status
    if candidate_fingerprint is not None:
        row["candidate_fingerprint"] = candidate_fingerprint
    if compare_used is not None:
        row["labelcritic_compare_used"] = compare_used
    if grade_used is not None:
        row["labelcritic_grade_used"] = grade_used
    organs[organ] = row


def _candidate_mask_path(seg_dir: Path, organ: str, model_key: str, alias_config: dict[str, Any]) -> tuple[Path, str]:
    direct = _mask_path(seg_dir, organ)
    if organ == "inferior_vena_cava" and not direct.exists():
        legacy = _mask_path(seg_dir, "postcava")
        if legacy.exists():
            return legacy, "canonical_alias:postcava->inferior_vena_cava"
    model_aliases = ((alias_config.get("models", {}) or {}).get(model_key, {}) or {})
    local_to_global = model_aliases.get("local_to_global", {}) or {}
    mapping_types = model_aliases.get("mapping_types", {}) or {}
    union_locals = [
        str(local_name) for local_name, global_name in local_to_global.items()
        if str(global_name) == organ and mapping_types.get(local_name) == "approved_union"
    ]
    if union_locals:
        union_paths = [_mask_path(seg_dir, local_name) for local_name in union_locals]
        if not all(path.exists() for path in union_paths):
            return direct, "missing_approved_union_components"
        try:
            import nibabel as nib
            import numpy as np
            from nibabel.processing import resample_from_to

            base = nib.load(str(union_paths[0]))
            union = np.zeros(base.shape, dtype=np.uint8)
            for path in union_paths:
                image = nib.load(str(path))
                if image.shape != base.shape or not np.allclose(image.affine, base.affine, atol=1e-4):
                    image = resample_from_to(image, base, order=0)
                union |= (np.asanyarray(image.dataobj) > 0).astype(np.uint8)
            union_path = seg_dir / "_canonical_unions" / f"{organ}.nii.gz"
            union_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(union, base.affine, base.header), str(union_path))
            return union_path, "approved_union:" + "+".join(union_locals)
        except Exception:
            return direct, "invalid_approved_union_components"
    if direct.exists():
        return direct, "direct"
    for local_name, global_name in local_to_global.items():
        if str(global_name) == organ:
            local_path = _mask_path(seg_dir, str(local_name))
            if local_path.exists():
                return local_path, f"local_alias:{local_name}"

    return direct, "missing"


def _candidate_identity_contract(
    *,
    taxonomy: dict[str, Any],
    alias_config: dict[str, Any],
    organ: str,
    model_key: str,
    alias_match: str,
    seg_dir: Path | None = None,
) -> dict[str, Any]:
    original_alias_match = alias_match
    if alias_match.startswith("post_shapekit_missing_fallback:"):
        alias_match = alias_match.split(":", 1)[1]
    source_local_label = organ
    resolved_organ = organ
    mapping_type = "exact_synonym"
    mapping_source = "canonical_filename"
    provenance_path = seg_dir / "identity_provenance.json" if seg_dir is not None else None
    if provenance_path is not None and provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            item = (provenance.get("organs", {}) or {}).get(organ)
            if isinstance(item, dict):
                source_labels = item.get("source_local_labels") or [item.get("source_local_label") or organ]
                return identity_contract(
                    taxonomy,
                    organ,
                    "+".join(str(x) for x in source_labels),
                    str(item.get("resolved_canonical_id") or organ),
                    mapping_type=str(item.get("mapping_type") or "exact_synonym"),
                    mapping_source=str(provenance_path),
                )
        except Exception:
            pass
    if alias_match.startswith("local_alias:"):
        source_local_label = alias_match.split(":", 1)[1]
        model_aliases = ((alias_config.get("models", {}) or {}).get(model_key, {}) or {})
        resolved_organ = str((model_aliases.get("local_to_global", {}) or {}).get(source_local_label, ""))
        mapping_type = str((model_aliases.get("mapping_types", {}) or {}).get(source_local_label, "exact_synonym"))
        mapping_source = "configs/model_label_aliases.json"
    elif alias_match.startswith("approved_union:"):
        source_local_label = alias_match.split(":", 1)[1]
        resolved_organ = organ
        mapping_type = "approved_union"
        mapping_source = "configs/model_label_aliases.json"
    elif alias_match == "missing" or alias_match.startswith("missing_") or alias_match.startswith("invalid_"):
        entry = taxonomy_entry(taxonomy, organ)
        return {
            "requested_canonical_id": _norm_organ_key(organ),
            "source_local_label": source_local_label,
            "resolved_canonical_id": _norm_organ_key(organ),
            "comparison_family": entry.get("comparison_family") if entry else None,
            "parent_ids": list(entry.get("parent_ids", [])) if entry else [],
            "mapping_type": "missing_mask",
            "mapping_source": "missing_mask",
            "identity_status": "missing_candidate",
            "identity_mismatch_reasons": ["candidate_mask_missing"],
            "alias_match": original_alias_match,
        }
    contract = identity_contract(
        taxonomy,
        organ,
        source_local_label,
        resolved_organ,
        mapping_type=mapping_type,
        mapping_source=mapping_source,
    )
    contract["alias_match"] = original_alias_match
    return contract


def _write_identity_provenance(seg_dir: Path, organ: str, contract: dict[str, Any]) -> None:
    path = seg_dir / "identity_provenance.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"organs": {}}
    except Exception:
        doc = {"organs": {}}
    doc.setdefault("organs", {})[organ] = {
        "source_local_labels": [contract.get("source_local_label") or organ],
        "resolved_canonical_id": contract.get("resolved_canonical_id") or organ,
        "mapping_type": contract.get("mapping_type") or "exact_synonym",
        "mapping_source": contract.get("mapping_source"),
    }
    write_json(path, doc)


def _usable_binary_mask(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import nibabel as nib
        import numpy as np

        return int((np.asanyarray(nib.load(str(path)).dataobj) > 0).sum()) > 10
    except Exception:
        return False


def _hierarchical_plan_cache_key(
    *,
    requested_organs: list[str],
    major_organs: list[str],
    child_organs: list[str],
    execution_plan: dict[str, Any],
) -> dict[str, Any]:
    """Return the semantic parts that make a hierarchical ROI cache reusable."""
    per_organ = execution_plan.get("per_organ", {}) or {}
    relevant_organs = sorted(set(requested_organs) | set(major_organs) | set(child_organs))
    return {
        "requested_organs": list(requested_organs),
        "major_organs": list(major_organs),
        "child_organs": list(child_organs),
        "execution_plan": {
            "candidate_mode": execution_plan.get("candidate_mode"),
            "teacher_run_list": list(execution_plan.get("teacher_run_list", []) or []),
            "preseeded_models": list(execution_plan.get("preseeded_models", []) or []),
            "per_organ": {organ: per_organ.get(organ, {}) for organ in relevant_organs},
        },
    }


def _hierarchical_cache_keys_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Treat the retired postcava target as the inferior_vena_cava alias."""
    def normalize(key: dict[str, Any]) -> dict[str, Any]:
        value = json.loads(json.dumps(key))
        for field in ("requested_organs", "major_organs", "child_organs"):
            value[field] = sorted({
                "inferior_vena_cava" if x == "postcava" else x
                for x in value.get(field, [])
            })
        plan = value.get("execution_plan") or {}
        per_organ = plan.get("per_organ") or {}
        normalized_per_organ = {}
        for organ, row in per_organ.items():
            canonical = "inferior_vena_cava" if organ == "postcava" else organ
            normalized_per_organ.setdefault(canonical, row)
        plan["per_organ"] = normalized_per_organ
        value["execution_plan"] = plan
        return value
    return normalize(left or {}) == normalize(right or {})


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_hierarchical_case_inference(
    *,
    ct: Path,
    case_id: str,
    case_raw: Path,
    case_out: Path,
    registry_path: str | Path,
    execution_plan: dict[str, Any],
    requested_organs: list[str],
    taxonomy: dict[str, Any],
    alias_config: dict[str, Any],
    timeout_sec: int,
    device: str | None,
    dry_run: bool,
    margin_mm: float,
    preseeded_seg_dirs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    registry_file = Path(registry_path).resolve()
    registry_sha256 = hashlib.sha256(registry_file.read_bytes()).hexdigest() if registry_file.is_file() else None
    per_organ = execution_plan.get("per_organ", {}) or {}
    requested_set = set(requested_organs)
    dependency_parents = {
        parent
        for organ in requested_organs
        for parent in ((taxonomy_entry(taxonomy, organ) or {}).get("parent_ids", []) or [])
    }
    # Preserve requested/display IDs while consulting normalized taxonomy keys.
    # Direct set membership against taxonomy["organs"] silently dropped targets
    # such as vertebrae_L2 and celiac_aa (celiac_artery).
    direct_full_volume_organs = {"kidney_left", "kidney_right"}
    requested_major_organs = {
        organ for organ in requested_organs
        if (
            (taxonomy_entry(taxonomy, organ) or {}).get("hierarchy_role") == "major"
            or _norm_organ_key(organ) in direct_full_volume_organs
        )
    }
    major_organs = sorted(requested_major_organs | dependency_parents)
    child_organs = sorted({
        organ for organ in requested_organs
        if (
            (taxonomy_entry(taxonomy, organ) or {}).get("hierarchy_role") == "child"
            and _norm_organ_key(organ) not in direct_full_volume_organs
        )
    })
    cache_key = _hierarchical_plan_cache_key(
        requested_organs=requested_organs,
        major_organs=major_organs,
        child_organs=child_organs,
        execution_plan=execution_plan,
    )
    cache_key_sha256 = _stable_json_sha256(cache_key)
    existing_manifest = case_out / "hierarchical_inference_plan.json"
    if existing_manifest.exists() and not dry_run:
        try:
            cached = json.loads(existing_manifest.read_text(encoding="utf-8"))
            ct_stat = ct.stat()
            cached_ct = cached.get("ct_fingerprint", {}) or {}
            cache_valid = (
                cached.get("teacher_inference_mode") == "hierarchical_roi"
                and cached.get("hierarchical_pipeline_version") == HIERARCHICAL_PIPELINE_VERSION
                and cached.get("taxonomy_source_sha256") == taxonomy.get("source_sha256")
                and cached.get("registry_sha256") == registry_sha256
                and float(cached.get("margin_mm")) == float(margin_mm)
                and _hierarchical_cache_keys_equivalent(
                    cached.get("hierarchical_plan_cache_key") or {}, cache_key
                )
                and int(cached_ct.get("size", -1)) == int(ct_stat.st_size)
                and int(cached_ct.get("mtime_ns", -1)) == int(ct_stat.st_mtime_ns)
            )
            for checkpoint in (cached.get("model_checkpoint_refs", {}) or {}).values():
                resolved = checkpoint.get("resolved_path")
                if not resolved:
                    continue
                checkpoint_path = Path(str(resolved))
                current_mtime = checkpoint_path.stat().st_mtime_ns if checkpoint_path.exists() else None
                if current_mtime != checkpoint.get("mtime_ns"):
                    cache_valid = False
                    break
            cached_dirs = {
                path.name: path / "segmentations"
                for path in (case_out / "hierarchical_predictions").glob("*")
                if path.is_dir() and (path / "segmentations").exists() and any((path / "segmentations").glob("*.nii.gz"))
            }
            if cache_valid and cached_dirs:
                return {
                    "model_seg_dirs": cached_dirs,
                    "inference_results": [{
                        "case_id": case_id,
                        "status": "success",
                        "cache_status": "reused_hierarchical_roi_cache",
                        "teacher_inference_mode": "hierarchical_roi",
                        "manifest": str(existing_manifest),
                    }],
                    "blocked": list(cached.get("blocked", [])),
                    "manifest": str(existing_manifest),
                }
        except Exception:
            pass

    merged_root = case_out / "hierarchical_predictions"
    merged_dirs: dict[str, Path] = {}
    raw_results: list[dict[str, Any]] = []
    full_runs: dict[str, tuple[dict[str, Any], Path]] = {}
    parent_masks: dict[str, Path] = {}
    blocked: list[dict[str, Any]] = []
    major_resolution: list[dict[str, Any]] = []
    hierarchy_qc_cache = _CaseMaskCache(max_arrays=48)

    def hard_qc_usable(mask: Path, organ: str) -> tuple[bool, dict[str, Any]]:
        qc = _compute_candidate_qc(ct=ct, mask=mask if mask.exists() else None, organ=organ, cache=hierarchy_qc_cache)
        return bool(_usable_binary_mask(mask) and qc.get("status") != "fail"), qc

    def merged_dir(model: str) -> Path:
        path = merged_dirs.setdefault(model, merged_root / model / "segmentations")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_full(model: str, target_organs: list[str]) -> tuple[dict[str, Any], Path]:
        if model in full_runs:
            return full_runs[model]
        preseeded_seg_dir = (preseeded_seg_dirs or {}).get(model)
        preseeded_summary: dict[str, Any] = {}
        if preseeded_seg_dir is not None:
            summary_candidates = [
                preseeded_seg_dir / "inference_summary.json",
                preseeded_seg_dir.parent / "inference_summary.json",
            ]
            for summary_path in summary_candidates:
                if summary_path.exists():
                    try:
                        preseeded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except Exception:
                        preseeded_summary = {}
                    break
        completed_zero_output = bool(
            preseeded_summary
            and preseeded_summary.get("return_code") == 0
            and not preseeded_summary.get("timed_out")
        )
        if (
            not dry_run
            and preseeded_seg_dir is not None
            and preseeded_seg_dir.exists()
            and (
                any((preseeded_seg_dir / f"{organ}.nii.gz").exists() for organ in target_organs)
                or completed_zero_output
            )
        ):
            result = {
                "status": "success",
                "model_key": model,
                "segmentation_output": str(preseeded_seg_dir),
                "num_masks": sum(1 for _ in preseeded_seg_dir.glob("*.nii.gz")),
                "cache_status": (
                    "reused_preseeded_completed_zero_output"
                    if completed_zero_output and not any((preseeded_seg_dir / f"{organ}.nii.gz").exists() for organ in target_organs)
                    else "reused_preseeded_major_parent_cache"
                ),
                "teacher_inference_mode": "hierarchical_roi",
                "inference_scope": "major_full_volume_parent_cache_only",
                "requested_organs": target_organs,
                "cache_policy": (
                    "Only major-organ parent masks are reused from preseeded full-volume predictions; "
                    "child/sub-organ masks are regenerated through hierarchical ROI."
                ),
            }
            full_runs[model] = (result, preseeded_seg_dir)
            raw_results.append({"case_id": case_id, "inference_scope": "major_full_volume_parent_cache_only", **result})
            return result, preseeded_seg_dir
        legacy_seg_dir = case_raw / model / case_id / "segmentations"
        if (
            not dry_run
            and legacy_seg_dir.exists()
            and any((legacy_seg_dir / f"{organ}.nii.gz").exists() for organ in target_organs)
        ):
            result = {
                "status": "success",
                "model_key": model,
                "segmentation_output": str(legacy_seg_dir),
                "num_masks": sum(1 for _ in legacy_seg_dir.glob("*.nii.gz")),
                "cache_status": "reused_legacy_full_volume_major_parent_cache",
                "teacher_inference_mode": "hierarchical_roi",
                "inference_scope": "major_full_volume_parent_cache_only",
                "requested_organs": target_organs,
                "cache_policy": (
                    "Only major-organ parent masks may be reused from legacy full-volume raw predictions; "
                    "child/sub-organ masks are regenerated through hierarchical ROI."
                ),
            }
            full_runs[model] = (result, legacy_seg_dir)
            raw_results.append({"case_id": case_id, "inference_scope": "major_full_volume_parent_cache_only", **result})
            return result, legacy_seg_dir
        result = run_registered_model(
            ct, case_raw / "hierarchical_full" / model, model, registry_path=registry_path, case_id=case_id,
            dry_run=dry_run, timeout_sec=timeout_sec, device=device,
            extra_context={"requested_organs": target_organs, "teacher_inference_mode": "hierarchical_roi", "inference_scope": "major_full_volume"},
        )
        result.setdefault("inference_scope", "major_full_volume")
        seg_dir = Path(str(result.get("segmentation_output", case_raw / model / case_id / "segmentations")))
        full_runs[model] = (result, seg_dir)
        raw_results.append({"case_id": case_id, "inference_scope": "major_full_volume", **result})
        return result, seg_dir

    model_major_targets: dict[str, set[str]] = {}
    for candidate_organ in major_organs:
        candidate_route = per_organ.get(candidate_organ, {}) or {}
        for candidate_model in [
            candidate_route.get("primary_teacher"),
            *candidate_route.get("backup_teachers", []),
            *candidate_route.get("competition_teachers", []),
        ]:
            if candidate_model:
                model_major_targets.setdefault(str(candidate_model), set()).add(candidate_organ)
    for organ in major_organs:
        route = per_organ.get(organ, {}) or {}
        ordered = [
            route.get("primary_teacher"),
            *route.get("backup_teachers", []),
            *route.get("competition_teachers", []),
        ]
        ordered = [str(x) for x in ordered if x]
        # Repair policy: exhaust reusable major-parent masks across the route
        # before launching any fresh full-volume teacher. This may prefer a
        # cached backup over an uncached primary, while preserving route order
        # within the cached and uncached groups.
        ordered = sorted(
            ordered,
            key=lambda model: 0 if (
                model in (preseeded_seg_dirs or {})
                or (case_raw / model / case_id / "segmentations").exists()
            ) else 1,
        )
        resolved = False
        for model in ordered:
            major_inference, seg_dir = run_full(model, sorted(model_major_targets.get(model, {organ})))
            source, match_mode = _candidate_mask_path(seg_dir, organ, model, alias_config)
            contract = _candidate_identity_contract(
                taxonomy=taxonomy, alias_config=alias_config, organ=organ, model_key=model, alias_match=match_mode,
                seg_dir=seg_dir,
            )
            usable, hierarchy_qc = hard_qc_usable(source, organ)
            if contract["identity_status"] != "valid" or not usable:
                major_resolution.append({
                    "organ": organ, "model": model, "status": "unusable", "mask": str(source),
                    "cache_status": major_inference.get("cache_status"),
                    "inference_scope": major_inference.get("inference_scope"),
                    "hierarchy_qc": hierarchy_qc, **contract,
                })
                continue
            destination = merged_dir(model) / f"{organ}.nii.gz"
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            _write_identity_provenance(merged_dir(model), organ, contract)
            if not resolved:
                parent_masks[organ] = destination
            major_resolution.append({
                "organ": organ, "model": model, "status": "resolved", "mask": str(destination),
                "cache_status": major_inference.get("cache_status"),
                "inference_scope": major_inference.get("inference_scope"),
                "hierarchy_qc": hierarchy_qc, **contract,
            })
            resolved = True
        if not resolved:
            blocked.append({"organ": organ, "status": "major_unresolved", "reason": "primary_and_backups_unusable"})

    child_routes: list[dict[str, Any]] = []
    child_fallbacks: dict[str, list[str]] = {}
    for organ in child_organs:
        entry = taxonomy_entry(taxonomy, organ) or {}
        route = per_organ.get(organ, {}) or {}
        primary = route.get("primary_teacher")
        fallbacks = [
            str(x)
            for x in [*route.get("backup_teachers", []), *route.get("competition_teachers", [])]
            if x
        ]
        if not primary:
            blocked.append({"organ": organ, "status": "unresolved_route", "reason": "no_exact_primary_teacher"})
            continue
        child_routes.append({"organ": organ, "parent_ids": entry.get("parent_ids", []), "model": str(primary)})
        child_fallbacks[organ] = fallbacks

    roi_root = case_out / "hierarchical_roi"

    def run_roi_model(image: Path, output: Path, model: str, target_organs: list[str], task_id: str) -> dict[str, Any]:
        result = run_registered_model(
            image, output, model, registry_path=registry_path, case_id=task_id,
            dry_run=dry_run, timeout_sec=timeout_sec, device=device,
            extra_context={"requested_organs": target_organs, "teacher_inference_mode": "hierarchical_roi", "inference_scope": "child_roi"},
        )
        result.setdefault("inference_scope", "child_roi")
        raw_results.append({"case_id": case_id, "inference_scope": "child_roi", "roi_task_id": task_id, **result})
        return result

    if dry_run:
        tasks = []
        parent_blocked = [{"organ": route["organ"], "status": "dry_run_pending_parent_mask"} for route in child_routes]
    else:
        tasks, parent_blocked = plan_roi_tasks(
            ct_path=ct, child_routes=child_routes, parent_masks=parent_masks, margin_mm=margin_mm,
            allow_cross_parent_merge=True,
        )
    blocked.extend(parent_blocked)
    attempted_child_pairs = {(str(task["model"]), str(organ)) for task in tasks for organ in task["organs"]}
    for task in tasks:
        merged_dir(str(task["model"]))
    task_results = execute_roi_tasks(
        ct_path=ct, tasks=tasks, work_root=roi_root, merged_model_dirs=merged_dirs, run_model=run_roi_model,
        alias_config=alias_config,
    ) if tasks and not dry_run else []

    # Competition/backup candidates must actually be produced, not only recorded
    # in the execution plan, otherwise LabelCritic/AutoLabelCore see an empty pool.
    proactive_fallback_routes = [
        {**route, "model": fallback_model}
        for organ, route in {str(route["organ"]): route for route in child_routes}.items()
        for fallback_model in child_fallbacks.get(organ, [])
    ]
    if dry_run:
        proactive_tasks, proactive_parent_blocked = [], []
    else:
        proactive_tasks, proactive_parent_blocked = plan_roi_tasks(
            ct_path=ct,
            child_routes=proactive_fallback_routes,
            parent_masks=parent_masks,
            margin_mm=margin_mm,
            allow_cross_parent_merge=True,
        )
    blocked.extend(proactive_parent_blocked)
    attempted_child_pairs.update((str(task["model"]), str(organ)) for task in proactive_tasks for organ in task["organs"])
    for task in proactive_tasks:
        merged_dir(str(task["model"]))
    proactive_results = execute_roi_tasks(
        ct_path=ct,
        tasks=proactive_tasks,
        work_root=roi_root / "candidate_fallbacks",
        merged_model_dirs=merged_dirs,
        run_model=run_roi_model,
        alias_config=alias_config,
    ) if proactive_tasks and not dry_run else []

    backup_results: list[dict[str, Any]] = []
    unresolved = {
        str(route["organ"]): route for route in child_routes
        if not hard_qc_usable(merged_dir(str(route["model"])) / f"{route['organ']}.nii.gz", str(route["organ"]))[0]
    }
    primary_retry_results: list[dict[str, Any]] = []
    if unresolved and not dry_run:
        primary_retry_tasks, primary_retry_blocked = plan_roi_tasks(
            ct_path=ct,
            child_routes=list(unresolved.values()),
            parent_masks=parent_masks,
            margin_mm=margin_mm,
            allow_cross_parent_merge=False,
        )
        blocked.extend(primary_retry_blocked)
        primary_retry_results = execute_roi_tasks(
            ct_path=ct,
            tasks=primary_retry_tasks,
            work_root=roi_root / "primary_parent_retry",
            merged_model_dirs=merged_dirs,
            run_model=run_roi_model,
            alias_config=alias_config,
        ) if primary_retry_tasks else []
        attempted_child_pairs.update(
            (str(task["model"]), str(organ))
            for task in primary_retry_tasks
            for organ in task["organs"]
        )
        unresolved = {
            organ: route for organ, route in unresolved.items()
            if not hard_qc_usable(merged_dir(str(route["model"])) / f"{organ}.nii.gz", organ)[0]
        }
    max_backups = max((len(child_fallbacks.get(organ, [])) for organ in unresolved), default=0)
    for backup_index in range(max_backups):
        backup_routes = [
            {**route, "model": child_fallbacks[organ][backup_index]}
            for organ, route in unresolved.items()
            if backup_index < len(child_fallbacks.get(organ, []))
            and (str(child_fallbacks[organ][backup_index]), organ) not in attempted_child_pairs
        ]
        if dry_run:
            backup_tasks, backup_parent_blocked = [], []
        else:
            backup_tasks, backup_parent_blocked = plan_roi_tasks(
                ct_path=ct, child_routes=backup_routes, parent_masks=parent_masks, margin_mm=margin_mm,
            )
        blocked.extend(backup_parent_blocked)
        attempted_child_pairs.update((str(task["model"]), str(organ)) for task in backup_tasks for organ in task["organs"])
        for task in backup_tasks:
            merged_dir(str(task["model"]))
        round_results = execute_roi_tasks(
            ct_path=ct, tasks=backup_tasks, work_root=roi_root / f"backup_{backup_index + 1}",
            merged_model_dirs=merged_dirs, run_model=run_roi_model, alias_config=alias_config,
        ) if backup_tasks and not dry_run else []
        backup_results.extend(round_results)
        for route in backup_routes:
            organ = str(route["organ"])
            if hard_qc_usable(merged_dir(str(route["model"])) / f"{organ}.nii.gz", organ)[0]:
                unresolved.pop(organ, None)
    for organ in sorted(unresolved):
        blocked.append({"organ": organ, "status": "child_unresolved", "reason": "primary_and_backups_unusable"})

    child_resolution: list[dict[str, Any]] = []
    for organ in child_organs:
        route = per_organ.get(organ, {}) or {}
        ordered_models = [
            route.get("primary_teacher"),
            *route.get("backup_teachers", []),
            *route.get("competition_teachers", []),
        ]
        resolved_record: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        for model_value in ordered_models:
            if not model_value:
                continue
            model = str(model_value)
            if (model, organ) not in attempted_child_pairs:
                attempts.append({"model": model, "attempted": False, "reason": "not_needed_after_prior_success"})
                continue
            candidate = merged_dir(model) / f"{organ}.nii.gz"
            usable, hierarchy_qc = hard_qc_usable(candidate, organ)
            attempt = {"model": model, "attempted": True, "mask": str(candidate), "usable": usable, "hierarchy_qc": hierarchy_qc}
            attempts.append(attempt)
            if usable and resolved_record is None:
                resolved_record = attempt
        child_resolution.append({
            "organ": organ,
            "status": "resolved" if resolved_record else "unresolved",
            "selected_model": resolved_record.get("model") if resolved_record else None,
            "selected_mask": resolved_record.get("mask") if resolved_record else None,
            "attempts": attempts,
        })

    manifest_path = case_out / "hierarchical_inference_plan.json"
    registry_doc = load_registry(registry_file) if registry_file.exists() else {"models": {}}
    checkpoint_refs: dict[str, Any] = {}
    project_root = registry_file.parent.parent
    for model in sorted(merged_dirs):
        raw_checkpoint = str(((registry_doc.get("models", {}) or {}).get(model, {}) or {}).get("checkpoint_path") or "")
        checkpoint_path = Path(raw_checkpoint)
        if raw_checkpoint and not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        checkpoint_refs[model] = {
            "configured_path": raw_checkpoint,
            "resolved_path": str(checkpoint_path.resolve()) if raw_checkpoint else None,
            "exists": bool(raw_checkpoint and checkpoint_path.exists()),
            "mtime_ns": checkpoint_path.stat().st_mtime_ns if raw_checkpoint and checkpoint_path.exists() else None,
        }
    write_hierarchical_manifest(manifest_path, {
        "case_id": case_id,
        "ct_path": str(ct),
        "teacher_inference_mode": "hierarchical_roi",
        "hierarchical_pipeline_version": HIERARCHICAL_PIPELINE_VERSION,
        "taxonomy_schema_version": taxonomy.get("schema_version"),
        "taxonomy_source_sha256": taxonomy.get("source_sha256"),
        "registry_sha256": registry_sha256,
        "model_checkpoint_refs": checkpoint_refs,
        "margin_mm": margin_mm,
        "requested_organs": requested_organs,
        "hierarchical_plan_cache_key": cache_key,
        "hierarchical_plan_cache_key_sha256": cache_key_sha256,
        "cross_parent_roi_merge": "adaptive_parent_support_validation",
        "major_organs": major_organs,
        "child_organs": child_organs,
        "major_resolution": major_resolution,
        "child_resolution": child_resolution,
        "roi_tasks": task_results,
        "candidate_fallback_tasks": proactive_results,
        "primary_parent_retry_tasks": primary_retry_results,
        "backup_roi_tasks": backup_results,
        "blocked": blocked,
    })
    return {
        "model_seg_dirs": {model: path for model, path in merged_dirs.items() if any(path.glob("*.nii.gz"))},
        "inference_results": raw_results,
        "blocked": blocked,
        "manifest": str(manifest_path),
    }


def _zero_mask_path_for_case(ct: Path, case_updated: Path) -> str | None:
    """Create/reuse a canonical all-zero target aligned to this case CT."""
    out = case_updated / "negative_targets" / "zero_mask.nii.gz"
    if out.exists():
        return str(out.resolve())
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(ct))
        zero = np.zeros(img.shape[:3], dtype=np.uint8)
        out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(zero, img.affine, img.header), str(out))
        return str(out.resolve())
    except Exception:
        return None


def _materialize_case_373_targets(
    *,
    case_id: str,
    ct: Path,
    organs: list[str],
    case_updated: Path,
    selection_rows: list[dict[str, Any]],
    selected_metadata: list[dict[str, Any]],
    presence_context: dict[str, Any],
    negative_absent_training_weight: float = 0.1,
) -> dict[str, Any]:
    """Ensure each formal target has a supervision record for this case."""
    formal_organs = set(organs)
    selection_rows[:] = [row for row in selection_rows if str(row.get("organ") or "") in formal_organs]
    selected_metadata[:] = [row for row in selected_metadata if str(row.get("organ") or "") in formal_organs]
    zero_mask = _zero_mask_path_for_case(ct, case_updated)

    def publish_absent_negative_mask(organ: str) -> str | None:
        """Publish an organ-named all-zero mask for absent-negative targets."""
        if not zero_mask:
            return None
        out = case_updated / f"{organ}.nii.gz"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                shutil.copy2(zero_mask, out)
            return str(out.resolve())
        except Exception:
            return zero_mask

    contract_type_map = {
        "hard": "positive_hard",
        "soft": "positive_soft",
        "provisional": "rejected",
        "review_gap": "unresolved_visible",
        "unresolved_review": "unresolved_visible",
        "absent_negative": "negative_absent",
    }
    for row in [*selection_rows, *selected_metadata]:
        legacy_type = str(row.get("target_type") or "hard")
        row.setdefault("legacy_target_type", legacy_type)
        row["target_type"] = contract_type_map.get(legacy_type, legacy_type)
        existing_fov = str(row.get("fov_status") or "")
        row["fov_status"] = (
            existing_fov
            if existing_fov in {"fully_visible", "partially_visible", "out_of_fov", "unknown"}
            else _fov_status_for_organ(str(row.get("organ") or ""), presence_context)
        )
        if (
            row["fov_status"] == "partially_visible"
            and row["target_type"] in {"positive_hard", "positive_soft"}
        ):
            row["legacy_target_type_before_partial_fov"] = row["target_type"]
            row["target_type"] = "partial_fov"
        if row["target_type"] == "negative_absent":
            if row["fov_status"] != "out_of_fov":
                row["target_type"] = (
                    "partial_fov"
                    if row["fov_status"] == "partially_visible"
                    else "unresolved_visible"
                )
                row["absence_confidence"] = "not_established"
                row["zero_mask_role"] = "io_placeholder_not_training_target"
                row["negative_source"] = None
                row["negative_reason"] = None
            else:
                organ_name = str(row.get("organ") or "")
                published_zero = publish_absent_negative_mask(organ_name) if organ_name else zero_mask
                row.update({
                    "absence_confidence": "high",
                    "grade_scope": "absence",
                    "record_type": "negative_absent",
                    "selection_method": "negative_absent",
                    "zero_mask_role": "negative_absent_target_mask",
                    "negative_source": "case_373_expected_absent",
                    "dataset_role": "semantic_absent_negative",
                    "fov_evidence": list(presence_context.get("coverage_evidence") or []),
                    "selected_prediction": published_zero,
                    "mask_path": published_zero,
                    "mask": published_zero,
                    "final_mask": published_zero,
                })
        if row["target_type"] in {"unresolved_visible", "partial_fov", "rejected"}:
            row["training_weight"] = 0.0
            row["distillation_eligible"] = False
            row["should_enter_student_training"] = False
        if (
            row["target_type"] in {"positive_hard", "positive_soft"}
            and (
                row.get("should_enter_student_training") is False
                or row.get("selection_method") in {
                    "label_critic_audit_only",
                    "label_critic_inconclusive",
                    "single_teacher_provisional",
                }
            )
        ):
            row["training_weight"] = 0.0
            row["distillation_eligible"] = False
            row["should_enter_student_training"] = False
            row["training_block_reason"] = (
                "selection_not_formally_eligible_under_current_gate"
            )
    selection_by_organ = {str(row.get("organ")): row for row in selection_rows if row.get("organ")}
    absent_added = 0
    review_gap_missing = 0
    geometry_failures = 0
    for organ in organs:
        if organ in selection_by_organ:
            continue
        fov_status = _fov_status_for_organ(organ, presence_context)
        expected_presence = _expected_presence_for_organ(organ, presence_context)
        if fov_status != "out_of_fov":
            review_gap_missing += 1
            reason = "no teacher candidate and absence is not proven by scan coverage"
            target_type = "partial_fov" if fov_status == "partially_visible" else "unresolved_visible"
            row = {
                "case_id": case_id, "ct_path": str(ct), "organ": organ,
                "expected_presence": expected_presence, "fov_status": fov_status,
                "target_type": target_type,
                "legacy_target_type": "unresolved_review",
                "record_type": target_type,
                "grade": "D", "grade_scope": "unresolved_target",
                "confidence": 0.0, "label_confidence": 0.0,
                "training_weight": 0.0, "distillation_eligible": False,
                "supervision_type": "none", "distillation_role": "review_only",
                "selection_method": "none", "selection_status": "review_required",
                "reason": reason, "selected_reason": reason,
                "rejected_reasons": {}, "failure_modes": ["missing_candidate"],
                "candidate_count": 0, "candidate_ids": [], "candidate_models": [],
                "teacher_names": [], "teacher_families": [],
                "selected_candidate": None, "selected_candidate_id": None,
                "selected_teacher": None, "selected_family": None,
                "selected_model": None, "source_model": None,
                "selected_prediction": zero_mask, "mask_path": zero_mask,
                "mask": zero_mask, "final_mask": zero_mask, "overlay_path": None,
                "zero_mask_role": "io_placeholder_not_training_target",
                "labelcritic_called": False,
                "labelcritic_skipped_reason": "no_candidate_and_absence_unproven",
                "labelcritic_prompt_version": None,
                "quality_flags": ["missing_candidate"],
                "review_flags": ["missing_candidate", "absence_unproven"],
                "quality_status": "review_required",
                "publication_status": "withheld_unresolved",
                "should_enter_student_training": False,
                "absence_confidence": "not_established",
                "fov_evidence": list(presence_context.get("coverage_evidence") or []),
                "metric_target": "none",
                "metric_subject": "E-step unresolved target",
                "metric_comparison": "none",
                "metric_interpretation": "not_evaluable",
            }
            selection_rows.append(dict(row))
            selected_metadata.append(dict(row))
            selection_by_organ[organ] = row
            continue
        if not zero_mask:
            geometry_failures += 1
            selection_rows.append({
                "case_id": case_id, "ct_path": str(ct), "organ": organ,
                "expected_presence": expected_presence, "fov_status": fov_status,
                "target_type": "unresolved_visible", "legacy_target_type": "review_gap",
                "selection_method": "none", "selection_status": "missing",
                "reason": "missing_geometry_for_negative", "candidate_count": 0,
                "candidate_models": [], "review_flags": ["missing_geometry_for_negative"],
                "quality_flags": ["missing_candidate"],
            })
            continue
        published_zero = publish_absent_negative_mask(organ) or zero_mask
        reason = "organ outside scan/body region; all-zero mask is valid negative target"
        row = {
            "case_id": case_id, "ct_path": str(ct), "organ": organ,
            "expected_presence": expected_presence, "fov_status": fov_status,
            "target_type": "negative_absent", "legacy_target_type": "absent_negative",
            "grade": "A", "grade_scope": "absence",
            "record_type": "negative_absent",
            "absence_confidence": "high",
            "fov_evidence": list(presence_context.get("coverage_evidence") or []),
            "confidence": 1.0, "label_confidence": 1.0,
            "training_weight": float(negative_absent_training_weight),
            "distillation_eligible": float(negative_absent_training_weight) > 0.0,
            "supervision_type": "negative", "distillation_role": "negative",
            "selection_method": "negative_absent", "selection_status": "selected",
            "reason": reason, "selected_reason": reason, "rejected_reasons": {}, "failure_modes": [],
            "candidate_count": 0, "candidate_ids": [], "candidate_models": [],
            "teacher_names": [], "teacher_families": [],
            "comparison_candidate_count": 0, "comparison_candidate_models": [],
            "selected_candidate": None, "selected_candidate_id": None,
            "selected_teacher": None, "selected_family": None,
            "selected_model": None, "source_model": None,
            "selected_prediction": published_zero,
            "labelcritic_called": False, "labelcritic_compare_used": False,
            "labelcritic_skipped_reason": "absent_negative_no_candidate_ranking_needed",
            "labelcritic_prompt_version": None,
            "labelcritic_grade_used": False,
            "mask_path": published_zero, "mask": published_zero, "final_mask": published_zero, "overlay_path": None,
            "zero_mask_role": "negative_absent_target_mask",
            "negative_reason": "out_of_scan_by_scan_coverage",
            "negative_source": "case_373_expected_absent",
            "dataset_role": "semantic_absent_negative",
            "ground_truth_status": "valid_absent_negative",
            "metric_family": "absence_supervision",
            "metric_scope": "all_zero_negative_target_for_absent_organ",
            "metric_target": "all-zero target",
            "metric_subject": "E-step output",
            "metric_comparison": "e_step_all_zero_absent_negative_target",
            "metric_interpretation": "negative_absence_quality",
            "accuracy_warning": "Absent-negative all-zero masks are scan-coverage supervision, not expert positive segmentations.",
            "scoring_schema_version": "autolabel_core_v3_absent_negative",
            "quality_flags": [], "review_flags": [], "quality_status": "ok",
            "publication_status": "accepted_absent_negative",
            "should_enter_student_training": float(negative_absent_training_weight) > 0.0,
        }
        selection_rows.append(dict(row))
        selected_metadata.append(dict(row))
        selection_by_organ[organ] = row
        absent_added += 1
    target_type_counts: dict[str, int] = {}
    for row in selection_rows:
        key = str(row.get("target_type") or "hard")
        target_type_counts[key] = target_type_counts.get(key, 0) + 1
    return {
        "case_id": case_id,
        "num_classes": len(organs),
        "expected_targets": len(organs),
        "selection_rows": len(selection_rows),
        "selected_metadata": len(selected_metadata),
        "absent_negative_added": absent_added,
        "review_gap_missing": review_gap_missing,
        "negative_geometry_failures": geometry_failures,
        "zero_mask": zero_mask,
        "target_type_counts": target_type_counts,
        "complete_case_373": (
            len(selection_rows) == len(organs)
            and len({str(r.get("organ")) for r in selection_rows if r.get("organ")}) == len(set(organs))
        ),
        "complete_case_373_semantics": "record_and_output_contract_only_not_visibility_or_accuracy",
        "target_identity_holds": sum(target_type_counts.values()) == len(organs),
    }


def _copy_annotation(src: Path | None, dst_dir: Path, organ: str) -> str | None:
    if src and src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{organ}.nii.gz"
        if src.resolve() != dst.resolve() and not _same_file_fingerprint(src, dst):
            shutil.copy2(src, dst)
        return str(dst.resolve())
    return None


def _copy_case_mask(src: Path, dst_case_root: Path, organ: str) -> Path | None:
    """Copy one selected organ mask into a case/segmentations layout."""
    if not src.exists():
        return None
    seg_dir = dst_case_root / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    dst = seg_dir / f"{organ}.nii.gz"
    if src.resolve() != dst.resolve() and not _same_file_fingerprint(src, dst):
        shutil.copy2(src, dst)
    return dst


def _export_standard_case_dataset(
    *,
    case_id: str,
    ct: Path,
    selected_metadata: list[dict[str, Any]],
    out_root: Path,
    student_target_ids: dict[str, Any],
) -> dict[str, Any]:
    case_root = out_root / case_id
    seg_dir = case_root / "segmentations"
    case_root.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    image_dst = case_root / "image.nii.gz"
    if ct.exists() and (not image_dst.exists() or not _same_file_fingerprint(ct, image_dst)):
        shutil.copy2(ct, image_dst)

    mappings: list[dict[str, Any]] = []
    copied_masks: list[str] = []
    exported_metadata: list[dict[str, Any]] = []
    for meta in selected_metadata:
        organ = str(meta.get("organ") or "").strip()
        final = Path(str(meta.get("final_mask") or meta.get("mask_path") or ""))
        if not organ or not final.exists():
            continue
        if str(meta.get("target_type")) == "absent_negative":
            exported_metadata.append({
                **meta,
                "ct_path": str(image_dst.resolve()),
                "image": str(image_dst.resolve()),
                "canonical_organ_name": organ,
                "student_target_id": student_target_ids.get(organ),
                "distillation_eligible": bool(meta.get("distillation_eligible", True)),
                "distillation_exclusion_reason": meta.get("distillation_exclusion_reason"),
                "standard_dataset_export_status": "metadata_only_absent_negative",
            })
            continue
        gate = _distillation_gate_for_selected_label(meta)
        export_as_positive = bool(gate["eligible"])
        if not export_as_positive:
            exported_metadata.append({
                **meta,
                "ct_path": str(image_dst.resolve()),
                "image": str(image_dst.resolve()),
                "canonical_organ_name": organ,
                "student_target_id": student_target_ids.get(organ),
                "distillation_eligible": False,
                "distillation_exclusion_reason": gate["reason"],
                "standard_dataset_export_status": "metadata_only_excluded_from_positive_masks",
            })
            continue
        dst = seg_dir / f"{organ}.nii.gz"
        if not dst.exists() or not _same_file_fingerprint(final, dst):
            shutil.copy2(final, dst)
        copied_masks.append(dst.name)
        selected_prediction = str(meta.get("selected_prediction") or meta.get("selected_pre_shapekit_prediction") or "")
        teacher_output_file = Path(selected_prediction).name if selected_prediction else None
        mappings.append({
            "teacher_target_id": meta.get("teacher_target_id"),
            "teacher_output_name": Path(teacher_output_file).name[:-7] if teacher_output_file and teacher_output_file.endswith(".nii.gz") else organ,
            "teacher_output_file": teacher_output_file,
            "teacher_model": meta.get("selected_model") or meta.get("source_model"),
            "canonical_organ_name": organ,
            "student_target_id": student_target_ids.get(organ),
            "selected_pseudo_label": str(dst.resolve()),
            "mapping_status": "selected_teacher_output_to_canonical_binary_mask",
        })
        exported_metadata.append({
            **meta,
            "ct_path": str(image_dst.resolve()),
            "image": str(image_dst.resolve()),
            "final_mask": str(dst.resolve()),
            "mask_path": str(dst.resolve()),
            "mask": str(dst.resolve()),
            "canonical_organ_name": organ,
            "student_target_id": student_target_ids.get(organ),
            "distillation_eligible": True,
            "distillation_exclusion_reason": None,
            "standard_dataset_export_status": "positive_mask_exported",
        })

    mapping_path = case_root / "label_mapping.json"
    write_json(mapping_path, {
        "case_id": case_id,
        "layout": "bdmap_pants_style_binary_masks",
        "image": str(image_dst.resolve()),
        "segmentations": str(seg_dir.resolve()),
        "mapping_layers": [
            "teacher target ID / output name",
            "canonical organ name",
            "student target ID",
        ],
        "target_mapping_policy": "Student target IDs come from configs/student_3d_prompt_target_organs.json and are not copied from teacher label IDs.",
        "ground_truth_status": "pseudo_label_candidate",
        "mappings": mappings,
    })
    write_json(case_root / "selection_metadata.json", {
        "case_id": case_id,
        "ct_path": str(image_dst.resolve()),
        "dataset_type": "pseudo_label_dataset",
        "ground_truth_status": "pseudo_label_candidate",
        "layout": "bdmap_pants_style_binary_masks",
        "label_mapping": str(mapping_path.resolve()),
        "selected_organs": exported_metadata,
    })
    return {
        "case_id": case_id,
        "case_folder": str(case_root.resolve()),
        "image": str(image_dst.resolve()),
        "segmentations": str(seg_dir.resolve()),
        "label_mapping": str(mapping_path.resolve()),
        "num_masks": len(copied_masks),
        "sample_masks": copied_masks[:30],
    }


def _resolve_preseeded_case_dir(seed_base: Path, case_id: str) -> Path | None:
    """Find a preseeded case mask directory across supported case layouts."""
    if "{case_id}" in str(seed_base):
        seed_base = Path(str(seed_base).replace("{case_id}", case_id))
    case_root = seed_base / case_id
    # Prefer directories of canonical binary masks. Some teacher roots also
    # contain a single combined_labels.nii.gz; selecting that root hides the
    # usable per-organ segmentations nested below it.
    for candidate in (case_root / "updated", case_root / "segmentations", seed_base / "updated", seed_base / "segmentations", case_root, seed_base):
        if candidate.exists() and any(candidate.glob("*.nii.gz")):
            return candidate
    return None


def _add_shapekit_calibration_masks(
    *,
    selected_case_root: Path,
    selected_organs: list[str],
    model_seg_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    """Add auxiliary masks required by ShapeKit but not by the final manifest.

    Some ShapeKit post-processors use the liver mask as an anatomical calibration
    standard for left/right reassignment.  We can provide that mask to ShapeKit
    without turning it into a selected pseudo-label for student training.
    """
    liver_dependent_families = {
        "femur": {"femur_left", "femur_right"},
        "kidney": {"kidney_left", "kidney_right"},
        "lung": {"lung_left", "lung_right"},
        "adrenal_gland": {"adrenal_gland_left", "adrenal_gland_right"},
    }
    requested = set(selected_organs)
    needs_liver = any(bool(requested & organs) for organs in liver_dependent_families.values())
    if not needs_liver:
        return []

    seg_dir = selected_case_root / "segmentations"
    liver_dst = seg_dir / "liver.nii.gz"
    if liver_dst.exists():
        return []

    for model_key, model_seg_dir in model_seg_dirs.items():
        liver_src = model_seg_dir / "liver.nii.gz"
        if liver_src.exists():
            copied = _copy_case_mask(liver_src, selected_case_root, "liver")
            if copied:
                return [{
                    "organ": "liver",
                    "source_model": model_key,
                    "mask": str(copied),
                    "reason": "ShapeKit calibration-only mask for liver-dependent post-processing",
                    "dataset_role": "shapekit_calibration_only",
                    "included_in_training_manifest": False,
                }]
    return []


def _prepare_candidate_shapekit_input(seg_dir: Path, input_root: Path, case_id: str) -> int:
    dst_seg = input_root / case_id / "segmentations"
    dst_seg.mkdir(parents=True, exist_ok=True)
    count = 0
    for mask in sorted(seg_dir.glob("*.nii.gz")):
        dst = dst_seg / mask.name
        if not _same_file_fingerprint(mask, dst):
            shutil.copy2(mask, dst)
        count += 1
    return count


def _safe_shapekit_targets_for_seg_dir(seg_dir: Path) -> list[str]:
    names = {p.name[:-7] for p in seg_dir.glob("*.nii.gz")} if seg_dir.exists() else set()
    targets: list[str] = []
    for target, reqs in _SHAPEKIT_TARGET_REQUIREMENTS.items():
        if target == "vertebrae":
            if any(name.startswith("vertebrae_") for name in names):
                targets.append(target)
        elif all(req in names for req in reqs):
            targets.append(target)
    return targets


def _postprocess_candidate_models_with_shapekit(
    *,
    model_seg_dirs: dict[str, Path],
    case_refined: Path,
    shapekit_root: str | Path,
    case_id: str,
    enable_shapekit: bool,
    dry_run: bool,
    timeout_sec: int,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    processed_dirs: dict[str, Path] = {}
    reports: dict[str, dict[str, Any]] = {}
    if not enable_shapekit:
        for model_key, seg_dir in model_seg_dirs.items():
            processed_dirs[model_key] = seg_dir
            reports[model_key] = {
                "stage": "candidate_preselection_shapekit",
                "status": "skipped_debug_only",
                "model": model_key,
                "raw_seg_dir": str(seg_dir),
                "processed_seg_dir": str(seg_dir),
                "fallback_used": True,
            }
        return processed_dirs, reports
    if dry_run:
        for model_key, seg_dir in model_seg_dirs.items():
            processed_dirs[model_key] = seg_dir
            reports[model_key] = {
                "stage": "candidate_preselection_shapekit",
                "status": "skipped_dry_run",
                "model": model_key,
                "raw_seg_dir": str(seg_dir),
                "processed_seg_dir": str(seg_dir),
                "fallback_used": True,
            }
        return processed_dirs, reports

    for model_key, seg_dir in model_seg_dirs.items():
        input_root = case_refined / "candidate_shapekit_input" / model_key
        output_root = case_refined / "candidate_shapekit" / model_key
        safe_targets = _safe_shapekit_targets_for_seg_dir(seg_dir)
        if not safe_targets:
            processed_dirs[model_key] = seg_dir
            reports[model_key] = {
                "stage": "candidate_preselection_shapekit",
                "status": "unsupported_target_skipped_by_policy",
                "reason": "No safe ShapeKit target organs detected before staging input",
                "model": model_key,
                "raw_seg_dir": str(seg_dir),
                "processed_seg_dir": str(seg_dir),
                "fallback_used": True,
                "safe_targets": [],
            }
            continue
        copied = _prepare_candidate_shapekit_input(seg_dir, input_root, case_id)
        if copied == 0:
            processed_dirs[model_key] = seg_dir
            reports[model_key] = {
                "stage": "candidate_preselection_shapekit",
                "status": "failed",
                "reason": "candidate model produced no masks for ShapeKit input",
                "model": model_key,
                "raw_seg_dir": str(seg_dir),
                "processed_seg_dir": str(seg_dir),
                "fallback_used": True,
                "safe_targets": safe_targets,
            }
            continue
        result = run_shapekit(
            shapekit_root,
            input_root,
            output_root,
            output_root / "logs",
            cpu_count=2,
            dry_run=False,
            auto_config=True,
            timeout_sec=min(timeout_sec, 900),
        )
        candidate_seg = output_root / case_id / "segmentations"
        has_processed_masks = candidate_seg.exists() and any(candidate_seg.glob("*.nii.gz"))
        use_processed = result.get("status") == "success" and has_processed_masks
        processed_dirs[model_key] = candidate_seg if use_processed else seg_dir
        reports[model_key] = {
            "stage": "candidate_preselection_shapekit",
            "status": "success" if use_processed else ("unsupported_target" if result.get("reason") == "No safe ShapeKit target organs detected" else "postprocess_failed"),
            "model": model_key,
            "raw_seg_dir": str(seg_dir),
            "processed_seg_dir": str(processed_dirs[model_key]),
            "fallback_used": not use_processed,
            "copied_masks": copied,
            "safe_targets": safe_targets,
            "result": result,
            "reason": result.get("reason"),
        }
    return processed_dirs, reports


def _add_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _file_fingerprint(path: str | Path | None) -> tuple[str, int, int] | None:
    if not path:
        return None
    try:
        p = Path(path)
        st = p.stat()
        return (str(p.resolve()), int(st.st_size), int(st.st_mtime_ns))
    except Exception:
        return None


def _same_file_fingerprint(src: Path, dst: Path) -> bool:
    src_fp = _file_fingerprint(src)
    dst_fp = _file_fingerprint(dst)
    return bool(src_fp and dst_fp and src_fp[1:] == dst_fp[1:])


class _CaseMaskCache:
    """Small case-scoped cache for NIfTI QC/Dice hot paths."""

    def __init__(self, max_arrays: int = 96) -> None:
        self.max_arrays = max(0, int(max_arrays))
        self._binary_arrays: dict[tuple[str, int, int], Any] = {}
        self._images: dict[tuple[str, int, int], Any] = {}
        self._qc: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._dice: dict[tuple[Any, Any], float | None] = {}
        self._lock = threading.RLock()

    def _evict_if_needed(self) -> None:
        while self.max_arrays and len(self._binary_arrays) > self.max_arrays:
            first = next(iter(self._binary_arrays))
            self._binary_arrays.pop(first, None)
            self._images.pop(first, None)

    def image(self, path: str | Path):
        fp = _file_fingerprint(path)
        if fp is None:
            return None
        with self._lock:
            if fp not in self._images:
                import nibabel as nib

                self._images[fp] = nib.load(str(path))
            return self._images[fp]

    def binary(self, path: str | Path):
        fp = _file_fingerprint(path)
        if fp is None:
            return None
        with self._lock:
            if fp in self._binary_arrays:
                return self._binary_arrays[fp]
        import numpy as np

        img = self.image(path)
        if img is None:
            return None
        arr = np.asanyarray(img.dataobj) > 0
        with self._lock:
            self._binary_arrays[fp] = arr
            self._evict_if_needed()
            return self._binary_arrays.get(fp, arr)

    def get_qc(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        with self._lock:
            cached = self._qc.get(key)
            return dict(cached) if cached is not None else None

    def set_qc(self, key: tuple[Any, ...], value: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._qc[key] = dict(value)
        return value

    def get_dice(self, key: tuple[Any, Any]) -> float | None | str:
        with self._lock:
            return self._dice[key] if key in self._dice else "__missing__"

    def set_dice(self, key: tuple[Any, Any], value: float | None) -> float | None:
        with self._lock:
            self._dice[key] = value
        return value


def _mask_dice_3d(a_path: str | Path, b_path: str | Path, cache: _CaseMaskCache | None = None) -> float | None:
    """3D binary Dice between two mask files, or None if unreadable/mismatched.

    Used as a correct pre-check before LabelCritic: LabelCritic's own 2D dice gate
    compares CT-window background images (background-dominated -> ~1.0 for any
    pair), so it skips every comparison. We gate on the real 3D mask overlap
    instead so near-identical candidates are skipped while genuine disagreements
    actually reach the VLM.
    """
    try:
        import nibabel as nib
        import numpy as np

        a_fp = _file_fingerprint(a_path)
        b_fp = _file_fingerprint(b_path)
        dice_key = tuple(sorted([a_fp, b_fp], key=lambda x: str(x))) if a_fp and b_fp else None
        if cache is not None and dice_key is not None:
            cached = cache.get_dice(dice_key)
            if cached != "__missing__":
                return cached  # type: ignore[return-value]
        a = cache.binary(a_path) if cache is not None else np.asanyarray(nib.load(str(a_path)).dataobj) > 0
        b = cache.binary(b_path) if cache is not None else np.asanyarray(nib.load(str(b_path)).dataobj) > 0
        if a is None or b is None:
            return cache.set_dice(dice_key, None) if cache is not None and dice_key is not None else None
        if a.shape != b.shape:
            return cache.set_dice(dice_key, None) if cache is not None and dice_key is not None else None
        total = int(a.sum()) + int(b.sum())
        if total == 0:
            value = 1.0
        else:
            value = float(2 * int(np.logical_and(a, b).sum()) / total)
        return cache.set_dice(dice_key, value) if cache is not None and dice_key is not None else value
    except Exception:
        return None


def _geometric_teacher_consensus_selection(
    candidates: list[dict[str, Any]],
    *,
    near_identical_dice: float,
    mask_cache: _CaseMaskCache | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Family-free complete-link teacher consensus selection.

    This is intentionally based only on original teacher masks, model keys,
    geometry/QC eligibility already enforced upstream, and the complete 3D Dice
    matrix. ``evidence_family`` is retained in the returned audit metadata but
    never participates in cluster membership, winner selection, or training
    eligibility.
    """
    excluded: list[dict[str, Any]] = []
    real_teachers: list[dict[str, Any]] = []
    non_teacher_families = {"voxtell_student", "prior_pseudo_label"}
    for candidate in candidates:
        model = str(candidate.get("model") or candidate.get("model_key") or "")
        family = str(candidate.get("evidence_family") or "")
        if (
            candidate.get("is_fusion")
            or family in non_teacher_families
            or model in {"fusion_consensus", "round_prev_selected", "previous_round_selected"}
            or "student" in model.lower()
        ):
            excluded.append({
                "model": model,
                "reason": "not_original_teacher_for_geometric_consensus",
                "evidence_family": family or None,
            })
            continue
        if not candidate.get("prediction") or not Path(str(candidate["prediction"])).exists():
            excluded.append({"model": model, "reason": "missing_prediction"})
            continue
        real_teachers.append(candidate)

    model_keys = {str(c.get("model") or c.get("model_key") or "") for c in real_teachers}
    base: dict[str, Any] = {
        "primary_selector": "complete_link_geometric_teacher_consensus",
        "geometric_consensus_threshold": float(near_identical_dice),
        "geometric_consensus_policy": "family_free_original_teacher_complete_link",
        "family_role": "audit_only_not_used_for_selection_or_training_gate",
        "geometric_consensus_excluded_candidates": excluded,
    }
    if len(model_keys) < 2 or len(real_teachers) < 2:
        return None, {**base, "geometric_consensus_status": "insufficient_teacher_models"}

    n = len(real_teachers)
    dice_by_pair: dict[tuple[int, int], float | None] = {}
    matrix: list[dict[str, Any]] = []
    geometry_failed = False
    for i in range(n):
        for j in range(i + 1, n):
            d3 = _mask_dice_3d(
                real_teachers[i]["prediction"],
                real_teachers[j]["prediction"],
                cache=mask_cache,
            )
            dice_by_pair[(i, j)] = d3
            if d3 is None:
                geometry_failed = True
            matrix.append({
                "candidate_a": real_teachers[i].get("model"),
                "candidate_b": real_teachers[j].get("model"),
                "dice_3d": round(float(d3), 6) if d3 is not None else None,
                "passes_threshold": bool(d3 is not None and d3 >= near_identical_dice),
            })
    if geometry_failed:
        return None, {
            **base,
            "geometric_consensus_status": "abstain_geometry_unreadable_or_mismatch",
            "geometric_pairwise_dice": matrix,
        }

    adjacency = {i: set() for i in range(n)}
    for (i, j), d3 in dice_by_pair.items():
        if d3 is not None and d3 >= near_identical_dice:
            adjacency[i].add(j)
            adjacency[j].add(i)

    # Bron-Kerbosch maximal clique enumeration is compact and gives exact
    # complete-link clusters for the small candidate sets used here.
    cliques: list[set[int]] = []

    def bronk(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            cliques.append(set(r))
            return
        pivot = next(iter(p | x), None)
        search = set(p - (adjacency[pivot] if pivot is not None else set()))
        for vertex in sorted(search, key=lambda idx: str(real_teachers[idx].get("model") or "")):
            bronk(r | {vertex}, p & adjacency[vertex], x & adjacency[vertex])
            p.remove(vertex)
            x.add(vertex)

    bronk(set(), set(range(n)), set())
    eligible_cliques = [
        clique for clique in cliques
        if len({str(real_teachers[idx].get("model") or "") for idx in clique}) >= 2
    ]
    if not eligible_cliques:
        return None, {
            **base,
            "geometric_consensus_status": "abstain_no_complete_link_cluster",
            "geometric_pairwise_dice": matrix,
        }
    max_size = max(len(clique) for clique in eligible_cliques)
    largest = [clique for clique in eligible_cliques if len(clique) == max_size]
    cluster_summaries = [
        {
            "models": sorted(str(real_teachers[idx].get("model") or "") for idx in clique),
            "size": len(clique),
        }
        for clique in eligible_cliques
    ]
    if len(largest) != 1:
        return None, {
            **base,
            "geometric_consensus_status": "abstain_tied_largest_clusters",
            "geometric_pairwise_dice": matrix,
            "geometric_consensus_clusters": cluster_summaries,
        }

    cluster = largest[0]
    medoid_scores: list[tuple[float, str, str, dict[str, Any]]] = []
    for idx in cluster:
        others = [other for other in cluster if other != idx]
        avg = sum(float(dice_by_pair[tuple(sorted((idx, other)))]) for other in others) / max(1, len(others))
        medoid_scores.append((
            avg,
            str(real_teachers[idx].get("model") or ""),
            str(Path(real_teachers[idx].get("prediction") or "").resolve()),
            real_teachers[idx],
        ))
    # Higher mean Dice wins; ties are deterministic by model/path.
    winner = sorted(medoid_scores, key=lambda item: (-item[0], item[1], item[2]))[0][3]
    return winner, {
        **base,
        "geometric_consensus_status": "selected",
        "geometric_pairwise_dice": matrix,
        "geometric_consensus_clusters": cluster_summaries,
        "geometric_consensus_cluster_models": sorted(str(real_teachers[idx].get("model") or "") for idx in cluster),
        "geometric_consensus_cluster_size": len(cluster),
        "geometric_consensus_medoid_mean_dice": round(float(sorted(medoid_scores, key=lambda item: (-item[0], item[1], item[2]))[0][0]), 6),
    }


def _pick_reference_fallback(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Fallback selection when LabelCritic is unavailable or inconclusive.

    Conservative v1: prefer original teacher candidates. Fusion is allowed to
    compete, but it must not automatically win fallback selection.
    """
    primary_pool = [c for c in candidates if not c.get("is_fusion")] or candidates
    with_dice = [c for c in primary_pool if c.get("dice") is not None]
    if with_dice:
        return max(with_dice, key=lambda c: float(c.get("dice") or -1))
    return primary_pool[0]


def _candidate_id(case_id: str, organ: str, candidate: dict[str, Any]) -> str:
    payload = "|".join([
        str(case_id),
        str(organ),
        str(candidate.get("model") or ""),
        str(Path(candidate.get("prediction") or "").resolve()),
    ])
    return f"cand_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _candidate_prompt_context(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return only anonymous, objective mask evidence for the VLM adapter."""
    qc = candidate.get("candidate_qc") or {}
    return {
        "candidate_id": candidate.get("candidate_id"),
        "qc_status": candidate.get("candidate_qc_status"),
        "qc_flags": candidate.get("candidate_qc_flags", []),
        "foreground_voxel_count": qc.get("mask_voxels"),
        "volume_mm3": qc.get("mask_volume_mm3"),
        "bbox_voxel": qc.get("bbox_voxel"),
        "connected_components": qc.get("connected_components"),
        "boundary_contacts": qc.get("boundary_contacts"),
        "centroid_ras_mm": qc.get("centroid_ras_mm"),
        "centroid_laterality": qc.get("centroid_laterality"),
        "truncation_suspected": qc.get("truncation_suspected"),
        "fov_status": candidate.get("fov_status") or candidate.get("expected_presence"),
    }


def _sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def _mask_voxel_count(path: str | Path | None, cache: _CaseMaskCache | None = None) -> int | None:
    if not path:
        return None
    try:
        import numpy as np

        arr = cache.binary(path) if cache is not None else None
        if arr is None:
            import nibabel as nib
            arr = np.asanyarray(nib.load(str(path)).dataobj) > 0
        return int(arr.sum())
    except Exception:
        return None


def _verify_annotation_cached(
    current_annotation: str | Path | None,
    model_prediction: str | Path | None,
    organ: str,
    *,
    dsc_replace_threshold: float = 0.0,
    dsc_vlm_threshold: float = 0.5,
    dsc_accept_threshold: float = 0.8,
    cache: _CaseMaskCache | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": "label_verifier",
        "organ": organ,
        "metric_family": "pseudo_consistency",
        "metric_scope": "prediction_vs_prior_or_selected_pseudo_reference",
        "metric_target": "pseudo-label",
        "metric_subject": "E-step output",
        "metric_comparison": "candidate_vs_prior_or_selected_pseudo_reference",
        "metric_interpretation": "pseudo_label_consistency",
        "ground_truth_status": "pseudo_label_candidate",
        "accuracy_warning": "DSC is pseudo-label consistency unless the caller explicitly supplies expert fine labels.",
        "dsc_replace_threshold": dsc_replace_threshold,
        "dsc_vlm_threshold": dsc_vlm_threshold,
        "dsc_accept_threshold": dsc_accept_threshold,
        "cache_status": "case_mask_cache",
    }
    ann_path = Path(current_annotation).resolve() if current_annotation else None
    pred_path = Path(model_prediction).resolve() if model_prediction else None
    ann_exists = ann_path is not None and ann_path.exists()
    pred_exists = pred_path is not None and pred_path.exists()
    result["current_annotation"] = str(ann_path) if ann_path else None
    result["model_prediction"] = str(pred_path) if pred_path else None
    result["current_annotation_exists"] = ann_exists
    result["model_prediction_exists"] = pred_exists

    if not pred_exists and not ann_exists:
        result.update({"status": "failed", "decision": "review_queue", "dice": None, "quality_bucket": "both_missing", "reason": "Both pseudo reference and prediction are missing."})
        return result
    if not pred_exists:
        result.update({"status": "failed", "decision": "review_queue", "dice": None, "quality_bucket": "missing_prediction", "reason": "Model prediction missing; cannot verify."})
        return result
    if not ann_exists:
        result.update({"status": "warning", "decision": "auto_replace_candidate", "quality_bucket": "no_reference", "reason": "No prior pseudo reference; prediction becomes pseudo-label candidate.", "dice": None})
        return result

    dice = _mask_dice_3d(ann_path, pred_path, cache=cache)
    result["dice"] = round(float(dice), 6) if dice is not None else None
    ann_voxels = _mask_voxel_count(ann_path, cache=cache)
    pred_voxels = _mask_voxel_count(pred_path, cache=cache)
    result["current_voxels"] = ann_voxels
    result["prediction_voxels"] = pred_voxels

    if dice is None:
        result.update({"status": "failed", "decision": "review_queue", "quality_bucket": "dice_failed", "reason": "DSC computation failed; check read error or shape mismatch."})
        return result

    pred_nonempty = (pred_voxels or 0) > 10
    ann_empty = (ann_voxels or 0) <= 10
    ann_nonempty = not ann_empty
    dice_value = float(dice)
    if dice_value == 0.0 and pred_nonempty and ann_empty:
        result.update({"status": "warning", "decision": "auto_replace_candidate", "quality_bucket": "empty_reference_nonempty_prediction", "reason": "Prior pseudo reference is empty but prediction is non-empty."})
    elif dice_value == 0.0 and pred_nonempty and ann_nonempty:
        result.update({"status": "warning", "decision": "send_to_vlm_label_expert", "quality_bucket": "critical_low_dice", "reason": "DSC=0 and both masks are non-empty; requires LabelCritic/VLM comparison."})
    elif dice_value < dsc_vlm_threshold:
        result.update({"status": "warning", "decision": "send_to_vlm_label_expert", "quality_bucket": "low_dice", "reason": f"DSC={round(dice_value, 6)} < {dsc_vlm_threshold}; send to LabelCritic/VLM."})
    elif dice_value < dsc_accept_threshold:
        result.update({"status": "warning", "decision": "uncertain_manual_check", "quality_bucket": "moderate_dice", "reason": f"{dsc_vlm_threshold} <= DSC={round(dice_value, 6)} < {dsc_accept_threshold}; keep best candidate but queue for sanity check."})
    else:
        result.update({"status": "success", "decision": "accept", "quality_bucket": "high_dice", "reason": f"DSC={round(dice_value, 6)} >= {dsc_accept_threshold}; accept."})
    return result


def _evaluate_candidate_for_organ(
    *,
    ct: Path,
    organ: str,
    model_key: str,
    seg_dir: Path,
    raw_seg_dir: Path,
    alias_config: dict[str, Any],
    shapekit_report: dict[str, Any],
    current_ref: Path | None,
    current_ref_exists: bool,
    vlm_threshold: float,
    case_id: str,
    mask_cache: _CaseMaskCache,
    taxonomy: dict[str, Any],
    presence_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_presence = _expected_presence_for_organ(organ, presence_context)
    raw_pred, raw_alias_match = _candidate_mask_path(raw_seg_dir, organ, model_key, alias_config)
    pred, alias_match = _candidate_mask_path(seg_dir, organ, model_key, alias_config)
    candidate_shapekit_status = shapekit_report.get("status", "skipped_debug_only")
    candidate_shapekit_reason = shapekit_report.get("reason")
    if not pred.exists() and seg_dir != raw_seg_dir and raw_pred.exists():
        pred = raw_pred
        alias_match = f"post_shapekit_missing_fallback:{raw_alias_match}"
        candidate_shapekit_status = "fallback_original"
        candidate_shapekit_reason = "ShapeKit did not produce this organ mask; using raw candidate for LabelCritic comparison"
    identity = _candidate_identity_contract(
        taxonomy=taxonomy,
        alias_config=alias_config,
        organ=organ,
        model_key=model_key,
        alias_match=alias_match,
        seg_dir=raw_seg_dir,
    )
    pred_for_verify = pred if pred.exists() else None
    candidate_qc = _compute_candidate_qc(
        ct=ct,
        mask=pred_for_verify,
        organ=organ,
        reference=current_ref if current_ref_exists else None,
        cache=mask_cache,
    )
    high_risk_postprocess = any(
        token in _norm_organ_key(organ)
        for token in ("duct", "artery", "vein", "vessel", "cava", "lesion", "tumor")
    )
    if candidate_shapekit_status in {"postprocess_failed", "fallback_original", "failed"}:
        candidate_qc = dict(candidate_qc)
        qc_flags = list(candidate_qc.get("flags", []))
        qc_flags.append("postprocess_failed")
        if high_risk_postprocess:
            qc_flags.append("high_risk_postprocess_required")
            candidate_qc.update({
                "status": "fail",
                "score": 0.0,
                "eligible_for_labelcritic": False,
            })
        else:
            candidate_qc["score"] = min(float(candidate_qc.get("score", 1.0)), 0.5)
        candidate_qc["flags"] = sorted(set(qc_flags))
        candidate_qc["reason"] = ";".join(candidate_qc["flags"])
    v = _verify_annotation_cached(
        current_ref if current_ref_exists else None,
        pred_for_verify,
        organ,
        dsc_replace_threshold=0.0,
        dsc_vlm_threshold=vlm_threshold,
        cache=mask_cache,
    )
    dice = v.get("dice")
    result = {
        "case_id": case_id,
        "organ": organ,
        "model": model_key,
        "prediction": str(pred),
        "pre_shapekit_prediction": str(raw_pred),
        "reference": str(current_ref) if current_ref else "",
        "reference_role": "historical_pseudo_label" if current_ref_exists else "none",
        "reference_provenance": _historical_reference_provenance(current_ref) if current_ref_exists else None,
        "metric_family": "pseudo_consistency",
        "metric_scope": "candidate_vs_prior_or_selected_pseudo_reference",
        "metric_target": "pseudo-label",
        "metric_subject": "E-step output",
        "metric_comparison": "candidate_vs_prior_or_selected_pseudo_reference",
        "metric_interpretation": "pseudo_label_consistency",
        "ground_truth_status": "pseudo_label_candidate",
        "accuracy_warning": "Dice is pseudo-label consistency, not true expert-label accuracy.",
        "dice": dice,
        "pseudo_consistency_dice": dice,
        "decision": v.get("decision"),
        "status": v.get("status"),
        "reason": v.get("reason"),
        "reference_quality_bucket": v.get("quality_bucket"),
        "expected_presence": expected_presence,
        "candidate_exists": pred.exists(),
        "alias_match": alias_match,
        "candidate_shapekit_status": candidate_shapekit_status,
        "candidate_shapekit_reason": candidate_shapekit_reason,
        "candidate_shapekit_report": shapekit_report,
        "candidate_qc": candidate_qc,
        "candidate_qc_status": candidate_qc.get("status"),
        "candidate_qc_score": candidate_qc.get("score"),
        "candidate_qc_flags": candidate_qc.get("flags", []),
        "fov_status": _candidate_adjusted_fov_status(
            organ, presence_context, candidate_qc
        ),
        "eligible_for_labelcritic": candidate_qc.get("eligible_for_labelcritic", True) and identity["identity_status"] == "valid",
        **identity,
    }
    if str(model_key).lower().startswith("student") or "voxtell" in str(model_key).lower():
        for provenance_path in (seg_dir / "oof_provenance.json", seg_dir.parent / "oof_provenance.json"):
            if provenance_path.exists():
                try:
                    result["oof_provenance"] = json.loads(provenance_path.read_text(encoding="utf-8"))
                    result["out_of_fold"] = True
                except Exception as exc:
                    result["oof_provenance_error"] = str(exc)
                break
    if identity["identity_status"] != "valid":
        result["status"] = "failed"
        result["decision"] = "identity_mismatch"
        result["reason"] = "; ".join(identity["identity_mismatch_reasons"])
    if expected_presence == "expected_present" and "zero_volume_mask" in result.get("candidate_qc_flags", []):
        result["candidate_qc_flags"] = list(dict.fromkeys([*result.get("candidate_qc_flags", []), "expected_present_zero_volume_mask"]))
    return result


def _candidate_qc_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    qc = candidate.get("candidate_qc") or {}
    return {
        "model": candidate.get("model"),
        "prediction": candidate.get("prediction"),
        "candidate_qc_status": qc.get("status", candidate.get("candidate_qc_status")),
        "candidate_qc_score": qc.get("score", candidate.get("candidate_qc_score")),
        "candidate_qc_flags": qc.get("flags", candidate.get("candidate_qc_flags", [])),
    }


def _compute_candidate_qc(
    *,
    ct: Path,
    mask: Path | None,
    organ: str,
    reference: Path | None = None,
    cache: _CaseMaskCache | None = None,
) -> dict[str, Any]:
    """Run cheap structural QC after candidate ShapeKit and before LabelCritic."""
    qc_key = (
        _file_fingerprint(ct),
        _file_fingerprint(mask),
        organ,
        _file_fingerprint(reference),
    )
    if cache is not None:
        cached = cache.get_qc(qc_key)
        if cached is not None:
            return cached
    flags: list[str] = []
    checks: dict[str, Any] = {
        "organ": organ,
        "ct_path": str(ct),
        "mask_path": str(mask) if mask else None,
        "reference_path": str(reference) if reference else None,
    }
    if not mask or not mask.exists():
        return {
            **checks,
            "status": "fail",
            "score": 0.0,
            "eligible_for_labelcritic": False,
            "mask_availability": "missing_file",
            "flags": ["missing_file"],
            "reason": "candidate mask file is missing",
        }

    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:
        return {
            **checks,
            "status": "review",
            "score": 0.5,
            "eligible_for_labelcritic": True,
            "flags": ["candidate_qc_dependency_missing"],
            "reason": f"candidate QC dependency missing: {exc}",
        }

    try:
        mask_img = cache.image(mask) if cache is not None else nib.load(str(mask))
        mask_arr = cache.binary(mask) if cache is not None else np.asarray(mask_img.dataobj) > 0
        if mask_img is None or mask_arr is None:
            raise RuntimeError("mask could not be loaded")
    except Exception as exc:
        return {
            **checks,
            "status": "fail",
            "score": 0.0,
            "eligible_for_labelcritic": False,
            "mask_availability": "unreadable_mask",
            "flags": ["unreadable_mask"],
            "reason": f"candidate mask is unreadable: {exc}",
        }

    mask_shape = tuple(int(x) for x in mask_img.shape[:3])
    checks["mask_shape"] = list(mask_shape)
    ct_img = None
    try:
        ct_img = cache.image(ct) if cache is not None else nib.load(str(ct))
        if ct_img is None:
            raise RuntimeError("ct could not be loaded")
        ct_shape = tuple(int(x) for x in ct_img.shape[:3])
        checks["ct_shape"] = list(ct_shape)
        checks["mask_orientation"] = list(nib.aff2axcodes(mask_img.affine))
        checks["ct_orientation"] = list(nib.aff2axcodes(ct_img.affine))
        checks["mask_spacing"] = [float(x) for x in mask_img.header.get_zooms()[:3]]
        checks["ct_spacing"] = [float(x) for x in ct_img.header.get_zooms()[:3]]
        if mask_shape != ct_shape:
            flags.extend(["shape_mismatch_ct", "geometry_mismatch"])
        if not np.allclose(mask_img.affine, ct_img.affine, atol=1e-3):
            flags.extend(["affine_mismatch_ct", "geometry_mismatch"])
        if nib.aff2axcodes(mask_img.affine) != nib.aff2axcodes(ct_img.affine):
            flags.extend(["orientation_mismatch_ct", "geometry_mismatch"])
    except Exception as exc:
        checks["ct_geometry_status"] = f"unreadable:{exc}"
        flags.append("ct_geometry_unavailable")

    voxels = int(mask_arr.sum())
    checks["mask_voxels"] = voxels
    if voxels == 0:
        flags.append("zero_volume_mask")
        checks["mask_availability"] = "zero_volume_mask"
    else:
        checks["mask_availability"] = "nonzero_mask"
        coordinates = np.argwhere(mask_arr)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0)
        centroid_voxel = coordinates.mean(axis=0)
        centroid_ras = nib.affines.apply_affine(mask_img.affine, centroid_voxel)
        shape = np.asarray(mask_arr.shape, dtype=int)
        ct_center_voxel = (shape.astype(float) - 1.0) / 2.0
        ct_center_ras = nib.affines.apply_affine(ct_img.affine, ct_center_voxel)
        boundary_contacts = {
            "axis0_min": bool(lower[0] == 0),
            "axis0_max": bool(upper[0] == shape[0] - 1),
            "axis1_min": bool(lower[1] == 0),
            "axis1_max": bool(upper[1] == shape[1] - 1),
            "axis2_min": bool(lower[2] == 0),
            "axis2_max": bool(upper[2] == shape[2] - 1),
        }
        checks.update({
            "bbox_voxel": {
                "min": [int(x) for x in lower],
                "max": [int(x) for x in upper],
            },
            "centroid_voxel": [round(float(x), 4) for x in centroid_voxel],
            "centroid_ras_mm": [round(float(x), 4) for x in centroid_ras],
            "ct_center_ras_mm": [round(float(x), 4) for x in ct_center_ras],
            "centroid_laterality": (
                "left" if float(centroid_ras[0]) > float(ct_center_ras[0])
                else "right" if float(centroid_ras[0]) < float(ct_center_ras[0])
                else "midline"
            ),
            "boundary_contacts": boundary_contacts,
            "touches_volume_boundary": any(boundary_contacts.values()),
            "touches_inferior_superior_boundary": bool(
                boundary_contacts["axis2_min"] or boundary_contacts["axis2_max"]
            ),
            "truncation_suspected": bool(
                boundary_contacts["axis2_min"] or boundary_contacts["axis2_max"]
            ),
        })

    try:
        voxel_volume = float(abs(np.linalg.det(mask_img.affine[:3, :3])))
        if voxel_volume <= 0:
            voxel_volume = float(np.prod(mask_img.header.get_zooms()[:3]))
        checks["mask_volume_mm3"] = float(voxels * voxel_volume)
    except Exception:
        checks["mask_volume_mm3"] = None

    try:
        from scipy.ndimage import label as scipy_label

        _, component_count = scipy_label(mask_arr)
        checks["connected_components"] = int(component_count)
        if int(component_count) > 20:
            flags.append("many_connected_components")
    except Exception:
        checks["connected_components"] = None
        flags.append("connected_components_unavailable")

    if reference and reference.exists():
        try:
            ref_img = cache.image(reference) if cache is not None else nib.load(str(reference))
            ref_arr = cache.binary(reference) if cache is not None else np.asarray(ref_img.dataobj) > 0
            if ref_img is None or ref_arr is None:
                raise RuntimeError("reference could not be loaded")
            ref_voxels = int(ref_arr.sum())
            checks["reference_voxels"] = ref_voxels
            if ref_img.shape[:3] != mask_img.shape[:3]:
                checks["reference_geometry_status"] = "shape_mismatch"
                flags.append("reference_geometry_mismatch")
            elif not np.allclose(ref_img.affine, mask_img.affine, atol=1e-3):
                # PanTS contains a small number of masks with a bad header but
                # correct voxel-index layout (notably aorta).  Never mutate the
                # source; record the anomaly and evaluate index-aligned only.
                checks["reference_geometry_status"] = "affine_mismatch_index_aligned"
                flags.append("reference_geometry_mismatch")
            else:
                checks["reference_geometry_status"] = "pass"
            if ref_voxels > 0 and voxels > 0:
                ratio = float(voxels / ref_voxels)
                checks["volume_ratio_to_reference"] = ratio
                if ratio < 0.25:
                    flags.append("volume_ratio_too_small_vs_reference")
                elif ratio > 4.0:
                    flags.append("volume_ratio_too_large_vs_reference")
                intersection = int(np.logical_and(mask_arr, ref_arr).sum())
                dice = float(2 * intersection / max(1, voxels + ref_voxels))
                checks["reference_dice"] = dice
                if organ == "pancreatic_duct" and dice < 0.1:
                    flags.append("pancreatic_duct_dice_hard_fail")
                try:
                    from scipy.ndimage import binary_erosion, distance_transform_edt

                    mask_surface = np.logical_xor(mask_arr, binary_erosion(mask_arr))
                    ref_surface = np.logical_xor(ref_arr, binary_erosion(ref_arr))
                    spacing = tuple(float(x) for x in ct_img.header.get_zooms()[:3])
                    d_ref = distance_transform_edt(~ref_surface, sampling=spacing)[mask_surface]
                    d_mask = distance_transform_edt(~mask_surface, sampling=spacing)[ref_surface]
                    distances = np.concatenate([d_ref, d_mask])
                    hd95 = float(np.percentile(distances, 95)) if distances.size else 0.0
                    checks["hd95_mm"] = hd95
                    if organ == "pancreatic_duct" and hd95 > 20.0:
                        flags.append("pancreatic_duct_hd95_hard_fail")
                except Exception as exc:
                    checks["surface_metric_status"] = f"unavailable:{type(exc).__name__}"
        except Exception as exc:
            checks["reference_status"] = f"unreadable:{exc}"

    if organ == "pancreatic_duct":
        component_count = checks.get("connected_components")
        if component_count is not None and int(component_count) > 5:
            flags.append("pancreatic_duct_many_components")
        ratio = checks.get("volume_ratio_to_reference")
        if ratio is not None and float(ratio) > 5.0:
            flags.append("pancreatic_duct_extreme_volume_ratio")
        pancreas_path = mask.parent / "pancreas.nii.gz"
        if pancreas_path.exists() and voxels > 0:
            try:
                pancreas_img = nib.load(str(pancreas_path))
                pancreas = np.asanyarray(pancreas_img.dataobj) > 0
                if pancreas.shape == mask_arr.shape and np.allclose(pancreas_img.affine, mask_img.affine, atol=1e-3):
                    from scipy.ndimage import binary_dilation

                    near_pancreas = binary_dilation(pancreas, iterations=3)
                    containment = float(np.logical_and(mask_arr, near_pancreas).sum() / voxels)
                    checks["pancreas_neighbourhood_containment"] = containment
                    if containment < 0.5:
                        flags.append("pancreatic_duct_outside_pancreas")
            except Exception as exc:
                checks["pancreas_containment_status"] = f"unavailable:{type(exc).__name__}"

    hard_fail_flags = {
        "missing_file", "unreadable_mask", "shape_mismatch_ct",
        "affine_mismatch_ct", "orientation_mismatch_ct", "geometry_mismatch",
        "pancreatic_duct_dice_hard_fail", "pancreatic_duct_hd95_hard_fail",
        "pancreatic_duct_many_components", "pancreatic_duct_extreme_volume_ratio",
        "pancreatic_duct_outside_pancreas",
    }
    review_flags = {
        "affine_mismatch_ct",
        "geometry_mismatch",
        "ct_geometry_unavailable",
        "many_connected_components",
        "connected_components_unavailable",
        "volume_ratio_too_small_vs_reference",
        "volume_ratio_too_large_vs_reference",
        "zero_volume_mask",
    }
    if hard_fail_flags & set(flags):
        status = "fail"
        score = 0.0
        eligible = False
    elif review_flags & set(flags):
        status = "review"
        score = max(0.25, 1.0 - 0.15 * len(set(flags) & review_flags))
        eligible = True
    else:
        status = "pass"
        score = 1.0
        eligible = True

    result = {
        **checks,
        "status": status,
        "score": float(score),
        "eligible_for_labelcritic": eligible,
        "flags": flags,
        "reason": "ok" if not flags else ";".join(flags),
    }
    return cache.set_qc(qc_key, result) if cache is not None else result


def _labelcritic_decision_path(records: list[dict[str, Any]] | None) -> str | None:
    for record in records or []:
        if record.get("output_json"):
            return str(record["output_json"])
    return None


def _quality_status(review_flags: list[str] | None, quality_flags: list[str] | None) -> str:
    flags = set(review_flags or []) | set(quality_flags or [])
    if not flags:
        return "ok"
    if {"missing_candidate", "missing_final_mask"} & flags:
        return "missing"
    if any(str(flag).startswith("candidate_qc_") for flag in flags):
        return "candidate_qc_review"
    if any(str(flag).startswith("shapekit_") for flag in flags):
        return "postprocess_review"
    if "selection_fallback" in flags:
        return "selection_review"
    return "review"


def _distillation_gate_for_selected_label(meta: dict[str, Any]) -> dict[str, Any]:
    grade_raw = meta.get("grade")
    grade = str(grade_raw or "").upper()
    target_type = str(meta.get("target_type") or "hard").lower()
    training_weight_raw = meta.get("training_weight")
    training_weight = float(training_weight_raw if training_weight_raw is not None else 1.0)
    qc_status = str(meta.get("selected_candidate_qc_status") or "")
    flags = set(meta.get("review_flags") or []) | set(meta.get("quality_flags") or []) | set(meta.get("selected_candidate_qc_flags") or [])
    hard_exclusion_flags = {
        "auto_grade_reject",
        "missing_final_mask",
        "missing_candidate",
        "volume_ratio_too_large_vs_reference",
        "volume_ratio_too_small_vs_reference",
        "many_connected_components",
        "zero_volume_mask",
        "expected_present_zero_volume_mask",
        "geometry_mismatch",
        "shape_mismatch_ct",
        "affine_mismatch_ct",
        "orientation_mismatch_ct",
        "postprocess_failed",
        "high_risk_postprocess_required",
        "pancreatic_duct_dice_hard_fail",
        "pancreatic_duct_hd95_hard_fail",
        "pancreatic_duct_many_components",
        "pancreatic_duct_extreme_volume_ratio",
        "pancreatic_duct_outside_pancreas",
    }
    expected_presence = str(meta.get("expected_presence") or "")
    if expected_presence in {"unknown", "expected_absent"}:
        return {"eligible": False, "reason": f"fov_not_confirmed_present:{expected_presence}"}
    if grade == "D":
        return {"eligible": False, "reason": "grade_D_or_zero_weight"}
    if training_weight <= 0.0:
        return {"eligible": False, "reason": "training_weight_zero"}
    if qc_status == "fail":
        return {"eligible": False, "reason": "candidate_qc_fail"}
    if flags & hard_exclusion_flags:
        return {"eligible": False, "reason": "hard_quality_flag_excluded:" + ",".join(sorted(flags & hard_exclusion_flags))}
    if grade == "C" and target_type != "soft":
        return {"eligible": False, "reason": "grade_C_requires_soft_probability_target"}
    return {"eligible": True, "reason": None}


_KEY_ABDOMINAL_ORGANS = {
    "liver",
    "spleen",
    "pancreas",
    "kidney_left",
    "kidney_right",
    "aorta",
    "adrenal_gland_left",
    "adrenal_gland_right",
    "stomach",
    "duodenum",
    "colon",
    "small_bowel",
    "bladder",
    "gall_bladder",
}


def _expected_presence_for_organ(organ: str, presence_context: dict[str, Any] | None = None) -> str:
    status = _fov_status_for_organ(organ, presence_context)
    return {
        "fully_visible": "expected_present",
        "out_of_fov": "expected_absent",
        "partially_visible": "unknown",
        "unknown": "unknown",
    }[status]


def _fov_status_for_organ(organ: str, presence_context: dict[str, Any] | None = None) -> str:
    """Return the conservative four-state case-organ FOV contract.

    Only explicit absence or a trusted, mutually exclusive region observation
    may produce ``out_of_fov``.  A missing teacher or an empty mask is never
    consulted here.
    """
    norm = _norm_organ_key(organ)
    context = presence_context or {}
    confirmed_absent = {_norm_organ_key(item) for item in context.get("confirmed_absent_organs", []) or []}
    if norm in confirmed_absent:
        return "out_of_fov"
    has_abdomen = bool(context.get("has_abdomen_coverage"))
    has_pelvis = bool(context.get("has_pelvis_coverage"))
    has_thorax = bool(context.get("has_thorax_coverage"))
    partial_thorax = bool(context.get("has_partial_thorax_coverage"))
    has_head = bool(context.get("has_head_coverage"))
    has_extremity = bool(context.get("has_extremity_coverage"))
    has_any_region = any(
        bool(context.get(key))
        for key in (
            "has_abdomen_coverage", "has_pelvis_coverage",
            "has_thorax_coverage", "has_partial_thorax_coverage",
            "has_head_coverage", "has_extremity_coverage",
        )
    )
    if not context.get("has_region_evidence") and not has_any_region:
        return "unknown"
    if norm == "esophagus":
        if has_thorax:
            return "fully_visible"
        return "partially_visible" if has_abdomen or partial_thorax else "unknown"
    if norm == "superior_vena_cava":
        if has_thorax:
            return "fully_visible"
        return "partially_visible" if partial_thorax else (
            "out_of_fov" if has_abdomen else "unknown"
        )
    if norm in {"hip_left", "hip_right"} and not has_pelvis:
        return "unknown"
    if norm in _KEY_ABDOMINAL_ORGANS:
        if norm == "bladder":
            if has_pelvis:
                return "fully_visible"
            return "unknown"
        if has_abdomen:
            return "fully_visible"
        if has_pelvis:
            return "partially_visible"
        return "unknown"
    hinted_region = _organ_region_hint(norm)
    appearance_regions = (
        set(_appearance_region_index().get(norm, ()))
        if hinted_region is None else set()
    )
    if "abdomen" in appearance_regions and has_abdomen:
        return "fully_visible"
    if "pelvis" in appearance_regions and has_pelvis:
        return "fully_visible"
    if "head_neck" in appearance_regions:
        return "fully_visible" if has_head else (
            "out_of_fov" if has_abdomen and not has_pelvis else "unknown"
        )
    if "thorax" in appearance_regions:
        if has_thorax:
            return "fully_visible"
        if partial_thorax:
            return "partially_visible"
        return "out_of_fov" if has_abdomen and not has_pelvis else "unknown"
    if "extremity" in appearance_regions:
        return "fully_visible" if has_extremity else (
            "out_of_fov" if has_abdomen and not has_pelvis else "unknown"
        )
    if hinted_region == "head_neck":
        return "fully_visible" if has_head else (
            "out_of_fov" if has_abdomen and not has_thorax and not has_pelvis else "unknown"
        )
    if hinted_region == "thorax":
        if has_thorax:
            return "fully_visible"
        if partial_thorax:
            return "partially_visible"
        return "unknown"
    if hinted_region == "abdomen":
        return "fully_visible" if has_abdomen else "unknown"
    if hinted_region == "pelvis":
        return "fully_visible" if has_pelvis else "unknown"
    if hinted_region == "extremity":
        return "fully_visible" if has_extremity else (
            "out_of_fov" if has_abdomen and not has_pelvis else "unknown"
        )
    region, _ = _infer_region_and_landmarks(norm)
    if region == "head and craniofacial region" or region == "neck and upper aerodigestive tract":
        return "fully_visible" if has_head else "unknown"
    if region == "thorax or upper mediastinum":
        return "fully_visible" if has_thorax else ("partially_visible" if partial_thorax else "unknown")
    if region == "abdomen and retroperitoneum":
        return "fully_visible" if has_abdomen else "unknown"
    if region == "pelvis and lower abdomen":
        return "fully_visible" if has_pelvis else "unknown"
    if region == "appendicular skeleton or extremity field of view":
        return "fully_visible" if has_extremity else "unknown"
    return "unknown"


def _candidate_adjusted_fov_status(
    organ: str,
    presence_context: dict[str, Any] | None,
    candidate_qc: dict[str, Any] | None,
) -> str:
    """Combine case coverage with candidate truncation evidence conservatively."""
    base = _fov_status_for_organ(organ, presence_context)
    qc = candidate_qc or {}
    if base != "out_of_fov" and qc.get("truncation_suspected"):
        return "partially_visible"
    return base


def _fov_pruned_organs(
    organs: list[str],
    taxonomy: dict[str, Any],
    presence_context: dict[str, Any],
) -> list[str]:
    """Keep organs supported by the detected FOV plus descendants/dependencies."""
    if not presence_context.get("has_region_evidence"):
        return list(organs)
    requested = set(organs)
    active = {
        organ for organ in organs
        if _expected_presence_for_organ(organ, presence_context) == "expected_present"
    }
    changed = True
    while changed:
        changed = False
        for organ in organs:
            parents = set((taxonomy_entry(taxonomy, organ) or {}).get("parent_ids", []) or [])
            if (
                organ not in active
                and parents & active
                and _expected_presence_for_organ(organ, presence_context) != "expected_absent"
            ):
                active.add(organ)
                changed = True
            for parent in parents:
                if organ in active and parent in requested and parent not in active:
                    active.add(parent)
                    changed = True
    return [organ for organ in organs if organ in active]


def _case_resume_state(
    *,
    case_out: Path,
    updated_root: Path,
    case_id: str,
    organs: list[str],
    enable_shapekit: bool,
) -> dict[str, Any]:
    pred_root = case_out / "raw_predictions"
    raw_ready = pred_root.exists() and (
        any(pred_root.glob(f"*/{case_id}/segmentations/*.nii.gz"))
        or any(pred_root.glob(f"*/{case_id}/inference_summary.json"))
        or any(pred_root.glob(f"hierarchical_full/*/{case_id}/segmentations/*.nii.gz"))
        or any(pred_root.glob(f"hierarchical_full/*/{case_id}/inference_summary.json"))
    )
    # Parent-cache-only hierarchical repair intentionally creates no new
    # raw/full-volume predictions. Its authoritative raw-stage evidence is the
    # hierarchical plan plus restored per-model predictions.
    if (
        (case_out / "hierarchical_inference_plan.json").exists()
        and any((case_out / "hierarchical_predictions").glob("*/segmentations/*.nii.gz"))
    ):
        raw_ready = True
    meta_path = updated_root / case_id / "selection_metadata.json"
    updated_dir = updated_root / case_id / "updated"
    if not raw_ready:
        return {"complete": False, "raw_ready": False, "reason": "raw_predictions_missing"}
    if not meta_path.exists():
        return {"complete": False, "raw_ready": True, "reason": "selection_metadata_missing"}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"complete": False, "raw_ready": True, "reason": f"selection_metadata_unreadable:{exc}"}
    if meta.get("quality_contract_version") != QUALITY_CONTRACT_VERSION:
        return {"complete": False, "raw_ready": True, "reason": "quality_contract_version_mismatch"}
    if meta.get("fov_policy_version") != FOV_POLICY_VERSION:
        return {"complete": False, "raw_ready": True, "reason": "fov_policy_version_mismatch"}
    selection_rows = meta.get("selection_rows") or []
    selected_organs = meta.get("selected_organs") or []
    seen = {str(row.get("organ")) for row in selection_rows if isinstance(row, dict) and row.get("organ")}
    missing_selection_rows = [organ for organ in organs if organ not in seen]
    if missing_selection_rows:
        return {
            "complete": False,
            "raw_ready": True,
            "reason": "selection_rows_incomplete",
            "missing_selection_rows": missing_selection_rows[:50],
        }
    pending = [
        item.get("organ")
        for item in selected_organs
        if isinstance(item, dict) and item.get("shapekit_status") == "pending"
    ]
    if pending:
        return {"complete": False, "raw_ready": True, "reason": "shapekit_metadata_pending", "pending_organs": pending[:50]}
    missing_final = [
        item.get("organ")
        for item in selected_organs
        if isinstance(item, dict)
        and item.get("organ")
        and item.get("publication_status") != "rejected_but_recorded"
        and not (updated_dir / f"{item['organ']}.nii.gz").exists()
    ]
    if missing_final:
        return {"complete": False, "raw_ready": True, "reason": "updated_masks_missing", "missing_final_masks": missing_final[:50]}
    try:
        import nibabel as nib
        import numpy as np

        ct_image = nib.load(str(meta.get("ct_path")))
        geometry_errors = []
        for item in selected_organs:
            if not isinstance(item, dict) or item.get("publication_status") == "rejected_but_recorded":
                continue
            organ = str(item.get("organ") or "")
            final_path = updated_dir / f"{organ}.nii.gz"
            if not final_path.exists():
                continue
            mask_image = nib.load(str(final_path))
            if mask_image.shape[:3] != ct_image.shape[:3] or not np.allclose(mask_image.affine, ct_image.affine, atol=1e-3):
                geometry_errors.append(organ)
        if geometry_errors:
            return {
                "complete": False,
                "raw_ready": True,
                "reason": "final_mask_geometry_mismatch",
                "organs": geometry_errors[:50],
            }
    except Exception as exc:
        return {"complete": False, "raw_ready": True, "reason": f"final_geometry_validation_failed:{exc}"}
    if enable_shapekit:
        unknown_postprocess = [
            item.get("organ")
            for item in selected_organs
            if isinstance(item, dict)
            and item.get("shapekit_status") in {None, "", "pending"}
        ]
        if unknown_postprocess:
            return {"complete": False, "raw_ready": True, "reason": "shapekit_status_incomplete", "organs": unknown_postprocess[:50]}
    return {"complete": True, "raw_ready": True, "reason": "complete"}


def _load_reusable_raw_inference_cache(
    *,
    cached_summary: Path,
    cached_seg_dir: Path,
    model_key: str,
) -> dict[str, Any] | None:
    if not cached_summary.exists() or not cached_seg_dir.exists():
        return None
    try:
        infer = json.loads(cached_summary.read_text(encoding="utf-8"))
    except Exception:
        return None
    if infer.get("timed_out") or infer.get("status") == "timed_out":
        return None
    cached_num_masks = sum(1 for _ in cached_seg_dir.glob("*.nii.gz"))
    summary_num_masks = infer.get("num_masks")
    executed_without_masks = (
        cached_num_masks == 0
        and str(infer.get("status")) in {"failed", "success"}
        and infer.get("return_code") == 0
        and int(summary_num_masks or 0) == 0
    )
    if cached_num_masks <= 0 and not executed_without_masks:
        return None
    infer.update({
        "status": infer.get("status", "success"),
        "model_key": model_key,
        "segmentation_output": str(cached_seg_dir),
        "num_masks": cached_num_masks,
        "cache_status": "reused_empty_raw_prediction" if cached_num_masks == 0 else "reused_raw_prediction",
    })
    return infer


def _build_gap_rows(
    *,
    case_id: str,
    ct: Path,
    organs: list[str],
    selection_rows: list[dict[str, Any]],
    selected_metadata: list[dict[str, Any]],
    presence_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    coverage_summary = _presence_context_summary(presence_context)

    def _gap_context(organ: str, selection: dict[str, Any] | None = None) -> dict[str, Any]:
        selection = selection or {}
        expected_presence = str(
            selection.get("expected_presence")
            or _expected_presence_for_organ(organ, presence_context)
            or "unknown"
        )
        region, landmarks = _infer_region_and_landmarks(organ)
        if expected_presence == "expected_absent":
            severity = "informational_out_of_fov"
        elif expected_presence == "unknown":
            severity = "review_fov_unknown"
        else:
            severity = "action_required_expected_present"
        return {
            "expected_presence": expected_presence,
            "fov_status": selection.get("fov_status") or _fov_status_for_organ(organ, presence_context),
            "anatomic_region": region,
            "region_landmarks": landmarks,
            "gap_severity": severity,
            "coverage_summary": coverage_summary,
        }

    def _missing_reason(selection: dict[str, Any]) -> str:
        expected_presence = str(selection.get("expected_presence") or "unknown")
        selected_qc_status = str(selection.get("selected_candidate_qc_status") or "")
        selected_qc_flags = set(selection.get("selected_candidate_qc_flags", []) or [])
        selection_status = str(selection.get("selection_status") or "missing")
        if expected_presence == "expected_absent":
            return "organ_expected_absent_or_out_of_fov"
        if expected_presence == "unknown":
            return "organ_fov_unknown_and_no_accepted_label"
        if {"missing_file", "unreadable_mask"} & selected_qc_flags:
            return "model_or_route_failed_to_produce_usable_mask"
        if "zero_volume_mask" in selected_qc_flags:
            return "expected_present_but_zero_volume_candidate"
        if selection_status != "selected":
            return "selection_inconclusive_or_rejected"
        if selected_qc_status and selected_qc_status != "pass":
            return "candidate_qc_not_pass"
        return "expected_present_missing_final_label"

    selected_by_organ = {str(item.get("organ")): item for item in selected_metadata if item.get("organ")}
    selection_by_organ = {str(item.get("organ")): item for item in selection_rows if item.get("organ")}
    rows: list[dict[str, Any]] = []
    for organ in organs:
        selected = selected_by_organ.get(organ)
        selection = selection_by_organ.get(organ, {})
        if not selected:
            gap_context = _gap_context(organ, selection)
            rows.append({
                "case_id": case_id,
                "ct_path": str(ct),
                "organ": organ,
                "gap_type": "missing_final_pseudo_label",
                "reason": selection.get("reason") or _missing_reason(selection),
                **gap_context,
                "candidate_count": selection.get("candidate_count", 0),
                "candidate_models": selection.get("candidate_models", []),
                "selection_method": selection.get("selection_method", "none"),
                "selection_status": selection.get("selection_status", "missing"),
                "dataset_type": "pseudo_label_dataset",
                "ground_truth_status": "pseudo_label_candidate",
            })
            continue
        status = selected.get("shapekit_status")
        if status in {"unsupported_target", "fallback_original", "postprocess_failed", "failed"}:
            mask_path = selected.get("mask_path") or selected.get("final_mask") or selected.get("selected_prediction")
            original_usable = bool(mask_path and Path(str(mask_path)).exists())
            if status in {"fallback_original", "postprocess_failed"}:
                gap_type = "shapekit_failed_but_original_usable" if original_usable else "shapekit_failed_and_no_usable_mask"
            elif status == "unsupported_target":
                gap_type = "shapekit_unsupported_but_original_usable" if original_usable else "shapekit_unsupported_and_no_usable_mask"
            else:
                gap_type = "shapekit_failed_but_original_usable" if original_usable else "shapekit_failed_and_no_usable_mask"
            gap_context = _gap_context(organ, selection)
            rows.append({
                "case_id": case_id,
                "ct_path": str(ct),
                "organ": organ,
                "gap_type": gap_type,
                "reason": selected.get("shapekit_reason") or status,
                **gap_context,
                "candidate_count": selected.get("candidate_count"),
                "candidate_models": selected.get("candidate_models", []),
                "selection_method": selected.get("selection_method"),
                "selection_status": selected.get("selection_status"),
                "shapekit_status": status,
                "original_mask_usable": original_usable,
                "selected_mask": str(mask_path or ""),
                "dataset_type": "pseudo_label_dataset",
                "ground_truth_status": "pseudo_label_candidate",
            })
        qc_status = selected.get("selected_candidate_qc_status")
        if qc_status and qc_status != "pass":
            gap_context = _gap_context(organ, selection)
            rows.append({
                "case_id": case_id,
                "ct_path": str(ct),
                "organ": organ,
                "gap_type": "candidate_qc_not_pass",
                "reason": ";".join(selected.get("selected_candidate_qc_flags", []) or []) or qc_status,
                **gap_context,
                "candidate_count": selected.get("candidate_count"),
                "candidate_models": selected.get("candidate_models", []),
                "comparison_candidate_models": selection.get("comparison_candidate_models", []),
                "selection_method": selected.get("selection_method"),
                "selection_status": selected.get("selection_status"),
                "candidate_qc_status": qc_status,
                "candidate_qc_flags": selected.get("selected_candidate_qc_flags", []),
                "dataset_type": "pseudo_label_dataset",
                "ground_truth_status": "pseudo_label_candidate",
            })
    return rows


def _summarize_gap_rows(gap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_fov_status: dict[str, int] = {}
    expected_present_actionable: list[dict[str, Any]] = []
    for row in gap_rows:
        reason = str(row.get("reason") or "unknown")
        gap_type = str(row.get("gap_type") or "unknown")
        severity = str(row.get("gap_severity") or "unknown")
        fov_status = str(row.get("fov_status") or row.get("expected_presence") or "unknown")
        by_reason[reason] = by_reason.get(reason, 0) + 1
        by_type[gap_type] = by_type.get(gap_type, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_fov_status[fov_status] = by_fov_status.get(fov_status, 0) + 1
        if severity == "action_required_expected_present":
            expected_present_actionable.append({
                "organ": row.get("organ"),
                "reason": reason,
                "gap_type": gap_type,
                "candidate_models": row.get("candidate_models", []),
                "candidate_count": row.get("candidate_count", 0),
            })
    return {
        "total_gap_rows": len(gap_rows),
        "by_reason": dict(sorted(by_reason.items(), key=lambda item: (-item[1], item[0]))),
        "by_gap_type": dict(sorted(by_type.items(), key=lambda item: (-item[1], item[0]))),
        "by_gap_severity": dict(sorted(by_severity.items(), key=lambda item: (-item[1], item[0]))),
        "by_fov_status": dict(sorted(by_fov_status.items(), key=lambda item: (-item[1], item[0]))),
        "num_action_required_expected_present": len(expected_present_actionable),
        "action_required_examples": expected_present_actionable[:50],
    }


def _summarize_case_quality(
    *,
    selection_rows: list[dict[str, Any]],
    selected_metadata: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grade_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    exclusion_counts: dict[str, int] = {}
    d_selected_or_fallback: list[dict[str, Any]] = []
    for row in selected_metadata:
        grade = str(row.get("grade") or "unknown").upper()
        status = str(row.get("selection_status") or "unknown")
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        if row.get("distillation_eligible") is False:
            reason = str(row.get("distillation_exclusion_reason") or "unknown")
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
        if grade == "D" and status in {"selected", "fallback"}:
            d_selected_or_fallback.append({
                "organ": row.get("organ"),
                "selection_status": status,
                "selected_model": row.get("selected_model") or row.get("source_model"),
                "reason": row.get("distillation_exclusion_reason") or row.get("selection_reason"),
                "selected_pseudo_consistency_dice": row.get("selected_pseudo_consistency_dice"),
                "selected_candidate_qc_flags": row.get("selected_candidate_qc_flags", []),
            })
    return {
        "num_selection_rows": len(selection_rows),
        "num_selected_organs": len(selected_metadata),
        "grade_counts": dict(sorted(grade_counts.items())),
        "selection_status_counts": dict(sorted(status_counts.items())),
        "distillation_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "num_distillation_eligible": sum(1 for row in selected_metadata if row.get("distillation_eligible") is True),
        "num_d_selected_or_fallback": len(d_selected_or_fallback),
        "d_selected_or_fallback_examples": d_selected_or_fallback[:50],
        "gap_summary": _summarize_gap_rows(gap_rows),
    }


def _summarize_preseeded_competition(
    *,
    preseeded_keys: list[str],
    selection_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Summarize whether Round2+ injected sources actually joined selection."""
    if not preseeded_keys:
        return None

    per_source: dict[str, dict[str, Any]] = {}
    for key in preseeded_keys:
        candidate_entries = [
            row for row in selection_rows
            if key in (row.get("candidate_models") or [])
        ]
        missing_entries = [
            row for row in selection_rows
            if key not in (row.get("candidate_models") or [])
        ]
        selected_entries = [
            row for row in selection_rows
            if row.get("selected_model") == key
        ]
        per_source[key] = {
            "candidate_entries": len(candidate_entries),
            "missing_entries": len(missing_entries),
            "selected_entries": len(selected_entries),
            "missing_examples": [
                {
                    "case_id": row.get("case_id"),
                    "organ": row.get("organ"),
                    "candidate_models": row.get("candidate_models", []),
                }
                for row in missing_entries[:20]
            ],
        }

    return {
        "status": "computed",
        "preseeded_sources": preseeded_keys,
        "selection_entries": len(selection_rows),
        "per_source": per_source,
        "note": (
            "Round2+ preseeded sources are candidates, not automatic winners. "
            "Missing entries usually mean that source had no mask for that "
            "case-organ layout and should be reviewed before formal claims."
        ),
    }


def _rebuild_selection_rows_from_artifacts(updated_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(updated_root.glob("*/selection_metadata.json")):
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_id = str(doc.get("case_id") or meta_path.parent.name)
        for row in doc.get("selection_rows", []) or []:
            if isinstance(row, dict):
                rows.append({"case_id": case_id, **row})
    return rows


def _rebuild_selected_metadata_from_artifacts(updated_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(updated_root.glob("*/selection_metadata.json")):
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_id = str(doc.get("case_id") or meta_path.parent.name)
        for row in doc.get("selected_organs", []) or []:
            if isinstance(row, dict):
                rows.append({"case_id": case_id, **row})
    return rows


def _build_formal_organ_audit_rows(
    *,
    selection_rows: list[dict[str, Any]],
    selected_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_key = {
        (str(row.get("case_id") or ""), str(row.get("organ") or "")): row
        for row in selected_metadata
        if row.get("case_id") and row.get("organ")
    }
    rows: list[dict[str, Any]] = []
    for selection in selection_rows:
        key = (str(selection.get("case_id") or ""), str(selection.get("organ") or ""))
        selected = selected_by_key.get(key, {})
        candidate_models = selection.get("candidate_models") or []
        family_membership = selection.get("family_membership") or {}
        rows.append({
            "case_id": selection.get("case_id"),
            "organ": selection.get("organ"),
            "fov_status": selection.get("fov_status", selected.get("fov_status", "unknown")),
            "route_primary_teacher": selection.get("route_primary_teacher"),
            "route_backup_teachers": selection.get("route_backup_teachers", []),
            "route_competition_teachers": selection.get("route_competition_teachers", []),
            "route_confidence": selection.get("route_confidence"),
            "candidate_count": len(candidate_models),
            "candidate_models": candidate_models,
            "independent_family_count": selection.get("independent_family_count", selected.get("independent_family_count", 0)),
            "family_membership": family_membership,
            "selected_model": selection.get("selected_model"),
            "selection_method": selection.get("selection_method"),
            "selection_status": selection.get("selection_status"),
            "candidate_qc_status": selection.get("selected_candidate_qc_status"),
            "candidate_qc_flags": selection.get("selected_candidate_qc_flags", []),
            "labelcritic_compare_used": selection.get("labelcritic_compare_used"),
            "labelcritic_grade_used": selection.get("labelcritic_grade_used"),
            "labelcritic_supported": selection.get("labelcritic_supported"),
            "labelcritic_tiebreak_adjustment": selection.get("labelcritic_tiebreak_adjustment"),
            "auto_grade": selection.get("auto_grade"),
            "evidence_confidence": selection.get("evidence_confidence"),
            "normalized_evidence_confidence": selection.get("normalized_evidence_confidence"),
            "winner_margin": selection.get("winner_margin"),
            "grade": selection.get("grade", selected.get("grade")),
            "training_weight": selection.get("training_weight", selected.get("training_weight")),
            "target_type": selection.get("target_type", selected.get("target_type")),
            "missing_evidence": selection.get("missing_evidence", []),
            "failed_evidence": selection.get("failed_evidence", []),
            "review_flags": selected.get("review_flags", selection.get("review_flags", [])),
            "quality_flags": selected.get("quality_flags", selection.get("quality_flags", [])),
            "missing_reason": (
                selection.get("reason")
                if not selected.get("final_mask")
                else None
            ),
            "final_mask": selected.get("final_mask"),
        })
    return rows




def _candidate_pairwise_agreement(
    candidates: list[dict[str, Any]],
    *,
    cache: _CaseMaskCache | None = None,
) -> dict[str, Any]:
    """Summarize 3D Dice agreement across candidate masks for gating work."""
    pairs: list[dict[str, Any]] = []
    dice_values: list[float] = []
    for i, cand_a in enumerate(candidates):
        for cand_b in candidates[i + 1:]:
            d3 = _mask_dice_3d(cand_a.get("prediction"), cand_b.get("prediction"), cache=cache)
            pairs.append({
                "candidate_a": cand_a.get("model"),
                "candidate_b": cand_b.get("model"),
                "dice_3d": round(float(d3), 4) if d3 is not None else None,
            })
            if d3 is not None:
                dice_values.append(float(d3))
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "min_dice_3d": min(dice_values) if dice_values else None,
        "mean_dice_3d": (sum(dice_values) / len(dice_values)) if dice_values else None,
    }


def _should_fuse_candidates(
    candidates: list[dict[str, Any]],
    *,
    cache: _CaseMaskCache | None = None,
    high_agreement_dice: float = 0.90,
) -> tuple[bool, dict[str, Any]]:
    """Conservative fusion gate.

    Fusion is allowed only for same-identity, same-family, QC-pass candidates
    with high 3D agreement. It is a consensus candidate, not an automatic winner.
    """
    if len(candidates) < 2:
        return False, {"reason": "fewer_than_two_candidates"}
    canonical_ids = {c.get("requested_canonical_id") for c in candidates}
    comparison_families = {c.get("comparison_family") for c in candidates}
    if any(c.get("identity_status") != "valid" for c in candidates) or len(canonical_ids) != 1 or len(comparison_families) != 1:
        return False, {"reason": "identity_mismatch_blocks_fusion", "canonical_ids": sorted(str(x) for x in canonical_ids)}
    if any(c.get("candidate_qc_status") not in (None, "pass") or c.get("candidate_qc_flags") for c in candidates):
        agreement = _candidate_pairwise_agreement(candidates, cache=cache)
        return False, {"reason": "candidate_qc_not_all_pass_blocks_fusion", **agreement}
    agreement = _candidate_pairwise_agreement(candidates, cache=cache)
    min_dice = agreement.get("min_dice_3d")
    if min_dice is not None and float(min_dice) >= high_agreement_dice:
        return True, {"reason": "high_candidate_agreement_allows_conservative_fusion", **agreement}
    return False, {"reason": "low_or_unknown_pairwise_agreement_blocks_fusion", **agreement}


def _official_pairwise_condorcet_selection(
    *,
    ct: Path,
    organ: str,
    candidates: list[dict[str, Any]],
    out: Path,
    case_id: str,
    critic_backend: str,
    critic_base_url: str,
    critic_port: int,
    timeout_sec: int,
    near_identical_dice: float,
    mask_cache: _CaseMaskCache | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run order-independent official pairwise comparisons and abstain on cycles."""
    from itertools import combinations

    status_path = Path(__file__).resolve().parents[4] / "configs" / "labelcritic_373_regression_status.json"
    regression_status = {}
    if status_path.is_file():
        try:
            regression_status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                .get("classes", {})
                .get(organ, {})
            )
        except Exception:
            regression_status = {}
    class_regression_failed = regression_status.get("status") == "failed"

    # Fused/consensus masks are audit artifacts, never LabelCritic competitors.
    # A winner must always trace to one original teacher mask.
    fusion_candidates = [row for row in candidates if row.get("is_fusion")]
    teacher_candidates = [row for row in candidates if not row.get("is_fusion")]
    if not teacher_candidates:
        return None, {
            "selection_method": "label_critic_inconclusive",
            "selection_status": "review_required",
            "candidate_count": len(candidates),
            "comparison_candidate_count": 0,
            "comparison_candidate_models": [],
            "excluded_fusion_candidates": [
                str(row.get("model") or "") for row in fusion_candidates
            ],
            "critic_records": [],
            "labelcritic_records": [],
            "fallback_reason": "no_original_teacher_candidate",
            "primary_selector": "official_labelcritic_pairwise_condorcet",
            "should_enter_student_training": False,
            "quality_flags": ["no_original_teacher_candidate"],
            "review_flags": ["automatic_abstention"],
        }
    if len(teacher_candidates) == 1:
        selected = teacher_candidates[0]
        selected.setdefault(
            "candidate_id", _candidate_id(case_id, organ, selected)
        )
        return selected, {
            "selection_method": "single_teacher_provisional",
            "selection_status": "provisional",
            "candidate_count": len(candidates),
            "comparison_candidate_count": 1,
            "comparison_candidate_models": [str(selected.get("model") or "")],
            "comparison_decisive_count": 0,
            "comparison_inconclusive_count": 0,
            "comparison_agreed_count": 0,
            "selected_model": selected.get("model"),
            "selected_prediction": selected.get("prediction"),
            "critic_records": [],
            "labelcritic_records": [],
            "excluded_fusion_candidates": [
                str(row.get("model") or "") for row in fusion_candidates
            ],
            "primary_selector": "deterministic_single_teacher_qc",
            "winner_is_original_teacher": True,
            "should_enter_student_training": False,
            "quality_flags": ["single_teacher_provisional"],
            "review_flags": ["single_teacher_no_pairwise_comparison"],
        }

    consensus_selected, consensus_evidence = _geometric_teacher_consensus_selection(
        teacher_candidates,
        near_identical_dice=near_identical_dice,
        mask_cache=mask_cache,
    )
    if consensus_selected is not None:
        consensus_selected.setdefault(
            "candidate_id", _candidate_id(case_id, organ, consensus_selected)
        )
        near_records = [
            {
                "candidate_a": row.get("candidate_a"),
                "candidate_b": row.get("candidate_b"),
                "status": "skipped_geometric_consensus",
                "decision": {
                    "winner": "agree" if row.get("passes_threshold") else "disagree",
                    "dice_3d": row.get("dice_3d"),
                    "threshold": near_identical_dice,
                },
            }
            for row in consensus_evidence.get("geometric_pairwise_dice", [])
        ]
        return consensus_selected, {
            "selection_method": "geometric_teacher_consensus",
            "legacy_selection_method": "near_identical_agreement",
            "selection_status": "selected",
            "candidate_count": len(candidates),
            "comparison_candidate_count": len(consensus_evidence.get("geometric_consensus_cluster_models", [])),
            "comparison_candidate_models": list(consensus_evidence.get("geometric_consensus_cluster_models", [])),
            "comparison_decisive_count": 0,
            "comparison_inconclusive_count": 0,
            "comparison_agreed_count": len([
                row for row in consensus_evidence.get("geometric_pairwise_dice", [])
                if row.get("passes_threshold")
            ]),
            "selected_model": consensus_selected.get("model"),
            "selected_prediction": consensus_selected.get("prediction"),
            "formal_winner": consensus_selected.get("model"),
            "audit_winner": None,
            "critic_records": near_records,
            "labelcritic_records": near_records,
            "excluded_fusion_candidates": [
                str(row.get("model") or "") for row in fusion_candidates
            ],
            "primary_selector": "complete_link_geometric_teacher_consensus",
            "winner_is_original_teacher": True,
            "candidate_identity_exposed_to_vlm": False,
            "should_enter_student_training": True,
            "family_role": "audit_only_not_used_for_selection_or_training_gate",
            "ground_truth_status": "geometric_consensus_pseudo_label_not_expert_accuracy",
            "accuracy_warning": "Geometric teacher consensus is a pseudo-label safety signal, not expert accuracy.",
            "quality_flags": ["geometric_teacher_consensus"],
            "review_flags": [],
            **consensus_evidence,
        }

    for candidate in teacher_candidates:
        candidate.setdefault(
            "candidate_id", _candidate_id(case_id, organ, candidate)
        )
    ordered = sorted(
        teacher_candidates,
        key=lambda row: (
            -float(row.get("candidate_qc_score") or 0.0),
            str(row.get("model") or ""),
            str(Path(row.get("prediction") or "").resolve()),
            str(row.get("candidate_id") or ""),
        ),
    )
    representatives: list[dict[str, Any]] = []
    near_records: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate_of = None
        duplicate_dice = None
        for representative in representatives:
            d3 = _mask_dice_3d(
                representative["prediction"],
                candidate["prediction"],
                cache=mask_cache,
            )
            if d3 is not None and d3 >= near_identical_dice:
                duplicate_of = representative
                duplicate_dice = d3
                break
        if duplicate_of is None:
            representatives.append(candidate)
        else:
            near_records.append(
                {
                    "candidate_a": duplicate_of.get("model"),
                    "candidate_b": candidate.get("model"),
                    "status": "skipped_near_identical",
                    "decision": {
                        "winner": "agree",
                        "dice_3d": round(float(duplicate_dice), 6),
                    },
                }
            )

    shortlist_truncated = len(representatives) > 4
    representatives = representatives[:4]
    if len(representatives) == 1:
        selected = representatives[0]
        return selected, {
            "selection_method": "geometric_teacher_consensus",
            "legacy_selection_method": "near_identical_agreement",
            "selection_status": "selected",
            "candidate_count": len(candidates),
            "comparison_candidate_count": 1,
            "comparison_candidate_models": [selected["model"]],
            "comparison_decisive_count": 0,
            "comparison_inconclusive_count": 0,
            "comparison_agreed_count": len(near_records),
            "selected_model": selected["model"],
            "selected_prediction": selected["prediction"],
            "critic_records": near_records,
            "labelcritic_records": near_records,
            "excluded_fusion_candidates": [
                str(row.get("model") or "") for row in fusion_candidates
            ],
            "primary_selector": "complete_link_geometric_teacher_consensus",
            "formal_winner": selected["model"],
            "audit_winner": None,
            "winner_is_original_teacher": True,
            "should_enter_student_training": True,
            "shortlist_truncated": shortlist_truncated,
            "family_role": "audit_only_not_used_for_selection_or_training_gate",
            "ground_truth_status": "geometric_consensus_pseudo_label_not_expert_accuracy",
            "accuracy_warning": "Geometric teacher consensus is a pseudo-label safety signal, not expert accuracy.",
            "quality_flags": ["geometric_teacher_consensus"],
            "review_flags": [],
        }

    benchmark_gate_path = (
        Path(__file__).resolve().parents[4]
        / "outputs" / "labelcritic_373_repair_20260703"
        / "stage8_benchmark_gate.json"
    )
    benchmark_gate = {}
    if benchmark_gate_path.is_file():
        try:
            benchmark_gate = json.loads(
                benchmark_gate_path.read_text(encoding="utf-8")
            )
        except Exception:
            benchmark_gate = {}
    requested_uncalibrated = (
        os.getenv("MEDAI_LABELCRITIC_ALLOW_UNCALIBRATED_SELECTION", "0")
        .strip().lower() in {"1", "true", "yes"}
    )
    # Formal 373 selection must not be enabled by an environment override when
    # the official LabelCritic benchmark is unavailable. The env var is kept
    # visible in metadata for debugging, but the formal path ignores it.
    allow_uncalibrated = False
    audit_only = (
        (benchmark_gate.get("status") != "ready" and not allow_uncalibrated)
        or class_regression_failed
    )

    wins = {str(row["candidate_id"]): set() for row in representatives}
    losses = {str(row["candidate_id"]): set() for row in representatives}
    uncertain_pairs = []
    critic_records = list(near_records)
    reverse_inconsistent_pairs = []
    for candidate_a, candidate_b in combinations(representatives, 2):
        output_json = (
            out
            / "critic"
            / case_id
            / f"{organ}_{candidate_a['model']}_vs_{candidate_b['model']}.json"
        )
        critic = run_labelcritic_compare(
            ct,
            Path(candidate_a["prediction"]),
            Path(candidate_b["prediction"]),
            organ,
            output_json,
            backend=critic_backend,
            base_url=critic_base_url,
            port=critic_port,
            dry_run=False,
            timeout_sec=min(timeout_sec, 900),
            candidate_context=[
                _candidate_prompt_context(candidate_a),
                _candidate_prompt_context(candidate_b),
            ],
        )
        winner = (critic.get("decision") or {}).get("winner")
        record = {
            "candidate_a": candidate_a["model"],
            "candidate_b": candidate_b["model"],
            "candidate_a_id": candidate_a["candidate_id"],
            "candidate_b_id": candidate_b["candidate_id"],
            "output_json": str(output_json),
            "status": critic.get("status"),
            "decision": critic.get("decision", {}),
            "vlm_vote": (critic.get("decision") or {}).get("winner"),
            "objective_qc_evidence": {
                "candidate_a": _candidate_prompt_context(candidate_a),
                "candidate_b": _candidate_prompt_context(candidate_b),
            },
            "vlm_rationale_available": bool(
                (critic.get("decision") or {}).get("reason")
                or (critic.get("decision") or {}).get("rationale")
            ),
        }
        critic_records.append(record)
        if audit_only:
            reverse_output_json = (
                out
                / "critic"
                / case_id
                / f"{organ}_{candidate_b['model']}_vs_{candidate_a['model']}_reverse.json"
            )
            reverse = run_labelcritic_compare(
                ct,
                Path(candidate_b["prediction"]),
                Path(candidate_a["prediction"]),
                organ,
                reverse_output_json,
                backend=critic_backend,
                base_url=critic_base_url,
                port=critic_port,
                dry_run=False,
                timeout_sec=min(timeout_sec, 900),
                candidate_context=[
                    _candidate_prompt_context(candidate_b),
                    _candidate_prompt_context(candidate_a),
                ],
            )
            reverse_winner = (reverse.get("decision") or {}).get("winner")
            reverse_record = {
                "candidate_a": candidate_b["model"],
                "candidate_b": candidate_a["model"],
                "candidate_a_id": candidate_b["candidate_id"],
                "candidate_b_id": candidate_a["candidate_id"],
                "output_json": str(reverse_output_json),
                "status": reverse.get("status"),
                "decision": reverse.get("decision", {}),
                "vlm_vote": reverse_winner,
                "objective_qc_evidence": {
                    "candidate_a": _candidate_prompt_context(candidate_b),
                    "candidate_b": _candidate_prompt_context(candidate_a),
                },
                "vlm_rationale_available": bool(
                    (reverse.get("decision") or {}).get("reason")
                    or (reverse.get("decision") or {}).get("rationale")
                ),
                "audit_reverse_order": True,
            }
            critic_records.append(reverse_record)
            forward_model = candidate_a["model"] if winner == "a" else candidate_b["model"] if winner == "b" else None
            reverse_model = candidate_b["model"] if reverse_winner == "a" else candidate_a["model"] if reverse_winner == "b" else None
            if forward_model and reverse_model and forward_model != reverse_model:
                reverse_inconsistent_pairs.append((candidate_a["candidate_id"], candidate_b["candidate_id"]))
        a_id = str(candidate_a["candidate_id"])
        b_id = str(candidate_b["candidate_id"])
        if critic.get("status") == "success" and winner == "a":
            wins[a_id].add(b_id)
            losses[b_id].add(a_id)
        elif critic.get("status") == "success" and winner == "b":
            wins[b_id].add(a_id)
            losses[a_id].add(b_id)
        else:
            uncertain_pairs.append((a_id, b_id))

    required_wins = len(representatives) - 1
    condorcet = [
        row
        for row in representatives
        if len(wins[str(row["candidate_id"])]) == required_wins
    ]
    selected = condorcet[0] if len(condorcet) == 1 else None
    if audit_only:
        audit_winner = selected.get("model") if selected else None
        return None, {
            "selection_method": "label_critic_audit_only",
            "selection_status": "review_required",
            "candidate_count": len(candidates),
            "comparison_candidate_count": len(representatives),
            "comparison_candidate_models": [row["model"] for row in representatives],
            "comparison_decisive_count": sum(len(value) for value in wins.values()),
            "comparison_inconclusive_count": len(uncertain_pairs),
            "comparison_agreed_count": len(near_records),
            "selected_model": None,
            "selected_prediction": None,
            "formal_winner": None,
            "audit_winner": audit_winner,
            "audit_condorcet_status": "unique_condorcet" if selected else "no_unique_condorcet",
            "audit_reverse_inconsistent_pair_count": len(reverse_inconsistent_pairs),
            "critic_records": critic_records,
            "labelcritic_records": critic_records,
            "excluded_fusion_candidates": [
                str(row.get("model") or "") for row in fusion_candidates
            ],
            "fallback_reason": (
                "class_corruption_regression_failed_labelcritic_audit_only"
                if class_regression_failed
                else "official_benchmark_not_ready_labelcritic_audit_only"
            ),
            "benchmark_gate_status": benchmark_gate.get("status", "missing"),
            "uncalibrated_selection_env_requested": requested_uncalibrated,
            "uncalibrated_selection_env_ignored": requested_uncalibrated,
            "class_regression_status": regression_status if class_regression_failed else None,
            "primary_selector": "official_labelcritic_pairwise_condorcet_audit_only",
            "fallback_selector": None,
            "should_enter_student_training": False,
            "candidate_identity_exposed_to_vlm": False,
            "winner_is_original_teacher": False,
            "quality_flags": ["labelcritic_audit_only"] + (["class_regression_failed"] if class_regression_failed else []),
            "review_flags": ["automatic_abstention"],
        }
    return selected, {
        "selection_method": "label_critic" if selected else "label_critic_inconclusive",
        "selection_status": "selected" if selected else "review_required",
        "candidate_count": len(candidates),
        "comparison_candidate_count": len(representatives),
        "comparison_candidate_models": [row["model"] for row in representatives],
        "comparison_decisive_count": sum(len(value) for value in wins.values()),
        "comparison_inconclusive_count": len(uncertain_pairs),
        "comparison_agreed_count": len(near_records),
        "selected_model": selected.get("model") if selected else None,
        "selected_prediction": selected.get("prediction") if selected else None,
        "formal_winner": selected.get("model") if selected else None,
        "audit_winner": None,
        "critic_records": critic_records,
        "labelcritic_records": critic_records,
        "excluded_fusion_candidates": [
            str(row.get("model") or "") for row in fusion_candidates
        ],
        "fallback_reason": None if selected else "no_unique_condorcet_winner",
        "primary_selector": "official_labelcritic_pairwise_condorcet",
        "fallback_selector": None,
        "should_enter_student_training": selected is not None,
        "shortlist_truncated": shortlist_truncated,
        "candidate_identity_exposed_to_vlm": False,
        "winner_is_original_teacher": selected is not None,
        "quality_flags": ["labelcritic_selected"] if selected else ["labelcritic_uncertain"],
        "review_flags": [] if selected else ["labelcritic_uncertain", "automatic_abstention"],
    }


def _select_candidate(
    *,
    ct: Path,
    organ: str,
    candidates: list[dict[str, Any]],
    out: Path,
    case_id: str,
    enable_critic: bool,
    critic_backend: str,
    critic_base_url: str,
    critic_port: int,
    timeout_sec: int,
    dry_run: bool,
    labelcritic_options: dict[str, Any] | None = None,
    near_identical_dice: float = 0.95,
    mask_cache: _CaseMaskCache | None = None,
    compare_batch_enabled: bool = True,
    compare_batch_max_candidates: int = 2,
    strict_labelcritic_selection: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select the pseudo-label candidate for one organ.

    Teacher outputs are pseudo-label candidates. If multiple candidates exist,
    LabelCritic is the primary selector; Dice is only a fallback/metadata signal.
    """
    qc_rejected = [
        c for c in candidates
        if not bool(c.get("eligible_for_labelcritic", True))
    ]
    eligible_candidates = [
        c for c in candidates
        if bool(c.get("eligible_for_labelcritic", True))
    ]
    comparison_candidates = eligible_candidates or candidates
    qc_review_flags: list[str] = []
    qc_quality_flags: list[str] = []
    if qc_rejected:
        qc_review_flags.append("candidate_qc_rejected")
        qc_quality_flags.append("candidate_qc_rejected")
    if candidates and not eligible_candidates:
        qc_review_flags.append("all_candidates_failed_qc")
        qc_quality_flags.append("all_candidates_failed_qc")

    if not candidates:
        return None, {
            "selection_method": "none",
            "selection_status": "missing",
            "reason": "no candidate masks produced",
            "candidate_count": 0,
            "comparison_candidate_count": 0,
            "comparison_candidate_models": [],
            "qc_rejected_candidates": [],
            "selected_model": None,
            "selected_prediction": None,
            "critic_records": [],
            "labelcritic_records": [],
            "fallback_reason": None,
            "quality_flags": ["missing_candidate"],
            "review_flags": ["missing_candidate"],
        }

    if not eligible_candidates:
        selected = _pick_reference_fallback(candidates)
        return selected, {
            "selection_method": "candidate_qc_fallback",
            "selection_status": "fallback",
            "candidate_count": len(candidates),
            "comparison_candidate_count": 0,
            "comparison_candidate_models": [],
            "qc_rejected_candidates": [_candidate_qc_summary(c) for c in qc_rejected],
            "candidate_qc_policy": "hard_fail_candidates_excluded_before_labelcritic",
            "selected_model": selected["model"],
            "selected_prediction": selected["prediction"],
            "critic_records": [],
            "labelcritic_records": [],
            "fallback_reason": "All candidates failed QC; selected only for review/fallback continuity",
            "quality_flags": ["fallback_selection", *qc_quality_flags],
            "review_flags": ["selection_fallback", *qc_review_flags],
        }

    if len(comparison_candidates) == 1:
        selected = comparison_candidates[0]
        return selected, {
            "selection_method": "single_teacher_provisional",
            "legacy_selection_method": "single_teacher_default",
            "selection_status": "provisional",
            "candidate_count": len(candidates),
            "comparison_candidate_count": 1,
            "comparison_candidate_models": [selected["model"]],
            "qc_rejected_candidates": [_candidate_qc_summary(c) for c in qc_rejected],
            "candidate_qc_policy": "hard_fail_candidates_excluded_before_labelcritic",
            "selected_model": selected["model"],
            "selected_prediction": selected["prediction"],
            "critic_records": [],
            "labelcritic_records": [],
            "fallback_reason": None,
            "quality_flags": ["single_candidate", "single_teacher_provisional", *qc_quality_flags],
            "review_flags": ["single_teacher_no_pairwise_comparison", *qc_review_flags],
        }

    if enable_critic and not dry_run:
        selected, selection = _official_pairwise_condorcet_selection(
            ct=ct,
            organ=organ,
            candidates=comparison_candidates,
            out=out,
            case_id=case_id,
            critic_backend=critic_backend,
            critic_base_url=critic_base_url,
            critic_port=critic_port,
            timeout_sec=timeout_sec,
            near_identical_dice=near_identical_dice,
            mask_cache=mask_cache,
        )
        selection["qc_rejected_candidates"] = [
            _candidate_qc_summary(candidate) for candidate in qc_rejected
        ]
        selection["candidate_qc_policy"] = (
            "hard_fail_candidates_excluded_before_labelcritic"
        )
        return selected, selection

    critic_records: list[dict[str, Any]] = []
    labelcritic_options = labelcritic_options or {}
    selected = comparison_candidates[0]
    selection_status = "selected"
    fallback_reason = None
    decisive = 0
    inconclusive = 0
    agreed = 0
    # LabelCritic's internal 2D dice gate compares background-dominated CT images
    # and skips every pair, so bypass it and gate on the real 3D mask overlap here.
    lc_opts = {**labelcritic_options, "no_dice_check": True}

    if enable_critic and not dry_run:
        all_pair_agreement = _candidate_pairwise_agreement(comparison_candidates, cache=mask_cache)
        all_pair_min_dice = all_pair_agreement.get("min_dice_3d")
        all_pair_threshold = float(os.getenv("MEDAI_COMPARE_HIGH_AGREEMENT_DICE", str(near_identical_dice)))
        skip_all_compare = bool(
            len(comparison_candidates) > 2
            and all_pair_min_dice is not None
            and float(all_pair_min_dice) >= all_pair_threshold
        )
        if skip_all_compare:
            agreed = max(0, len(comparison_candidates) - 1)
            critic_records.append({
                "status": "skipped_high_pairwise_agreement",
                "decision": {
                    "winner": "agree",
                    "min_dice_3d": round(float(all_pair_min_dice), 4),
                    "threshold": all_pair_threshold,
                },
                "batch_status": "all_pair_agreement_gate",
                "pairwise_agreement": all_pair_agreement,
            })
        compare_fn_patched = getattr(run_labelcritic_compare, "__module__", "") != "cli_anything.medai.core.labelcritic_wrapper"
        use_batched_pairwise = bool(
            not skip_all_compare
            and compare_batch_enabled
            and not compare_fn_patched
            and 2 <= len(comparison_candidates) <= max(2, int(compare_batch_max_candidates))
        )
        if use_batched_pairwise:
            batch_jobs: list[dict[str, Any]] = []
            # Batch only the exact two-candidate tournament case. For 3+ candidates
            # the incumbent can change after each comparison, so pre-batching all
            # adjacent pairs would alter old tournament semantics.
            pair_meta: list[tuple[dict[str, Any], dict[str, Any], float | None, Path | None]] = []
            incumbent = comparison_candidates[0]
            for challenger in comparison_candidates[1:]:
                d3 = _mask_dice_3d(incumbent["prediction"], challenger["prediction"], cache=mask_cache)
                critic_out = None
                if d3 is None or d3 < near_identical_dice:
                    critic_out = out / "critic" / case_id / f"{organ}_{incumbent['model']}_vs_{challenger['model']}.json"
                    batch_jobs.append({
                        "ct_image": ct,
                        "mask_a": Path(incumbent["prediction"]),
                        "mask_b": Path(challenger["prediction"]),
                        "organ": organ,
                        "output_json": critic_out,
                        "candidate_context": [
                            _candidate_prompt_context(incumbent),
                            _candidate_prompt_context(challenger),
                        ],
                        **lc_opts,
                    })
                pair_meta.append((incumbent, challenger, d3, critic_out))
            batch_results = run_labelcritic_compare_batch(
                batch_jobs,
                backend=critic_backend,
                base_url=critic_base_url,
                port=critic_port,
                dry_run=False,
                timeout_sec=min(timeout_sec, 900),
            ) if batch_jobs else []
            result_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
            for job, critic in zip(batch_jobs, batch_results):
                result_by_pair[(str(Path(job["mask_a"])), str(Path(job["mask_b"])))] = critic
            for cand_a, cand_b, d3, critic_out in pair_meta:
                if d3 is not None and d3 >= near_identical_dice:
                    agreed += 1
                    critic_records.append({
                        "candidate_a": cand_a["model"],
                        "candidate_b": cand_b["model"],
                        "status": "skipped_near_identical",
                        "decision": {"winner": "agree", "dice_3d": round(d3, 4)},
                        "batch_status": "batched_tournament",
                    })
                    continue
                key = (str(Path(cand_a["prediction"])), str(Path(cand_b["prediction"])))
                critic = result_by_pair.get(key, {})
                record = {
                    "candidate_a": cand_a["model"],
                    "candidate_b": cand_b["model"],
                    "output_json": critic.get("output_json") or (str(critic_out) if critic_out else None),
                    "status": critic.get("status"),
                    "decision": critic.get("decision", {}),
                    "batch_status": "batched_tournament",
                }
                critic_records.append(record)
                winner = (critic.get("decision", {}) or {}).get("winner")
                if critic.get("status") == "success" and winner == "b":
                    selected = cand_b
                    decisive += 1
                elif critic.get("status") == "success" and winner == "a":
                    decisive += 1
                else:
                    inconclusive += 1
        elif not skip_all_compare:
            for challenger in comparison_candidates[1:]:
                d3 = _mask_dice_3d(selected["prediction"], challenger["prediction"], cache=mask_cache)
                if d3 is not None and d3 >= near_identical_dice:
                    # Candidates effectively agree; no VLM judgment needed (efficiency).
                    agreed += 1
                    critic_records.append({
                        "candidate_a": selected["model"],
                        "candidate_b": challenger["model"],
                        "status": "skipped_near_identical",
                        "decision": {"winner": "agree", "dice_3d": round(d3, 4)},
                    })
                    continue
                critic_out = out / "critic" / case_id / f"{organ}_{selected['model']}_vs_{challenger['model']}.json"
                critic = run_labelcritic_compare(
                    ct,
                    Path(selected["prediction"]),
                    Path(challenger["prediction"]),
                    organ,
                    critic_out,
                    backend=critic_backend,
                    base_url=critic_base_url,
                    port=critic_port,
                    dry_run=False,
                    timeout_sec=min(timeout_sec, 300),
                    candidate_context=[
                        _candidate_prompt_context(selected),
                        _candidate_prompt_context(challenger),
                    ],
                    **lc_opts,
                )
                record = {
                    "candidate_a": selected["model"],
                    "candidate_b": challenger["model"],
                    "output_json": str(critic_out),
                    "status": critic.get("status"),
                    "decision": critic.get("decision", {}),
                }
                critic_records.append(record)
                winner = (critic.get("decision", {}) or {}).get("winner")
                if critic.get("status") == "success" and winner == "b":
                    selected = challenger
                    decisive += 1
                elif critic.get("status") == "success" and winner == "a":
                    decisive += 1
                else:
                    # Inconclusive: VLM undecided, the dice-check skipped a near-identical
                    # pair (LabelCritic's efficiency gate produces no rows), or the call
                    # errored. Keep the incumbent and continue the tournament — a single
                    # inconclusive pair must not abort it and discard decisive results or
                    # the consensus pick.
                    inconclusive += 1
        if decisive > 0:
            method = "label_critic"
            selection_status = "selected"
            fallback_reason = (
                f"{inconclusive} pairwise comparison(s) inconclusive; kept decisive LabelCritic result"
                if inconclusive else None
            )
        elif agreed == max(0, len(comparison_candidates) - 1) and inconclusive == 0:
            # All candidates were near-identical in 3D, so no VLM tie-break is
            # needed. Treat this as positive multi-teacher agreement rather than
            # an inconclusive fallback, otherwise highly consistent labels get
            # unfairly down-weighted in the dashboard/training manifest.
            method = "near_identical_agreement"
            selection_status = "selected"
            fallback_reason = "All eligible candidates agree above the near-identical 3D Dice threshold; VLM tie-break skipped"
        else:
            method = "label_critic_inconclusive"
            selection_status = "review_required" if strict_labelcritic_selection else "fallback"
            fallback_reason = "LabelCritic produced no decisive comparison (all pairs inconclusive or skipped)"
            selected = None if strict_labelcritic_selection else _pick_reference_fallback(comparison_candidates)
    else:
        selected = None if strict_labelcritic_selection else _pick_reference_fallback(comparison_candidates)
        method = "critic_disabled_fallback"
        selection_status = "review_required" if strict_labelcritic_selection else "fallback"
        fallback_reason = "LabelCritic disabled, unavailable, or dry-run"

    explanation = _labelcritic_explanation_fields(critic_records, selected.get("model") if selected else None, fallback_reason=fallback_reason)

    return selected, {
        "selection_method": method,
        "selection_status": selection_status,
        "candidate_count": len(candidates),
        "comparison_candidate_count": len(comparison_candidates),
        "comparison_decisive_count": decisive,
        "comparison_inconclusive_count": inconclusive,
        "comparison_agreed_count": agreed,
        "comparison_candidate_models": [c["model"] for c in comparison_candidates],
        "qc_rejected_candidates": [_candidate_qc_summary(c) for c in qc_rejected],
        "candidate_qc_policy": "hard_fail_candidates_excluded_before_labelcritic",
        "selected_model": selected["model"] if selected else None,
        "selected_prediction": selected["prediction"] if selected else None,
        "critic_records": critic_records,
        "labelcritic_records": critic_records,
        "fallback_reason": fallback_reason,
        "primary_selector": "labelcritic",
        "labelcritic_decisive": bool(method == "label_critic" and decisive > 0),
        "labelcritic_confidence": explanation.get("labelcritic_confidence") if explanation.get("labelcritic_confidence") is not None else (0.75 if method == "label_critic" and decisive > 0 else None),
        "fallback_selector": None if method in {"label_critic", "near_identical_agreement", "single_teacher_default", "single_teacher_provisional"} else "family_evidence_or_reference",
        "evidence_used_for": "audit_only" if method == "label_critic" and decisive > 0 else "fallback_selection",
        "selected_reason": explanation.get("selected_reason"),
        "rejected_reasons": explanation.get("rejected_reasons", {}),
        "failure_modes": explanation.get("failure_modes", []),
        "should_enter_student_training": selection_status == "selected" and selected is not None,
        "quality_flags": (
            ["labelcritic_selected"] if method == "label_critic"
            else ["multi_teacher_agreement"] if method == "near_identical_agreement"
            else ["fallback_selection", "labelcritic_uncertain"]
        ) + qc_quality_flags,
        "review_flags": ([] if selection_status == "selected" else ["selection_fallback", "labelcritic_uncertain"]) + qc_review_flags,
    }




def _compact_labelcritic_reason(value: Any, *, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "LabelCritic did not provide a textual reason."
    return text[:limit] + ("..." if len(text) > limit else "")


def _labelcritic_explanation_fields(
    critic_records: list[dict[str, Any]],
    selected_model: str | None,
    *,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Build explainable selection fields from pairwise LabelCritic records.

    LabelCritic's current parser returns pairwise winner/confidence/reason.  The
    E-step manifest still needs stable selected/rejected reasons even before the
    future structured JSON VLM response is available.
    """
    rejected: dict[str, str] = {}
    failure_modes: list[str] = []
    selected_reasons: list[str] = []
    confidences: list[float] = []
    for record in critic_records or []:
        decision = record.get("decision") or {}
        winner = decision.get("winner")
        parse_status = decision.get("parse_status")
        if parse_status and parse_status not in {"better_path_parse", "success"}:
            failure_modes.append(str(parse_status))
        try:
            if decision.get("confidence") is not None:
                confidences.append(float(decision.get("confidence")))
        except Exception:
            pass
        winner_model = record.get("candidate_a") if winner == "a" else record.get("candidate_b") if winner == "b" else None
        loser_model = record.get("candidate_b") if winner == "a" else record.get("candidate_a") if winner == "b" else None
        reason = _compact_labelcritic_reason(decision.get("reason") or parse_status or record.get("status"))
        if winner_model and str(winner_model) == str(selected_model):
            selected_reasons.append(f"LabelCritic preferred {winner_model} over {loser_model}: {reason}")
        if loser_model:
            rejected[str(loser_model)] = f"Rejected in LabelCritic pairwise comparison against {winner_model}: {reason}"
        elif winner == "uncertain":
            for candidate_key in ("candidate_a", "candidate_b"):
                candidate = record.get(candidate_key)
                if candidate and str(candidate) != str(selected_model):
                    rejected.setdefault(str(candidate), f"LabelCritic was uncertain for this pair: {reason}")
    if not selected_reasons:
        selected_reasons.append(fallback_reason or "Selected by LabelCritic/fallback policy; no decisive textual LabelCritic rationale was parsed.")
    return {
        "selected_reason": selected_reasons[0],
        "rejected_reasons": rejected,
        "failure_modes": sorted(set(failure_modes)),
        "labelcritic_confidence": max(confidences) if confidences else None,
    }

def _labelcritic_locks_selection(selection_record: dict[str, Any]) -> bool:
    """True when a decisive LabelCritic tournament winner owns final selection.

    AutoLabelCore/family evidence may still audit reliability and training
    weight, but must not replace the selected mask in this state.
    """
    return bool(
        selection_record.get("selection_method") == "label_critic"
        and int(selection_record.get("comparison_decisive_count") or 0) > 0
    )

def _fusion_weight(tracker: OrganModelPerformance | None, organ: str, model: str) -> float:
    """Reliability weight for one (organ, model) in weighted-vote fusion.

    Uses the tracker's mean pseudo-consistency DSC when available (round 2+);
    falls back to equal weight (1.0) during exploration / round 1. A small floor
    keeps a currently-low-scoring teacher contributing rather than vanishing.
    """
    if tracker is None:
        return 1.0
    try:
        stats = tracker.get_stats(organ, model)
    except Exception:
        stats = None
    if stats and stats.get("n_cases", 0) >= 1:
        return max(float(stats.get("estimated_reliability", stats.get("mean_estimated_reliability", 0.0))), 0.05)
    return 1.0


_HIGH_RISK_GRADE_KEYWORDS = (
    "tumor", "lesion", "duct", "vessel", "artery", "vein", "nerve",
    "pancreas", "adrenal", "lymph", "node", "bowel", "intestine",
    "duodenum", "colon", "prostate", "postcava", "portal", "hepatic",
)


def _is_high_risk_grade_organ(organ: str) -> bool:
    key = _norm_organ_key(organ)
    return any(term in key for term in _HIGH_RISK_GRADE_KEYWORDS)


def _labelcritic_grade_policy_decision(
    *,
    organ: str,
    selected: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    selection: dict[str, Any],
    route_info: dict[str, Any],
    best_dice: float | None,
    vlm_threshold: float,
    empty_reference_nonempty_prediction: bool,
    policy: str,
) -> dict[str, Any]:
    if not selected:
        return {"run": False, "reason": None, "skipped_reason": "no_selected_candidate"}
    policy = (policy or "risk_aware").strip().lower()
    if policy in {"all", "always", "formal_full_grade_all"}:
        return {"run": True, "reason": "policy_grade_all", "skipped_reason": None}
    if policy in {"off", "disabled", "none"}:
        return {"run": False, "reason": None, "skipped_reason": "policy_grade_disabled"}

    non_fusion_count = len([c for c in candidates if not c.get("is_fusion")])
    selected_qc_status = selected.get("candidate_qc_status")
    selected_qc_flags = selected.get("candidate_qc_flags", []) or []
    selection_status = selection.get("selection_status")
    selection_method = selection.get("selection_method")
    compare_used = bool(selection.get("labelcritic_records") or selection.get("critic_records"))
    high_risk = _is_high_risk_grade_organ(organ)
    route_confidence = str(route_info.get("route_confidence", "low"))
    low_dice = bool(
        best_dice is not None
        and float(best_dice) < vlm_threshold
        and not empty_reference_nonempty_prediction
    )

    hard_qc_flags = {"missing_file", "unreadable_mask", "missing_prediction", "geometry_mismatch", "shape_mismatch", "shape_mismatch_ct", "all_candidates_failed_qc"}
    hard_qc = (selected_qc_status == "fail") or bool(set(map(str, selected_qc_flags)) & hard_qc_flags)
    review_qc = selected_qc_status not in (None, "pass") or bool(selected_qc_flags)

    if hard_qc:
        return {
            "run": False,
            "reason": None,
            "skipped_reason": "hard_qc_structural_reject_no_vlm_needed",
        }

    reasons: list[str] = []
    if low_dice:
        reasons.append("low_pseudo_consistency_dice")
    # LC-2 is an absolute safety-net, not a tax on every routed comparison.
    # Compare evidence already chose among candidates; grade only when the organ
    # is intrinsically risky or has non-structural warning evidence where VLM can help.
    if high_risk:
        reasons.append("high_risk_organ")
    if review_qc and high_risk:
        reasons.append("candidate_qc_review_high_risk")
    if (
        compare_used
        and selection_method not in {"near_identical_agreement", "geometric_teacher_consensus"}
        and (high_risk or low_dice or (review_qc and route_confidence != "high"))
    ):
        reasons.append("post_compare_absolute_quality_gate")
    if (
        route_confidence == "low"
        and (high_risk or low_dice or review_qc)
    ):
        reasons.append("route_confidence_low_with_risk")
    if (
        non_fusion_count > 1
        and selection_method not in {"near_identical_agreement", "geometric_teacher_consensus"}
        and (high_risk or low_dice or review_qc)
    ):
        reasons.append("multi_candidate_conflict_or_diversity")

    if selection_status != "selected" and not reasons:
        return {"run": False, "reason": None, "skipped_reason": "fallback_structural_low_weight_no_vlm_needed"}

    if reasons:
        return {"run": True, "reason": ";".join(dict.fromkeys(reasons)), "skipped_reason": None}

    if non_fusion_count <= 1 and selected_qc_status in (None, "pass") and route_confidence == "high":
        return {"run": False, "reason": None, "skipped_reason": "stable_single_candidate_qc_pass"}
    if selection_method in {"near_identical_agreement", "geometric_teacher_consensus"} and selected_qc_status in (None, "pass"):
        return {"run": False, "reason": None, "skipped_reason": "high_agreement_qc_pass"}
    return {"run": False, "reason": None, "skipped_reason": "risk_aware_grade_not_required"}


def _auto_arbitrate_organ(
    *,
    ct: Path,
    organ: str,
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    out: Path,
    case_id: str,
    base_url: str,
    port: int,
    vlm_model: str | None,
    accept_grade: float,
    reject_grade: float,
    timeout_sec: int,
    grade_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Conservative automated absolute-quality arbitration (de-human) for one organ.

    The VLM absolute single-mask grade is noisy (it reliably flags empty/garbage
    masks at ~0.0 but is promptable and unreliable at distinguishing decent vs
    wrong-organ masks). So this acts as a SAFETY NET, not a strict filter: it only
    treats a pick as rejected when the grade is confidently bad (<= reject_grade,
    e.g. empty/clearly-wrong), and only then swaps to an alternative that grades
    clearly good (>= accept_grade). Everything else is recorded as an advisory
    verdict and the original (fusion/LabelCritic) selection is kept. ``_selected``
    is the final chosen candidate; ``final_accept`` is the machine verdict
    (True unless confidently bad). No-ops gracefully when no VLM is available.
    """
    grade_dir = out / "critic" / case_id

    def _grade(cand: dict[str, Any]) -> dict[str, Any]:
        out_json = grade_dir / f"{organ}_{cand['model']}_grade.json"
        if grade_cache is not None and str(out_json.resolve()) in grade_cache:
            cached = dict(grade_cache[str(out_json.resolve())])
            cached["cache_status"] = cached.get("cache_status", "reused_batched_grade")
            return cached
        return run_labelcritic_grade(
            ct, Path(cand["prediction"]), organ,
            out_json,
            base_url=base_url, port=port, vlm_model=vlm_model,
            accept_grade=accept_grade, dry_run=False, timeout_sec=min(timeout_sec, 300),
        )

    def _confident_bad(g: dict[str, Any]) -> bool:
        if g.get("status") != "success":
            return False
        grade_label = str(g.get("grade_label") or "").strip().lower()
        if grade_label == "bad":
            return True
        return g.get("grade") is not None and float(g["grade"]) <= reject_grade

    def _clearly_good(g: dict[str, Any]) -> bool:
        if g.get("status") != "success":
            return False
        grade_label = str(g.get("grade_label") or "").strip().lower()
        if grade_label in {"good", "acceptable"}:
            return True
        return g.get("grade") is not None and float(g["grade"]) >= accept_grade

    g0 = _grade(selected)
    record: dict[str, Any] = {
        "case_id": case_id,
        "organ": organ,
        "selected_model_before": selected["model"],
        "selected_model_after": selected["model"],
        "grade_before": g0.get("grade"),
        "grade_label_before": g0.get("grade_label"),
        "grade_after": g0.get("grade"),
        "grade_label_after": g0.get("grade_label"),
        "grade_status": g0.get("status"),
        "final_accept": not _confident_bad(g0),
        "confident_bad": _confident_bad(g0),
        "reason": g0.get("reason"),
        "hard_failure_reason": g0.get("hard_failure_reason"),
        "swapped": False,
        "_selected": selected,
    }
    # Only act when the selected pick is confidently bad (the regime where the VLM
    # is reliable). Try up to two alternatives; swap to the first clearly-good one.
    if _confident_bad(g0):
        alts = [
            c for c in candidates
            if c is not selected and c.get("candidate_exists") and c.get("eligible_for_labelcritic", True)
        ]
        alts.sort(key=lambda c: (0 if c.get("is_fusion") else 1, -(c.get("dice") or 0.0)))
        for alt in alts[:2]:
            g1 = _grade(alt)
            if _clearly_good(g1) and not _confident_bad(g1):
                record.update({
                    "swapped": True,
                    "selected_model_after": alt["model"],
                    "grade_after": g1.get("grade"),
                    "grade_label_after": g1.get("grade_label"),
                    "final_accept": True,
                    "reason": g1.get("reason"),
                    "hard_failure_reason": g1.get("hard_failure_reason"),
                    "_selected": alt,
                })
                break
    return record


def run_multimodel_annotation_loop(
    case_list: str | Path,
    output_folder: str | Path,
    models: list[str] | None = None,
    organs: list[str] | None = None,
    registry_path: str | Path = "configs/model_registry.yaml",
    checkpoint_map_models: bool = False,
    shapekit_root: str | Path = "third_party/ShapeKit-main",
    enable_shapekit: bool = True,
    enable_critic: bool = True,
    critic_backend: str = "labelcritic",
    critic_base_url: str = "http://localhost",
    critic_port: int = 8000,
    vlm_threshold: float = 0.5,
    accept_threshold: float = 0.8,
    dry_run: bool = False,
    timeout_sec: int = 1800,
    device: str | None = None,
    perf_tracker_path: str | Path | None = None,
    resume: bool = True,
    preseeded_model_dirs: dict[str, Path] | None = None,
    labelcritic_options: dict[str, Any] | None = None,
    enable_fusion: bool = False,
    fusion_method: str = "weighted_vote",
    enable_auto_arbitration: bool = True,
    arbitration_accept_grade: float = 0.5,
    arbitration_reject_grade: float = 0.2,
    vlm_model: str | None = None,
    candidate_mode: str = "route_pruned_with_competition",
    teacher_inference_mode: str = "hierarchical_roi",
    roi_margin_mm: float = 20.0,
    preseeded_parent_only: bool = False,
    reuse_preseeded_only: bool = False,
    strict_labelcritic_selection: bool = True,
) -> dict[str, Any]:
    """
    preseeded_model_dirs: mapping of model_key -> base directory where
        per-case predictions already exist as one of:
        <base>/<case_id>/<organ>.nii.gz,
        <base>/<case_id>/updated/<organ>.nii.gz, or
        <base>/<case_id>/segmentations/<organ>.nii.gz.
        By default these models are injected directly into model_seg_dirs. When
        preseeded_parent_only=True they are visible only to hierarchical major
        parent resolution; stale child/sub-organ masks never enter selection.
    """
    """Run the teacher-requested multi-model annotation refinement loop.

    Required case_list columns:
      case_id, ct_path, annotation_folder
    Optional columns:
      report_path, clinical_path, pathology_path
    """
    case_csv = Path(case_list).resolve()
    out = Path(output_folder).resolve()
    labelcritic_options = labelcritic_options or {}
    if teacher_inference_mode not in {"full_volume", "hierarchical_roi"}:
        raise ValueError("teacher_inference_mode must be 'full_volume' or 'hierarchical_roi'")
    out.mkdir(parents=True, exist_ok=True)
    cases = _read_case_list(case_csv)
    registry = load_registry(registry_path)
    project_root = Path(__file__).resolve().parents[4]
    alias_config = _load_model_label_aliases(project_root)
    taxonomy = _load_organ_taxonomy(project_root)
    organ_prompts = _load_organ_prompts(project_root)
    if organs is None or not organs:
        organs = _load_default_target_organs(project_root)
    if models is None:
        models = ["mock_seg"] if dry_run else ["totalsegmentator"]
    target_validation = _target_validation_for_run(project_root, organs)
    student_target_ids = _load_student_target_ids(project_root)
    target_blocking = target_validation.get("blocking", {}) if isinstance(target_validation, dict) else {}
    if target_blocking.get("requested_non_target_organs"):
        raise ValueError(
            "Requested organs include non-target organs for the formal 373-organ mainline: "
            + json.dumps(target_blocking.get("requested_non_target_organs"), ensure_ascii=False)
        )
    if len(organs) == 373 and target_validation.get("status") != "success":
        raise ValueError(
            "Formal 373-organ run failed target-space validation: "
            + json.dumps(target_validation.get("blocking", target_validation), ensure_ascii=False)
        )

    # Initialise organ-model performance tracker (None = disabled)
    tracker: OrganModelPerformance | None = (
        OrganModelPerformance(perf_tracker_path) if perf_tracker_path else None
    )

    dice_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    inference_results: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    all_gap_rows: list[dict[str, Any]] = []
    resume_rows: list[dict[str, Any]] = []
    updated_root = out / "annotation_versions"
    standard_dataset_root = out / "standard_dataset"
    standard_dataset_cases: list[dict[str, Any]] = []
    review_queue = out / "review_queue.jsonl"
    timing_rows: list[dict[str, Any]] = []
    vlm_decisions = out / "vlm_decisions.jsonl"
    traces_jsonl = out / "patient_traces.jsonl"
    report_supervision_jsonl = out / "report_supervision.jsonl"

    # Reset append-only outputs for a clean run.
    for p in (review_queue, vlm_decisions, traces_jsonl, report_supervision_jsonl):
        if p.exists():
            p.unlink()

    for idx, case in enumerate(cases, start=1):
        case_id = case.get("case_id") or Path(case.get("ct_path", f"case_{idx}")).parent.name
        ct = Path(case.get("ct_path", "")).resolve()
        ref_dir = Path(case.get("annotation_folder", "")).resolve() if case.get("annotation_folder") else None
        case_out = out / "cases" / case_id
        case_raw = case_out / "raw_predictions"
        case_refined = case_out / "refined_predictions"
        case_updated = updated_root / case_id / "updated"
        case_updated.mkdir(parents=True, exist_ok=True)
        presence_context = _load_case_presence_context(case, ct, case_id, case_out)

        import time as _time
        _case_start = _time.time()
        grade_batch_concurrency = 1
        compare_batch_enabled = True
        stage_timing = {
            "teacher_inference_sec": 0.0,
            "candidate_shapekit_sec": 0.0,
            "candidate_qc_verify_sec": 0.0,
            "fusion_sec": 0.0,
            "labelcritic_compare_sec": 0.0,
            "labelcritic_grade_sec": 0.0,
            "copy_final_sec": 0.0,
            "labelcritic_batch_queue_sec": 0.0,
        }
        stage_counts = {
            "candidate_qc_verify_count": 0,
            "fusion_count": 0,
            "labelcritic_compare_records": 0,
            "labelcritic_grade_count": 0,
            "copy_final_count": 0,
            "labelcritic_grade_batch_concurrency": grade_batch_concurrency,
            "labelcritic_compare_batch_enabled": compare_batch_enabled,
        }

        fov_organs = _fov_pruned_organs(list(organs), taxonomy, presence_context)
        execution_organs = list(fov_organs)
        if teacher_inference_mode == "hierarchical_roi" and not reuse_preseeded_only:
            for requested_organ in fov_organs:
                for parent in ((taxonomy_entry(taxonomy, requested_organ) or {}).get("parent_ids", []) or []):
                    if parent not in execution_organs:
                        execution_organs.append(parent)
        case_execution_plan = _build_case_execution_plan(
            registry=registry,
            project_root=project_root,
            organs=execution_organs,
            requested_models=list(models),
            preseeded_model_dirs=preseeded_model_dirs,
            candidate_mode=candidate_mode,
        )
        hierarchy_plan_cache_key: dict[str, Any] | None = None
        hierarchy_plan_cache_key_sha256: str | None = None
        if teacher_inference_mode == "hierarchical_roi":
            requested_set_for_cache = set(fov_organs)
            dependency_parents_for_cache = {
                parent
                for organ in fov_organs
                for parent in ((taxonomy_entry(taxonomy, organ) or {}).get("parent_ids", []) or [])
            }
            major_organs_for_cache = sorted({
                organ for organ in fov_organs
                if (taxonomy_entry(taxonomy, organ) or {}).get("hierarchy_role") == "major"
            } | dependency_parents_for_cache)
            child_organs_for_cache = sorted({
                organ for organ in fov_organs
                if (taxonomy_entry(taxonomy, organ) or {}).get("hierarchy_role") == "child"
            })
            hierarchy_plan_cache_key = _hierarchical_plan_cache_key(
                requested_organs=fov_organs,
                major_organs=major_organs_for_cache,
                child_organs=child_organs_for_cache,
                execution_plan=case_execution_plan,
            )
            hierarchy_plan_cache_key_sha256 = _stable_json_sha256(hierarchy_plan_cache_key)

        # Resume only when the full E-step artifact set is complete.  Raw
        # predictions alone are not enough because a prior run may have stopped
        # before LabelCritic selection, ShapeKit, or manifest metadata.
        if resume and not dry_run:
            resume_state = _case_resume_state(
                case_out=case_out,
                updated_root=updated_root,
                case_id=case_id,
                organs=organs,
                enable_shapekit=enable_shapekit,
            )
            if teacher_inference_mode == "hierarchical_roi":
                hierarchy_manifest = case_out / "hierarchical_inference_plan.json"
                if not hierarchy_manifest.exists():
                    resume_state = {**resume_state, "complete": False, "reason": "hierarchical_roi_manifest_missing"}
                else:
                    try:
                        hierarchy_cached = json.loads(hierarchy_manifest.read_text(encoding="utf-8"))
                    except Exception as exc:
                        resume_state = {
                            **resume_state,
                            "complete": False,
                            "reason": f"hierarchical_roi_manifest_unreadable:{exc}",
                        }
                    else:
                        if (
                            hierarchy_cached.get("teacher_inference_mode") != "hierarchical_roi"
                            or hierarchy_cached.get("hierarchical_pipeline_version") != HIERARCHICAL_PIPELINE_VERSION
                            or not _hierarchical_cache_keys_equivalent(
                                hierarchy_cached.get("hierarchical_plan_cache_key") or {},
                                hierarchy_plan_cache_key,
                            )
                        ):
                            resume_state = {
                                **resume_state,
                                "complete": False,
                                "reason": "hierarchical_roi_plan_cache_mismatch",
                            }
            resume_rows.append({"case_id": case_id, **resume_state})
            if resume_state["complete"]:
                print(f"[{_time.strftime('%H:%M:%S')}] Case {idx}/{len(cases)}: {case_id} 已完整完成，跳过", flush=True)
                continue
            if resume_state.get("raw_ready"):
                print(
                    f"[{_time.strftime('%H:%M:%S')}] Case {idx}/{len(cases)}: {case_id} raw 已存在但 {resume_state.get('reason')}，继续补齐后续阶段",
                    flush=True,
                )

        print(f"[{_time.strftime('%H:%M:%S')}] Case {idx}/{len(cases)}: {case_id} 开始推理...", flush=True)

        if not dry_run and not ct.exists():
            _append_jsonl(review_queue, {"case_id": case_id, "reason": "ct_path missing", "ct_path": str(ct)})
            continue

        write_json(updated_root / case_id / "case_execution_plan.json", {
            "case_id": case_id,
            "ct_path": str(ct),
            "teacher_inference_mode": teacher_inference_mode,
            "roi_margin_mm": roi_margin_mm,
            **case_execution_plan,
        })
        organ_task_state_path = updated_root / case_id / "organ_task_state.json"
        organ_task_state = _load_organ_task_state(organ_task_state_path)
        mask_cache = _CaseMaskCache(max_arrays=int(os.getenv("MEDAI_MASK_CACHE_MAX_ARRAYS", "96")))
        # Formal selection is pairwise. The legacy project single-mask grader
        # used a non-official centered-slice fallback and is audit-only.
        grade_policy = os.getenv("MEDAI_LABELCRITIC_GRADE_POLICY", "off")
        grade_batch_concurrency = max(1, int(os.getenv("MEDAI_LABELCRITIC_GRADE_CONCURRENCY", "2")))
        compare_batch_enabled = os.getenv("MEDAI_LABELCRITIC_COMPARE_BATCH", "1").strip().lower() not in {"0", "false", "no"}
        stage_counts["labelcritic_grade_batch_concurrency"] = grade_batch_concurrency
        stage_counts["labelcritic_compare_batch_enabled"] = compare_batch_enabled

        case_models = list(case_execution_plan.get("teacher_run_list", [])) if teacher_inference_mode == "full_volume" else []
        if checkpoint_map_models and candidate_mode == "formal_full_legacy":
            mapped = candidate_models_for_organs(registry, organs)
            for organ_models in mapped.values():
                for m in organ_models:
                    if m not in case_models:
                        case_models.append(m)

        model_seg_dirs: dict[str, Path] = {}
        resolved_preseeded_seg_dirs: dict[str, Path] = {}
        hierarchy_blocked: list[dict[str, Any]] = []
        hierarchy_models_used: list[str] = []

        # Inject preseeded predictions (e.g. student from previous round) directly
        # into model_seg_dirs without running inference.
        if preseeded_model_dirs:
            for seed_key, seed_base in preseeded_model_dirs.items():
                seed_seg = _resolve_preseeded_case_dir(Path(seed_base), case_id)
                if seed_seg:
                    resolved_preseeded_seg_dirs[seed_key] = seed_seg
                    if not preseeded_parent_only:
                        model_seg_dirs[seed_key] = seed_seg
                    print(f"[{_time.strftime('%H:%M:%S')}]   [preseeded] {seed_key} ✓ ({sum(1 for _ in seed_seg.glob('*.nii.gz'))} masks)", flush=True)
        # A resolved preseed is the authoritative cached output for this run;
        # never launch the same teacher again.
        case_models = [
            model_key for model_key in case_models
            if model_key not in resolved_preseeded_seg_dirs
        ]

        if teacher_inference_mode == "hierarchical_roi" and not reuse_preseeded_only:
            _teacher_t0 = _time.time()
            hierarchy_result = _run_hierarchical_case_inference(
                ct=ct, case_id=case_id, case_raw=case_raw, case_out=case_out,
                registry_path=registry_path, execution_plan=case_execution_plan,
                requested_organs=fov_organs, taxonomy=taxonomy, alias_config=alias_config,
                timeout_sec=timeout_sec, device=device, dry_run=dry_run, margin_mm=roi_margin_mm,
                preseeded_seg_dirs=resolved_preseeded_seg_dirs,
            )
            model_seg_dirs.update(hierarchy_result["model_seg_dirs"])
            inference_results.extend(hierarchy_result["inference_results"])
            hierarchy_blocked = list(hierarchy_result["blocked"])
            hierarchy_models_used = sorted({
                str(item.get("model_key")) for item in hierarchy_result["inference_results"] if item.get("model_key")
            })
            for item in hierarchy_blocked:
                _append_jsonl(review_queue, {"case_id": case_id, "ct_path": str(ct), **item})
            stage_timing["teacher_inference_sec"] += _time.time() - _teacher_t0
        elif teacher_inference_mode == "hierarchical_roi" and reuse_preseeded_only:
            if not resolved_preseeded_seg_dirs:
                raise RuntimeError(
                    f"Preseeded-only replay requested for {case_id}, but no cached "
                    "teacher/student segmentation directories were resolved."
                )
            stage_counts["hierarchical_inference_skipped_preseeded_only"] = True

        for model_idx, model_key in enumerate(case_models, start=1):
            _teacher_t0 = _time.time()
            print(f"[{_time.strftime('%H:%M:%S')}]   [{model_idx}/{len(case_models)}] {model_key}...", flush=True)
            cached_root = case_raw / model_key / case_id
            cached_summary = cached_root / "inference_summary.json"
            cached_seg_dir = cached_root / "segmentations"
            infer = None
            if resume and not dry_run:
                infer = _load_reusable_raw_inference_cache(
                    cached_summary=cached_summary,
                    cached_seg_dir=cached_seg_dir,
                    model_key=model_key,
                )
            if infer is not None:
                cache_label = "cached-empty" if infer.get("num_masks", 0) == 0 else "cached"
                print(f"[{_time.strftime('%H:%M:%S')}]   [{model_idx}/{len(case_models)}] {model_key} {cache_label} ({infer.get('num_masks',0)} masks)", flush=True)
            else:
                infer = run_registered_model(
                    ct,
                    case_raw / model_key,
                    model_key,
                    registry_path=registry_path,
                    case_id=case_id,
                    dry_run=dry_run,
                    timeout_sec=timeout_sec,
                    device=device,
                    extra_context={"requested_organs": organs},
                )
            inference_results.append({"case_id": case_id, **infer})
            seg_dir = Path(infer.get("segmentation_output", case_raw / model_key / case_id / "segmentations"))
            actual_num_masks = sum(1 for _ in seg_dir.glob("*.nii.gz")) if seg_dir.exists() else int(infer.get("num_masks", 0) or 0)
            infer["num_masks"] = actual_num_masks
            if infer.get("status") in {"success", "dry_run"}:
                model_seg_dirs[model_key] = seg_dir
                print(f"[{_time.strftime('%H:%M:%S')}]   [{model_idx}/{len(case_models)}] {model_key} ✓ ({actual_num_masks} masks)", flush=True)
            else:
                print(f"[{_time.strftime('%H:%M:%S')}]   [{model_idx}/{len(case_models)}] {model_key} ✗ ({infer.get('status')})", flush=True)
            stage_timing["teacher_inference_sec"] += _time.time() - _teacher_t0

        if not dry_run:
            presence_context = _augment_presence_from_model_landmarks(
                presence_context, model_seg_dirs, case_out
            )

        _shapekit_t0 = _time.time()
        candidate_model_seg_dirs, candidate_shapekit_reports = _postprocess_candidate_models_with_shapekit(
            model_seg_dirs=model_seg_dirs,
            case_refined=case_refined,
            shapekit_root=shapekit_root,
            case_id=case_id,
            enable_shapekit=enable_shapekit,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )
        stage_timing["candidate_shapekit_sec"] += _time.time() - _shapekit_t0

        checked = accepted = low_dice = uncertain = updated = critic_count = 0
        selected_input_root = case_out / "selected_after_candidate_shapekit"
        selected_case_root = selected_input_root / case_id
        selected_seg_dir = selected_case_root / "segmentations"
        if selected_seg_dir.exists() and not dry_run:
            shutil.rmtree(selected_seg_dir)
        if case_updated.exists() and not dry_run:
            shutil.rmtree(case_updated)
        case_updated.mkdir(parents=True, exist_ok=True)
        selected_metadata: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        pending_organ_decisions: list[dict[str, Any]] = []
        ordered_organs = topological_order_organs(taxonomy, list(fov_organs), strict=False)

        for organ in ordered_organs:
            current_ref = _mask_path(ref_dir, organ) if ref_dir else None
            current_ref_exists = bool(current_ref and (not dry_run) and current_ref.exists())
            organ_rows: list[dict[str, Any]] = []
            candidates: list[dict[str, Any]] = []

            route_info = (case_execution_plan.get("per_organ", {}) or {}).get(organ, {})
            eligible_teachers = set(route_info.get("eligible_teachers", []))
            preseeded_keys = set() if preseeded_parent_only else set((preseeded_model_dirs or {}).keys())
            # Round2+ student predictions are diagnostic-only.  Even if a
            # legacy caller passes ``student_prev`` in preseeded_model_dirs,
            # do not allow it to become an E-step candidate.
            preseeded_keys = {
                key for key in preseeded_keys
                if key != "student_prev" and "student" not in key.lower()
            }
            if tracker and not tracker.should_run_all(organ):
                top_models = tracker.get_top_k_models(organ, k=2)
                organ_model_seg_dirs = {
                    k: v for k, v in candidate_model_seg_dirs.items()
                    if (k in eligible_teachers or k in preseeded_keys)
                    and (
                        k in top_models
                        or k.replace("_shapekit", "") in top_models
                        or k in preseeded_keys
                    )
                }
                if not organ_model_seg_dirs:
                    organ_model_seg_dirs = {
                        k: v for k, v in candidate_model_seg_dirs.items()
                        if k in eligible_teachers or k in preseeded_keys
                    }
            else:
                organ_model_seg_dirs = {
                    k: v for k, v in candidate_model_seg_dirs.items()
                    if k in eligible_teachers or k in preseeded_keys
                }

            # Round 2+ remains teacher-cache selection.  The previous selected
            # mask is a replay/reference candidate, but it must not collapse the
            # pool into a student-vs-previous challenger match.  If there are
            # teacher candidates, keep them; if not, carry the previous selected
            # mask forward as historical replay.
            if (
                candidate_mode == "route_pruned_with_competition"
                and "round_prev_selected" in organ_model_seg_dirs
            ):
                previous_dir = organ_model_seg_dirs["round_prev_selected"]
                previous_mask = _mask_path(previous_dir, organ)
                if previous_mask.exists():
                    teacher_dirs = {
                        key: value
                        for key, value in organ_model_seg_dirs.items()
                        if key != "round_prev_selected" and "student" not in key.lower()
                    }
                    organ_model_seg_dirs = {
                        **teacher_dirs,
                        "round_prev_selected": previous_dir,
                    } if teacher_dirs else {"round_prev_selected": previous_dir}

            _qc_t0 = _time.time()
            organ_worker_count = max(1, int(os.getenv("MEDAI_ORGAN_WORKER_COUNT", "4")))
            organ_items = list(organ_model_seg_dirs.items())
            evaluated_rows: list[dict[str, Any]] = []
            if organ_worker_count > 1 and len(organ_items) > 1:
                future_to_index = {}
                with ThreadPoolExecutor(max_workers=min(organ_worker_count, len(organ_items))) as pool:
                    for item_idx, (model_key, seg_dir) in enumerate(organ_items):
                        raw_seg_dir = model_seg_dirs.get(model_key, seg_dir)
                        future = pool.submit(
                            _evaluate_candidate_for_organ,
                            ct=ct,
                            organ=organ,
                            model_key=model_key,
                            seg_dir=seg_dir,
                            raw_seg_dir=raw_seg_dir,
                            alias_config=alias_config,
                            shapekit_report=candidate_shapekit_reports.get(model_key, {}),
                            current_ref=current_ref,
                            current_ref_exists=current_ref_exists,
                        vlm_threshold=vlm_threshold,
                        case_id=case_id,
                        mask_cache=mask_cache,
                        taxonomy=taxonomy,
                        presence_context=presence_context,
                    )
                        future_to_index[future] = item_idx
                    ordered: dict[int, dict[str, Any]] = {}
                    for future in as_completed(future_to_index):
                        ordered[future_to_index[future]] = future.result()
                    evaluated_rows = [ordered[i] for i in range(len(organ_items)) if i in ordered]
            else:
                for model_key, seg_dir in organ_items:
                    raw_seg_dir = model_seg_dirs.get(model_key, seg_dir)
                    evaluated_rows.append(_evaluate_candidate_for_organ(
                        ct=ct,
                        organ=organ,
                        model_key=model_key,
                        seg_dir=seg_dir,
                        raw_seg_dir=raw_seg_dir,
                        alias_config=alias_config,
                        shapekit_report=candidate_shapekit_reports.get(model_key, {}),
                        current_ref=current_ref,
                        current_ref_exists=current_ref_exists,
                        vlm_threshold=vlm_threshold,
                        case_id=case_id,
                        mask_cache=mask_cache,
                        taxonomy=taxonomy,
                        presence_context=presence_context,
                    ))
            stage_timing["candidate_qc_verify_sec"] += _time.time() - _qc_t0
            stage_counts["candidate_qc_verify_count"] += len(evaluated_rows)

            for row in evaluated_rows:
                dice_rows.append(row)
                organ_rows.append(row)
                if row.get("candidate_exists") and row.get("identity_status") == "valid":
                    candidates.append(row)

            # v2 model reliability is updated only from a leave-one-evidence-
            # family-out consensus. Candidate-vs-prior pseudo Dice is never
            # written to the ranking tracker.
            if tracker and not dry_run and candidates:
                for observation in leave_one_family_out_observations(candidates, ct, registry=registry):
                    if observation.get("status") == "success":
                        tracker.update_leave_one_family_out(
                            organ=organ,
                            organ_family=str((taxonomy_entry(taxonomy, organ) or {}).get("comparison_family") or "unknown"),
                            model=str(observation["model"]),
                            evidence_family=str(observation["evidence_family"]),
                            dice=float(observation["dice"]),
                            nsd=float(observation["nsd"]),
                            ct_support=float(observation["ct_support"]),
                            anatomy_plausibility=float(observation["anatomy_plausibility"]),
                            other_family_count=int(observation["other_family_count"]),
                            case_id=case_id,
                        )
                    else:
                        tracker.record_insufficient(
                            organ=organ,
                            model=str(observation.get("model")),
                            evidence_family=str(observation.get("evidence_family")),
                            other_family_count=int(observation.get("other_family_count") or 0),
                            case_id=case_id,
                            reason=str(observation.get("status")),
                        )
                for candidate in candidates:
                    candidate["estimated_model_reliability"] = tracker.get_estimated_reliability(
                        organ,
                        str(candidate.get("model")),
                        str((taxonomy_entry(taxonomy, organ) or {}).get("comparison_family") or "unknown"),
                    )

            # Multi-teacher consensus fusion: when >=2 eligible teacher
            # candidates exist for this organ, fuse them (STAPLE / reliability-
            # weighted vote) into a consensus mask that then competes as its own
            # candidate. This denoises individual teacher errors automatically.
            if enable_fusion and not dry_run:
                fuse_inputs = [
                    c for c in candidates
                    if c.get("candidate_exists") and c.get("eligible_for_labelcritic", True) and not c.get("is_fusion")
                ]
                should_fuse, fusion_gate = _should_fuse_candidates(
                    fuse_inputs,
                    cache=mask_cache,
                    high_agreement_dice=float(os.getenv("MEDAI_FUSION_HIGH_AGREEMENT_DICE", "0.92")),
                )
                if len(fuse_inputs) >= 2 and not should_fuse:
                    stage_counts["fusion_skipped_high_agreement_count"] = stage_counts.get("fusion_skipped_high_agreement_count", 0) + 1
                    for c in fuse_inputs:
                        c["fusion_gate"] = fusion_gate
                        c["fusion_skipped_reason"] = fusion_gate.get("reason")
                if len(fuse_inputs) >= 2 and should_fuse:
                    fused_path = case_refined / "fusion" / case_id / "segmentations" / f"{organ}.nii.gz"
                    print(f"[{_time.strftime('%H:%M:%S')}]     fusion {organ}: {len(fuse_inputs)} candidates via {fusion_method}", flush=True)
                    _fusion_t0 = _time.time()
                    fusion_meta = fuse_candidate_masks(
                        [c["prediction"] for c in fuse_inputs],
                        fused_path,
                        weights=[_fusion_weight(tracker, organ, c["model"]) for c in fuse_inputs],
                        reference_image=ct,
                        method=fusion_method,
                        vote_threshold=float(os.getenv("MEDAI_FUSION_VOTE_THRESHOLD", "0.5")),
                    )
                    stage_timing["fusion_sec"] += _time.time() - _fusion_t0
                    stage_counts["fusion_count"] += 1
                    if fusion_meta.get("status") == "success" and fused_path.exists():
                        _fused_qc_t0 = _time.time()
                        fused_qc = _compute_candidate_qc(
                            ct=ct, mask=fused_path, organ=organ,
                            reference=current_ref if current_ref_exists else None,
                            cache=mask_cache,
                        )
                        fv = _verify_annotation_cached(
                            current_ref if current_ref_exists else None, fused_path, organ,
                            dsc_replace_threshold=0.0, dsc_vlm_threshold=vlm_threshold,
                            cache=mask_cache,
                        )
                        stage_timing["candidate_qc_verify_sec"] += _time.time() - _fused_qc_t0
                        stage_counts["candidate_qc_verify_count"] += 1
                        fused_row = {
                            "case_id": case_id,
                            "organ": organ,
                            "model": "fusion_consensus",
                            "prediction": str(fused_path),
                            "pre_shapekit_prediction": str(fused_path),
                            "reference": str(current_ref) if current_ref else "",
                            "reference_role": "prior_or_selected_pseudo_reference" if current_ref_exists else "none",
                            "metric_family": "pseudo_consistency",
                            "metric_scope": "candidate_vs_prior_or_selected_pseudo_reference",
                            "metric_target": "pseudo-label",
                            "metric_subject": "E-step output",
                            "metric_comparison": "candidate_vs_prior_or_selected_pseudo_reference",
                            "metric_interpretation": "pseudo_label_consistency",
                            "ground_truth_status": "pseudo_label_candidate",
                            "accuracy_warning": "Dice is pseudo-label consistency, not true expert-label accuracy.",
                            "dice": fv.get("dice"),
                            "pseudo_consistency_dice": fv.get("dice"),
                            "decision": fv.get("decision"),
                            "status": fv.get("status"),
                            "reason": fv.get("reason"),
                            "reference_quality_bucket": fv.get("quality_bucket"),
                            "candidate_exists": True,
                            "alias_match": "fusion_consensus",
                            "candidate_shapekit_status": "fusion_consensus",
                            "candidate_shapekit_reason": f"{fusion_meta.get('method')} of {fusion_meta.get('n_inputs')} candidates",
                            "candidate_shapekit_report": {},
                            "candidate_qc": fused_qc,
                            "candidate_qc_status": fused_qc.get("status"),
                            "candidate_qc_score": fused_qc.get("score"),
                            "candidate_qc_flags": fused_qc.get("flags", []),
                            "eligible_for_labelcritic": fused_qc.get("eligible_for_labelcritic", True),
                            "is_fusion": True,
                            "fusion_method": fusion_meta.get("method"),
                            "fusion_status": fusion_meta.get("status"),
                            "fusion_inputs": [c["model"] for c in fuse_inputs],
                            "fusion_weights": fusion_meta.get("weights"),
                            "fusion_vote_threshold": float(os.getenv("MEDAI_FUSION_VOTE_THRESHOLD", "0.5")),
                            "fusion_gate": fusion_gate,
                            **identity_contract(
                                taxonomy,
                                organ,
                                "fusion_consensus",
                                organ,
                                mapping_type="exact_synonym",
                                mapping_source="same_canonical_candidate_fusion",
                            ),
                        }
                        dice_rows.append(fused_row)
                        if fused_row.get("candidate_qc_status") == "pass" and fused_row.get("eligible_for_labelcritic", True):
                            # Ablation-only artifact. A fused mask must never enter
                            # the formal LabelCritic candidate set.
                            fused_row["fusion_ablation_only"] = True
                        else:
                            fused_row["fusion_rejected"] = True
                            fused_row["fusion_rejected_reason"] = "fusion_qc_not_pass_or_ineligible"
                            stage_counts["fusion_rejected_count"] = stage_counts.get("fusion_rejected_count", 0) + 1
                            dice_rows[-1] = fused_row

            _select_t0 = _time.time()
            for candidate in candidates:
                candidate["candidate_id"] = _candidate_id(case_id, organ, candidate)
                candidate["mask_sha256"] = _sha256_file(candidate.get("prediction"))
            selected, selection = _select_candidate(
                ct=ct,
                organ=organ,
                candidates=candidates,
                out=out,
                case_id=case_id,
                enable_critic=enable_critic,
                critic_backend=critic_backend,
                critic_base_url=critic_base_url,
                critic_port=critic_port,
                timeout_sec=timeout_sec,
                dry_run=dry_run,
                labelcritic_options=labelcritic_options,
                mask_cache=mask_cache,
                compare_batch_enabled=compare_batch_enabled,
                compare_batch_max_candidates=int(os.getenv("MEDAI_LABELCRITIC_COMPARE_BATCH_MAX_CANDIDATES", "2")),
                strict_labelcritic_selection=strict_labelcritic_selection,
            )
            stage_timing["labelcritic_compare_sec"] += _time.time() - _select_t0
            stage_counts["labelcritic_compare_records"] += len(selection.get("critic_records", []) or [])
            critic_count += len(selection.get("critic_records", []) or [])
            best_dice = selected.get("dice") if selected else None
            selected_reference_quality_bucket = selected.get("reference_quality_bucket") if selected else None
            empty_reference_nonempty_prediction = selected_reference_quality_bucket == "empty_reference_nonempty_prediction"
            if best_dice is not None and float(best_dice) < vlm_threshold and not empty_reference_nonempty_prediction:
                low_dice += 1

            # Phase 2 — automated absolute-quality arbitration (de-human).
            # LC-2 is preserved as an absolute-quality gate, but risk-aware mode
            # avoids sending stable single-candidate QC-pass organs through a
            # costly VLM projection call with no extra conflict evidence.
            auto_grade_record = None
            grade_policy_decision = _labelcritic_grade_policy_decision(
                organ=organ,
                selected=selected,
                candidates=candidates,
                selection=selection,
                route_info=route_info,
                best_dice=best_dice,
                vlm_threshold=vlm_threshold,
                empty_reference_nonempty_prediction=empty_reference_nonempty_prediction,
                policy=grade_policy,
            )
            should_run_grade = bool(
                enable_auto_arbitration
                and enable_critic
                and not dry_run
                and selected
                and grade_policy_decision.get("run")
            )
            compare_used = bool(selection.get("labelcritic_records") or selection.get("critic_records"))
            compare_reason = None
            compare_skipped_reason = None
            if compare_used:
                compare_reason = "multi_candidate_conflict"
            else:
                if len(candidates) <= 1:
                    compare_skipped_reason = "single_candidate_or_route_unique"
                elif selection.get("selection_method") in {"near_identical_agreement", "geometric_teacher_consensus"}:
                    compare_skipped_reason = "high_agreement"
                else:
                    compare_skipped_reason = "critic_disabled_or_not_needed"

            selection_record = {
                "case_id": case_id,
                "ct_path": str(ct),
                "organ": organ,
                "prompt": organ_prompts.get(organ, organ.replace("_", " ")),
                "route_primary_teacher": route_info.get("primary_teacher"),
                "route_backup_teachers": route_info.get("backup_teachers", []),
                "route_competition_teachers": route_info.get("competition_teachers", []),
                "route_confidence": route_info.get("route_confidence", "low"),
                "candidate_mode": candidate_mode,
                "labelcritic_compare_used": compare_used,
                "labelcritic_called": compare_used,
                "labelcritic_compare_reason": compare_reason,
                "labelcritic_compare_skipped_reason": compare_skipped_reason,
                "candidate_models": [c["model"] for c in candidates],
                "candidate_predictions": [
                    {
                        "model": c["model"],
                        "candidate_id": c.get("candidate_id"),
                        "mask_sha256": c.get("mask_sha256"),
                        "prediction": c["prediction"],
                        "pre_shapekit_prediction": c.get("pre_shapekit_prediction"),
                        "dice": c.get("dice"),
                        "status": c.get("status"),
                        "reason": c.get("reason"),
                        "reference_quality_bucket": c.get("reference_quality_bucket"),
                        "candidate_shapekit_status": c.get("candidate_shapekit_status"),
                        "candidate_shapekit_reason": c.get("candidate_shapekit_reason"),
                        "candidate_qc_status": c.get("candidate_qc_status"),
                        "candidate_qc_score": c.get("candidate_qc_score"),
                        "candidate_qc_flags": c.get("candidate_qc_flags", []),
                        "eligible_for_labelcritic": c.get("eligible_for_labelcritic", True),
                        "fusion_gate": c.get("fusion_gate"),
                        "fusion_skipped_reason": c.get("fusion_skipped_reason"),
                        "requested_canonical_id": c.get("requested_canonical_id"),
                        "source_local_label": c.get("source_local_label"),
                        "resolved_canonical_id": c.get("resolved_canonical_id"),
                        "comparison_family": c.get("comparison_family"),
                        "parent_ids": c.get("parent_ids", []),
                        "mapping_type": c.get("mapping_type"),
                        "mapping_source": c.get("mapping_source"),
                        "identity_status": c.get("identity_status"),
                        "identity_mismatch_reasons": c.get("identity_mismatch_reasons", []),
                    }
                    for c in candidates
                ],
                "candidate_count": len(candidates),
                "reference": str(current_ref) if current_ref else "",
                "reference_role": "historical_pseudo_label" if current_ref_exists else "none",
                "reference_provenance": _historical_reference_provenance(current_ref) if current_ref_exists else None,
                "selected_model": selection.get("selected_model"),
                "source_model": selection.get("selected_model"),
                "selected_prediction": selected.get("prediction") if selected else None,
                "selected_pre_shapekit_prediction": selected.get("pre_shapekit_prediction") if selected else None,
                "selected_candidate_shapekit_status": selected.get("candidate_shapekit_status") if selected else None,
                "selected_candidate_shapekit_reason": selected.get("candidate_shapekit_reason") if selected else None,
                "selected_candidate_qc_status": selected.get("candidate_qc_status") if selected else None,
                "selected_candidate_qc_score": selected.get("candidate_qc_score") if selected else None,
                "selected_candidate_qc_flags": selected.get("candidate_qc_flags", []) if selected else [],
                "selected_candidate_qc_checks": selected.get("candidate_qc", {}) if selected else {},
                "selected_reference_quality_bucket": selected_reference_quality_bucket,
                "selected_dice": best_dice,
                "selected_pseudo_consistency_dice": best_dice,
                "expected_presence": selected.get("expected_presence") if selected else _expected_presence_for_organ(organ, presence_context),
                "fov_status": selected.get("fov_status") if selected else _fov_status_for_organ(organ, presence_context),
                "requested_canonical_id": selected.get("requested_canonical_id") if selected else organ,
                "source_local_label": selected.get("source_local_label") if selected else None,
                "resolved_canonical_id": selected.get("resolved_canonical_id") if selected else None,
                "comparison_family": selected.get("comparison_family") if selected else (taxonomy_entry(taxonomy, organ) or {}).get("comparison_family"),
                "parent_ids": selected.get("parent_ids", []) if selected else (taxonomy_entry(taxonomy, organ) or {}).get("parent_ids", []),
                "mapping_type": selected.get("mapping_type") if selected else None,
                "mapping_source": selected.get("mapping_source") if selected else None,
                "identity_status": selected.get("identity_status") if selected else "missing",
                "identity_mismatch_reasons": selected.get("identity_mismatch_reasons", []) if selected else ["no_selected_candidate"],
                "metric_family": "pseudo_consistency",
                "metric_scope": "selected_candidate_vs_prior_or_selected_pseudo_reference",
                "metric_target": "pseudo-label",
                "metric_subject": "E-step output",
                "metric_comparison": "selected_candidate_vs_prior_or_selected_pseudo_reference",
                "metric_interpretation": "pseudo_label_consistency",
                "accuracy_warning": "Selected Dice is pseudo-label consistency, not true expert-label accuracy.",
                "comparison_input_stage": "post_shapekit_candidate" if enable_shapekit and not dry_run else "raw_candidate",
                "dataset_type": "pseudo_label_dataset",
                "ground_truth_status": "pseudo_label_candidate",
                **selection,
            }
            selection_record["labelcritic_records"] = selection_record.get("critic_records", [])
            selection_record["labelcritic_decision_path"] = _labelcritic_decision_path(selection_record["labelcritic_records"])
            selection_record["labelcritic_decisive"] = _labelcritic_locks_selection(selection_record)
            selection_record["primary_selector"] = "labelcritic"
            selection_record["evidence_used_for"] = "audit_only" if selection_record["labelcritic_decisive"] else "fallback_selection"
            selection_record["selected_candidate"] = selection_record.get("selected_model")
            selection_record["selected_candidate_id"] = selected.get("candidate_id") if selected else None
            selection_record["candidate_ids"] = [c.get("candidate_id") for c in candidates if c.get("candidate_id")]
            selection_record["teacher_names"] = [c.get("model") for c in candidates if c.get("model")]
            selection_record["teacher_families"] = [
                c.get("evidence_family") or c.get("comparison_family")
                for c in candidates
                if c.get("evidence_family") or c.get("comparison_family")
            ]
            selection_record["record_type"] = "candidate_pseudo" if selected else "unresolved_review"
            selection_record["selected_source_mask_sha256"] = selected.get("mask_sha256") if selected else None
            selection_record["selected_teacher"] = selection_record.get("selected_model")
            selection_record["selected_family"] = (selected or {}).get("evidence_family") if selected else None
            selection_record["labelcritic_prompt_version"] = selection_record.get("labelcritic_prompt_source") or selection_record.get("prompt_source")
            selection_record.setdefault("selected_reason", selection_record.get("fallback_reason") or selection_record.get("selection_method"))
            selection_record.setdefault("rejected_reasons", {})
            selection_record.setdefault("failure_modes", [])
            selection_record.setdefault("should_enter_student_training", selection_record.get("selection_status") == "selected")
            prompt_sources = [r.get("prompt_source") for r in selection_record["labelcritic_records"] if r.get("prompt_source")]
            if prompt_sources:
                selection_record["labelcritic_prompt_source"] = prompt_sources[0]
                selection_record["prompt_source"] = prompt_sources[0]
            prompt_hashes = [
                r.get("rendered_organ_prompt_hash")
                for r in selection_record["labelcritic_records"]
                if r.get("rendered_organ_prompt_hash")
            ]
            selection_record["labelcritic_prompt_version"] = (
                prompt_hashes[0] if prompt_hashes
                else selection_record.get("labelcritic_prompt_source")
                or selection_record.get("prompt_source")
            )
            review_flags = [r.get("requires_manual_review") for r in selection_record["labelcritic_records"] if "requires_manual_review" in r]
            if review_flags:
                selection_record["requires_manual_review"] = any(bool(x) for x in review_flags)
            else:
                selection_record.setdefault("requires_manual_review", False)
            selection_record["label_critic_decision_path"] = selection_record["labelcritic_decision_path"]
            selection_record["labelcritic_grade_policy"] = grade_policy
            selection_record["labelcritic_grade_used"] = bool(should_run_grade)
            selection_record["labelcritic_grade_reason"] = grade_policy_decision.get("reason") if should_run_grade else None
            selection_record["labelcritic_grade_skipped_reason"] = None if should_run_grade else grade_policy_decision.get("skipped_reason", "grade_not_required_by_policy")
            pending_organ_decisions.append({
                "organ": organ,
                "selected": selected,
                "candidates": candidates,
                "selection": selection,
                "selection_record": selection_record,
                "should_run_grade": should_run_grade,
                "grade_policy_decision": grade_policy_decision,
                "best_dice": best_dice,
                "selected_reference_quality_bucket": selected_reference_quality_bucket,
                "empty_reference_nonempty_prediction": empty_reference_nonempty_prediction,
            })

        grade_cache: dict[str, dict[str, Any]] = {}
        grade_jobs: list[dict[str, Any]] = []
        for decision in pending_organ_decisions:
            selected = decision.get("selected")
            if not (decision.get("should_run_grade") and selected):
                continue
            organ = str(decision["organ"])
            out_json = out / "critic" / case_id / f"{organ}_{selected['model']}_grade.json"
            grade_jobs.append({
                "ct_image": ct,
                "mask": Path(selected["prediction"]),
                "organ": organ,
                "output_json": out_json,
                "accept_grade": arbitration_accept_grade,
            })
        grade_fn_patched = getattr(run_labelcritic_grade, "__module__", "") != "cli_anything.medai.core.labelcritic_wrapper"
        if grade_jobs and not grade_fn_patched:
            _grade_t0 = _time.time()
            grade_results = run_labelcritic_grade_batch(
                grade_jobs,
                base_url=critic_base_url,
                port=critic_port,
                vlm_model=vlm_model,
                accept_grade=arbitration_accept_grade,
                dry_run=False,
                timeout_sec=min(timeout_sec, 300),
                concurrency=grade_batch_concurrency,
            )
            stage_timing["labelcritic_grade_sec"] += _time.time() - _grade_t0
            stage_counts["labelcritic_grade_count"] += len(grade_jobs)
            for result in grade_results:
                out_json = result.get("output_json")
                if out_json:
                    grade_cache[str(Path(out_json).resolve())] = result

        for decision in pending_organ_decisions:
            organ = str(decision["organ"])
            selected = decision.get("selected")
            candidates = decision["candidates"]
            selection = decision["selection"]
            selection_record = decision["selection_record"]
            best_dice = decision["best_dice"]
            selected_reference_quality_bucket = decision["selected_reference_quality_bucket"]
            empty_reference_nonempty_prediction = decision["empty_reference_nonempty_prediction"]
            auto_grade_record = None

            if decision.get("should_run_grade") and selected:
                auto_grade_record = _auto_arbitrate_organ(
                    ct=ct, organ=organ, selected=selected, candidates=candidates,
                    out=out, case_id=case_id, base_url=critic_base_url, port=critic_port,
                    vlm_model=vlm_model, accept_grade=arbitration_accept_grade,
                    reject_grade=arbitration_reject_grade, timeout_sec=timeout_sec,
                    grade_cache=grade_cache,
                )
                chosen = auto_grade_record.get("_selected")
                if chosen is not None and chosen is not selected:
                    selected = chosen
                    best_dice = selected.get("dice")
                    selected_reference_quality_bucket = selected.get("reference_quality_bucket")
                    empty_reference_nonempty_prediction = selected_reference_quality_bucket == "empty_reference_nonempty_prediction"
                    selection_record["selected_model"] = selected.get("model")
                    selection_record["selected_candidate_id"] = selected.get("candidate_id")
                    selection_record["selected_source_mask_sha256"] = selected.get("mask_sha256")
                    selection_record["source_model"] = selected.get("model")
                    selection_record["selected_prediction"] = selected.get("prediction")
                    selection_record["selected_pre_shapekit_prediction"] = selected.get("pre_shapekit_prediction")
                    selection_record["selected_candidate_shapekit_status"] = selected.get("candidate_shapekit_status")
                    selection_record["selected_candidate_shapekit_reason"] = selected.get("candidate_shapekit_reason")
                    selection_record["selected_candidate_qc_status"] = selected.get("candidate_qc_status")
                    selection_record["selected_candidate_qc_score"] = selected.get("candidate_qc_score")
                    selection_record["selected_candidate_qc_flags"] = selected.get("candidate_qc_flags", [])
                    selection_record["selected_candidate_qc_checks"] = selected.get("candidate_qc", {})
                    selection_record["selected_reference_quality_bucket"] = selected_reference_quality_bucket
                    selection_record["selected_dice"] = best_dice
                    selection_record["selected_pseudo_consistency_dice"] = best_dice
                    for identity_field in (
                        "requested_canonical_id", "source_local_label", "resolved_canonical_id",
                        "comparison_family", "parent_ids", "mapping_type", "mapping_source",
                        "identity_status", "identity_mismatch_reasons",
                    ):
                        selection_record[identity_field] = selected.get(identity_field)
                _append_jsonl(out / "auto_arbitration_log.jsonl", {k: v for k, v in auto_grade_record.items() if not k.startswith("_")})

            if auto_grade_record is not None:
                arb = {k: v for k, v in auto_grade_record.items() if not k.startswith("_")}
                selection_record["auto_arbitration"] = arb
                selection_record["auto_grade"] = arb.get("grade_after")
                selection_record["auto_grade_accept"] = arb.get("final_accept")
                selection_record["auto_grade_swapped"] = arb.get("swapped")
                selection_record["labelcritic_called"] = True

            # LabelCritic is the primary 373-target candidate selector.
            # AutoLabelCore/family evidence is retained for audit, grading, review
            # priority, and fallback only; it must not overwrite a decisive
            # LabelCritic tournament winner.
            labelcritic_locked_selection = _labelcritic_locks_selection(selection_record)
            critic_records = selection_record.get("labelcritic_records", []) or []
            successful_critic = [r for r in critic_records if r.get("status") == "success"]
            selected_wins = 0
            selected_losses = 0
            for record in successful_critic:
                winner = (record.get("decision") or {}).get("winner")
                winner_model = record.get("candidate_a") if winner == "a" else record.get("candidate_b") if winner == "b" else None
                if winner_model == (selected or {}).get("model"):
                    selected_wins += 1
                elif winner_model:
                    selected_losses += 1
            lc_adjustment = 0.0
            if selected_wins > selected_losses:
                lc_adjustment = 0.03
            elif selected_losses > selected_wins:
                lc_adjustment = -0.03
            labelcritic_calibrated_score = None
            if auto_grade_record is not None:
                try:
                    labelcritic_calibrated_score = float(auto_grade_record.get("grade_after"))
                except Exception:
                    labelcritic_calibrated_score = None
            prior_path = current_ref if current_ref_exists else None
            prior_candidate = next(
                (
                    c for c in candidates
                    if str(c.get("model")) in {"round_prev_selected", "previous_round_selected"}
                    and c.get("prediction") and Path(c["prediction"]).exists()
                ),
                None,
            )
            if prior_candidate:
                prior_path = Path(prior_candidate["prediction"])
            autolabel_decision = score_candidate_set(
                case_id=case_id,
                organ=organ,
                ct_path=ct,
                candidates=candidates,
                prior_round=prior_path,
                labelcritic_result={
                    "supported": bool(successful_critic),
                    "tiebreak_adjustment": lc_adjustment,
                    "calibrated_score": labelcritic_calibrated_score,
                },
                selected_model=(selected or {}).get("model"),
                output_dir=out / "autolabel_core" / case_id / organ,
                registry=registry,
                parent_masks={
                    parent: selected_case_root / f"{parent}.nii.gz"
                    for parent in ((taxonomy_entry(taxonomy, organ) or {}).get("parent_ids", []) or [])
                    if (selected_case_root / f"{parent}.nii.gz").exists()
                },
            )
            autolabel_record = autolabel_decision.to_dict()
            selection_record.update(autolabel_record)
            if strict_labelcritic_selection:
                # AutoLabelCore is audit-only in the formal path. Its serialized
                # decision must not overwrite the strict LabelCritic selection,
                # including the explicit no-selection/review state.
                selection_record["selected_model"] = selected.get("model") if selected else None
                selection_record["source_model"] = selected.get("model") if selected else None
                selection_record["selected_prediction"] = selected.get("prediction") if selected else None
                selection_record["selected_candidate_id"] = selected.get("candidate_id") if selected else None
                selection_record["selected_source_mask_sha256"] = selected.get("mask_sha256") if selected else None
                selection_record["should_enter_student_training"] = bool(
                    selected is not None and selection.get("selection_status") == "selected"
                )
                if (
                    selected is not None
                    and selection.get("selection_status") == "selected"
                    and selection.get("selection_method") == "geometric_teacher_consensus"
                ):
                    # Geometric consensus is the formal, family-free safety
                    # signal. Preserve AutoLabelCore/family output as audit
                    # fields, but do not let family-weighted reliability
                    # scoring alter training eligibility for this path.
                    selection_record["autolabel_grade_audit"] = autolabel_record.get("grade")
                    selection_record["autolabel_training_weight_audit"] = autolabel_record.get("training_weight")
                    selection_record["autolabel_evidence_confidence_audit"] = autolabel_record.get("evidence_confidence")
                    selection_record["grade"] = "A"
                    selection_record["training_weight"] = 1.0
                    selection_record["target_type"] = "hard"
                    selection_record["decision_status"] = "accepted"
                    selection_record["distillation_eligible"] = True
                    selection_record["distillation_exclusion_reason"] = None
                    selection_record["evidence_confidence"] = 1.0
                    selection_record["auto_fine_label_reliability_score"] = 1.0
                    selection_record["evidence_scores"] = {
                        **(selection_record.get("evidence_scores") or {}),
                        "geometric_teacher_consensus": 1.0,
                    }
                    selection_record["missing_evidence"] = []
                    selection_record["decision_reasons"] = [
                        "family_free_complete_link_geometric_teacher_consensus",
                        "winner_is_original_teacher_medoid",
                    ]
                    selection_record["scoring_schema_version"] = "autolabel_core_v3"
                    selection_record["ground_truth_status"] = "geometric_consensus_pseudo_label_not_expert_accuracy"
                    selection_record["accuracy_warning"] = (
                        "Geometric teacher consensus is a pseudo-label safety signal, not expert accuracy."
                    )
            selection_record["labelcritic_supported"] = bool(successful_critic)
            selection_record["labelcritic_tiebreak_adjustment"] = lc_adjustment
            selection_record["labelcritic_decisive"] = labelcritic_locked_selection
            selection_record["evidence_used_for"] = (
                "audit_only" if labelcritic_locked_selection
                else "formal_selection" if selection_record.get("selection_method") == "geometric_teacher_consensus"
                else "fallback_selection"
            )
            if labelcritic_locked_selection:
                selection_record["primary_selector"] = "labelcritic"
                selection_record["fallback_selector"] = None
                selection_record["fallback_reason"] = selection_record.get("fallback_reason")
            selection_record["case_fold"] = stable_case_fold(case_id)
            selection_record["estimated_reliability"] = autolabel_record["evidence_confidence"]
            selection_record["autolabel_candidate_scores"] = [
                {
                    "model": c.get("model"),
                    "evidence_family": c.get("evidence_family"),
                    "relative_score": c.get("autolabel_candidate_relative_score"),
                    "ct_support_score": c.get("ct_support_score"),
                    "anatomy_plausibility_score": c.get("anatomy_plausibility_score"),
                    "perturbation_stability_score": c.get("perturbation_stability_score"),
                    "structural_corruption_probability": c.get("structural_corruption_probability"),
                    "tta_plan": c.get("tta_plan"),
                }
                for c in candidates
            ]
            selection_record["metric_family"] = "pseudo_consistency_and_evidence_reliability"
            selection_record["metric_target"] = "pseudo-label"
            selection_record["metric_subject"] = "E-step output"
            selection_record["metric_comparison"] = "selected_candidate_vs_prior_or_selected_pseudo_reference"
            selection_record["metric_interpretation"] = "pseudo_label_consistency"
            selection_record["accuracy_warning"] = "AutoLabelCore evidence confidence is not expert accuracy or ground-truth DSC."
            evidence_selected = next((c for c in candidates if str(c.get("model")) == str(autolabel_decision.selected_model)), None)
            if (not strict_labelcritic_selection) and (not labelcritic_locked_selection) and evidence_selected is not None and evidence_selected is not selected:
                selected = evidence_selected
                selection_record["selected_model"] = selected.get("model")
                selection_record["source_model"] = selected.get("model")
                selection_record["selected_prediction"] = selected.get("prediction")
                selection_record["selected_pre_shapekit_prediction"] = selected.get("pre_shapekit_prediction")
                selection_record["selected_candidate_shapekit_status"] = selected.get("candidate_shapekit_status")
                selection_record["selected_candidate_shapekit_reason"] = selected.get("candidate_shapekit_reason")
                selection_record["selected_candidate_qc_status"] = selected.get("candidate_qc_status")
                selection_record["selected_candidate_qc_score"] = selected.get("candidate_qc_score")
                selection_record["selected_candidate_qc_flags"] = selected.get("candidate_qc_flags", [])
                selection_record["selected_candidate_qc_checks"] = selected.get("candidate_qc", {})
                selection_record["selected_reference_quality_bucket"] = selected.get("reference_quality_bucket")
                selection_record["selected_dice"] = selected.get("dice")
                selection_record["selected_pseudo_consistency_dice"] = selected.get("dice")
                selection_record["expected_presence"] = selected.get("expected_presence")
                selection_record["selection_method"] = "autolabel_core_evidence"
                for identity_field in (
                    "requested_canonical_id", "source_local_label", "resolved_canonical_id",
                    "comparison_family", "parent_ids", "mapping_type", "mapping_source",
                    "identity_status", "identity_mismatch_reasons",
                ):
                    selection_record[identity_field] = selected.get(identity_field)
            # AutoLabelCore fusion remains an audit artifact. It must never
            # replace a real teacher candidate or an abstention.
            selection_record["family_balanced_consensus_policy"] = (
                "audit_only_never_selected"
            )

            if labelcritic_locked_selection:
                selection_record["selected_candidate"] = selection_record.get("selected_model")
                selection_record["selected_reason"] = selection_record.get("selected_reason") or "LabelCritic decisive tournament winner retained; AutoLabelCore evidence used for audit only"
                selection_record["autolabel_selected_model_audit"] = autolabel_decision.selected_model
                selection_record["autolabel_hard_mask_path_audit"] = autolabel_decision.hard_mask_path
                selection_record["autolabel_target_type_audit"] = autolabel_decision.target_type
                selection_record["should_enter_student_training"] = selection_record.get("training_weight", 0.0) > 0.0

            selection_rows.append(selection_record)
            all_selection_rows.append(selection_record)
            for critic_record in selection_record.get("labelcritic_records", []) or []:
                _append_jsonl(vlm_decisions, {
                    "case_id": case_id,
                    "ct_path": str(ct),
                    "organ": organ,
                    "selection_method": selection_record.get("selection_method"),
                    "selected_model_after_pairwise": selection_record.get("selected_model"),
                    **critic_record,
                })

            if not selected:
                _append_jsonl(review_queue, {"case_id": case_id, "organ": organ, **selection_record})
                uncertain += 1
                continue

            _copy_t0 = _time.time()
            copied = _copy_case_mask(Path(selected["prediction"]), selected_case_root, organ)
            stage_timing["copy_final_sec"] += _time.time() - _copy_t0
            if copied:
                stage_counts["copy_final_count"] += 1
                review_flags: list[str] = list(selection_record.get("review_flags", []) or [])
                quality_flags: list[str] = list(selection_record.get("quality_flags", []) or [])
                if auto_grade_record is not None and auto_grade_record.get("final_accept") is False:
                    _add_unique(review_flags, "auto_grade_reject")
                    _add_unique(quality_flags, "auto_grade_reject")
                    _append_jsonl(review_queue, {
                        "case_id": case_id,
                        "organ": organ,
                        "reason": "automated VLM quality gate rejected the selected pseudo-label and found no better alternative; kept as low-grade candidate (no human required)",
                        **selection_record,
                        "review_flags": review_flags,
                        "quality_flags": quality_flags,
                    })
                selected_qc_status = selected.get("candidate_qc_status") if selected else None
                if selected_qc_status and selected_qc_status != "pass":
                    _add_unique(review_flags, f"candidate_qc_{selected_qc_status}")
                    _add_unique(quality_flags, f"candidate_qc_{selected_qc_status}")
                if empty_reference_nonempty_prediction:
                    _add_unique(review_flags, "empty_reference_nonempty_prediction")
                    _add_unique(quality_flags, "empty_reference_nonempty_prediction")
                    _append_jsonl(review_queue, {
                        "case_id": case_id,
                        "organ": organ,
                        "reason": "prior pseudo/reference mask is empty but selected teacher prediction is non-empty; do not treat Dice=0 as true low-quality evidence",
                        **selection_record,
                        "review_flags": review_flags,
                        "quality_flags": quality_flags,
                    })
                elif best_dice is not None and float(best_dice) < vlm_threshold:
                    _add_unique(review_flags, "low_pseudo_consistency_dice")
                    _add_unique(quality_flags, "low_pseudo_consistency_dice")
                    _append_jsonl(review_queue, {
                        "case_id": case_id,
                        "organ": organ,
                        "reason": "selected pseudo-label has low pseudo-consistency Dice against available pseudo reference",
                        **selection_record,
                        "review_flags": review_flags,
                        "quality_flags": quality_flags,
                    })
                updated += 1
                if best_dice is not None and float(best_dice) >= accept_threshold:
                    accepted += 1
                if selection.get("selection_status") != "selected":
                    uncertain += 1
                    _add_unique(review_flags, "selection_fallback")
                    _add_unique(quality_flags, "selection_fallback")
                    _append_jsonl(review_queue, {
                        "case_id": case_id,
                        "organ": organ,
                        "reason": "fallback pseudo-label selection requires review",
                        **selection_record,
                        "review_flags": review_flags,
                        "quality_flags": quality_flags,
                    })
                if any(str(flag).startswith("candidate_qc_") for flag in review_flags):
                    _append_jsonl(review_queue, {
                        "case_id": case_id,
                        "organ": organ,
                        "reason": "candidate QC flagged selection or rejected competing candidates",
                        **selection_record,
                        "selected_candidate_qc_status": selected_qc_status,
                        "selected_candidate_qc_flags": selected.get("candidate_qc_flags", []) if selected else [],
                        "comparison_candidate_models": selection_record.get("comparison_candidate_models", []),
                        "qc_rejected_candidates": selection_record.get("qc_rejected_candidates", []),
                        "review_flags": review_flags,
                        "quality_flags": quality_flags,
                    })
                _record_organ_task_state(
                    organ_task_state,
                    organ,
                    status="selected",
                    candidate_fingerprint=sorted(str(c.get("model")) for c in candidates),
                    compare_used=selection_record.get("labelcritic_compare_used"),
                    grade_used=selection_record.get("labelcritic_grade_used"),
                )
                selected_metadata.append({
                    **selection_record,
                    "teacher_lineage": [c["model"] for c in candidates],
                    "pre_shapekit_mask": str(copied),
                    "mask_path": None,
                    "mask": None,
                    "dataset_role": "pseudo_label",
                    "ground_truth_status": "pseudo_label_candidate",
                    "shapekit_status": selected.get("candidate_shapekit_status") if selected else ("skipped_dry_run" if dry_run else "skipped_debug_only"),
                    "shapekit_reason": selected.get("candidate_shapekit_reason") if selected else None,
                    "shapekit_report": selected.get("candidate_shapekit_report") if selected else None,
                    "shapekit_attempted": bool(enable_shapekit and not dry_run),
                    "final_mask": None,
                    "review_flags": review_flags,
                    "quality_flags": quality_flags,
                    "quality_status": _quality_status(review_flags, quality_flags),
                })

        shapekit_result: dict[str, Any] = {
            "stage": "candidate_preselection_shapekit",
            "status": "completed",
            "order": "ShapeKit candidates before LabelCritic selection",
            "candidate_reports": candidate_shapekit_reports,
        }

        for meta in selected_metadata:
            organ = meta["organ"]
            pre_mask = Path(meta["pre_shapekit_mask"])
            _final_copy_t0 = _time.time()
            final = _copy_annotation(pre_mask, case_updated, organ)
            stage_timing["copy_final_sec"] += _time.time() - _final_copy_t0
            meta["final_mask"] = final
            meta["mask_path"] = final
            meta["mask"] = final
            meta["final_mask_sha256"] = _sha256_file(final)
            meta["mask_lineage_verified"] = bool(
                meta.get("selected_source_mask_sha256")
                and meta.get("selected_source_mask_sha256") == meta.get("final_mask_sha256")
            )
            if not meta["mask_lineage_verified"]:
                _add_unique(meta.setdefault("review_flags", []), "mask_lineage_mismatch")
                _add_unique(meta.setdefault("quality_flags", []), "mask_lineage_mismatch")
                meta["training_weight"] = 0.0
                meta["should_enter_student_training"] = False
            if enable_shapekit and not dry_run and meta.get("shapekit_status") != "success":
                shapekit_unsupported = meta.get("shapekit_status") in {"unsupported_target", "unsupported_target_skipped_by_policy"}
                _add_unique(meta.setdefault("review_flags", []), "shapekit_fallback")
                _add_unique(meta.setdefault("quality_flags", []), "shapekit_fallback")
                if shapekit_unsupported:
                    _add_unique(meta.setdefault("review_flags", []), "shapekit_unsupported_target")
                    _add_unique(meta.setdefault("quality_flags", []), "shapekit_unsupported_target")
                _append_jsonl(review_queue, {
                    "case_id": case_id,
                    "organ": organ,
                    "reason": "Candidate ShapeKit fallback before LabelCritic selection",
                    "selected_model": meta.get("selected_model"),
                    "candidate_models": meta.get("candidate_models", []),
                    "selection_method": meta.get("selection_method"),
                    "selection_status": meta.get("selection_status"),
                    "shapekit_status": meta.get("shapekit_status"),
                    "shapekit_reason": meta.get("shapekit_reason"),
                    "shapekit_unsupported_target": shapekit_unsupported,
                    "review_flags": meta.get("review_flags", []),
                    "quality_flags": meta.get("quality_flags", []),
                })
            else:
                if not enable_shapekit:
                    meta["shapekit_status"] = "skipped_debug_only"
                elif dry_run:
                    meta["shapekit_status"] = "skipped_dry_run"
            meta["quality_status"] = _quality_status(meta.get("review_flags", []), meta.get("quality_flags", []))
            passport = build_label_passport({
                **meta,
                "case_id": case_id,
                "ct_path": str(ct),
                "mask_path": final,
            })
            meta.update({
                "label_maturity_level": passport["label_maturity_level"],
                "auto_fine_label_status": passport["auto_fine_label_status"],
                "auto_fine_label_reliability_score": passport["auto_fine_label_reliability_score"],
                "grade": passport["grade"],
                "training_weight": passport["training_weight"],
                "label_passport_path": str(passport_path_for_mask(final)) if final else None,
                "ground_truth_status": "machine_generated_candidate",
                "distillation_eligible": float(passport["training_weight"]) > 0.0,
                "distillation_exclusion_reason": None if float(passport["training_weight"]) > 0.0 else "training_weight_zero",
                "student_training_priority": passport["grade"],
                "label_confidence": passport["auto_fine_label_reliability_score"],
                "estimated_reliability": passport["evidence_confidence"],
                "evidence_confidence": passport["evidence_confidence"],
                "evidence_scores": passport["evidence_scores"],
                "missing_evidence": passport["missing_evidence"],
                "decision_status": passport["decision_status"],
                "decision_reasons": passport["decision_reasons"],
                "target_type": passport["target_type"],
                "probability_mask_path": passport.get("probability_mask_path"),
                "voxel_uncertainty_path": passport.get("voxel_uncertainty_path"),
                "independent_family_count": passport.get("independent_family_count", 0),
                "family_membership": passport.get("family_membership", {}),
                "scoring_schema_version": passport["scoring_schema_version"],
            })
            distillation_gate = _distillation_gate_for_selected_label(meta)
            meta["distillation_eligible"] = bool(distillation_gate["eligible"])
            meta["distillation_exclusion_reason"] = distillation_gate["reason"]
            grade = str(meta.get("grade") or "D").upper()
            method = str(meta.get("selection_method") or "")
            status = str(meta.get("selection_status") or "")
            if grade == "D":
                publication_status = "rejected_but_recorded"
            elif method in {"near_identical_agreement", "geometric_teacher_consensus", "consensus"}:
                publication_status = "accepted_by_consensus"
            elif status == "fallback":
                publication_status = "accepted_by_fallback"
            else:
                publication_status = "selected_by_decision"
            meta["publication_status"] = publication_status
            if publication_status == "rejected_but_recorded" and final and Path(final).exists():
                rejected_dir = case_updated.parent / "rejected"
                rejected_dir.mkdir(parents=True, exist_ok=True)
                rejected_mask = rejected_dir / Path(final).name
                shutil.move(str(final), str(rejected_mask))
                meta["audit_mask_path"] = str(rejected_mask.resolve())
                meta["final_mask"] = None
                meta["mask_path"] = None
                meta["mask"] = None
            passport_mask = meta.get("audit_mask_path") or meta.get("final_mask")
            if passport_mask:
                passport["mask_path"] = str(passport_mask)
                meta["label_passport_path"] = str(passport_path_for_mask(passport_mask))
                write_json(passport_path_for_mask(passport_mask), passport)

        case_373_summary = _materialize_case_373_targets(
            case_id=case_id,
            ct=ct,
            organs=list(organs),
            case_updated=case_updated,
            selection_rows=selection_rows,
            selected_metadata=selected_metadata,
            presence_context=presence_context,
            negative_absent_training_weight=float(os.getenv("MEDAI_NEGATIVE_ABSENT_TRAINING_WEIGHT", "0.1")),
        )
        write_json(updated_root / case_id / "case_373_target_summary.json", case_373_summary)

        case_gap_rows = _build_gap_rows(
            case_id=case_id,
            ct=ct,
            organs=organs,
            selection_rows=selection_rows,
            selected_metadata=selected_metadata,
            presence_context=presence_context,
        )
        case_gap_summary = _summarize_gap_rows(case_gap_rows)
        case_quality_summary = _summarize_case_quality(
            selection_rows=selection_rows,
            selected_metadata=selected_metadata,
            gap_rows=case_gap_rows,
        )
        all_gap_rows.extend(case_gap_rows)

        write_json(organ_task_state_path, organ_task_state)
        case_timing = {
            "case_id": case_id,
            "candidate_mode": candidate_mode,
            "teacher_inference_mode": teacher_inference_mode,
            "teacher_inference_models": hierarchy_models_used or case_models,
            "teacher_inference_count": len(hierarchy_models_used or case_models),
            "hierarchy_blocked_count": len(hierarchy_blocked),
            "selected_organs": len(selected_metadata),
            "compare_used_count": sum(1 for row in selection_rows if row.get("labelcritic_compare_used")),
            "grade_used_count": sum(1 for row in selection_rows if row.get("labelcritic_grade_used")),
            "runtime_sec": round(time.time() - _case_start, 3),
            "stage_timing_sec": {k: round(v, 3) for k, v in stage_timing.items()},
            "stage_counts": stage_counts,
        }
        timing_rows.append(case_timing)
        write_json(updated_root / case_id / "case_timing_breakdown.json", case_timing)
        write_json(updated_root / case_id / "selection_metadata.json", {
            "quality_contract_version": QUALITY_CONTRACT_VERSION,
            "fov_policy_version": FOV_POLICY_VERSION,
            "case_id": case_id,
            "ct_path": str(ct),
            "dataset_type": "pseudo_label_dataset",
            "ground_truth_status": "pseudo_label_candidate",
            "candidate_mode": candidate_mode,
            "scan_coverage": (presence_context.get("metadata") or {}).get("scan_coverage"),
            "coverage_regions": (presence_context.get("metadata") or {}).get("coverage_regions"),
            "body_region": (presence_context.get("metadata") or {}).get("body_region"),
            "ct_region": (presence_context.get("metadata") or {}).get("ct_region"),
            "case_presence_context": presence_context,
            "case_execution_plan": case_execution_plan,
            "case_timing": case_timing,
            "case_373_target_summary": case_373_summary,
            "selected_organs": selected_metadata,
            "selection_rows": selection_rows,
            "gap_rows": case_gap_rows,
            "gap_summary": case_gap_summary,
            "case_quality_summary": case_quality_summary,
            "shapekit": shapekit_result,
        })
        if not dry_run:
            try:
                build_case_quality_report(
                    case_id=case_id,
                    ct_path=ct,
                    selected=selected_metadata,
                    gap_summary=case_gap_summary,
                    output_dir=updated_root / case_id / "automatic_quality_report",
                )
            except Exception as exc:
                _append_jsonl(review_queue, {
                    "case_id": case_id,
                    "reason": f"automatic_quality_report_failed:{type(exc).__name__}:{exc}",
                })
        standard_case = _export_standard_case_dataset(
            case_id=case_id,
            ct=ct,
            selected_metadata=selected_metadata,
            out_root=standard_dataset_root,
            student_target_ids=student_target_ids,
        )
        standard_dataset_cases.append(standard_case)
        write_json(updated_root / case_id / "standard_dataset_case.json", standard_case)
        write_json(updated_root / case_id / "shapekit_report.json", {
            "case_id": case_id,
            "ct_path": str(ct),
            "stage": "shapekit_report",
            "dataset_type": "pseudo_label_dataset",
            "ground_truth_status": "pseudo_label_candidate",
            "enable_shapekit": enable_shapekit,
            "result": shapekit_result,
            "selected_organs": [
                {
                    "organ": item.get("organ"),
                    "shapekit_status": item.get("shapekit_status"),
                    "shapekit_reason": item.get("shapekit_reason"),
                    "final_mask": item.get("final_mask"),
                    "review_flags": item.get("review_flags", []),
                    "quality_flags": item.get("quality_flags", []),
                }
                for item in selected_metadata
            ],
        })

        write_json(case_out / "pseudo_label_selection.json", {
            "case_id": case_id,
            "ct_path": str(ct),
            "dataset_type": "pseudo_label_dataset",
            "scan_coverage": (presence_context.get("metadata") or {}).get("scan_coverage"),
            "coverage_regions": (presence_context.get("metadata") or {}).get("coverage_regions"),
            "body_region": (presence_context.get("metadata") or {}).get("body_region"),
            "ct_region": (presence_context.get("metadata") or {}).get("ct_region"),
            "case_presence_context": presence_context,
            "case_373_target_summary": case_373_summary,
            "selection_rows": selection_rows,
            "selected_organs": selected_metadata,
            "gap_rows": case_gap_rows,
            "gap_summary": case_gap_summary,
            "case_quality_summary": case_quality_summary,
            "shapekit": shapekit_result,
        })

        # Report supervision: compare tumor mask against report if both are available.
        if not dry_run and case.get("report_path"):
            try:
                from .report_supervision import verify_tumor_with_report
                tumor_mask = case_updated / "pancreatic_lesion.nii.gz"
                if not tumor_mask.exists() and ref_dir:
                    tumor_mask = ref_dir / "pancreatic_lesion.nii.gz"
                if tumor_mask.exists():
                    report_decision = verify_tumor_with_report(Path(case["report_path"]).resolve(), tumor_mask, "pancreas", Path(case["clinical_path"]).resolve() if case.get("clinical_path") else None)
                    _append_jsonl(report_supervision_jsonl, {"case_id": case_id, **report_decision})
            except Exception as exc:
                _append_jsonl(report_supervision_jsonl, {"case_id": case_id, "stage": "report_supervision", "status": "failed", "reason": str(exc)})

        # Reasoning trace grounded in available case paths — one entry per organ with updated mask.
        if not dry_run:
            for trace_organ in organs:
                organ_mask = case_updated / f"{trace_organ}.nii.gz"
                if not organ_mask.exists():
                    continue
                try:
                    trace = build_reasoning_trace(
                        patient_folder=None, scan_id=case_id, ct_image=ct,
                        current_mask=organ_mask,
                        previous_mask=None, organ=trace_organ,
                        report_path=Path(case["report_path"]).resolve() if case.get("report_path") else None,
                        clinical_path=Path(case["clinical_path"]).resolve() if case.get("clinical_path") else None,
                        pathology_path=Path(case["pathology_path"]).resolve() if case.get("pathology_path") else None,
                        output_json=None,
                    )
                    _append_jsonl(traces_jsonl, {"case_id": case_id, "organ": trace_organ, "trace": trace})
                except Exception as exc:
                    _append_jsonl(traces_jsonl, {"case_id": case_id, "organ": trace_organ, "trace_status": "failed", "reason": str(exc)})

        round_rows.append({
            "case_id": case_id,
            "checked_masks": checked,
            "accepted_masks": accepted,
            "low_dice_masks": low_dice,
            "vlm_reviewed": critic_count,
            "updated_masks": updated,
            "remaining_uncertain": uncertain,
        })
        _case_elapsed = round(_time.time() - _case_start, 1)
        print(f"[{_time.strftime('%H:%M:%S')}] Case {idx}/{len(cases)}: {case_id} 完成 "
              f"(耗时{_case_elapsed}s, accepted={accepted}, updated={updated}, critic={critic_count})", flush=True)

    dice_csv = out / "dice_metrics.csv"
    round_csv = out / "round_metrics.csv"
    _write_csv(dice_csv, dice_rows, [
        "case_id", "organ", "model", "prediction", "reference", "reference_role",
        "metric_family", "metric_scope", "ground_truth_status", "accuracy_warning",
        "metric_target", "metric_subject", "metric_comparison", "metric_interpretation",
        "dice", "pseudo_consistency_dice", "decision", "status", "reason",
        "candidate_exists", "alias_match", "requested_canonical_id", "source_local_label",
        "resolved_canonical_id", "comparison_family", "parent_ids", "mapping_type",
        "mapping_source", "identity_status", "identity_mismatch_reasons",
    ])
    _write_csv(round_csv, round_rows, ["case_id", "checked_masks", "accepted_masks", "low_dice_masks", "vlm_reviewed", "updated_masks", "remaining_uncertain"])
    write_json(out / "inference_results.json", inference_results)
    write_json(out / "standard_dataset_index.json", {
        "stage": "standard_dataset_index",
        "status": "success",
        "layout": "bdmap_pants_style_binary_masks",
        "root": str(standard_dataset_root.resolve()),
        "case_count": len(standard_dataset_cases),
        "cases": standard_dataset_cases,
        "mapping_policy": "teacher target ID / output name -> canonical organ name -> student target ID",
    })
    all_selection_rows = _rebuild_selection_rows_from_artifacts(updated_root)
    rebuilt_selected_metadata = _rebuild_selected_metadata_from_artifacts(updated_root)
    case_373_summaries = []
    for case in cases:
        sid = case.get("case_id")
        sp = updated_root / str(sid) / "case_373_target_summary.json" if sid else None
        if sp and sp.exists():
            try:
                case_373_summaries.append(json.loads(sp.read_text(encoding="utf-8")))
            except Exception:
                pass
    target_type_counts_373: dict[str, int] = {}
    for row in all_selection_rows:
        key = str(row.get("target_type") or "hard")
        target_type_counts_373[key] = target_type_counts_373.get(key, 0) + 1
    case_373_dataset_summary = {
        "stage": "case_373_target_summary",
        "status": "success",
        "num_cases": len(cases),
        "num_classes": len(organs),
        "expected_targets": len(cases) * len(organs),
        "manifest_targets": len(all_selection_rows),
        "candidate_pseudo_targets": sum(1 for r in all_selection_rows if str(r.get("record_type")) == "candidate_pseudo"),
        "absent_negative_targets": sum(1 for r in all_selection_rows if str(r.get("target_type")) in {"absent_negative", "negative_absent"}),
        "unresolved_review_targets": sum(1 for r in all_selection_rows if str(r.get("target_type")) in {"unresolved_review", "unresolved_visible", "partial_fov"}),
        "all_zero_masks": len({str(r.get("final_mask") or r.get("mask_path") or "") for r in all_selection_rows if str(r.get("target_type")) in {"absent_negative", "negative_absent"} and (r.get("final_mask") or r.get("mask_path"))}),
        "target_type_counts": target_type_counts_373,
        "complete_case_373": len(all_selection_rows) == len(cases) * len(organs),
        "cases": case_373_summaries,
    }
    write_json(out / "case_373_target_summary.json", case_373_dataset_summary)
    write_json(out / "full_case_373_manifest.json", {
        "stage": "full_case_373_estep_manifest",
        "status": "success" if case_373_dataset_summary["complete_case_373"] else "failed",
        "summary": case_373_dataset_summary,
        "items": all_selection_rows,
    })
    manifest = build_training_manifest(standard_dataset_root, out / "training_manifest.json", organs=organs)
    gap_report = {
        "stage": "pseudo_label_gap_report",
        "status": "success",
        "dataset_type": "pseudo_label_dataset",
        "ground_truth_status": "pseudo_label_candidate",
        "num_gap_rows": len(all_gap_rows),
        "gap_rows": all_gap_rows,
        "target_space_policy": _load_target_space_policy(project_root, organs),
        "formal_373_target_validation": target_validation,
        "case_373_target_summary": case_373_dataset_summary,
        "teacher_inference_mode": teacher_inference_mode,
        "roi_margin_mm": roi_margin_mm,
        "note": "Rows here are missing selected pseudo labels, ShapeKit fallbacks, or target-policy exclusions; they are not silently dropped.",
    }
    write_json(out / "pseudo_label_gap_report.json", gap_report)
    shapekit_reports = [
        str((updated_root / case.get("case_id", "") / "shapekit_report.json").resolve())
        for case in cases
        if case.get("case_id") and (updated_root / case.get("case_id", "") / "shapekit_report.json").exists()
    ]
    write_json(out / "shapekit_report.json", {
        "stage": "shapekit_report",
        "status": "success",
        "dataset_type": "pseudo_label_dataset",
        "ground_truth_status": "pseudo_label_candidate",
        "enable_shapekit": enable_shapekit,
        "num_case_reports": len(shapekit_reports),
        "case_reports": shapekit_reports,
        "note": "Per-case ShapeKit reports live under annotation_versions/<case_id>/shapekit_report.json.",
    })
    _write_csv(
        out / "pseudo_label_gap_report.csv",
        all_gap_rows,
        ["case_id", "ct_path", "organ", "gap_type", "reason", "candidate_count", "candidate_models", "selection_method", "selection_status", "shapekit_status", "dataset_type", "ground_truth_status"],
    )
    organ_audit_rows = _build_formal_organ_audit_rows(
        selection_rows=all_selection_rows,
        selected_metadata=rebuilt_selected_metadata,
    )
    _write_csv(
        out / "organ_audit_table.csv",
        organ_audit_rows,
        [
            "case_id", "organ", "fov_status", "route_primary_teacher", "route_backup_teachers",
            "route_competition_teachers", "route_confidence", "candidate_count", "candidate_models",
            "independent_family_count", "family_membership", "selected_model", "selection_method",
            "selection_status", "candidate_qc_status", "candidate_qc_flags", "labelcritic_compare_used",
            "labelcritic_grade_used", "labelcritic_supported", "labelcritic_tiebreak_adjustment",
            "auto_grade", "evidence_confidence", "normalized_evidence_confidence", "winner_margin",
            "grade", "training_weight", "target_type", "missing_evidence", "failed_evidence",
            "review_flags", "quality_flags", "missing_reason", "final_mask",
        ],
    )
    write_json(out / "organ_audit_table.json", {
        "stage": "formal_organ_audit_table",
        "status": "success",
        "rows": organ_audit_rows,
    })
    # Selected-model-aware M-step routing: record which primary model should be
    # updated for each organ, instead of implying that a generic model is always
    # the M-step target.
    mstep_routing = recommend_primary_models_for_organs(registry, organs)
    write_json(out / "mstep_model_routing.json", mstep_routing)
    mcfg = write_mstep_config(
        out / "mstep_config.json",
        out / "training_manifest.json",
        base_model="selected_model_aware",
        notes="Use `mstep-update --target-model <primary_model>` for each trainable primary model in mstep_model_routing.json. TotalSegmentator is baseline-only in this project.",
    )

    summary = {
        "stage": "run_loop", "status": "success", "case_list": str(case_csv), "output_folder": str(out),
        "num_cases": len(cases), "models_requested": models, "organs": organs,
        "target_space_policy": _load_target_space_policy(project_root, organs),
        "formal_373_target_validation": target_validation,
        "dry_run": dry_run, "enable_shapekit": enable_shapekit, "enable_critic": enable_critic, "critic_backend": critic_backend, "critic_base_url": critic_base_url, "critic_port": critic_port,
        "labelcritic_options": labelcritic_options,
        "dice_metrics_csv": str(dice_csv), "round_metrics_csv": str(round_csv), "review_queue_jsonl": str(review_queue),
        "vlm_decisions_jsonl": str(vlm_decisions), "patient_traces_jsonl": str(traces_jsonl), "report_supervision_jsonl": str(report_supervision_jsonl),
        "standard_dataset_root": str(standard_dataset_root.resolve()),
        "standard_dataset_index": str((out / "standard_dataset_index.json").resolve()),
        "training_manifest": manifest, "pseudo_label_gap_report": gap_report, "case_373_target_summary": case_373_dataset_summary, "shapekit_report": str((out / "shapekit_report.json").resolve()), "mstep_config": mcfg, "mstep_model_routing": mstep_routing,
        "resume_audit": resume_rows,
        "preseeded_parent_only": preseeded_parent_only,
        "reuse_preseeded_only": reuse_preseeded_only,
        "round2_competition_audit": _summarize_preseeded_competition(
            preseeded_keys=[] if preseeded_parent_only else sorted((preseeded_model_dirs or {}).keys()),
            selection_rows=all_selection_rows,
        ),
        "round_rows": round_rows,
        "total_updated": len(rebuilt_selected_metadata),
        "total_labelcritic_decisions": sum(int(r.get("vlm_reviewed", 0) or 0) for r in round_rows),
    }
    timing_rows = _merge_case_timing_rows(cases, updated_root, timing_rows)
    _write_csv(
        out / "case_timing_breakdown.csv",
        timing_rows,
        ["case_id", "candidate_mode", "teacher_inference_mode", "teacher_inference_models", "teacher_inference_count", "hierarchy_blocked_count", "selected_organs", "compare_used_count", "grade_used_count", "runtime_sec", "stage_timing_sec", "stage_counts"],
    )
    write_json(out / "case_timing_breakdown.json", {"stage": "case_timing_breakdown", "status": "success", "rows": timing_rows})
    summary["case_timing_breakdown"] = timing_rows
    summary["candidate_mode"] = candidate_mode
    summary["organ_audit_table"] = str((out / "organ_audit_table.csv").resolve())
    summary["selection_rows_rebuilt"] = len(all_selection_rows)
    summary["selected_organs_rebuilt"] = len(rebuilt_selected_metadata)
    summary["summary_rebuilt_from_artifacts"] = True
    write_json(out / "run_summary.json", summary)
    return summary
