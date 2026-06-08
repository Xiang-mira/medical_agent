from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import resolve_path


EXPECTED_FORMAL_TARGET_COUNT = 373


def load_student_target_space(target_config: str | Path = "configs/student_3d_prompt_target_organs.json") -> dict[str, Any]:
    path = resolve_path(target_config)
    if not path.exists():
        raise FileNotFoundError(f"Student target config not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"Student target config is not a JSON object: {path}")
    doc["_resolved_path"] = str(path)
    return doc


def validate_formal_373_target_space(
    target_config: str | Path = "configs/student_3d_prompt_target_organs.json",
    *,
    requested_organs: list[str] | None = None,
    require_full_target: bool = False,
) -> dict[str, Any]:
    """Validate the current formal 373-organ target contract.

    This is intentionally strict for formal runs. Historical spaces such as
    384 global organs, 377 teacher direction, 358 teacher coverage, or VISTA3D
    127 labels must not silently become the mainline target.
    """
    doc = load_student_target_space(target_config)
    target_organs = [str(x) for x in doc.get("target_organs", [])]
    target_set = set(target_organs)
    organ_to_prompt = doc.get("organ_to_prompt", {}) or {}
    organ_to_student_id = doc.get("organ_to_student_id", {}) or {}
    policy_skipped = {str(item.get("organ")) for item in doc.get("policy_skipped_organs", []) if isinstance(item, dict)}
    no_route = {str(item.get("organ")) for item in doc.get("no_enabled_route_organs", []) if isinstance(item, dict)}

    duplicate_targets = sorted({organ for organ in target_organs if target_organs.count(organ) > 1})
    missing_prompts = sorted([organ for organ in target_organs if not organ_to_prompt.get(organ)])
    missing_student_ids = sorted([organ for organ in target_organs if organ not in organ_to_student_id])
    target_contains_skipped = sorted([organ for organ in target_organs if organ in policy_skipped])
    target_contains_no_route = sorted([organ for organ in target_organs if organ in no_route])

    requested = [str(x) for x in (requested_organs or [])]
    requested_set = set(requested)
    requested_non_target = sorted(requested_set - target_set)
    target_not_requested = sorted(target_set - requested_set) if requested else []

    blocking: dict[str, Any] = {
        "target_count_mismatch": [] if len(target_organs) == EXPECTED_FORMAL_TARGET_COUNT else [len(target_organs)],
        "unique_target_count_mismatch": [] if len(target_set) == EXPECTED_FORMAL_TARGET_COUNT else [len(target_set)],
        "duplicate_targets": duplicate_targets,
        "missing_prompts": missing_prompts,
        "missing_student_ids": missing_student_ids,
        "target_contains_policy_skipped_organs": target_contains_skipped,
        "target_contains_no_enabled_route_organs": target_contains_no_route,
        "requested_non_target_organs": requested_non_target,
        "target_organs_not_requested": target_not_requested if require_full_target else [],
    }
    status = "success" if not any(blocking.values()) else "failed"
    return {
        "stage": "formal_373_target_validation",
        "status": status,
        "target_config": doc.get("_resolved_path"),
        "expected_formal_target_count": EXPECTED_FORMAL_TARGET_COUNT,
        "counts": {
            "target_organs": len(target_organs),
            "unique_target_organs": len(target_set),
            "requested_organs": len(requested) if requested else None,
            "global_label_space_organs": (doc.get("counts") or {}).get("global_label_space_organs"),
            "policy_skipped_organs": len(policy_skipped),
            "no_enabled_route_organs": len(no_route),
            "historical_teacher_direction": (doc.get("counts") or {}).get("historical_teacher_direction"),
        },
        "requested_is_full_373_target": requested_set == target_set if requested else None,
        "blocking": blocking,
        "historical_count_explanation": {
            "384": "global_label_space_organs, not the formal current target",
            "8": "SAROS coarse-label policy-skipped organs",
            "3": "organs with no enabled route",
            "373": "current accepted exact 3D prompt target organs",
            "377": "historical teacher direction only",
            "358_or_127": "legacy/side target spaces; not the mainline 373-organ student target",
        },
        "terminology_policy": {
            "dataset_name": "auto_fine_label_dataset",
            "allowed_statuses": [
                "machine_label_candidate",
                "auto_fine_label_candidate",
                "auto_fine_label_accepted",
                "unresolved",
                "expert_verified",
            ],
            "expert_verified_note": "Use only for future expert-confirmed labels.",
            "accuracy_warning": "Pseudo-consistency is not true accuracy or expert ground-truth DSC.",
        },
    }


def require_formal_373_target_space(
    target_config: str | Path = "configs/student_3d_prompt_target_organs.json",
    *,
    requested_organs: list[str] | None = None,
    require_full_target: bool = False,
) -> dict[str, Any]:
    result = validate_formal_373_target_space(
        target_config,
        requested_organs=requested_organs,
        require_full_target=require_full_target,
    )
    if result["status"] != "success":
        raise ValueError(json.dumps(result, indent=2, ensure_ascii=False))
    return result

