from __future__ import annotations

from copy import deepcopy
from typing import Any

PROJECT_PROMPT_STUDENT = "project_voxtell_prompt_distillation_student"
PROJECT_PROMPT_STUDENT_ALIAS = "project_prompt_student"
LEGACY_PROJECT_DISTILLATION = "project_distillation_experimental"
OFFICIAL_VOXTELL_PRETRAINED = "official_voxtell_pretrained"
OFFICIAL_NNUNET_BASELINE = "official_voxtell_nnunet_encoder_baseline"
LEGACY_OFFICIAL_NNUNET_FINETUNE = "official_voxtell_nnunet_encoder_finetune"
LEGACY_AMBIGUOUS_OFFICIAL_FINETUNE = "official_voxtell_finetune"

OFFICIAL_VOXTELL_MODES = {"disabled", "baseline_only", "candidate"}
DEFAULT_EXPERIMENT_PROFILE = "advisor_aligned_default"

EXPERIMENT_PROFILES: dict[str, dict[str, Any]] = {
    "advisor_aligned_default": {
        "official_voxtell_pretrained": {"mode": "baseline_only"},
        "mstep_backend": PROJECT_PROMPT_STUDENT,
        "official_voxtell_nnunet_encoder_baseline": {
            "enabled": False,
            "allow_as_teacher_candidate": False,
            "explicit_baseline_mode": False,
        },
    },
    "enhanced_candidate_pool": {
        "official_voxtell_pretrained": {"mode": "candidate"},
        "mstep_backend": PROJECT_PROMPT_STUDENT,
        "official_voxtell_nnunet_encoder_baseline": {
            "enabled": False,
            "allow_as_teacher_candidate": False,
            "explicit_baseline_mode": False,
        },
    },
}

BACKEND_CAPABILITIES: dict[str, dict[str, Any]] = {
    OFFICIAL_VOXTELL_PRETRAINED: {
        "source_name": OFFICIAL_VOXTELL_PRETRAINED,
        "role": "candidate_teacher_or_baseline",
        "teacher_like_candidate_source": True,
        "official": True,
        "checkpoint_origin": "official_huggingface",
        "prompt_conditioned": True,
        "is_project_student": False,
        "can_train": False,
        "main_mstep_allowed": False,
        "eligible_for_next_round_prompt_student": False,
        "eligible_as_teacher_candidate": True,
    },
    PROJECT_PROMPT_STUDENT: {
        "source_name": PROJECT_PROMPT_STUDENT,
        "role": "main_mstep_student",
        "official": False,
        "uses_official_voxtell_model_components": True,
        "prompt_conditioned": True,
        "is_project_student": True,
        "can_train": True,
        "main_mstep_allowed": True,
        "eligible_for_next_round_prompt_student": True,
        "eligible_as_teacher_candidate": True,
    },
    OFFICIAL_NNUNET_BASELINE: {
        "source_name": OFFICIAL_NNUNET_BASELINE,
        "role": "official_baseline",
        "official": True,
        "prompt_conditioned": False,
        "label_format": "nnunet_multiclass",
        "can_train": True,
        "main_mstep_allowed": False,
        "eligible_for_next_round_prompt_student": False,
        "eligible_as_teacher_candidate": True,
        "baseline_only": True,
        "not_used_as_main_prompt_student": True,
    },
}


def canonical_backend_name(name: str | None) -> str | None:
    if not name:
        return None
    value = str(name).strip()
    if value == PROJECT_PROMPT_STUDENT_ALIAS:
        return PROJECT_PROMPT_STUDENT
    if value == LEGACY_PROJECT_DISTILLATION:
        return PROJECT_PROMPT_STUDENT
    if value == LEGACY_OFFICIAL_NNUNET_FINETUNE:
        return OFFICIAL_NNUNET_BASELINE
    return value


def backend_capability(name: str) -> dict[str, Any]:
    canonical = canonical_backend_name(name)
    if canonical not in BACKEND_CAPABILITIES:
        raise KeyError(f"Unknown VoxTell backend capability: {name}")
    item = deepcopy(BACKEND_CAPABILITIES[canonical])
    item["canonical_backend"] = canonical
    if canonical != name:
        item["alias_used"] = name
    return item


def resolve_experiment_profile(name: str | None = None) -> dict[str, Any]:
    profile_name = (name or DEFAULT_EXPERIMENT_PROFILE).strip() or DEFAULT_EXPERIMENT_PROFILE
    if profile_name not in EXPERIMENT_PROFILES:
        raise KeyError(f"Unknown experiment profile: {profile_name}")
    doc = deepcopy(EXPERIMENT_PROFILES[profile_name])
    doc["experiment_profile"] = profile_name
    return doc


def official_voxtell_runtime_policy(mode: str | None) -> dict[str, Any]:
    resolved = (mode or "baseline_only").strip().lower() or "baseline_only"
    if resolved not in OFFICIAL_VOXTELL_MODES:
        raise ValueError(f"Unsupported official_voxtell_pretrained mode: {mode}")
    active = resolved == "candidate"
    return {
        **backend_capability(OFFICIAL_VOXTELL_PRETRAINED),
        "official_voxtell_mode": resolved,
        "active_as_teacher_candidate": active,
        "allow_selection_by_autolabelcore": active,
        "used_for_selected_pseudo_label": False if resolved in {"disabled", "baseline_only"} else None,
        "used_for_training_manifest": False if resolved in {"disabled", "baseline_only"} else None,
        "purpose": "zero_shot_baseline_only" if resolved == "baseline_only" else ("disabled" if resolved == "disabled" else "candidate_pool"),
    }


def profile_runtime_policy(profile_name: str | None = None) -> dict[str, Any]:
    profile = resolve_experiment_profile(profile_name)
    official_mode = (profile.get("official_voxtell_pretrained") or {}).get("mode", "baseline_only")
    mstep_backend = canonical_backend_name(profile.get("mstep_backend"))
    nnunet = profile.get("official_voxtell_nnunet_encoder_baseline") or {}
    return {
        "experiment_profile": profile["experiment_profile"],
        "official_voxtell_pretrained": official_voxtell_runtime_policy(official_mode),
        "mstep_backend": mstep_backend,
        "mstep_backend_capability": backend_capability(mstep_backend),
        "official_voxtell_nnunet_encoder_baseline": {
            **backend_capability(OFFICIAL_NNUNET_BASELINE),
            "enabled": bool(nnunet.get("enabled", False)),
            "allow_as_teacher_candidate": bool(nnunet.get("allow_as_teacher_candidate", False)),
            "explicit_baseline_mode": bool(nnunet.get("explicit_baseline_mode", False)),
        },
    }
