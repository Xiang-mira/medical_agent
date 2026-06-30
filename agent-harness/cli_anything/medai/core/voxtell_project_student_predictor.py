from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .voxtell_student import VoxTellStudent


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _checkpoint_metadata(model: Path) -> dict[str, Any]:
    candidates = [
        model / "project_student_metadata.json",
        model / "student_checkpoint_metadata.json",
        model.parent / "voxtell_prompt_train_result.json",
        model.parent / "voxtell_prompt_mstep_result.json",
        model.parent / "final_student_export.json",
    ]
    merged: dict[str, Any] = {}
    for path in candidates:
        merged.update(_read_json(path))
    return merged


def project_student_predictor_preflight(model_dir: str | Path) -> dict[str, Any]:
    model = Path(model_dir).resolve()
    plans = model / "plans.json"
    checkpoint = model / "fold_0" / "checkpoint_final.pth"
    metadata = _checkpoint_metadata(model)
    prompt_conditioned = metadata.get("prompt_conditioned", metadata.get("is_prompt_conditioned_student", True)) is not False
    reload_ok = bool(metadata.get("checkpoint_reload_success") or metadata.get("reload_success"))
    prompt_smoke_ok = bool(metadata.get("prompt_inference_smoke_success") or metadata.get("prompt_inference_success"))
    quality_gate_ok = bool(metadata.get("quality_gate_success") or (metadata.get("quality_gate") or {}).get("status") == "success")
    result: dict[str, Any] = {
        "stage": "project_student_predictor_preflight",
        "model_dir": str(model),
        "plans_exists": plans.exists(),
        "checkpoint_exists": checkpoint.exists(),
        "prompt_conditioned": bool(prompt_conditioned),
        "metadata_prompt_conditioned": metadata.get("prompt_conditioned", metadata.get("is_prompt_conditioned_student")),
        "checkpoint_reload_success": reload_ok,
        "prompt_inference_smoke_success": prompt_smoke_ok,
        "quality_gate_success": quality_gate_ok,
        "official_predictor_compatibility": "not_checked",
        "student_predictor_backend": None,
        "eligible_for_next_round_prompt_student": False,
        "is_project_student": True,
        "checkpoint_origin": "project_distillation",
    }
    if not plans.exists() or not checkpoint.exists():
        result.update({
            "status": "failed",
            "student_predictor_backend": "unavailable",
            "failure_reason": "missing_plans_or_checkpoint",
        })
        return result
    if not prompt_conditioned:
        result.update({
            "status": "failed",
            "student_predictor_backend": "unavailable",
            "failure_reason": "metadata_prompt_conditioned_false",
        })
        return result
    try:
        from voxtell.inference.predictor import VoxTellPredictor  # noqa: F401
        result.update({
            "official_predictor_compatibility": "importable_not_loaded",
            "student_predictor_backend": "official_voxtell_predictor_compatible",
        })
    except Exception as exc:
        result.update({
            "official_predictor_compatibility": "failed",
            "student_predictor_backend": "project_student_predictor_using_official_voxtell_model",
            "fallback_reason": str(exc),
        })
    if reload_ok and prompt_smoke_ok and quality_gate_ok:
        result.update({
            "status": "passed",
            "eligible_for_next_round_prompt_student": True,
        })
    else:
        missing = []
        if not reload_ok:
            missing.append("checkpoint_reload_success")
        if not prompt_smoke_ok:
            missing.append("prompt_inference_smoke_success")
        if not quality_gate_ok:
            missing.append("quality_gate_success")
        result.update({
            "status": "requires_smoke_and_quality_gate",
            "failure_reason": "missing_" + ",".join(missing),
        })
    return result


class ProjectVoxTellStudentAdapter(VoxTellStudent):
    """Project-distilled prompt Student inference adapter."""

    pass
