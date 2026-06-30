from __future__ import annotations

from pathlib import Path
from typing import Any

from .backend_capabilities import official_voxtell_runtime_policy
from .voxtell_student import VoxTellStudent


class OfficialVoxTellPretrainedAdapter:
    """Official pretrained VoxTell baseline/candidate adapter.

    This wrapper keeps the official pretrained predictor separate from the
    project-distilled Student. By default it is baseline_only: it may write masks
    and audit files, but it must not enter AutoLabelCore selection or the M-step
    manifest unless mode='candidate' is explicitly selected.
    """

    def __init__(
        self,
        model_dir: str | Path,
        target_config: str | Path,
        mode: str = "baseline_only",
        device: str = "cuda",
        backend: str = "official_python_api",
        text_encoding_model: str | Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.target_config = Path(target_config).resolve()
        self.mode = mode
        self.policy = official_voxtell_runtime_policy(mode)
        self.student = VoxTellStudent(
            model_dir=self.model_dir,
            target_config=self.target_config,
            device=device,
            backend=backend,
            text_encoding_model=text_encoding_model,
        )

    def segment(
        self,
        ct_image: str | Path,
        output_dir: str | Path,
        prompts: list[str] | None = None,
        dry_run: bool = False,
        timeout_sec: int = 1800,
        prompt_batch_size: int = 16,
        prompt_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self.mode == "disabled":
            return {
                "status": "skipped",
                "reason": "official_voxtell_pretrained mode is disabled",
                "provider": "official_voxtell_pretrained",
                "source_name": "official_voxtell_pretrained",
                "model_role": "disabled",
                "teacher_like_candidate_source": True,
                "checkpoint_origin": "official_huggingface",
                "checkpoint_version": "voxtell_v1.1",
                "is_project_student": False,
                "prompt_conditioned": True,
                "uses_official_voxtell_predictor": False,
                "official_voxtell_mode": self.policy["official_voxtell_mode"],
                "active_as_teacher_candidate": False,
                "allow_selection_by_autolabelcore": False,
                "used_for_selected_pseudo_label": False,
                "used_for_training_manifest": False,
                "eligible_for_next_round_prompt_student": False,
                "eligible_as_teacher_candidate": False,
                "purpose": self.policy["purpose"],
            }
        result = self.student.segment(
            ct_image=ct_image,
            output_dir=output_dir,
            prompts=prompts,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
            prompt_batch_size=prompt_batch_size,
            prompt_overrides=prompt_overrides,
        )
        result.update({
            "provider": "official_voxtell_pretrained",
            "source_name": "official_voxtell_pretrained",
            "model_role": "zero_shot_baseline" if self.mode == "baseline_only" else "candidate_teacher_or_baseline",
            "teacher_like_candidate_source": True,
            "checkpoint_origin": "official_huggingface",
            "checkpoint_version": "voxtell_v1.1",
            "is_project_student": False,
            "prompt_conditioned": True,
            "uses_official_voxtell_predictor": True,
            "official_voxtell_mode": self.policy["official_voxtell_mode"],
            "active_as_teacher_candidate": self.policy["active_as_teacher_candidate"],
            "allow_selection_by_autolabelcore": self.policy["allow_selection_by_autolabelcore"],
            "used_for_selected_pseudo_label": False if self.mode in {"baseline_only", "disabled"} else result.get("used_for_selected_pseudo_label"),
            "used_for_training_manifest": False if self.mode in {"baseline_only", "disabled"} else result.get("used_for_training_manifest"),
            "eligible_for_next_round_prompt_student": False,
            "eligible_as_teacher_candidate": bool(self.policy["active_as_teacher_candidate"]),
            "purpose": self.policy["purpose"],
        })
        return result
