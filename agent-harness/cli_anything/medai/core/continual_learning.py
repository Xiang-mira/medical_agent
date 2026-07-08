from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


TRAINING_CONTRACT_VERSION = "medai.training-label.v1"
RUN_SPEC_VERSION = "medai.run-spec.v1"
NOVELTY_POLICY_VERSION = "medai.novelty.v1"
PROMOTION_POLICY_VERSION = "medai.promotion.no-regression.v1"
SAMPLING_POLICY_VERSION = "medai.organ-balanced-sampling.v1"
RETENTION_POLICY_VERSION = "medai.foreground-anchor-retention.v1"

PROMOTION_STATUSES = {
    "candidate",
    "promoted",
    "competition_blocked",
    "no_material_update",
}
ROUND_STATE_STAGES = (
    "preflight",
    "estep",
    "manifest",
    "novelty_decision",
    "mstep",
    "inference",
    "evaluation",
    "promotion",
)


class TrainingContractError(ValueError):
    pass


def sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_target_type(value: Any) -> str:
    raw = str(value or "positive_hard").strip().lower()
    aliases = {
        "hard": "positive_hard",
        "soft": "positive_soft",
        "absent_negative": "negative_absent",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"positive_hard", "positive_soft", "negative_absent"}:
        raise TrainingContractError(f"unsupported target_type: {raw}")
    return normalized


def _git_commit(project_root: str | Path | None) -> str | None:
    if not project_root:
        return None
    proc = subprocess.run(
        ["git", "-C", str(Path(project_root).resolve()), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def probability_mask_audit(path: str | Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path) if path else None,
        "sha256": sha256_file(path),
        "status": "failed",
        "finite": False,
        "in_unit_interval": False,
        "has_intermediate_probability": False,
    }
    if not path or not Path(path).is_file():
        result["reason"] = "probability_mask_missing"
        return result
    try:
        import nibabel as nib
        import numpy as np

        array = np.asanyarray(nib.load(str(path)).dataobj)
        finite = bool(np.isfinite(array).all())
        minimum = float(np.nanmin(array))
        maximum = float(np.nanmax(array))
        # NIfTI scaling can introduce tiny float overshoots.
        in_range = finite and minimum >= -1e-6 and maximum <= 1.0 + 1e-6
        intermediate = bool(((array > 1e-6) & (array < 1.0 - 1e-6)).any())
        result.update(
            {
                "status": "passed" if in_range else "failed",
                "finite": finite,
                "in_unit_interval": in_range,
                "has_intermediate_probability": intermediate,
                "min": minimum,
                "max": maximum,
            }
        )
        if not in_range:
            result["reason"] = "probability_values_out_of_contract"
    except Exception as exc:
        result["reason"] = f"probability_mask_unreadable:{exc}"
    return result


def mask_foreground_audit(path: str | Path | None) -> dict[str, Any]:
    result = {"path": str(path) if path else None, "status": "failed", "nonempty": False}
    if not path or not Path(path).is_file():
        result["reason"] = "mask_missing"
        return result
    try:
        import nibabel as nib
        import numpy as np

        array = np.asanyarray(nib.load(str(path)).dataobj)
        nonempty = bool(np.isfinite(array).all() and (array > 0).any())
        result.update({"status": "passed" if nonempty else "failed", "nonempty": nonempty})
        if not nonempty:
            result["reason"] = "positive_mask_empty_or_nonfinite"
    except Exception as exc:
        result["reason"] = f"mask_unreadable:{exc}"
    return result


def _source_role(row: dict[str, Any]) -> tuple[str, str | None]:
    provider = str(
        row.get("origin_provider")
        or row.get("selected_provider")
        or row.get("selected_model")
        or row.get("source_model")
        or row.get("distillation_source")
        or row.get("teacher_output_name")
        or ""
    ).strip()
    dataset_role = str(row.get("dataset_role") or "").lower()
    gt_status = str(row.get("ground_truth_status") or "").lower()
    if "ground_truth" in dataset_role or dataset_role in {"gt", "expert_gt"}:
        return "ground_truth", provider or None
    if "ground_truth" in gt_status or gt_status in {"gt", "expert_gt"}:
        return "ground_truth", provider or None
    provider_lower = provider.lower()
    if "student" in provider_lower:
        return "student", provider or None
    if provider_lower in {"round_prev_selected", "previous_selected", "selected_previous_round"}:
        # These aliases discard the true origin. They are not sufficient
        # provenance for a formal positive label.
        return "unknown", provider or None
    if not provider:
        return "unknown", None
    return "teacher", provider


def _verified_student_replacement(row: dict[str, Any], provider: str | None) -> bool:
    provider_lower = str(provider or "").lower()
    if "student" not in provider_lower:
        return False
    if str(row.get("selection_status") or "").lower() != "selected":
        return False
    if str(row.get("selection_method") or "").lower() != "label_critic":
        return False
    qc_status = str(
        row.get("selected_candidate_qc_status")
        or row.get("candidate_qc_status")
        or ""
    ).lower()
    if qc_status and qc_status not in {"pass", "passed", "success", "ok"}:
        return False
    if row.get("labelcritic_decisive") is not True:
        return False
    records = row.get("labelcritic_records") or row.get("critic_records") or []
    if not any(isinstance(record, dict) and record.get("status") == "success" for record in records):
        return False
    return True


def _previous_pseudo_carry_forward(row: dict[str, Any], provider: str | None) -> bool:
    provider_lower = str(provider or "").lower()
    if provider_lower not in {
        "round_prev_selected",
        "previous_round_selected",
        "previous_selected",
        "selected_previous_round",
    }:
        return False
    if str(row.get("selection_status") or "").lower() != "selected":
        return False
    method = str(row.get("selection_method") or "").lower()
    return method in {
        "em_student_vs_previous_carry_forward",
        "label_critic",
        "near_identical_agreement",
        "single_teacher_provisional",
    }


def canonicalize_training_record(
    row: dict[str, Any],
    *,
    round_index: int | None = None,
    project_root: str | Path | None = None,
    strict_soft: bool = True,
) -> dict[str, Any]:
    item = dict(row)
    reasons: list[str] = []
    target_type = canonical_target_type(item.get("target_type"))
    supervision_type = "negative" if target_type == "negative_absent" else "positive"
    grade = str(item.get("grade") or "D").upper()
    source_role, provider = _source_role(item)

    mask = item.get("mask_path") or item.get("mask") or item.get("final_mask")
    image = item.get("image") or item.get("ct_path")
    image_hash = sha256_file(image)
    if not image_hash:
        reasons.append("image_missing_or_unreadable")
    probability_path = item.get("probability_mask_path")
    probability_audit: dict[str, Any] | None = None
    if target_type == "positive_soft":
        probability_audit = probability_mask_audit(probability_path)
        if probability_audit.get("status") != "passed":
            reasons.append(str(probability_audit.get("reason") or "invalid_probability_mask"))
        elif strict_soft and not probability_audit.get("has_intermediate_probability"):
            reasons.append("soft_target_has_no_intermediate_probabilities")
        mask = probability_path

    mask_hash = sha256_file(mask)
    if not mask_hash:
        reasons.append("mask_missing_or_unreadable")
    foreground_audit = None
    if supervision_type == "positive":
        foreground_audit = mask_foreground_audit(mask)
        if foreground_audit.get("status") != "passed":
            reasons.append(str(foreground_audit.get("reason") or "positive_mask_invalid"))
    verified_student = supervision_type == "positive" and _verified_student_replacement(item, provider)
    previous_carry_forward = supervision_type == "positive" and _previous_pseudo_carry_forward(item, provider)
    if (
        supervision_type == "positive"
        and source_role != "teacher"
        and not verified_student
        and not previous_carry_forward
    ):
        reasons.append(f"positive_source_role_forbidden:{source_role}")
    if supervision_type == "positive" and grade not in {"A", "B", "C"}:
        reasons.append(f"positive_grade_not_trainable:{grade}")
    if grade == "C" and target_type != "positive_soft":
        reasons.append("grade_C_requires_probability_target")
    qc_status = str(
        item.get("selected_candidate_qc_status")
        or item.get("candidate_qc_status")
        or ""
    ).lower()
    if qc_status and qc_status not in {"pass", "passed", "success", "ok"}:
        reasons.append(f"candidate_qc_not_pass:{qc_status}")
    identity_status = str(item.get("identity_status") or "").lower()
    if identity_status and identity_status != "valid":
        reasons.append(f"identity_not_valid:{identity_status}")
    flags = {
        str(flag).lower()
        for key in ("quality_flags", "review_flags", "selected_candidate_qc_flags")
        for flag in (item.get(key) or [])
    }
    hard_flags = {
        "zero_volume_mask",
        "empty_mask",
        "geometry_mismatch",
        "shape_mismatch_ct",
        "affine_mismatch_ct",
        "orientation_mismatch_ct",
        "candidate_qc_fail",
        "postprocess_failed",
    }
    if flags & hard_flags:
        reasons.append(f"hard_quality_flag:{sorted(flags & hard_flags)[0]}")
    fov_status = str(item.get("fov_status") or "").lower()
    if supervision_type == "negative":
        fov_evidence = item.get("fov_evidence") or item.get("negative_evidence")
        if fov_status not in {"full", "covered", "organ_absent_confirmed"}:
            reasons.append("negative_absent_missing_full_fov")
        if not fov_evidence:
            reasons.append("negative_absent_missing_fov_evidence")

    item.update(
        {
            "contract_version": TRAINING_CONTRACT_VERSION,
            "policy_version": TRAINING_CONTRACT_VERSION,
            "case_id": str(item.get("case_id") or ""),
            "canonical_organ": str(
                item.get("canonical_organ")
                or item.get("resolved_canonical_id")
                or item.get("organ")
                or ""
            ),
            "organ": str(item.get("organ") or item.get("canonical_organ") or ""),
            "target_type": target_type,
            "supervision_type": supervision_type,
            "image": str(image) if image else None,
            "image_sha256": image_hash,
            "mask": str(mask) if mask else None,
            "mask_path": str(mask) if mask else None,
            "mask_sha256": mask_hash,
            "mask_foreground_audit": foreground_audit,
            "probability_mask_path": str(probability_path) if probability_path else None,
            "probability_mask_sha256": (
                probability_audit.get("sha256") if probability_audit else None
            ),
            "probability_audit": probability_audit,
            "origin_provider": provider,
            "source_role": (
                "student" if verified_student else
                "previous_pseudo_label" if previous_carry_forward else
                source_role
            ),
            "verified_student_replacement": verified_student,
            "previous_pseudo_carry_forward": previous_carry_forward,
            "source_round": round_index if round_index is not None else item.get("source_round"),
            "memory_role": (
                "historical"
                if item.get("historical_replay")
                else str(item.get("memory_role") or "current")
            ),
            "lineage": list(
                item.get("lineage")
                or item.get("teacher_lineage")
                or item.get("candidate_models")
                or ([provider] if provider else [])
            ),
            "code_commit_sha": item.get("code_commit_sha") or _git_commit(project_root),
            "training_eligible": not reasons and item.get("distillation_eligible") is not False,
            "contract_failures": reasons,
        }
    )
    if item.get("distillation_eligible") is False:
        item["contract_failures"].append("distillation_eligible_false")
        item["training_eligible"] = False
    return item


def _record_quality(record: dict[str, Any]) -> tuple[float, ...]:
    grade_score = {"A": 3.0, "B": 2.0, "C": 1.0}.get(
        str(record.get("grade") or "D").upper(), 0.0
    )
    confidence = float(
        record.get("selected_candidate_qc_score")
        or record.get("auto_fine_label_reliability_score")
        or record.get("evidence_confidence")
        or 0.0
    )
    stable = 1.0 if record.get("historical_replay") else 0.0
    source_diversity = float(len(set(record.get("lineage") or [])))
    return grade_score, confidence, source_diversity, stable


def resolve_canonical_memory(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    winners: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for record in records:
        if record.get("supervision_type") != "positive":
            continue
        if not record.get("training_eligible"):
            rejected.append({**record, "memory_rejection": "contract_ineligible"})
            continue
        key = (str(record.get("case_id") or ""), str(record.get("organ") or ""))
        if not all(key):
            rejected.append({**record, "memory_rejection": "missing_case_or_organ"})
            continue
        incumbent = winners.get(key)
        if incumbent is None or _record_quality(record) > _record_quality(incumbent):
            if incumbent is not None:
                rejected.append({**incumbent, "memory_rejection": "lower_quality_than_winner"})
            winners[key] = record
        else:
            rejected.append({**record, "memory_rejection": "lower_quality_than_winner"})
    return [winners[key] for key in sorted(winners)], rejected


def novelty_audit(
    current: Iterable[dict[str, Any]],
    previous: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    def index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (str(row.get("case_id") or ""), str(row.get("organ") or "")): row
            for row in rows
            if row.get("supervision_type") == "positive" and row.get("training_eligible")
        }

    current_index = index(current)
    previous_index = index(previous)
    changed: list[list[str]] = []
    added: list[list[str]] = []
    total_weight = 0.0
    changed_weight = 0.0
    for key, row in current_index.items():
        weight = max(0.0, float(row.get("training_weight") or 1.0))
        total_weight += weight
        old = previous_index.get(key)
        if old is None:
            added.append(list(key))
            changed_weight += weight
        elif old.get("mask_sha256") != row.get("mask_sha256"):
            changed.append(list(key))
            changed_weight += weight
    removed = [list(key) for key in sorted(set(previous_index) - set(current_index))]
    ratio = changed_weight / total_weight if total_weight else 0.0
    if ratio < 0.05 and not added:
        decision, max_steps = "no_material_update", 0
    elif ratio < 0.20:
        decision, max_steps = "limited_update", 500
    else:
        decision, max_steps = "full_update", 2000
    return {
        "stage": "continual_learning_novelty",
        "policy_version": NOVELTY_POLICY_VERSION,
        "status": "success",
        "weighted_change_ratio": ratio,
        "changed_keys": changed,
        "added_keys": added,
        "removed_keys": removed,
        "decision": decision,
        "max_steps": max_steps,
    }


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    round_index: int
    backend: str
    case_list_path: str
    case_list_sha256: str
    case_count: int
    target_space_path: str
    target_space_sha256: str
    baseline_checkpoint_sha256: str | None
    previous_checkpoint_sha256: str | None
    evaluation_protocol_sha256: str
    training_split_sha256: str | None = None
    evaluation_split_sha256: str | None = None
    reference_protocol_sha256: str | None = None
    code_commit_sha: str | None = None
    protected_organs: tuple[str, ...] = ()
    training_contract_version: str = TRAINING_CONTRACT_VERSION
    novelty_policy_version: str = NOVELTY_POLICY_VERSION
    promotion_policy_version: str = PROMOTION_POLICY_VERSION
    sampling_policy_version: str = SAMPLING_POLICY_VERSION
    retention_policy_version: str = RETENTION_POLICY_VERSION
    schema_version: str = RUN_SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["protected_organs"] = list(self.protected_organs)
        return data


class RoundStateMachine:
    """Cohort/backend-independent, append-only formal round state."""

    def __init__(self, path: str | Path, run_spec: dict[str, Any]):
        self.path = Path(path)
        self.run_spec = deepcopy(run_spec)
        if self.path.exists():
            self.document = json.loads(self.path.read_text(encoding="utf-8"))
            if self.document.get("run_spec") != self.run_spec:
                raise ValueError("round state RunSpec is immutable")
        else:
            self.document = {
                "stage": "formal_round_state_machine",
                "status": "pending",
                "run_spec": self.run_spec,
                "transitions": [],
            }
            write_json(self.path, self.document)

    def advance(
        self,
        stage: str,
        status: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage not in ROUND_STATE_STAGES:
            raise ValueError(f"unknown round state stage: {stage}")
        transitions = self.document.setdefault("transitions", [])
        prior_stage = transitions[-1]["stage"] if transitions else None
        requested_index = ROUND_STATE_STAGES.index(stage)
        prior_index = ROUND_STATE_STAGES.index(prior_stage) if prior_stage else -1
        if requested_index < prior_index:
            existing = next(
                row for row in transitions if row.get("stage") == stage
            )
            if existing.get("status") in {
                "passed",
                "success",
                "completed",
                "full_update",
                "limited_update",
                "no_material_update",
            }:
                return existing
            raise ValueError(f"cannot retry failed {stage} after advancing to {prior_stage}")
        if requested_index not in {prior_index, prior_index + 1}:
            raise ValueError(
                f"illegal round transition {prior_stage or '<start>'} -> {stage}"
            )
        transition = {
            "stage": stage,
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
            "artifacts": deepcopy(artifacts or {}),
        }
        if requested_index == prior_index:
            transitions[-1] = transition
        else:
            transitions.append(transition)
        self.document["current_stage"] = stage
        self.document["status"] = (
            "completed"
            if stage == "promotion" and status in PROMOTION_STATUSES
            else status
        )
        write_json(self.path, self.document)
        return transition


def derive_round_seed(run_id: str, round_index: int) -> int:
    digest = hashlib.sha256(f"{run_id}:{round_index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def promotion_decision(
    *,
    requested_status: str,
    hard_checks: dict[str, bool],
    regression: dict[str, Any] | None,
    novelty: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if requested_status not in PROMOTION_STATUSES:
        raise ValueError(f"unsupported promotion status: {requested_status}")
    if novelty and novelty.get("decision") == "no_material_update":
        return {
            "status": "no_material_update",
            "policy_version": PROMOTION_POLICY_VERSION,
            "failures": [],
            "novelty": novelty,
        }
    failures = [name for name, passed in hard_checks.items() if not passed]
    if requested_status == "promoted":
        if not regression:
            failures.append("quality_regression_audit_missing")
        else:
            thresholds = {
                "mean_delta": -0.002,
                "paired_median_delta": -0.002,
                "worst_protected_delta": -0.01,
                "pseudo_delta": -0.005,
            }
            for name, minimum in thresholds.items():
                value = regression.get(name)
                if value is None or not math.isfinite(float(value)) or float(value) < minimum:
                    failures.append(f"{name}_below_{minimum}")
    return {
        "status": "competition_blocked" if failures else requested_status,
        "policy_version": PROMOTION_POLICY_VERSION,
        "failures": failures,
        "regression": regression,
        "novelty": novelty,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def retroactively_block_rounds(
    registry: dict[str, Any],
    round_indices: Iterable[int],
    *,
    reason: str = "retrospective_quality_regression",
    evidence: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    """Preserve prior entries and fail-closed block regressed checkpoints."""
    updated = deepcopy(registry)
    updated.setdefault("stage", "checkpoint_promotion_registry")
    updated["status_values"] = sorted(PROMOTION_STATUSES)
    rounds = updated.setdefault("rounds", {})
    history = updated.setdefault("history", [])
    changed: list[int] = []
    timestamp = datetime.now(UTC).isoformat()
    for round_index in round_indices:
        key = str(round_index)
        current = rounds.get(key)
        if not isinstance(current, dict):
            continue
        if (
            current.get("status") == "competition_blocked"
            and current.get("reason") == reason
            and current.get("retrospective_block") is True
        ):
            continue
        history.append(
            {
                **deepcopy(current),
                "history_index": len(history),
                "superseded_at": timestamp,
                "superseded_by": reason,
            }
        )
        blocked = deepcopy(current)
        blocked.update(
            {
                "status": "competition_blocked",
                "reason": reason,
                "eligible_for_next_round_prompt_student": False,
                "retrospective_block": True,
                "retrospective_blocked_at": timestamp,
                "retrospective_evidence": deepcopy(evidence or {}),
            }
        )
        rounds[key] = blocked
        history.append({**deepcopy(blocked), "history_index": len(history)})
        changed.append(round_index)
    updated["updated_at"] = timestamp
    return updated, changed
