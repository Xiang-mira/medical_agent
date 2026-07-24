from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import SchedulerConfig, resolve_path
from .manifest import (
    audit_case_masks,
    resolved_manifest_path,
    validate_split,
    validate_target_config,
    validate_target_mapping,
)
from .utils import SchedulerError, git_snapshot


def _require_path(path: Path | None, label: str, *, must_exist: bool) -> dict[str, Any]:
    if path is None:
        raise SchedulerError(f"Missing path: {label}")
    exists = path.exists()
    if must_exist and not exists:
        raise SchedulerError(f"Required path does not exist for {label}: {path}")
    return {"label": label, "path": str(path), "exists": exists}


def run_preflight(config: SchedulerConfig, *, require_hpc_paths: bool = False) -> dict[str, Any]:
    paths = config.paths
    target_config = resolve_path(paths.get("target_config"))
    mapping_path = resolve_path(paths.get("target_mapping"))
    pilot_target_config = resolve_path(paths.get("pilot_target_config"))
    image_root = resolve_path(paths.get("image_root"))
    mask_root = resolve_path(paths.get("mask_root"))
    report: dict[str, Any] = {
        "status": "success",
        "git": git_snapshot(),
        "paths": [],
        "checks": {},
        "warnings": [],
    }
    for label, path in (
        ("target_config", target_config),
        ("target_mapping", mapping_path),
        ("pilot_target_config", pilot_target_config),
        ("image_root", image_root),
        ("mask_root", mask_root),
        ("work_root", resolve_path(paths.get("work_root"))),
    ):
        report["paths"].append(_require_path(path, label, must_exist=label in {"target_config", "target_mapping"} or require_hpc_paths))

    assert target_config is not None
    target = validate_target_config(target_config, expected_count=373)
    assert mapping_path is not None
    mapping = validate_target_mapping(mapping_path, target["targets"], pilot_target_count=int(config.project.get("pilot_target_count", 338)))
    assert pilot_target_config is not None
    pilot_target = validate_target_config(pilot_target_config, expected_count=338)
    report["checks"]["target_config"] = {k: v for k, v in target.items() if k != "targets"}
    report["checks"]["pilot_target_config"] = {k: v for k, v in pilot_target.items() if k != "targets"}
    report["checks"]["target_mapping"] = mapping

    train = resolved_manifest_path(paths, "pilot_train_input_case_list")
    test = resolved_manifest_path(paths, "pilot_test_input_case_list")
    reserve = resolved_manifest_path(paths, "pilot_reserve_case_list")
    if train.exists() and test.exists() and reserve.exists():
        split = validate_split(train, test, reserve)
        report["checks"]["pilot_split"] = split
        if image_root and mask_root and image_root.exists() and mask_root.exists():
            all_case_ids = split["splits"]["train"]["case_ids"] + split["splits"]["test"]["case_ids"] + split["splits"]["reserve"]["case_ids"]
            report["checks"]["data_audit"] = audit_case_masks(image_root, mask_root, all_case_ids)
    elif require_hpc_paths:
        missing = [str(p) for p in (train, test, reserve) if not p.exists()]
        raise SchedulerError(f"Missing pilot manifests: {missing}")
    else:
        report["warnings"].append("Pilot manifests not present locally; HPC preflight will require them.")

    for cmd in ("sbatch", "squeue", "sacct"):
        report["checks"][f"slurm_{cmd}"] = {"available": bool(shutil.which(cmd))}
    return report
