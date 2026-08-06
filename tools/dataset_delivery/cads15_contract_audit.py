#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.model_registry import candidate_models_for_organs, get_model_entry, load_registry  # noqa: E402
from cli_anything.medai.core.runtime_resolver import (  # noqa: E402
    DEFAULT_HPC_CHECKPOINT_ROOT,
    DEFAULT_NNUNETV2_PREDICT,
    executable_status,
    resolve_checkpoint_root,
    resolve_nnunet_predictor,
    resolve_registry_path,
)
from tools.dataset_delivery.delivery_lib import read_csv_rows, write_csv, write_json  # noqa: E402


DEFAULT_CONTRACT = REPO_ROOT / "configs" / "cads15_target_contract.json"
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "model_registry.yaml"
DEFAULT_ALIAS_CONFIG = REPO_ROOT / "configs" / "model_label_aliases.json"
DEFAULT_TAXONOMY = REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json"
DEFAULT_TASK2_TARGETS = REPO_ROOT / "configs" / "dataset_delivery" / "task1" / "task2_generate_targets_23.csv"


CADS15_TARGETS = [
    "blood",
    "cerebrospinal_fluid",
    "common_iliac_artery_left",
    "common_iliac_artery_right",
    "common_iliac_vein_left",
    "common_iliac_vein_right",
    "compact_bone",
    "eyeball",
    "face",
    "gland_structure",
    "gray_matter",
    "muscle_of_head",
    "scalp",
    "spongy_bone",
    "white_matter",
]

ARTERY_TARGETS = {"common_iliac_artery_left", "common_iliac_artery_right"}
VEIN_TARGETS = {"common_iliac_vein_left", "common_iliac_vein_right"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    doc = read_json(path)
    targets = doc.get("targets")
    if not isinstance(targets, list):
        raise ValueError(f"CADS15 contract has no target list: {path}")
    return doc


def contract_targets(path: Path = DEFAULT_CONTRACT) -> list[dict[str, Any]]:
    return list(load_contract(path).get("targets") or [])


def load_taxonomy_ids(path: Path = DEFAULT_TAXONOMY) -> set[str]:
    doc = read_json(path)
    return {str(item) for item in doc.get("target_organs", [])}


def load_task2_targets(path: Path = DEFAULT_TASK2_TARGETS) -> set[str]:
    return {row.get("target_name", "") for row in read_csv_rows(path) if row.get("target_name")}


def _alias_maps(path: Path = DEFAULT_ALIAS_CONFIG) -> dict[str, dict[str, str]]:
    doc = read_json(path)
    return {
        model: {
            str(local): str(global_name)
            for local, global_name in ((entry or {}).get("local_to_global", {}) or {}).items()
        }
        for model, entry in ((doc.get("models") or {}).items())
    }


def dataset_labels(dataset_json: Path) -> dict[str, int]:
    doc = read_json(dataset_json)
    labels = doc.get("labels") or {}
    if not isinstance(labels, dict):
        raise ValueError(f"Unsupported dataset.json labels format: {dataset_json}")
    return {str(name): int(value) for name, value in labels.items()}


def trainer_dir_for_entry(entry: dict[str, Any], *, repo_root: Path, checkpoint_root: Path) -> Path:
    dataset_json = Path(resolve_registry_path(
        entry.get("dataset_json_path"),
        repo_root=repo_root,
        checkpoint_root=checkpoint_root,
        path_kind="dataset_json",
    ).resolved)
    return dataset_json.parent


def checkpoint_file_for_entry(entry: dict[str, Any], *, trainer_dir: Path) -> Path:
    fold = str(entry.get("folds") or "all")
    name = str(entry.get("checkpoint_name") or "checkpoint_final")
    if not name.endswith(".pth"):
        name += ".pth"
    return trainer_dir / f"fold_{fold}" / name


def _source_maps_to_target(source_label: str, target: str, model_key: str, aliases: dict[str, dict[str, str]], covered: set[str]) -> bool:
    if source_label == target and target in covered:
        return True
    return aliases.get(model_key, {}).get(source_label) == target


def _semantic_guard(target: str, source_label: str) -> list[str]:
    errors: list[str] = []
    if target in ARTERY_TARGETS and "vena" in source_label:
        errors.append("artery_target_mapped_to_vein_source")
    if target in VEIN_TARGETS and "artery" in source_label:
        errors.append("vein_target_mapped_to_artery_source")
    if target.endswith("_left") and source_label.endswith("_right"):
        errors.append("left_target_mapped_to_right_source")
    if target.endswith("_right") and source_label.endswith("_left"):
        errors.append("right_target_mapped_to_left_source")
    if target == "compact_bone" and source_label == "spongy bone":
        errors.append("compact_bone_mapped_to_spongy_bone")
    if target == "spongy_bone" and source_label == "compact bone":
        errors.append("spongy_bone_mapped_to_compact_bone")
    return errors


def audit_contract(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    registry_path: Path = DEFAULT_REGISTRY,
    alias_config_path: Path = DEFAULT_ALIAS_CONFIG,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    task2_targets_path: Path = DEFAULT_TASK2_TARGETS,
    checkpoint_root_arg: Path | None = None,
    predictor_arg: str | None = None,
    require_runtime_files: bool = True,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    registry = load_registry(registry_path)
    aliases = _alias_maps(alias_config_path)
    taxonomy = load_taxonomy_ids(taxonomy_path)
    task2 = load_task2_targets(task2_targets_path)
    checkpoint_root = resolve_checkpoint_root(
        explicit=checkpoint_root_arg,
        registry_checkpoint_root=registry.get("checkpoint_root"),
        repo_root=REPO_ROOT,
    )
    predictor = resolve_nnunet_predictor(explicit=predictor_arg, require_exists=False)
    predictor_status = executable_status(predictor)
    predictor_ok = bool(predictor_status["exists"] and predictor_status["is_file"] and predictor_status["is_executable"])
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    contracts = {str(row.get("canonical_id")): row for row in contract_targets(contract_path)}
    target_set = set(contracts)
    if sorted(target_set) != sorted(CADS15_TARGETS):
        errors.append({
            "type": "contract_target_set_mismatch",
            "expected": CADS15_TARGETS,
            "actual": sorted(target_set),
        })
    candidate_routes = candidate_models_for_organs(registry, CADS15_TARGETS)
    for target in CADS15_TARGETS:
        contract_row = contracts.get(target, {})
        row_errors: list[str] = []
        candidates = [model for model in candidate_routes.get(target, []) if str(model).startswith("cads")]
        if len(candidates) != 1:
            row_errors.append("missing_or_ambiguous_cads_primary_route")
        model_key = str(contract_row.get("primary_model") or (candidates[0] if candidates else ""))
        if candidates and model_key != candidates[0]:
            row_errors.append("contract_primary_model_differs_from_registry_route")
        try:
            entry = get_model_entry(registry, model_key)
        except Exception:
            entry = {}
            row_errors.append("registry_entry_missing")
        covered = {str(item) for item in (entry.get("covered_organs") or [])}
        source_label = str(contract_row.get("source_label") or "")
        if not _source_maps_to_target(source_label, target, model_key, aliases, covered):
            row_errors.append("source_label_does_not_map_to_canonical_target")
        row_errors.extend(_semantic_guard(target, source_label))
        if target not in taxonomy:
            row_errors.append("target_not_in_373_taxonomy")
        if target not in task2:
            row_errors.append("target_not_in_task2_generate_targets")
        dataset_json = Path(resolve_registry_path(
            entry.get("dataset_json_path"),
            repo_root=REPO_ROOT,
            checkpoint_root=checkpoint_root,
            path_kind="dataset_json",
        ).resolved) if entry else Path()
        labels: dict[str, int] = {}
        label_id: int | None = None
        if dataset_json.exists():
            try:
                labels = dataset_labels(dataset_json)
                label_id = labels.get(source_label)
                if label_id is None:
                    row_errors.append("source_label_missing_from_dataset_json")
            except Exception as exc:
                row_errors.append(f"dataset_json_unreadable:{type(exc).__name__}")
        elif require_runtime_files:
            row_errors.append("dataset_json_missing")
        contract_label_id = contract_row.get("source_label_id")
        if label_id is not None and contract_label_id != label_id:
            row_errors.append("contract_source_label_id_mismatch_dataset_json")
        trainer_dir = trainer_dir_for_entry(entry, repo_root=REPO_ROOT, checkpoint_root=checkpoint_root) if entry else Path()
        plans_json = trainer_dir / "plans.json" if trainer_dir else Path()
        checkpoint_file = checkpoint_file_for_entry(entry, trainer_dir=trainer_dir) if entry else Path()
        checkpoint_path_resolved = Path(resolve_registry_path(
            entry.get("checkpoint_path"),
            repo_root=REPO_ROOT,
            checkpoint_root=checkpoint_root,
            path_kind="checkpoint",
        ).resolved) if entry else Path()
        if require_runtime_files:
            if not checkpoint_path_resolved.exists():
                row_errors.append("checkpoint_root_missing")
            if not plans_json.exists() or plans_json.stat().st_size <= 0:
                row_errors.append("plans_json_missing_or_empty")
            if not checkpoint_file.exists() or checkpoint_file.stat().st_size <= 0:
                row_errors.append("checkpoint_file_missing_or_empty")
            if not predictor_ok:
                row_errors.append("predictor_missing_or_not_executable")
        route_status = "verified" if not row_errors else "blocked"
        status = "READY_FOR_HPC_SMOKE" if not row_errors else (
            "BLOCKED_MISSING_LABEL_MAPPING" if any("label" in error for error in row_errors)
            else "BLOCKED_MISSING_CHECKPOINT" if any("checkpoint" in error or "plans" in error for error in row_errors)
            else "BLOCKED_ROUTE_AMBIGUOUS" if any("route" in error for error in row_errors)
            else "BLOCKED_RUNTIME_PRECHECK"
        )
        row = {
            **contract_row,
            "model": model_key,
            "dataset_id": str(entry.get("dataset_id") or contract_row.get("nnunet_dataset_id") or ""),
            "source_label_id_dataset_json": label_id,
            "checkpoint_resolved": str(checkpoint_path_resolved) if entry else "",
            "dataset_json_resolved": str(dataset_json) if entry else "",
            "plans_json_resolved": str(plans_json) if entry else "",
            "checkpoint_file_resolved": str(checkpoint_file) if entry else "",
            "predictor_resolved": str(predictor),
            "predictor_executable": predictor_ok,
            "route_status": route_status,
            "status": status,
            "errors": row_errors,
        }
        rows.append(row)
        for error in row_errors:
            errors.append({"canonical_id": target, "type": error})
    report = {
        "status": "READY_FOR_HPC_SMOKE" if not errors else "BLOCKED",
        "contract_path": str(contract_path),
        "registry_path": str(registry_path),
        "alias_config_path": str(alias_config_path),
        "taxonomy_path": str(taxonomy_path),
        "task2_targets_path": str(task2_targets_path),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_root_default": str(DEFAULT_HPC_CHECKPOINT_ROOT),
        "predictor": str(predictor),
        "predictor_default": str(DEFAULT_NNUNETV2_PREDICT),
        "predictor_status": predictor_status,
        "target_count": len(rows),
        "verified_count": sum(1 for row in rows if row["route_status"] == "verified"),
        "errors": errors,
        "rows": rows,
    }
    return report


def write_audit_outputs(report: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "cads15_route_audit.json", report)
    write_csv(
        output_root / "cads15_route_audit.csv",
        report["rows"],
        [
            "canonical_id",
            "model",
            "dataset_id",
            "source_label",
            "source_label_id",
            "source_label_id_dataset_json",
            "checkpoint_resolved",
            "dataset_json_resolved",
            "plans_json_resolved",
            "checkpoint_file_resolved",
            "predictor_resolved",
            "route_status",
            "status",
            "errors",
        ],
    )
    lines = [
        "# CADS15 Route Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Targets: `{report['target_count']}`",
        f"- Verified: `{report['verified_count']}`",
        f"- Checkpoint root: `{report['checkpoint_root']}`",
        f"- Predictor: `{report['predictor']}`",
        "",
        "## Routes",
        "",
    ]
    for row in report["rows"]:
        lines.append(
            f"- `{row['canonical_id']}` -> `{row['model']}` / "
            f"`{row['source_label']}` id `{row.get('source_label_id_dataset_json')}`: "
            f"`{row['status']}`"
        )
        if row.get("errors"):
            lines.append(f"  errors: `{';'.join(row['errors'])}`")
    (output_root / "cads15_route_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the CADS15 Task 2 target contract.")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, type=Path)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, type=Path)
    parser.add_argument("--alias-config", default=DEFAULT_ALIAS_CONFIG, type=Path)
    parser.add_argument("--taxonomy", default=DEFAULT_TAXONOMY, type=Path)
    parser.add_argument("--task2-targets", default=DEFAULT_TASK2_TARGETS, type=Path)
    parser.add_argument("--checkpoint-root", default=None, type=Path)
    parser.add_argument("--nnunet-predict-executable", default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--allow-missing-runtime-files", action="store_true")
    args = parser.parse_args()
    report = audit_contract(
        contract_path=args.contract.resolve(),
        registry_path=args.registry.resolve(),
        alias_config_path=args.alias_config.resolve(),
        taxonomy_path=args.taxonomy.resolve(),
        task2_targets_path=args.task2_targets.resolve(),
        checkpoint_root_arg=args.checkpoint_root,
        predictor_arg=args.nnunet_predict_executable,
        require_runtime_files=not args.allow_missing_runtime_files,
    )
    write_audit_outputs(report, args.output_root.resolve())
    print(json.dumps({"status": report["status"], "verified_count": report["verified_count"]}, indent=2))
    return 0 if report["status"] == "READY_FOR_HPC_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
