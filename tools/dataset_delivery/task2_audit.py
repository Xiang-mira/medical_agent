#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.model_registry import candidate_models_for_organs, load_registry
from cli_anything.medai.core.multimodel_loop import _build_case_execution_plan, _expected_presence_for_organ, _fov_pruned_organs
from cli_anything.medai.core.organ_taxonomy import load_taxonomy, taxonomy_entry
from cli_anything.medai.core.registered_infer import run_registered_model


TASK2_CASE_ID = "BDMAP_00000120"
TASK2_WORK_DIR = Path("/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work")
IMAGE_ROOT = Path("/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro")
MASK_ROOT = Path("/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro")
CHECKPOINT_ROOT = Path("/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints")
FORMAL_CADS_DIR = TASK2_WORK_DIR / "formal_teacher_22_full_volume_20260730/results/cads"

TASK2_SCOPE: list[dict[str, str]] = [
    *[
        {"organ": organ, "model_group": "CADS"}
        for organ in [
            "cerebrospinal_fluid",
            "eyeball",
            "face",
            "gray_matter",
            "muscle_of_head",
            "scalp",
            "white_matter",
            "brain",
            "trachea",
            "brainstem",
            "oral_cavity",
            "compact_bone",
            "spongy_bone",
            "blood",
            "larynx",
        ]
    ],
    {"organ": "airway_tree", "model_group": "ATM"},
    {"organ": "airway_wall", "model_group": "AirRC"},
    {"organ": "lung_pulmonary_arteries", "model_group": "AirRC"},
    {"organ": "lung_pulmonary_veins", "model_group": "AirRC"},
    {"organ": "kidney_cortex", "model_group": "UNEST"},
    {"organ": "kidney_medulla", "model_group": "UNEST"},
    {"organ": "kidney_pelvicalyceal_system", "model_group": "UNEST"},
    {"organ": "pulmonary_artery", "model_group": "TotalSegmentator"},
]

TASK2_DEBUG_ORGANS = [
    "airway_tree",
    "airway_wall",
    "lung_pulmonary_arteries",
    "lung_pulmonary_veins",
    "kidney",
    "kidney_cortex",
    "kidney_medulla",
    "kidney_pelvicalyceal_system",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(k).strip(): str(v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
            if any(str(v or "").strip() for v in row.values())
        ]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def path_record(logical_name: str, configured: str | Path, consumer: str) -> dict[str, Any]:
    path = Path(configured).expanduser()
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = path.absolute()
    exists = path.exists()
    is_symlink = path.is_symlink()
    return {
        "logical_name": logical_name,
        "configured_value": str(configured),
        "resolved_absolute_path": str(resolved),
        "exists": exists,
        "readable": os.access(path, os.R_OK) if exists else False,
        "writable": os.access(path, os.W_OK) if exists else False,
        "is_symlink": is_symlink,
        "symlink_target": os.readlink(path) if is_symlink else "",
        "runtime_consumer": consumer,
        "status": "ok" if exists else "missing",
        "notes": "",
    }


def build_path_inventory(output_dir: Path) -> list[dict[str, Any]]:
    paths = [
        ("code_repo", REPO_ROOT, "git/python imports"),
        ("hpc_code_repo_expected", "/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent", "JHU HPC code checkout"),
        ("task2_work_dir", TASK2_WORK_DIR, "dataset delivery output"),
        ("cases_100_manifest", TASK2_WORK_DIR / "cases_100_manifest.csv", "fixed 100-case manifest"),
        ("ct_example", IMAGE_ROOT / TASK2_CASE_ID / "ct.nii.gz", "read-only CT source"),
        ("mask_example", MASK_ROOT / TASK2_CASE_ID / "segmentations", "read-only reference masks"),
        ("checkpoint_root_hpc", CHECKPOINT_ROOT, "registry checkpoint root"),
        ("code_checkpoints", REPO_ROOT / "checkpoints", "registry relative checkpoint root"),
        ("medical_agent_python", "/home/xhan74/envs/medical_agent/bin/python", "outer run-loop"),
        ("nnunet_python", "/home/xhan74/envs/medical_agent_train_py311/bin/python", "nnUNet environment audit"),
        ("nnunet_predict", "/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict", "nnUNet subprocess"),
        ("nnunet_sitecustomize", "/home/xhan74/nnunet_torch22_compat/sitecustomize.py", "nnUNet subprocess only"),
        ("slurm_entry", REPO_ROOT / "slurm/teacher_missing_labels_100.sbatch", "Slurm wrapper"),
        ("run_teacher_case", REPO_ROOT / "tools/dataset_delivery/run_teacher_case.py", "per-case launcher"),
        ("delivery_lib", REPO_ROOT / "tools/dataset_delivery/delivery_lib.py", "delivery orchestration"),
        ("run_medai_cli", REPO_ROOT / "run_medai_cli.py", "CLI entrypoint"),
        ("medai_cli", REPO_ROOT / "agent-harness/cli_anything/medai/medai_cli.py", "run-loop CLI"),
        ("multimodel_loop", REPO_ROOT / "agent-harness/cli_anything/medai/core/multimodel_loop.py", "teacher planning/execution"),
        ("hierarchical_roi", REPO_ROOT / "agent-harness/cli_anything/medai/core/hierarchical_roi.py", "ROI crop/restore"),
        ("registered_infer", REPO_ROOT / "agent-harness/cli_anything/medai/core/registered_infer.py", "registered teacher runner"),
        ("nnunet_wrapper_script", REPO_ROOT / "scripts/nnunetv2_predict_and_split.py", "ATM/AirRC wrapper"),
        ("unest_wrapper_script", REPO_ROOT / "scripts/unest_predict_and_split.py", "UNEST wrapper"),
    ]
    rows = [path_record(*item) for item in paths]
    git_cmds = [
        ["pwd", "-P"],
        ["git", "rev-parse", "--show-toplevel"],
        ["git", "rev-parse", "HEAD"],
        ["git", "branch", "--show-current"],
        ["git", "status", "--short"],
    ]
    git_rows = []
    for cmd in git_cmds:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        git_rows.append({"command": " ".join(cmd), "return_code": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})
    payload = {"status": "success", "paths": rows, "git": git_rows}
    write_json(output_dir / "task2_resolved_path_inventory.json", payload)
    md = ["# Task2 Resolved Path Inventory", "", "| logical_name | exists | readable | writable | resolved_absolute_path |", "|---|---:|---:|---:|---|"]
    for row in rows:
        md.append(f"| {row['logical_name']} | {row['exists']} | {row['readable']} | {row['writable']} | `{row['resolved_absolute_path']}` |")
    (output_dir / "task2_resolved_path_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def _prompt_entry(target_config: dict[str, Any], organ: str) -> dict[str, Any]:
    bank = target_config.get("organ_prompt_bank") or {}
    entry = bank.get(organ) or {}
    return entry if isinstance(entry, dict) else {}


def _resolve_registry_checkpoint(project_root: Path, entry: dict[str, Any], key: str = "checkpoint_path") -> str:
    value = entry.get(key) or ""
    if not value:
        return ""
    path = Path(str(value))
    return str(path if path.is_absolute() else (project_root / path).resolve())


def build_registry_taxonomy_audit(output_dir: Path) -> list[dict[str, Any]]:
    registry = load_registry(REPO_ROOT / "configs/model_registry.yaml")
    taxonomy = load_taxonomy(REPO_ROOT / "configs/organ_taxonomy.json")
    target_config = json.loads((REPO_ROOT / "configs/student_3d_prompt_target_organs.json").read_text(encoding="utf-8"))
    rows = []
    for organ in TASK2_DEBUG_ORGANS:
        tax = taxonomy_entry(taxonomy, organ) or {}
        prompt = _prompt_entry(target_config, organ)
        model = model_for_organ(registry, organ, next((item["model_group"] for item in TASK2_SCOPE if item["organ"] == organ), ""))
        entry = (registry.get("models", {}) or {}).get(model, {}) if model else {}
        rows.append({
            "organ": organ,
            "canonical_id": tax.get("canonical_id", organ),
            "hierarchy_role": tax.get("hierarchy_role", ""),
            "parent_ids": ";".join(tax.get("parent_ids", []) or []),
            "body_region": prompt.get("region", ""),
            "fov_region": prompt.get("region", ""),
            "aliases": ";".join(prompt.get("aliases", []) or []),
            "primary_teacher": tax.get("primary_teacher", ""),
            "candidate_models": ";".join(tax.get("candidate_models", []) or []),
            "registry_model": model,
            "registry_enabled": entry.get("enabled", ""),
            "registry_status": entry.get("status", ""),
            "covered_organs": ";".join(entry.get("covered_organs", []) or []),
            "checkpoint_path": _resolve_registry_checkpoint(REPO_ROOT, entry),
            "dataset_json_path": _resolve_registry_checkpoint(REPO_ROOT, entry, "dataset_json_path"),
            "runner": entry.get("runner", ""),
            "command_template": entry.get("command_template", ""),
            "routing_aliases": ";".join(entry.get("routing_aliases", []) or []),
        })
    fields = [
        "organ", "canonical_id", "hierarchy_role", "parent_ids", "body_region", "fov_region", "aliases",
        "primary_teacher", "candidate_models", "registry_model", "registry_enabled", "registry_status",
        "covered_organs", "checkpoint_path", "dataset_json_path", "runner", "command_template", "routing_aliases",
    ]
    write_csv(output_dir / "task2_registry_taxonomy_audit.csv", rows, fields)
    write_json(output_dir / "task2_registry_taxonomy_audit.json", {"status": "success", "rows": rows})
    return rows


def model_for_organ(registry: dict[str, Any], organ: str, group: str) -> str:
    if group == "ATM":
        return "atm"
    if group == "AirRC":
        return "airrc"
    if group == "UNEST":
        return "unest"
    if group == "TotalSegmentator":
        return "totalsegmentator"
    candidates = candidate_models_for_organs(registry, [organ]).get(organ, [])
    return next((m for m in candidates if str(m).startswith("cads")), candidates[0] if candidates else "")


def build_scope(output_dir: Path) -> list[dict[str, Any]]:
    registry = load_registry(REPO_ROOT / "configs/model_registry.yaml")
    taxonomy = load_taxonomy(REPO_ROOT / "configs/organ_taxonomy.json")
    rows = []
    completed = {"cerebrospinal_fluid", "eyeball", "face", "gray_matter", "muscle_of_head", "scalp", "white_matter"}
    for item in TASK2_SCOPE:
        organ = item["organ"]
        group = item["model_group"]
        model = model_for_organ(registry, organ, group)
        entry = (registry.get("models", {}) or {}).get(model, {}) if model else {}
        checkpoint = entry.get("checkpoint_path") or ""
        checkpoint_path = Path(str(checkpoint))
        if checkpoint and not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        blocked_reason = ""
        execution_status = "pending"
        current_status = "pending"
        if group == "TotalSegmentator":
            execution_status = "blocked"
            current_status = "blocked"
            blocked_reason = "TotalSegmentator target retained in 23-class scope but not runnable in current phase"
        elif organ in completed:
            execution_status = "completed_existing_cads_100"
            current_status = "completed"
        rows.append({
            "organ": organ,
            "canonical_name": (taxonomy_entry(taxonomy, organ) or {}).get("canonical_id", organ),
            "teacher_model": model,
            "model_group": group,
            "current_status": current_status,
            "checkpoint_status": "exists" if checkpoint and checkpoint_path.exists() else "missing_or_unmounted",
            "execution_status": execution_status,
            "blocked_reason": blocked_reason,
            "expected_case_count": 100,
            "expected_mask_count": 100,
        })
    if len(rows) != 23:
        raise RuntimeError(f"Task2 scope must have 23 rows, got {len(rows)}")
    fields = ["organ", "canonical_name", "teacher_model", "model_group", "current_status", "checkpoint_status", "execution_status", "blocked_reason", "expected_case_count", "expected_mask_count"]
    write_csv(output_dir / "task2_23class_scope.csv", rows, fields)
    return rows


def kidney_inventory(manifest: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    if not manifest.exists():
        write_csv(output_dir / "task2_kidney_parent_inventory.csv", [], ["case_id", "kidney_combined", "kidney_left", "kidney_right", "resolved_parent_sources", "can_build_parent_roi", "failure_reason"])
        return rows
    for case in read_csv_rows(manifest):
        case_id = case.get("case_id") or ""
        ref_dir = Path(case.get("reference_mask_dir") or MASK_ROOT / case_id / "segmentations")
        sources = []
        flags = {}
        for key, names in {
            "kidney_combined": ["kidney.nii.gz"],
            "kidney_left": ["kidney_left.nii.gz", "left_kidney.nii.gz"],
            "kidney_right": ["kidney_right.nii.gz", "right_kidney.nii.gz"],
        }.items():
            found = next((ref_dir / name for name in names if (ref_dir / name).exists()), None)
            flags[key] = bool(found)
            if found:
                sources.append(str(found))
        rows.append({
            "case_id": case_id,
            **flags,
            "resolved_parent_sources": ";".join(sources),
            "can_build_parent_roi": bool(flags["kidney_combined"] or (flags["kidney_left"] and flags["kidney_right"])),
            "failure_reason": "" if sources else "no_kidney_parent_or_left_right_masks_found",
        })
    write_csv(output_dir / "task2_kidney_parent_inventory.csv", rows, ["case_id", "kidney_combined", "kidney_left", "kidney_right", "resolved_parent_sources", "can_build_parent_roi", "failure_reason"])
    return rows


def parent_mask_candidates(case_id: str, organs: list[str], taxonomy: dict[str, Any]) -> dict[str, Any]:
    ref_dir = MASK_ROOT / case_id / "segmentations"
    parent_ids = sorted({
        str(parent)
        for organ in organs
        for parent in ((taxonomy_entry(taxonomy, organ) or {}).get("parent_ids", []) or [])
        if parent
    })
    result: dict[str, Any] = {}
    for parent in parent_ids:
        names = [f"{parent}.nii.gz"]
        if parent == "kidney":
            names.extend(["kidney_left.nii.gz", "kidney_right.nii.gz", "left_kidney.nii.gz", "right_kidney.nii.gz"])
        found = [ref_dir / name for name in names if (ref_dir / name).exists()]
        result[parent] = {
            "reference_mask_dir": str(ref_dir),
            "candidate_names": names,
            "found": [str(path) for path in found],
            "exists": bool(found),
            "can_build_union": parent == "kidney" and bool(found),
        }
    return result


def audit_cads_existing(output_dir: Path) -> dict[str, Any]:
    completed = {"cerebrospinal_fluid", "eyeball", "face", "gray_matter", "muscle_of_head", "scalp", "white_matter"}
    if not FORMAL_CADS_DIR.exists():
        payload = {
            "status": "missing",
            "path": str(FORMAL_CADS_DIR),
            "formal_mask_count": 0,
            "auxiliary_image_count": 0,
            "auxiliary_zero_mask_count": 0,
            "completed_organs": sorted(completed),
            "note": "CADS history is not mounted on this development machine; rerun on JHU HPC to verify the 700 masks.",
        }
        write_json(output_dir / "task2_cads_existing_audit.json", payload)
        return payload
    masks = sorted(FORMAL_CADS_DIR.rglob("*.nii.gz"))
    formal = [p for p in masks if p.name not in {"image.nii.gz", "zero_mask.nii.gz"}]
    by_organ = {organ: 0 for organ in sorted(completed)}
    for path in formal:
        organ = path.name.removesuffix(".nii.gz")
        if organ in by_organ:
            by_organ[organ] += 1
    payload = {
        "status": "success" if all(count == 100 for count in by_organ.values()) else "failed",
        "path": str(FORMAL_CADS_DIR),
        "total_nifti_count": len(masks),
        "formal_mask_count": len(formal),
        "auxiliary_image_count": sum(1 for p in masks if p.name == "image.nii.gz"),
        "auxiliary_zero_mask_count": sum(1 for p in masks if p.name == "zero_mask.nii.gz"),
        "completed_organs": sorted(completed),
        "formal_count_by_completed_organ": by_organ,
    }
    write_json(output_dir / "task2_cads_existing_audit.json", payload)
    return payload


def debug_plan(case_id: str, organs: list[str], models: list[str], output_path: Path) -> dict[str, Any]:
    registry_path = REPO_ROOT / "configs/model_registry.yaml"
    registry = load_registry(registry_path)
    taxonomy = load_taxonomy(REPO_ROOT / "configs/organ_taxonomy.json")
    ct = IMAGE_ROOT / case_id / "ct.nii.gz"
    presence_context = {"has_region_evidence": False, "metadata": {}}
    fov_organs = _fov_pruned_organs(organs, taxonomy, presence_context)
    execution_organs = list(fov_organs)
    for organ in fov_organs:
        for parent in ((taxonomy_entry(taxonomy, organ) or {}).get("parent_ids", []) or []):
            if parent not in execution_organs:
                execution_organs.append(parent)
    plan = _build_case_execution_plan(
        registry=registry,
        project_root=REPO_ROOT,
        organs=execution_organs,
        requested_models=models,
        preseeded_model_dirs=None,
        candidate_mode="route_pruned_with_competition",
    )
    dry_runs = {}
    for model in models:
        dry_runs[model] = run_registered_model(
            ct,
            output_path.parent / "dry_run_commands" / model,
            model,
            registry_path=registry_path,
            case_id=case_id,
            dry_run=True,
            extra_context={"requested_organs": organs},
        )
    payload = {
        "case_id": case_id,
        "requested_organs": organs,
        "fov_organs": fov_organs,
        "pruned_organs": [o for o in organs if o not in set(fov_organs)],
        "prune_reasons": [
            {
                "organ": organ,
                "reason": "expected_absent_by_fov",
                "expected_presence": _expected_presence_for_organ(organ, presence_context),
                "presence_context": presence_context,
            }
            for organ in organs
            if organ not in set(fov_organs)
        ],
        "execution_organs": execution_organs,
        "requested_models": models,
        "teacher_run_list": plan.get("teacher_run_list", []),
        "major_organs": [o for o in execution_organs if (taxonomy_entry(taxonomy, o) or {}).get("hierarchy_role") == "major"],
        "child_organs": [o for o in execution_organs if (taxonomy_entry(taxonomy, o) or {}).get("hierarchy_role") == "child"],
        "hierarchical_tasks": [
            {
                "organ": organ,
                "model": ((plan.get("per_organ", {}) or {}).get(organ, {}) or {}).get("primary_teacher"),
                "parent_ids": (taxonomy_entry(taxonomy, organ) or {}).get("parent_ids", []) or [],
                "planned_scope": "full_volume_child" if ((plan.get("per_organ", {}) or {}).get(organ, {}) or {}).get("primary_teacher") in {"atm", "airrc"} else "parent_roi",
            }
            for organ in fov_organs
            if (taxonomy_entry(taxonomy, organ) or {}).get("hierarchy_role") == "child"
        ],
        "blocked": [],
        "parent_masks_found": parent_mask_candidates(case_id, fov_organs, taxonomy),
        "resolved_model_command": {m: dry_runs[m].get("command") for m in dry_runs},
        "resolved_python": {m: dry_runs[m].get("resolved_python") for m in dry_runs},
        "resolved_checkpoint": {
            m: str((REPO_ROOT / ((registry.get("models", {}) or {}).get(m, {}) or {}).get("checkpoint_path", "")).resolve())
            for m in models
        },
        "expected_outputs": {m: dry_runs[m].get("expected_outputs", []) for m in dry_runs},
        "case_execution_plan": plan,
    }
    write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Task2 fixed 100-case delivery audit and dry-run planner.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument("--manifest", type=Path, default=TASK2_WORK_DIR / "cases_100_manifest.csv")
    parser.add_argument("--case-id", default=TASK2_CASE_ID)
    args = parser.parse_args()
    out = args.output_dir
    debug_root = out / "task2_debug" / args.case_id
    build_path_inventory(out)
    build_scope(out)
    build_registry_taxonomy_audit(out)
    kidney_inventory(args.manifest, out)
    audit_cads_existing(out)
    debug_plan(args.case_id, ["airway_tree"], ["atm"], debug_root / "atm_plan.json")
    debug_plan(args.case_id, ["airway_wall", "lung_pulmonary_arteries", "lung_pulmonary_veins"], ["airrc"], debug_root / "airrc_plan.json")
    debug_plan(args.case_id, ["kidney_cortex", "kidney_medulla", "kidney_pelvicalyceal_system"], ["unest"], debug_root / "unest_plan.json")
    print(json.dumps({"status": "success", "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
