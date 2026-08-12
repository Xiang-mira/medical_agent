#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.model_registry import candidate_models_for_organs, load_registry  # noqa: E402
from tools.dataset_delivery.delivery_lib import load_taxonomy_names, write_json  # noqa: E402
from tools.dataset_delivery.hierarchy_config import validate_hierarchy_config  # noqa: E402
from tools.dataset_delivery.labelcritic_resource_preflight import build_labelcritic_preflight  # noqa: E402
from tools.dataset_delivery.task2_formal_manifest import FORMAL_CASE_COUNT, FORMAL_GROUP_MODELS, FORMAL_MODEL_TARGETS, FORMAL_TARGETS, validate_formal_manifest  # noqa: E402
from tools.dataset_delivery.task2_workspace_staging import WORKSPACE_DIRS  # noqa: E402


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _check(name: str, ok: bool, reason: str = "", **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "reason": reason, **extra}


def build_formal_round1_preflight(
    *,
    workspace_root: Path,
    cases_manifest: Path,
    base_manifest: Path,
    code_root: Path = REPO_ROOT,
    registry_path: Path = REPO_ROOT / "configs" / "model_registry.yaml",
    taxonomy_path: Path = REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json",
    rename_mapping: Path = REPO_ROOT / "configs" / "dataset_delivery" / "task1" / "organ_rename_mapping_373.csv",
    boundary_classification: Path = REPO_ROOT / "configs" / "dataset_delivery" / "task1" / "task_boundary_classification.csv",
    alias_groups: Path = REPO_ROOT / "configs" / "dataset_delivery" / "task1" / "task1_alias_groups.csv",
    task2_targets: Path = REPO_ROOT / "configs" / "dataset_delivery" / "task1" / "task2_generate_targets_23.csv",
    checkpoint_root: Path | None = None,
    hierarchy_config: Path | None = None,
    require_hierarchy: bool = False,
    run_labelcritic_request: bool = False,
    labelcritic_ct: Path | None = None,
    labelcritic_mask_a: Path | None = None,
    labelcritic_mask_b: Path | None = None,
    output_json: Path | None = None,
    allow_dirty_tracked: bool = False,
) -> dict[str, Any]:
    from tools.dataset_delivery.delivery_lib import validate_rename_mapping

    workspace_root = workspace_root.resolve()
    checks: list[dict[str, Any]] = []
    branch = _git(["branch", "--show-current"], code_root)
    commit = _git(["rev-parse", "HEAD"], code_root)
    dirty = _git(["status", "--porcelain", "--untracked-files=no"], code_root)
    checks.append(_check("git_branch_main", branch == "main", branch))
    checks.append(_check("git_commit_recorded", bool(commit), commit))
    checks.append(_check("working_tree_expected_clean", allow_dirty_tracked or not dirty, dirty))
    checks.append(_check("workspace_root_not_public_data", not str(workspace_root).startswith("/projects/bodymaps/Data"), str(workspace_root)))
    checks.append(_check("workspace_root_exists", workspace_root.exists(), str(workspace_root)))
    checks.append(_check("workspace_root_writable", workspace_root.exists() and os.access(workspace_root, os.W_OK), str(workspace_root)))
    for rel in WORKSPACE_DIRS:
        checks.append(_check(f"workspace_dir_{rel.replace('/', '_')}", (workspace_root / rel).is_dir(), str(workspace_root / rel)))
    try:
        manifest = validate_formal_manifest(manifest=cases_manifest, base_manifest=base_manifest, check_exists=True)
        checks.append(_check("cases_103_manifest", manifest["rows"] == FORMAL_CASE_COUNT and manifest["unique_case_count"] == FORMAL_CASE_COUNT, ""))
    except Exception as exc:
        manifest = {"status": "failed", "errors": [str(exc)]}
        checks.append(_check("cases_103_manifest", False, str(exc)))
    try:
        rename = validate_rename_mapping(
            rename_mapping,
            taxonomy_path,
            task2_targets=task2_targets,
            alias_groups=alias_groups,
            boundary_classification=boundary_classification,
        )
        checks.append(_check("task1_mapping", rename["status"] == "success", ""))
    except Exception as exc:
        rename = {"status": "failed", "errors": [str(exc)]}
        checks.append(_check("task1_mapping", False, str(exc)))
    registry = load_registry(registry_path)
    routes = candidate_models_for_organs(registry, list(FORMAL_TARGETS))
    missing_routes = [target for target in FORMAL_TARGETS if not routes.get(target)]
    checks.append(_check("multi_teacher_candidate_routes", not missing_routes, ",".join(missing_routes)))
    if checkpoint_root is not None:
        checks.append(_check("checkpoint_root_exists", checkpoint_root.exists(), str(checkpoint_root)))
    known_nodes = set(load_taxonomy_names(taxonomy_path))
    hierarchy = validate_hierarchy_config(hierarchy_config, known_nodes=known_nodes, required=require_hierarchy)
    checks.append(_check("hierarchy_config", hierarchy["status"] in {"READY", "DISABLED"}, hierarchy.get("reason", ""), hierarchy=hierarchy))
    labelcritic = build_labelcritic_preflight(
        output_json=None,
        run_request=run_labelcritic_request,
        ct_path=labelcritic_ct,
        mask_a=labelcritic_mask_a,
        mask_b=labelcritic_mask_b,
    )
    labelcritic_profile = ((labelcritic.get("resource_selection") or {}).get("selected_profile") or {})
    checks.append(
        _check(
            "labelcritic_72b_resource",
            labelcritic["status"] == "READY" and not bool(labelcritic_profile.get("smoke_only")),
            labelcritic.get("status", ""),
            labelcritic=labelcritic,
        )
    )
    checks.append(
        _check(
            "fixed_round1_manifest_not_100",
            manifest.get("rows") == FORMAL_CASE_COUNT
            and manifest.get("unique_case_count") == FORMAL_CASE_COUNT
            and "100cases" not in cases_manifest.name,
            str(cases_manifest),
        )
    )
    status = "READY" if all(check["ok"] for check in checks) else "BLOCKED"
    report = {
        "status": status,
        "branch": branch,
        "commit": commit,
        "workspace_root": str(workspace_root),
        "cases_manifest": str(cases_manifest),
        "case_count": manifest.get("rows", 0),
        "expected_case_count": FORMAL_CASE_COUNT,
        "formal_groups": FORMAL_GROUP_MODELS,
        "formal_targets": FORMAL_MODEL_TARGETS,
        "checks": checks,
        "blocked_checks": [check for check in checks if not check["ok"]],
        "manifest": manifest,
        "rename_mapping": rename,
        "hierarchy": hierarchy,
        "labelcritic": labelcritic,
    }
    if output_json:
        write_json(output_json, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Blocking formal preflight for the 103-case EM Round 1 launch.")
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--cases-manifest", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--code-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--hierarchy-config", type=Path)
    parser.add_argument("--require-hierarchy", action="store_true")
    parser.add_argument("--run-labelcritic-request", action="store_true")
    parser.add_argument("--labelcritic-ct", type=Path)
    parser.add_argument("--labelcritic-mask-a", type=Path)
    parser.add_argument("--labelcritic-mask-b", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--allow-dirty-tracked", action="store_true")
    args = parser.parse_args()
    report = build_formal_round1_preflight(
        workspace_root=args.workspace_root,
        cases_manifest=args.cases_manifest.resolve(),
        base_manifest=args.base_manifest.resolve(),
        code_root=args.code_root.resolve(),
        checkpoint_root=args.checkpoint_root.resolve() if args.checkpoint_root else None,
        hierarchy_config=args.hierarchy_config.resolve() if args.hierarchy_config else None,
        require_hierarchy=bool(args.require_hierarchy),
        run_labelcritic_request=bool(args.run_labelcritic_request),
        labelcritic_ct=args.labelcritic_ct.resolve() if args.labelcritic_ct else None,
        labelcritic_mask_a=args.labelcritic_mask_a.resolve() if args.labelcritic_mask_a else None,
        labelcritic_mask_b=args.labelcritic_mask_b.resolve() if args.labelcritic_mask_b else None,
        output_json=args.output_json.resolve() if args.output_json else None,
        allow_dirty_tracked=bool(args.allow_dirty_tracked),
    )
    print(json.dumps({"status": report["status"], "blocked": [c["name"] for c in report["blocked_checks"]]}, indent=2))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
