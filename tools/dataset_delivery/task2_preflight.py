#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.model_registry import get_model_entry, load_registry  # noqa: E402
from cli_anything.medai.core.runtime_resolver import (  # noqa: E402
    DEFAULT_CANONICAL_CODE_ROOT,
    DEFAULT_HPC_CHECKPOINT_ROOT,
    DEFAULT_NNUNETV2_PREDICT,
    DEFAULT_OUTER_PYTHON,
    DEFAULT_UNEST_PYTHON,
    executable_status,
    resolve_checkpoint_root,
    resolve_nnunet_predictor,
    resolve_registry_path,
    resolve_unest_python,
    resolve_unest_python_details,
    run_help_check,
)
from tools.dataset_delivery.delivery_lib import read_csv_rows, write_json  # noqa: E402


NNUNET_MODELS = {"atm", "airrc"}
CADS_MODELS = {f"cads{i}" for i in range(551, 560)}
UNEST_MODELS = {"unest"}
NNUNET_EXPECTED_LABELS: dict[str, dict[str, int]] = {
    "atm": {"airway_tree": 1},
    "airrc": {
        "airway_wall": 2,
        "lung_pulmonary_arteries": 3,
        "lung_pulmonary_veins": 4,
    },
}
REQUIRED_UNEST_FILES = [
    "models/model.pt",
    "configs/metadata.json",
    "configs/inference.json",
    "configs/logging.conf",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_check(path: Path, *, must_be_file: bool | None = None, sha256: bool = False) -> dict[str, Any]:
    exists = path.exists()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "readable": os.access(path, os.R_OK) if exists else False,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "ok": exists and os.access(path, os.R_OK),
    }
    if must_be_file is True:
        result["is_file"] = path.is_file() if exists else False
        result["ok"] = bool(result["ok"] and result["is_file"] and int(result["size_bytes"] or 0) > 0)
    elif must_be_file is False:
        result["is_dir"] = path.is_dir() if exists else False
        result["ok"] = bool(result["ok"] and result["is_dir"])
    if sha256 and result["ok"] and path.is_file():
        result["sha256"] = _sha256_file(path)
    return result


def _case_id(row: dict[str, str], index: int) -> str:
    return row.get("case_id") or row.get("id") or f"case_{index:03d}"


def _ct_path(row: dict[str, str]) -> str:
    return row.get("ct_path") or row.get("image_path") or ""


def _annotation_folder(row: dict[str, str]) -> str:
    return row.get("annotation_folder") or row.get("reference_mask_dir") or row.get("mask_dir") or ""


def _repo_checks(repo_root: Path, canonical_code_root: Path, formal_mode: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    rc, top, err = _run_git(["rev-parse", "--show-toplevel"], repo_root)
    git_repo_ok = rc == 0 and bool(top)
    checks.append({"name": "git_repository", "ok": git_repo_ok, "stdout": top, "stderr": err})
    rc_branch, branch, _ = _run_git(["branch", "--show-current"], repo_root)
    checks.append({"name": "git_branch_main", "ok": rc_branch == 0 and branch == "main", "branch": branch})
    rc_head, head, _ = _run_git(["rev-parse", "HEAD"], repo_root)
    checks.append({"name": "git_head", "ok": rc_head == 0 and bool(head), "commit": head})
    rc_status, status, _ = _run_git(["status", "--short", "--untracked-files=no"], repo_root)
    checks.append({"name": "tracked_working_tree_clean", "ok": rc_status == 0 and status == "", "status_short": status})
    rc_common, common_dir, _ = _run_git(["rev-parse", "--path-format=absolute", "--git-common-dir"], repo_root)
    is_worktree = rc_common == 0 and common_dir and Path(common_dir).resolve() != (repo_root / ".git").resolve()
    checks.append({"name": "worktree_detected", "ok": True, "is_worktree": bool(is_worktree), "git_common_dir": common_dir})
    canonical_ok = repo_root.resolve() == canonical_code_root.resolve()
    checks.append({
        "name": "canonical_main_root",
        "ok": (not formal_mode) or canonical_ok,
        "repo_root": str(repo_root.resolve()),
        "canonical_code_root": str(canonical_code_root.resolve()),
        "formal_mode": formal_mode,
    })
    meta = {
        "repository_root": str(repo_root.resolve()),
        "canonical_code_root": str(canonical_code_root.resolve()),
        "branch": branch,
        "commit": head,
        "is_worktree": bool(is_worktree),
        "git_common_dir": common_dir,
    }
    return checks, meta


def _data_checks(case_list: Path, output_root: Path, formal_mode: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append({"name": "case_manifest_exists", **_path_check(case_list, must_be_file=True)})
    cases: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    if case_list.exists():
        rows = read_csv_rows(case_list)
        for index, row in enumerate(rows):
            case_id = _case_id(row, index)
            if case_id in seen:
                duplicate_ids.append(case_id)
            seen.add(case_id)
            ct = Path(_ct_path(row))
            ann = Path(_annotation_folder(row))
            cases.append({
                "case_id": case_id,
                "ct_path": str(ct),
                "annotation_folder": str(ann),
                "ct_ok": ct.exists() and ct.is_file() and ct.stat().st_size > 0,
                "annotation_ok": ann.exists() and os.access(ann, os.R_OK),
            })
    checks.append({"name": "case_ids_unique", "ok": not duplicate_ids, "duplicates": duplicate_ids})
    checks.append({"name": "case_ct_files_readable", "ok": all(case["ct_ok"] for case in cases) if cases else False, "bad_cases": [case for case in cases if not case["ct_ok"]][:20]})
    checks.append({"name": "annotation_folders_readable", "ok": all(case["annotation_ok"] for case in cases) if cases else False, "bad_cases": [case for case in cases if not case["annotation_ok"]][:20]})
    preexisting = output_root.exists()
    checks.append({
        "name": "output_root_new",
        "ok": (not formal_mode) or not preexisting,
        "output_root": str(output_root),
        "preexisting": preexisting,
        "formal_mode": formal_mode,
    })
    return checks, {"cases": cases, "case_count": len(cases), "output_root": str(output_root)}


def _trainer_dir_from_entry(entry: dict[str, Any], repo_root: Path, checkpoint_root: Path) -> Path | None:
    dataset_json = entry.get("dataset_json_path")
    if dataset_json:
        return Path(resolve_registry_path(dataset_json, repo_root=repo_root, checkpoint_root=checkpoint_root, path_kind="dataset_json").resolved).parent
    checkpoint_path = entry.get("checkpoint_path")
    if checkpoint_path:
        return Path(resolve_registry_path(checkpoint_path, repo_root=repo_root, checkpoint_root=checkpoint_root, path_kind="checkpoint").resolved)
    return None


def _nnunet_model_checks(
    *,
    model_key: str,
    entry: dict[str, Any],
    repo_root: Path,
    checkpoint_root: Path,
    predictor: Path,
    run_predictor_help: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    resolved_checkpoint = resolve_registry_path(entry.get("checkpoint_path"), repo_root=repo_root, checkpoint_root=checkpoint_root, path_kind="checkpoint")
    resolved_dataset_json = resolve_registry_path(entry.get("dataset_json_path"), repo_root=repo_root, checkpoint_root=checkpoint_root, path_kind="dataset_json")
    trainer_dir = _trainer_dir_from_entry(entry, repo_root, checkpoint_root)
    checks.append({"name": f"{model_key}_checkpoint_root", **_path_check(Path(resolved_checkpoint.resolved), must_be_file=False)})
    checks.append({"name": f"{model_key}_dataset_json", **_path_check(Path(resolved_dataset_json.resolved), must_be_file=True, sha256=True)})
    dataset_labels: dict[str, Any] = {}
    if Path(resolved_dataset_json.resolved).exists():
        try:
            dataset_doc = json.loads(Path(resolved_dataset_json.resolved).read_text(encoding="utf-8"))
            raw_labels = dataset_doc.get("labels") or {}
            if isinstance(raw_labels, dict):
                dataset_labels = raw_labels
        except Exception as exc:
            checks.append({"name": f"{model_key}_dataset_json_parse", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    if trainer_dir is not None:
        checks.append({"name": f"{model_key}_plans_json", **_path_check(trainer_dir / "plans.json", must_be_file=True)})
        fold = str(entry.get("folds") or "all")
        checkpoint_name = str(entry.get("checkpoint_name") or "checkpoint_final")
        if not checkpoint_name.endswith(".pth"):
            checkpoint_name += ".pth"
        fold_dir = trainer_dir / f"fold_{fold}"
        checks.append({"name": f"{model_key}_fold_dir", **_path_check(fold_dir, must_be_file=False)})
        checks.append({"name": f"{model_key}_checkpoint_file", **_path_check(fold_dir / checkpoint_name, must_be_file=True, sha256=True)})
    for label_name, expected_id in NNUNET_EXPECTED_LABELS.get(model_key, {}).items():
        actual = dataset_labels.get(label_name)
        checks.append({
            "name": f"{model_key}_label_{label_name}",
            "ok": actual == expected_id,
            "label": label_name,
            "expected_label_id": expected_id,
            "dataset_label_id": actual,
        })
    predictor_check = {"name": f"{model_key}_predictor_executable", **executable_status(predictor)}
    predictor_check["ok"] = bool(predictor_check["exists"] and predictor_check["is_file"] and predictor_check["is_executable"])
    checks.append(predictor_check)
    help_result: dict[str, Any] = {"skipped": not run_predictor_help}
    if run_predictor_help and predictor_check["ok"]:
        try:
            help_result = run_help_check(predictor)
        except Exception as exc:
            help_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        checks.append({"name": f"{model_key}_predictor_help", "ok": bool(help_result.get("ok")), **help_result})
    meta = {
        "checkpoint_path": resolved_checkpoint.as_dict(),
        "dataset_json_path": resolved_dataset_json.as_dict(),
        "trainer_dir": str(trainer_dir) if trainer_dir else None,
        "predictor_executable": str(predictor),
        "predictor_help": help_result,
        "dataset_id": entry.get("dataset_id"),
        "trainer": entry.get("trainer"),
        "plans": entry.get("plans"),
        "folds": entry.get("folds"),
        "checkpoint_name": entry.get("checkpoint_name"),
        "expected_labels": NNUNET_EXPECTED_LABELS.get(model_key, {}),
        "dataset_labels": dataset_labels,
    }
    return checks, meta


def _run_python_probe(python: Path, code: str, *, timeout_sec: int = 60) -> dict[str, Any]:
    proc = subprocess.run(
        [str(python), "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_sec,
    )
    return {
        "command": [str(python), "-c", code],
        "return_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr_tail": (proc.stderr or "")[-2000:],
        "ok": proc.returncode == 0,
    }


def _unest_checks(
    *,
    entry: dict[str, Any],
    repo_root: Path,
    checkpoint_root: Path,
    unest_python: Path,
    unest_python_details: dict[str, str],
    require_cuda: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    root = Path(resolve_registry_path(entry.get("checkpoint_path"), repo_root=repo_root, checkpoint_root=checkpoint_root, path_kind="checkpoint").resolved)
    source = Path(resolve_registry_path(entry.get("source_code_path"), repo_root=repo_root, checkpoint_root=checkpoint_root, path_kind="checkpoint").resolved)
    py_status = executable_status(unest_python)
    py_status["ok"] = bool(py_status["exists"] and py_status["is_file"] and py_status["is_executable"])
    checks.append({"name": "unest_python_executable", **py_status})
    checks.append({"name": "unest_checkpoint_root", **_path_check(root, must_be_file=False)})
    checks.append({"name": "unest_source_code_path", **_path_check(source, must_be_file=False)})
    for rel in REQUIRED_UNEST_FILES:
        checks.append({"name": f"unest_{rel.replace('/', '_')}", **_path_check(root / rel, must_be_file=True, sha256=rel == "models/model.pt")})
    import_probe: dict[str, Any] = {"skipped": not py_status["ok"]}
    monai_help: dict[str, Any] = {"skipped": not py_status["ok"]}
    cuda_probe: dict[str, Any] = {"skipped": not py_status["ok"] or not require_cuda}
    if py_status["ok"]:
        try:
            import_probe = _run_python_probe(
                unest_python,
                "import json, sys, monai, torch; print(json.dumps({'sys_executable': sys.executable, 'sys_prefix': sys.prefix, 'sys_base_prefix': sys.base_prefix, 'monai_version': monai.__version__, 'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda}))",
            )
        except Exception as exc:
            import_probe = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if import_probe.get("ok") and import_probe.get("stdout"):
            try:
                import_probe["parsed"] = json.loads(str(import_probe["stdout"]))
            except Exception:
                pass
        checks.append({"name": "unest_import_monai_torch", "ok": bool(import_probe.get("ok")), **import_probe})
        try:
            proc = subprocess.run(
                [str(unest_python), "-m", "monai.bundle", "--help"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
            )
            monai_help = {"return_code": proc.returncode, "stdout_tail": (proc.stdout or "")[-1000:], "stderr_tail": (proc.stderr or "")[-1000:], "ok": proc.returncode == 0}
        except Exception as exc:
            monai_help = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        checks.append({"name": "unest_monai_bundle_help", "ok": bool(monai_help.get("ok")), **monai_help})
        if require_cuda:
            try:
                cuda_probe = _run_python_probe(unest_python, "import torch; raise SystemExit(0 if torch.cuda.is_available() else 3)")
            except Exception as exc:
                cuda_probe = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            checks.append({"name": "unest_cuda_available", "ok": bool(cuda_probe.get("ok")), **cuda_probe})
    return checks, {
        "checkpoint_path": str(root),
        "source_code_path": str(source),
        "python_executable": str(unest_python),
        "python_resolution": unest_python_details,
        "import_probe": import_probe,
        "monai_bundle_help": monai_help,
        "cuda_probe": cuda_probe,
    }


def build_preflight(
    *,
    models: list[str],
    case_list: Path,
    output_root: Path,
    registry_path: Path,
    target_config: Path,
    checkpoint_root_arg: Path | None,
    json_output: Path,
    canonical_code_root: Path,
    formal_mode: bool,
    outer_python: Path | None = None,
    nnunet_predict_executable: str | None = None,
    unest_python_executable: str | None = None,
    run_predictor_help: bool = True,
    require_cuda: bool = False,
) -> dict[str, Any]:
    repo_root = REPO_ROOT.resolve()
    registry = load_registry(registry_path)
    checkpoint_root = resolve_checkpoint_root(
        explicit=checkpoint_root_arg,
        registry_checkpoint_root=registry.get("checkpoint_root"),
        repo_root=repo_root,
    )
    predictor = resolve_nnunet_predictor(explicit=nnunet_predict_executable, require_exists=False)
    unest_python_resolution = resolve_unest_python_details(
        explicit=unest_python_executable,
        registry_value=(registry.get("models", {}).get("unest", {}) or {}).get("unest_python_executable"),
        require_exists=False,
    )
    unest_python = Path(unest_python_resolution.execution_path)
    resolved_outer_python = Path(outer_python or DEFAULT_OUTER_PYTHON).expanduser().resolve()
    checks: list[dict[str, Any]] = []
    repo_checks, repo_meta = _repo_checks(repo_root, canonical_code_root, formal_mode)
    data_checks, data_meta = _data_checks(case_list, output_root, formal_mode)
    checks.extend(repo_checks)
    checks.extend(data_checks)
    checks.append({"name": "target_config", **_path_check(target_config, must_be_file=True)})
    checks.append({"name": "registry", **_path_check(registry_path, must_be_file=True)})
    checks.append({"name": "checkpoint_root", **_path_check(checkpoint_root, must_be_file=False)})
    outer_status = executable_status(resolved_outer_python)
    outer_status["ok"] = bool(outer_status["exists"] and outer_status["is_file"] and outer_status["is_executable"])
    checks.append({"name": "outer_python_executable", **outer_status})

    model_meta: dict[str, Any] = {}
    for model_key in models:
        try:
            entry = get_model_entry(registry, model_key)
        except Exception as exc:
            checks.append({"name": f"{model_key}_registry_entry", "ok": False, "error": str(exc)})
            continue
        checks.append({"name": f"{model_key}_registry_entry", "ok": True})
        if model_key in NNUNET_MODELS or model_key in CADS_MODELS:
            model_checks, meta = _nnunet_model_checks(
                model_key=model_key,
                entry=entry,
                repo_root=repo_root,
                checkpoint_root=checkpoint_root,
                predictor=predictor,
                run_predictor_help=run_predictor_help,
            )
            checks.extend(model_checks)
            model_meta[model_key] = meta
        elif model_key in UNEST_MODELS:
            model_checks, meta = _unest_checks(
                entry=entry,
                repo_root=repo_root,
                checkpoint_root=checkpoint_root,
                unest_python=unest_python,
                unest_python_details=unest_python_resolution.as_dict(),
                require_cuda=require_cuda,
            )
            checks.extend(model_checks)
            model_meta[model_key] = meta
        else:
            resolved = resolve_registry_path(entry.get("checkpoint_path"), repo_root=repo_root, checkpoint_root=checkpoint_root, path_kind="checkpoint")
            checks.append({"name": f"{model_key}_checkpoint_path", **_path_check(Path(resolved.resolved), must_be_file=None)})
            model_meta[model_key] = {"checkpoint_path": resolved.as_dict()}

    blocked_checks = [check for check in checks if not bool(check.get("ok", False))]
    status = "READY" if not blocked_checks else "BLOCKED"
    env_manifest = {
        "PATH": os.getenv("PATH", ""),
        "PYTHONPATH": os.getenv("PYTHONPATH", ""),
        "MEDAI_CHECKPOINT_ROOT": os.getenv("MEDAI_CHECKPOINT_ROOT", ""),
        "NNUNETV2_PREDICT_EXECUTABLE": os.getenv("NNUNETV2_PREDICT_EXECUTABLE", ""),
        "MEDAI_NNUNETV2_PREDICT": os.getenv("MEDAI_NNUNETV2_PREDICT", ""),
        "UNEST_PYTHON_EXECUTABLE": os.getenv("UNEST_PYTHON_EXECUTABLE", ""),
        "MEDAI_UNEST_PYTHON": os.getenv("MEDAI_UNEST_PYTHON", ""),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "nnUNet_raw": os.getenv("nnUNet_raw", ""),
        "nnUNet_preprocessed": os.getenv("nnUNet_preprocessed", ""),
        "nnUNet_results": os.getenv("nnUNet_results", ""),
    }
    manifest = {
        "timestamp": _utc_now(),
        **repo_meta,
        "formal_mode": formal_mode,
        "model_keys": models,
        "case_list": str(case_list),
        "target_config": str(target_config),
        "registry_path": str(registry_path),
        "checkpoint_root": str(checkpoint_root),
        "default_hpc_checkpoint_root": str(DEFAULT_HPC_CHECKPOINT_ROOT),
        "outer_python": str(resolved_outer_python),
        "outer_python_default": str(DEFAULT_OUTER_PYTHON),
        "nnunet_predictor": str(predictor),
        "nnunet_predictor_default": str(DEFAULT_NNUNETV2_PREDICT),
        "unest_python": str(unest_python),
        "unest_python_resolution": unest_python_resolution.as_dict(),
        "unest_python_default": str(DEFAULT_UNEST_PYTHON),
        "output_root": str(output_root),
        "environment": env_manifest,
        "data": data_meta,
        "models": model_meta,
        "preflight_status": status,
    }
    report = {
        "status": status,
        "blocked_count": len(blocked_checks),
        "blocked_checks": blocked_checks,
        "checks": checks,
        "runtime_manifest": manifest,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(json_output, report)
    write_json(json_output.parent / "runtime_manifest.json", manifest)
    return report


def parse_models(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 2 Teacher runtime preflight.")
    parser.add_argument("--models", required=True, help="Comma-separated model keys, e.g. atm,cads553,cads557,cads559,unest")
    parser.add_argument("--case-list", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", default=None, type=Path)
    parser.add_argument("--registry", default=REPO_ROOT / "configs" / "model_registry.yaml", type=Path)
    parser.add_argument("--target-config", default=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json", type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--canonical-code-root", default=DEFAULT_CANONICAL_CODE_ROOT, type=Path)
    parser.add_argument("--formal-mode", action="store_true")
    parser.add_argument("--python-executable", default=DEFAULT_OUTER_PYTHON, type=Path, help="Outer run-loop Python executable.")
    parser.add_argument("--nnunet-predict-executable", default=None)
    parser.add_argument("--unest-python-executable", default=None)
    parser.add_argument("--require-cuda", action="store_true", help="Require torch.cuda.is_available() for UNEST; intended for compute-node GPU smoke checks.")
    parser.add_argument("--skip-predictor-help", action="store_true", help="Skip nnUNetv2_predict --help probe for unit tests or offline diagnostics.")
    args = parser.parse_args()
    report = build_preflight(
        models=parse_models(args.models),
        case_list=args.case_list.resolve(),
        output_root=args.output_root.resolve(),
        registry_path=args.registry.resolve(),
        target_config=args.target_config.resolve(),
        checkpoint_root_arg=args.checkpoint_root,
        json_output=args.json_output.resolve(),
        canonical_code_root=args.canonical_code_root,
        formal_mode=bool(args.formal_mode),
        outer_python=args.python_executable,
        nnunet_predict_executable=args.nnunet_predict_executable,
        unest_python_executable=args.unest_python_executable,
        run_predictor_help=not args.skip_predictor_help,
        require_cuda=bool(args.require_cuda),
    )
    print(json.dumps({"status": report["status"], "blocked_count": report["blocked_count"], "json_output": str(args.json_output)}, indent=2))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
