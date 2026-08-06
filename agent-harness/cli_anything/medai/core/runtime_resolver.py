from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CANONICAL_CODE_ROOT = Path("/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent")
DEFAULT_HPC_CHECKPOINT_ROOT = Path("/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints")
DEFAULT_NNUNETV2_PREDICT = Path("/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict")
DEFAULT_NNUNETV2_PREDICT_FROM_MODELFOLDER = Path("/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict_from_modelfolder")
DEFAULT_UNEST_PYTHON = Path("/home/xhan74/envs/medical_agent_train_py311/bin/python")
DEFAULT_OUTER_PYTHON = Path("/home/xhan74/envs/medical_agent/bin/python")


@dataclass(frozen=True)
class ResolvedPath:
    raw: str
    base_kind: str
    base_root: str
    resolved: str

    def as_dict(self) -> dict[str, str]:
        return {
            "raw": self.raw,
            "base_kind": self.base_kind,
            "base_root": self.base_root,
            "resolved": self.resolved,
        }


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[4]


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _is_under_checkpoint_namespace(value: str) -> bool:
    parts = Path(value).parts
    return bool(parts and parts[0] == "checkpoints")


def resolve_checkpoint_root(
    *,
    explicit: str | Path | None = None,
    registry_checkpoint_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    repo = Path(repo_root).resolve() if repo_root else repo_root_from_module()
    raw = (
        str(explicit).strip()
        if explicit
        else _env_first("MEDAI_CHECKPOINT_ROOT")
        or str(registry_checkpoint_root or "").strip()
    )
    if not raw or raw == "checkpoints":
        env_value = _env_first("MEDAI_CHECKPOINT_ROOT")
        if env_value:
            return Path(env_value).expanduser().resolve()
        if DEFAULT_HPC_CHECKPOINT_ROOT.exists():
            return DEFAULT_HPC_CHECKPOINT_ROOT
        return (repo / "checkpoints").resolve()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    if _is_under_checkpoint_namespace(raw):
        return (repo / path).resolve()
    return (repo / path).resolve()


def resolve_registry_path(
    value: str | Path | None,
    *,
    repo_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    path_kind: str = "repo",
) -> ResolvedPath:
    raw = str(value or "").strip()
    repo = Path(repo_root).resolve() if repo_root else repo_root_from_module()
    ckpt = Path(checkpoint_root).resolve() if checkpoint_root else resolve_checkpoint_root(repo_root=repo)
    if not raw:
        return ResolvedPath(raw="", base_kind="none", base_root="", resolved="")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return ResolvedPath(raw=raw, base_kind="absolute", base_root="", resolved=str(path.resolve()))
    if path_kind in {"checkpoint", "dataset_json", "model"} or _is_under_checkpoint_namespace(raw):
        parts = path.parts
        rel = Path(*parts[1:]) if parts and parts[0] == "checkpoints" else path
        resolved = (ckpt / rel).resolve()
        return ResolvedPath(raw=raw, base_kind="checkpoint_root", base_root=str(ckpt), resolved=str(resolved))
    resolved = (repo / path).resolve()
    return ResolvedPath(raw=raw, base_kind="repo_root", base_root=str(repo), resolved=str(resolved))


def executable_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "is_file": path.is_file() if exists else False,
        "is_executable": os.access(path, os.X_OK) if exists else False,
        "readable": os.access(path, os.R_OK) if exists else False,
    }


def resolve_executable(
    *,
    explicit: str | Path | None = None,
    env_names: tuple[str, ...] = (),
    registry_value: str | Path | None = None,
    fallback: str | Path | None = None,
    name_for_path_lookup: str | None = None,
    require_exists: bool = False,
) -> Path:
    raw = (
        str(explicit).strip()
        if explicit
        else _env_first(*env_names)
        or (str(registry_value).strip() if registry_value else "")
    )
    if raw:
        path = Path(raw).expanduser()
        if path.is_absolute() or any(sep in raw for sep in ("/", "\\")):
            resolved = path.resolve()
        else:
            found = shutil.which(raw)
            resolved = Path(found).resolve() if found else Path(fallback or raw).expanduser().resolve()
    elif name_for_path_lookup:
        found = shutil.which(name_for_path_lookup)
        resolved = Path(found).resolve() if found else Path(fallback or name_for_path_lookup).expanduser().resolve()
    else:
        resolved = Path(fallback or "").expanduser().resolve()
    if require_exists:
        status = executable_status(resolved)
        if not status["exists"]:
            raise FileNotFoundError(f"executable not found: {resolved}")
        if not status["is_file"]:
            raise FileNotFoundError(f"executable is not a file: {resolved}")
        if not status["is_executable"]:
            raise PermissionError(f"executable is not executable: {resolved}")
    return resolved


def resolve_nnunet_predictor(
    *,
    explicit: str | Path | None = None,
    registry_value: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    return resolve_executable(
        explicit=explicit,
        env_names=("NNUNETV2_PREDICT_EXECUTABLE", "MEDAI_NNUNETV2_PREDICT"),
        registry_value=registry_value,
        fallback=DEFAULT_NNUNETV2_PREDICT,
        name_for_path_lookup="nnUNetv2_predict",
        require_exists=require_exists,
    )


def resolve_nnunet_predict_from_modelfolder(
    *,
    explicit: str | Path | None = None,
    registry_value: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    return resolve_executable(
        explicit=explicit,
        env_names=("NNUNETV2_PREDICT_FROM_MODELFOLDER_EXECUTABLE", "MEDAI_NNUNETV2_PREDICT_FROM_MODELFOLDER"),
        registry_value=registry_value,
        fallback=DEFAULT_NNUNETV2_PREDICT_FROM_MODELFOLDER,
        name_for_path_lookup="nnUNetv2_predict_from_modelfolder",
        require_exists=require_exists,
    )


def resolve_unest_python(
    *,
    explicit: str | Path | None = None,
    registry_value: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    return resolve_executable(
        explicit=explicit,
        env_names=("UNEST_PYTHON_EXECUTABLE", "MEDAI_UNEST_PYTHON"),
        registry_value=registry_value,
        fallback=DEFAULT_UNEST_PYTHON,
        name_for_path_lookup=None,
        require_exists=require_exists,
    )


def run_help_check(executable: Path, *, timeout_sec: int = 30) -> dict[str, Any]:
    proc = subprocess.run(
        [str(executable), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_sec,
    )
    return {
        "command": [str(executable), "--help"],
        "return_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
        "ok": proc.returncode == 0,
    }


def scrub_env_for_runner(env: dict[str, str], *, runner: str) -> dict[str, str]:
    cleaned = dict(env)
    if runner == "unest":
        for key in list(cleaned):
            if key.startswith("MEDAI_NNUNET_") or key in {
                "NNUNETV2_PREDICT_EXECUTABLE",
                "NNUNETV2_PREDICT_FROM_MODELFOLDER_EXECUTABLE",
                "MEDAI_NNUNETV2_PREDICT",
                "MEDAI_NNUNETV2_PREDICT_FROM_MODELFOLDER",
            }:
                cleaned.pop(key, None)
    return cleaned
