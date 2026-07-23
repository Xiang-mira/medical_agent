from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .utils import ROOT, SchedulerError


@dataclass(frozen=True)
class SchedulerConfig:
    path: Path
    data: dict[str, Any]

    @property
    def project(self) -> dict[str, Any]:
        return self.data.get("project", {})

    @property
    def paths(self) -> dict[str, Any]:
        return self.data.get("paths", {})

    @property
    def slurm_defaults(self) -> dict[str, Any]:
        return self.data.get("slurm_defaults", {})


def resolve_path(value: str | Path | None, *, base: Path = ROOT) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SchedulerError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SchedulerError(f"Config must be a YAML mapping: {path}")
    return data


def load_config(path: str | Path) -> SchedulerConfig:
    resolved = resolve_path(path)
    assert resolved is not None
    data = load_yaml(resolved)
    validate_config_dict(data, source=resolved)
    return SchedulerConfig(path=resolved, data=data)


def validate_config_dict(data: dict[str, Any], *, source: Path | None = None) -> None:
    required_top = {"project", "paths", "resource_policy"}
    missing = sorted(required_top - set(data))
    if missing:
        raise SchedulerError(f"Missing config sections {missing} in {source or '<dict>'}")
    project = data.get("project") or {}
    if project.get("full_target_count") != 373:
        raise SchedulerError("project.full_target_count must be 373")
    if project.get("pilot_target_count") != 338:
        raise SchedulerError("project.pilot_target_count must be 338 for abdomenatlaspro_pilot338")
    if str(project.get("target_terminology")) != "target_anatomical_structures":
        raise SchedulerError("project.target_terminology must be target_anatomical_structures")


def load_pipeline(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    assert resolved is not None
    data = load_yaml(resolved)
    if "tasks" not in data or not isinstance(data["tasks"], dict):
        raise SchedulerError(f"Pipeline must define tasks mapping: {resolved}")
    return data


def load_resource_profiles(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    assert resolved is not None
    data = load_yaml(resolved)
    profiles = data.get("resource_profiles")
    if not isinstance(profiles, dict):
        raise SchedulerError(f"Resource profile file must define resource_profiles: {resolved}")
    return profiles
