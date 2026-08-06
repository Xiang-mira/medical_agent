from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


EXPLICIT_MODEL_ALIASES = {
    "airrc": "airrc",
    "air_rc": "airrc",
    "airway_rc": "airrc",
    "dataset1380_airrc": "airrc",
    "dataset_1380_airrc": "airrc",
    "atm": "atm",
    "dataset1370_atm": "atm",
    "dataset_1370_atm": "atm",
    "unest": "unest",
    "cads553": "cads553",
    "dataset553_totalseg253": "cads553",
    "cads557": "cads557",
    "dataset557_brain257": "cads557",
    "cads559": "cads559",
    "dataset559_saros259": "cads559",
}


@dataclass(frozen=True)
class ModelKeyResolution:
    raw: str
    resolved: str | None
    ok: bool
    source: str
    reason: str = ""


def normalize_model_token(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _registry_aliases(registry: dict[str, Any] | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    models = (registry or {}).get("models", {}) or {}
    for key, entry in models.items():
        model_key = str(key)
        aliases[normalize_model_token(model_key)] = model_key
        for alias in entry.get("routing_aliases", []) or []:
            aliases[normalize_model_token(alias)] = model_key
        name = entry.get("name")
        if name:
            aliases[normalize_model_token(str(name))] = model_key
        dataset_id = entry.get("dataset_id")
        if dataset_id:
            aliases[f"dataset{dataset_id}_{normalize_model_token(model_key)}"] = model_key
            aliases[f"dataset_{dataset_id}_{normalize_model_token(model_key)}"] = model_key
            dataset_json = str(entry.get("dataset_json_path") or "")
            match = re.search(r"(Dataset\d+_[^/]+)", dataset_json)
            if match:
                aliases[normalize_model_token(match.group(1))] = model_key
    return aliases


def resolve_model_key(raw: Any, registry: dict[str, Any] | None = None) -> ModelKeyResolution:
    text = str(raw or "").strip()
    if not text:
        return ModelKeyResolution(raw="", resolved=None, ok=False, source="empty", reason="empty_model_key")
    models = (registry or {}).get("models", {}) or {}
    if text in models:
        return ModelKeyResolution(raw=text, resolved=text, ok=True, source="registry_key")
    token = normalize_model_token(text)
    if token in EXPLICIT_MODEL_ALIASES:
        resolved = EXPLICIT_MODEL_ALIASES[token]
        if not models or resolved in models:
            return ModelKeyResolution(raw=text, resolved=resolved, ok=True, source="explicit_alias")
        return ModelKeyResolution(raw=text, resolved=resolved, ok=False, source="explicit_alias", reason=f"alias_target_not_in_registry:{resolved}")
    registry_aliases = _registry_aliases(registry)
    if token in registry_aliases:
        return ModelKeyResolution(raw=text, resolved=registry_aliases[token], ok=True, source="registry_alias")
    return ModelKeyResolution(raw=text, resolved=None, ok=False, source="unresolved", reason=f"unresolved_model_key:{text}")


def canonical_model_keys(values: list[Any] | tuple[Any, ...] | set[Any], registry: dict[str, Any] | None = None) -> tuple[list[str], list[dict[str, str | bool | None]]]:
    keys: list[str] = []
    seen: set[str] = set()
    diagnostics: list[dict[str, str | bool | None]] = []
    for value in values:
        resolved = resolve_model_key(value, registry)
        diagnostics.append({
            "raw": resolved.raw,
            "resolved": resolved.resolved,
            "ok": resolved.ok,
            "source": resolved.source,
            "reason": resolved.reason,
        })
        if resolved.ok and resolved.resolved and resolved.resolved not in seen:
            keys.append(resolved.resolved)
            seen.add(resolved.resolved)
    return keys, diagnostics
