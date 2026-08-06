from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .json_utils import read_json
from .model_key_resolver import resolve_model_key
from .paths import resolve_path


@dataclass(frozen=True)
class RoutedCandidate:
    organ: str
    token: str
    model_key: str | None
    subtask: str | None
    enabled: bool
    reason: str | None = None
    note: str | None = None


def _load_json_required(path: str | Path) -> dict[str, Any]:
    data = read_json(resolve_path(path), default=None)
    if not isinstance(data, dict) or not data:
        raise FileNotFoundError(f"Missing or empty JSON config: {resolve_path(path)}")
    return data


def _route_cads_fallback(organ: str, token_entry: dict[str, Any], token_map: dict[str, Any]) -> dict[str, Any]:
    overrides = token_map.get("bare_cads_organ_overrides", {}) or {}
    chosen = overrides.get(organ)
    if chosen:
        return {
            "model": chosen,
            "subtask": None,
            "enabled": True,
            "reason": None,
            "note": "Resolved from bare_cads_organ_overrides",
        }
    return {
        "model": None,
        "subtask": None,
        "enabled": False,
        "reason": f"Token 'CADS' requires an organ-specific override, but '{organ}' is not mapped",
        "note": token_entry.get("note"),
    }


def _route_organ_token_override(organ: str, token: str, token_map: dict[str, Any]) -> dict[str, Any] | None:
    overrides = token_map.get("organ_token_overrides", {}) or {}
    organ_overrides = overrides.get(organ, {}) or {}
    chosen = organ_overrides.get(token)
    return chosen if isinstance(chosen, dict) else None


def route_organs(
    organs: list[str] | None = None,
    routing_path: str | Path = "configs/organ_routing_from_xlsx.json",
    token_map_path: str | Path = "configs/routing_token_to_model.json",
    target_config_path: str | Path | None = "configs/student_3d_prompt_target_organs.json",
) -> dict[str, Any]:
    routing = _load_json_required(routing_path)
    token_map_doc = _load_json_required(token_map_path)
    organ_to_models = routing.get("organ_to_models", {}) or {}
    token_map = token_map_doc.get("tokens", {}) or {}

    if organs is None and target_config_path is not None:
        target_config = read_json(resolve_path(target_config_path), default={})
        configured_targets = target_config.get("target_organs") if isinstance(target_config, dict) else None
        requested_organs = list(configured_targets or sorted(organ_to_models.keys()))
    else:
        requested_organs = list(organs or sorted(organ_to_models.keys()))
    ranked_candidates: dict[str, list[dict[str, Any]]] = {}
    selected_models: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str | None]] = set()
    missing_organs: list[str] = []
    disabled_organs: dict[str, list[dict[str, Any]]] = {}

    for organ in requested_organs:
        tokens = organ_to_models.get(organ)
        if not tokens:
            missing_organs.append(organ)
            ranked_candidates[organ] = []
            continue

        organ_candidates: list[dict[str, Any]] = []
        organ_disabled: list[dict[str, Any]] = []

        for token in tokens:
            token_entry = token_map.get(token)
            override_entry = _route_organ_token_override(organ, token, token_map_doc)
            if override_entry:
                token_entry = override_entry
            if not token_entry:
                resolved = resolve_model_key(token)
                if resolved.ok and resolved.resolved:
                    token_entry = {
                        "model": resolved.resolved,
                        "subtask": None,
                        "enabled": True,
                        "reason": None,
                        "note": f"Resolved by canonical model-key resolver from {resolved.source}",
                    }
                else:
                    organ_disabled.append({
                        "token": token,
                        "model_key": None,
                        "subtask": None,
                        "enabled": False,
                        "reason": "Token not found in routing_token_to_model.json",
                    })
                    continue

            if token == "CADS" and token_entry.get("model") is None:
                token_entry = _route_cads_fallback(organ, token_entry, token_map_doc)
            resolved_model = resolve_model_key(token_entry.get("model"))
            model_key = resolved_model.resolved if resolved_model.ok else token_entry.get("model")

            candidate = RoutedCandidate(
                organ=organ,
                token=token,
                model_key=model_key,
                subtask=token_entry.get("subtask"),
                enabled=bool(token_entry.get("enabled", False)) and bool(model_key),
                reason=token_entry.get("reason") or (resolved_model.reason if not resolved_model.ok else None),
                note=token_entry.get("note"),
            )
            item = {
                "token": candidate.token,
                "model_key": candidate.model_key,
                "subtask": candidate.subtask,
                "enabled": candidate.enabled,
                "reason": candidate.reason,
                "note": candidate.note,
            }
            if candidate.enabled:
                organ_candidates.append(item)
                key = (candidate.model_key or "", candidate.subtask)
                if key not in selected_keys and candidate.model_key:
                    selected_keys.add(key)
                    selected_models.append({
                        "model_key": candidate.model_key,
                        "subtask": candidate.subtask,
                    })
            else:
                organ_disabled.append(item)

        ranked_candidates[organ] = organ_candidates
        if organ_disabled:
            disabled_organs[organ] = organ_disabled

    return {
        "status": "success",
        "requested_organs": requested_organs,
        "total_requested_organs": len(requested_organs),
        "ranked_candidates": ranked_candidates,
        "selected_models": selected_models,
        "selected_model_keys": [m["model_key"] for m in selected_models],
        "missing_organs": missing_organs,
        "disabled_candidates": disabled_organs,
    }
