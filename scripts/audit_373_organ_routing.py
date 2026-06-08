#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in (text or "")).strip("_")


def local_labels_from_dataset_json(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = load_json(path)
    labels = data.get("labels") or {}
    return [str(name) for name in labels if str(name).lower() != "background"]


def local_labels_from_label_map(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = load_json(path)
    out = []
    for name, value in data.items():
        if str(name).lower() == "background":
            continue
        if isinstance(value, int) and value != 0:
            out.append(str(name))
    return out


def local_labels_for_model(model_key: str, entry: dict[str, Any], totalseg_subtask: str | None, totalseg_config: dict[str, Any]) -> list[str]:
    if model_key == "totalsegmentator":
        if totalseg_subtask:
            task_entry = (totalseg_config.get("subtasks", {}) or {}).get(totalseg_subtask, {})
            return [str(x) for x in task_entry.get("organs", [])]
        return [str(x) for x in entry.get("covered_organs", [])]

    dataset_json = entry.get("dataset_json_path")
    if dataset_json:
        labels = local_labels_from_dataset_json(ROOT / dataset_json)
        if labels:
            return labels

    label_map = entry.get("label_map_path")
    if label_map:
        labels = local_labels_from_label_map(ROOT / label_map)
        if labels:
            return labels

    return [str(x) for x in entry.get("covered_organs", [])]


def resolve_static_local_name(
    organ: str,
    available: list[str],
    model_entry: dict[str, Any],
    model_aliases: dict[str, Any],
) -> tuple[str | None, str]:
    available_set = set(available)
    candidates = [organ, organ.lower(), organ.upper(), organ.replace("_", " ")]

    for local_name, global_name in (model_entry.get("supported_organ_aliases", {}) or {}).items():
        if str(global_name) == organ:
            candidates.insert(0, str(local_name))

    for local_name, global_name in (model_aliases.get("local_to_global", {}) or {}).items():
        if str(global_name) == organ:
            candidates.insert(0, str(local_name))

    for candidate in candidates:
        if candidate in available_set:
            return candidate, "direct_or_declared_alias"

    organ_norm = norm(organ)
    for name in available:
        if norm(name) == organ_norm:
            return name, "normalized"

    return None, "unresolved"


def main() -> int:
    import sys

    sys.path.insert(0, str(ROOT / "agent-harness"))
    from cli_anything.medai.core.organ_router import route_organs

    target_config = load_json(ROOT / "configs/student_3d_prompt_target_organs.json")
    global_space = load_json(ROOT / "configs/global_label_space.json")
    alias_config = load_json(ROOT / "configs/model_label_aliases.json")
    totalseg_config = load_json(ROOT / "configs/totalseg_subtask_organs.json")
    registry = yaml.safe_load((ROOT / "configs/model_registry.yaml").read_text(encoding="utf-8"))

    target_organs = [str(x) for x in target_config.get("target_organs", [])]
    organ_to_id = global_space.get("organ_to_id", {}) or {}
    organ_to_student_id = target_config.get("organ_to_student_id", {}) or {}
    organ_to_prompt = target_config.get("organ_to_prompt", {}) or {}
    skipped = {item.get("organ") for item in target_config.get("policy_skipped_organs", [])}
    no_route = {item.get("organ") for item in target_config.get("no_enabled_route_organs", [])}

    route = route_organs(target_organs)

    duplicate_targets = sorted({organ for organ in target_organs if target_organs.count(organ) > 1})
    missing_global_ids = sorted([organ for organ in target_organs if organ not in organ_to_id])
    missing_student_ids = sorted([organ for organ in target_organs if organ not in organ_to_student_id])
    missing_prompts = sorted([organ for organ in target_organs if not organ_to_prompt.get(organ)])
    target_contains_skipped = sorted([organ for organ in target_organs if organ in skipped])
    target_contains_no_route = sorted([organ for organ in target_organs if organ in no_route])
    route_missing = sorted(route.get("missing_organs", []))
    route_no_enabled = sorted([organ for organ, candidates in (route.get("ranked_candidates", {}) or {}).items() if not candidates])

    per_organ: dict[str, Any] = {}
    statically_unresolvable: list[dict[str, Any]] = []
    disabled_registry_candidates: list[dict[str, Any]] = []

    for organ in target_organs:
        candidates = route.get("ranked_candidates", {}).get(organ, []) or []
        resolved_candidates = []
        for candidate in candidates:
            model_key = candidate.get("model_key")
            model_entry = (registry.get("models", {}) or {}).get(model_key, {})
            if not model_entry or model_entry.get("enabled") is False:
                disabled_registry_candidates.append({"organ": organ, "model_key": model_key})
                continue
            model_aliases = (alias_config.get("models", {}) or {}).get(model_key, {})
            skip_reason = (model_aliases.get("skip_global_organs", {}) or {}).get(organ)
            if skip_reason:
                resolved_candidates.append({
                    "model_key": model_key,
                    "subtask": candidate.get("subtask"),
                    "resolvable": False,
                    "reason": f"policy_skipped: {skip_reason}",
                })
                continue
            available = local_labels_for_model(model_key, model_entry, candidate.get("subtask"), totalseg_config)
            local_name, mode = resolve_static_local_name(organ, available, model_entry, model_aliases)
            resolved_candidates.append({
                "model_key": model_key,
                "subtask": candidate.get("subtask"),
                "resolvable": local_name is not None,
                "local_name": local_name,
                "match_mode": mode,
                "available_label_count": len(available),
            })
        per_organ[organ] = {"candidates": resolved_candidates}
        if candidates and not any(item.get("resolvable") for item in resolved_candidates):
            statically_unresolvable.append({
                "organ": organ,
                "candidates": resolved_candidates,
            })

    status = "success"
    blocking = {
        "duplicate_targets": duplicate_targets,
        "missing_global_ids": missing_global_ids,
        "missing_student_ids": missing_student_ids,
        "missing_prompts": missing_prompts,
        "target_contains_policy_skipped_organs": target_contains_skipped,
        "target_contains_no_enabled_route_organs": target_contains_no_route,
        "route_missing_organs": route_missing,
        "route_no_enabled_candidate_organs": route_no_enabled,
        "disabled_registry_candidates": disabled_registry_candidates,
        "statically_unresolvable_target_organs": statically_unresolvable,
    }
    if any(blocking.values()):
        status = "failed"

    out_dir = ROOT / "outputs/audit_21_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": status,
        "target_config": "configs/student_3d_prompt_target_organs.json",
        "counts": {
            "target_organs": len(target_organs),
            "unique_target_organs": len(set(target_organs)),
            "route_requested_organs": route.get("total_requested_organs"),
            "selected_model_keys": len(set(route.get("selected_model_keys", []))),
            "statically_unresolvable_target_organs": len(statically_unresolvable),
        },
        "blocking": blocking,
        "per_organ": per_organ,
    }
    json_path = out_dir / "routing_373_audit.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# 373 Organ Routing Audit",
        "",
        f"- Status: {status}",
        f"- Target organs: {len(target_organs)}",
        f"- Unique target organs: {len(set(target_organs))}",
        f"- Route requested organs: {route.get('total_requested_organs')}",
        f"- Static unresolvable target organs: {len(statically_unresolvable)}",
        "",
        "## Blocking Findings",
        "",
    ]
    for key, value in blocking.items():
        count = len(value) if isinstance(value, list) else 0
        md_lines.append(f"- `{key}`: {count}")
        if isinstance(value, list) and value:
            for item in value[:30]:
                md_lines.append(f"- `{key}` sample: `{item}`")
    md_path = out_dir / "routing_373_audit.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": status,
        "report": str(json_path.relative_to(ROOT)),
        "markdown_report": str(md_path.relative_to(ROOT)),
        **report["counts"],
        "blocking_keys": [key for key, value in blocking.items() if value],
    }, indent=2, ensure_ascii=False))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
