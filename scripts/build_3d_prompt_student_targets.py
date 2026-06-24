#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_OUT = ROOT / "configs/student_3d_prompt_target_organs.json"
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.organ_prompt_bank import build_prompt_bank, flatten_prompt_bank_entry


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_enabled_routes() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    routing = load_json(ROOT / "configs/organ_routing_from_xlsx.json")
    token_doc = load_json(ROOT / "configs/routing_token_to_model.json")
    tokens = token_doc.get("tokens", {})
    best: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    for organ in sorted((routing.get("organ_to_models", {}) or {}).keys()):
        candidates = []
        for token in routing["organ_to_models"].get(organ, []):
            entry = tokens.get(token)
            override = ((token_doc.get("organ_token_overrides", {}) or {}).get(organ, {}) or {}).get(token)
            if isinstance(override, dict):
                entry = override
            if token == "CADS" and entry and entry.get("model") is None:
                chosen = (token_doc.get("bare_cads_organ_overrides", {}) or {}).get(organ)
                entry = {"model": chosen, "enabled": bool(chosen), "subtask": None, "note": "bare CADS override"}
            if not entry:
                candidates.append({"token": token, "model_key": None, "enabled": False, "reason": "token not mapped"})
                continue
            model = entry.get("model")
            enabled = bool(entry.get("enabled")) and bool(model)
            item = {
                "token": token,
                "model_key": model,
                "subtask": entry.get("subtask"),
                "enabled": enabled,
                "reason": entry.get("reason"),
                "note": entry.get("note"),
            }
            candidates.append(item)
            if enabled and organ not in best:
                best[organ] = item
        if organ not in best:
            unresolved.append({"organ": organ, "candidates": candidates})
    return best, unresolved


def main() -> int:
    global_space = load_json(ROOT / "configs/global_label_space.json")
    alias_config = load_json(ROOT / "configs/model_label_aliases.json")
    organ_to_id = global_space.get("organ_to_id", {})
    best, unresolved = best_enabled_routes()

    policy_skipped = []
    target_organs = []
    for organ, route in best.items():
        model_key = route.get("model_key")
        model_aliases = (alias_config.get("models", {}) or {}).get(model_key, {})
        skip_reason = (model_aliases.get("skip_global_organs", {}) or {}).get(organ)
        if skip_reason:
            policy_skipped.append({
                "organ": organ,
                "model_key": model_key,
                "reason": str(skip_reason),
                "policy": "skip_unresolvable_coarse_label",
            })
            continue
        if organ in organ_to_id:
            target_organs.append(organ)

    target_organs = sorted(target_organs, key=lambda o: int(organ_to_id[o]))
    organ_prompt_bank = build_prompt_bank(target_organs)
    prompt_variants = {
        organ: flatten_prompt_bank_entry(organ_prompt_bank[organ])
        for organ in target_organs
    }
    doc = {
        "version": 2,
        "student_backend": "voxtell_style_3d_prompt",
        "status": "prompt_bank_baseline",
        "note": (
            "3D prompt-based student target space. This intentionally does not "
            "use VISTA3D's 127-label space as the system limit. Each target organ "
            "now has an organ-level prompt bank with simple instructions, anatomical "
            "location, CT appearance, medical terms/synonyms, and natural-language variants."
        ),
        "source_configs": [
            "configs/global_label_space.json",
            "configs/organ_routing_from_xlsx.json",
            "configs/routing_token_to_model.json",
            "configs/model_label_aliases.json",
        ],
        "counts": {
            "global_label_space_organs": len(organ_to_id),
            "enabled_routed_organs": len(best),
            "policy_skipped_organs": len(policy_skipped),
            "no_enabled_route_organs": len(unresolved),
            "current_exact_prompt_target_organs": len(target_organs),
            "accepted_current_exact_prompt_targets": len(target_organs),
            "organs_with_prompt_bank": len(organ_prompt_bank),
            "prompt_bank_total_prompts": sum(len(v) for v in prompt_variants.values()),
            "prompt_bank_min_prompts_per_organ": min((len(v) for v in prompt_variants.values()), default=0),
            "historical_teacher_direction": 377,
        },
        "target_organs": target_organs,
        "organ_to_student_id": {organ: idx for idx, organ in enumerate(target_organs, start=1)},
        "organ_to_prompt": {organ: organ_prompt_bank[organ]["canonical_prompt"] for organ in target_organs},
        "organ_prompt_bank": organ_prompt_bank,
        "prompt_variants": prompt_variants,
        "prompt_sampling_policy": {
            "training_manifest": "expand prompt_variants by default; each variant keeps the same mask but records prompt_source/category",
            "inference_default": "canonical prompt for reproducibility",
            "inference_optional": "set MEDAI_PROMPT_SAMPLING=random or hash, or pass prompt_overrides, to use alternative expressions",
        },
        "prompt_bank_review": {
            "generation_method": "curated common-organ facts plus conservative anatomy/type heuristics from organ names",
            "required_human_action": "spot-check expanded prompts before formal training; correct organ-specific descriptions if a target has unusual dataset semantics",
            "spot_checked_organs": [
                "liver",
                "pancreas",
                "spleen",
                "kidney_left",
                "kidney_right",
                "aorta",
                "inferior_vena_cava",
                "stomach",
                "gall_bladder",
                "duodenum",
                "colon",
                "bladder",
                "heart",
                "lung",
            ],
        },
        "policy_skipped_organs": policy_skipped,
        "no_enabled_route_organs": unresolved,
    }
    TARGET_OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "output": str(TARGET_OUT.relative_to(ROOT)),
        **doc["counts"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
