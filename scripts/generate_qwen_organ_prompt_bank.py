#!/usr/bin/env python3
"""Generate a grounded, hybrid organ prompt bank with an OpenAI-compatible Qwen server.

The script is intentionally offline from the EM loop. Importing it or using
--dry-run never contacts vLLM, loads a model, or changes the active target config.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.organ_prompt_bank import (
    PROMPT_BANK_CATEGORIES,
    flatten_prompt_bank_entry,
)

DEFAULT_TARGET = ROOT / "configs" / "student_3d_prompt_target_organs.json"
DEFAULT_TAXONOMY = ROOT / "configs" / "organ_taxonomy.json"
DEFAULT_OUTPUT = ROOT / "configs" / "student_3d_prompt_target_organs_llm_candidate.json"
DEFAULT_CACHE = ROOT / "outputs" / "prompt_bank_generation_cache.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate grounded free-form prompts for the 373-organ student.")
    ap.add_argument("--target-config", default=str(DEFAULT_TARGET))
    ap.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="Served model id; default resolves the first /models entry.")
    ap.add_argument("--organs", default="", help="Optional comma-separated canonical organ IDs.")
    ap.add_argument("--prompts-per-category", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", clean_text(value).lower()))


def strip_model_wrapping(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    text = re.sub(r"^\x60\x60\x60(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*\x60\x60\x60$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response does not contain a JSON object")
    return text[start:end + 1]


def resolve_model(base_url: str, explicit: str | None, timeout: int) -> str:
    if explicit:
        return explicit
    import requests

    response = requests.get(
        f"{base_url.rstrip('/')}/models",
        timeout=min(timeout, 15),
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    models = response.json().get("data", [])
    if not models or not models[0].get("id"):
        raise RuntimeError("vLLM /models returned no served model id")
    return str(models[0]["id"])


def call_vllm(
    *,
    base_url: str,
    model: str,
    user_prompt: str,
    temperature: float,
    timeout: int,
) -> tuple[dict[str, Any], str]:
    import requests

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You create English text prompts for 3D CT segmentation. "
                    "Use only supplied facts, preserve exact anatomical identity, and output JSON only."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2400,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=timeout,
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    raw = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = json.loads(strip_model_wrapping(raw))
    if not isinstance(parsed, dict):
        raise ValueError("model JSON must be an object")
    return parsed, raw


def taxonomy_context(organ: str, taxonomy: dict[str, Any]) -> dict[str, Any]:
    organs = taxonomy.get("organs", {}) or {}
    entry = organs.get(organ, {}) or {}
    parents = [str(x) for x in entry.get("parent_ids", [])]
    children = sorted(
        name for name, item in organs.items()
        if organ in (item.get("parent_ids", []) or [])
    )
    siblings = sorted(
        name for name, item in organs.items()
        if name != organ and set(parents) & set(item.get("parent_ids", []) or [])
    )
    opposite = None
    if organ.endswith("_left"):
        opposite = organ[:-5] + "_right"
    elif organ.endswith("_right"):
        opposite = organ[:-6] + "_left"
    forbidden = siblings + children
    if opposite and opposite in organs:
        forbidden.append(opposite)
    return {
        "hierarchy_role": entry.get("hierarchy_role"),
        "comparison_family": entry.get("comparison_family"),
        "parent_ids": parents,
        "child_ids": children,
        "forbidden_targets": sorted(set(forbidden)),
    }


def generation_request(
    organ: str,
    entry: dict[str, Any],
    hierarchy: dict[str, Any],
    count: int,
    existing: dict[str, list[str]],
) -> str:
    facts = {
        "canonical_id": organ,
        "display_name": entry.get("display_name"),
        "canonical_prompt": entry.get("canonical_prompt"),
        "aliases": entry.get("aliases", []),
        "region": entry.get("region"),
        "landmarks": entry.get("landmarks", []),
        "ct_appearance": entry.get("ct_appearance"),
        **hierarchy,
        "existing_prompts_to_avoid": existing,
    }
    schema = {category: [f"exactly {count} distinct strings"] for category in PROMPT_BANK_CATEGORIES}
    return (
        "Generate diverse prompts grounded strictly in the facts below. Every prompt must explicitly "
        "name the target or one supplied alias. Do not broaden to a parent, child, sibling, opposite "
        "side, lesion, tumor, duct, vessel, or other target. Location and appearance prompts may "
        "mention landmarks only as exclusions/context. Simple instructions must stay concise; "
        "natural-language prompts may vary tone and syntax. Return exactly this five-key JSON shape.\n\n"
        f"FACTS:\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        f"OUTPUT SHAPE:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def identity_phrases(organ: str, entry: dict[str, Any]) -> list[str]:
    values = [entry.get("display_name"), organ.replace("_", " "), *(entry.get("aliases", []) or [])]
    return sorted({normalize_phrase(str(x)) for x in values if normalize_phrase(str(x))}, key=len, reverse=True)


def validate_prompt(
    *,
    text: Any,
    category: str,
    organ: str,
    entry: dict[str, Any],
    hierarchy: dict[str, Any],
    seen: list[str],
) -> tuple[str | None, list[str]]:
    prompt = clean_text(text)
    errors: list[str] = []
    normalized = normalize_phrase(prompt)
    if not prompt:
        errors.append("empty")
    if len(prompt.split()) < 3 or len(prompt.split()) > 80:
        errors.append("word_count_out_of_range")
    if not any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in identity_phrases(organ, entry)):
        errors.append("target_identity_not_explicit")
    if organ.endswith("_left") and re.search(r"\bright\b", normalized):
        errors.append("opposite_side_right")
    if organ.endswith("_right") and re.search(r"\bleft\b", normalized):
        errors.append("opposite_side_left")
    segment = re.search(r"(?:liver|hepatic|couinaud) segment (\d+)", normalized)
    expected_segment = re.search(r"liver_segment_(\d+)", organ)
    if segment and (not expected_segment or segment.group(1) != expected_segment.group(1)):
        errors.append("segment_identity_mismatch")
    for forbidden in hierarchy.get("forbidden_targets", []):
        phrase = normalize_phrase(forbidden.replace("_", " "))
        if phrase and re.search(rf"\b{re.escape(phrase)}\b", normalized):
            errors.append(f"forbidden_target:{forbidden}")
            break
    for prior in seen:
        if normalize_phrase(prior) == normalized or SequenceMatcher(None, normalize_phrase(prior), normalized).ratio() > 0.96:
            errors.append("duplicate_or_near_duplicate")
            break
    if category == "simple_instruction" and len(prompt.split()) > 18:
        errors.append("simple_instruction_too_long")
    return (prompt if not errors else None), errors


def validate_response(
    response: dict[str, Any],
    *,
    organ: str,
    entry: dict[str, Any],
    hierarchy: dict[str, Any],
    needed: int,
    accepted: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    seen = [
        *flatten_prompt_bank_entry(entry),
        *(prompt for values in accepted.values() for prompt in values),
    ]
    for category in PROMPT_BANK_CATEGORIES:
        if not isinstance(values, list):
        values = response.get(category)
            rejected.append({"category": category, "text": None, "reasons": ["missing_or_non_list_category"]})
            continue
        for value in values:
            prompt, errors = validate_prompt(
                text=value,
                category=category,
                organ=organ,
                entry=entry,
                hierarchy=hierarchy,
                seen=seen,
            )
            if errors:
                rejected.append({"category": category, "text": clean_text(value), "reasons": errors})
                continue
            if len(accepted[category]) < needed and prompt:
                accepted[category].append(prompt)
                seen.append(prompt)
    return accepted, rejected


def template_records(entry: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    canonical = clean_text(entry.get("canonical_prompt"))
    if canonical:
        records.append({"text": canonical, "category": "canonical", "source": "template", "validation_status": "accepted"})
    for category in PROMPT_BANK_CATEGORIES:
        for prompt in entry.get("prompts", {}).get(category, []) or []:
            records.append({
                "text": clean_text(prompt),
                "category": category,
                "source": "template",
                "validation_status": "accepted",
            })
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["text"].lower(), record)
    return list(unique.values())


def merge_generated_entry(
    entry: dict[str, Any],
    accepted: dict[str, list[str]],
    *,
    model: str,
    attempts: int,
    needed: int,
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    merged = copy.deepcopy(entry)
    merged.setdefault("prompts", {})
    records = template_records(entry)
    for category in PROMPT_BANK_CATEGORIES:
        current = [clean_text(x) for x in merged["prompts"].get(category, []) if clean_text(x)]
        for prompt in accepted.get(category, []):
            if prompt.lower() not in {x.lower() for x in current}:
                current.append(prompt)
            records.append({
                "text": prompt,
                "category": category,
                "source": "qwen_freeform",
                "model": model,
                "validation_status": "accepted",
            })
        merged["prompts"][category] = current
    merged["prompt_records"] = records
    merged["generation_status"] = (
        "success" if all(len(accepted[c]) >= needed for c in PROMPT_BANK_CATEGORIES) else "template_fallback"
    )
    merged["generation_attempts"] = attempts
    merged["generation_rejections"] = rejected
    return merged


def main() -> int:
    args = parse_args()
    target_path = Path(args.target_config).resolve()
    taxonomy_path = Path(args.taxonomy).resolve()
    output_path = Path(args.output).resolve()
    cache_path = Path(args.cache).resolve()
    doc = load_json(target_path)
    taxonomy = load_json(taxonomy_path, default={"organs": {}})
    if not isinstance(doc, dict) or not isinstance(doc.get("organ_prompt_bank"), dict):
        raise SystemExit(f"Invalid target config: {target_path}")
    selected = [x.strip() for x in args.organs.split(",") if x.strip()] or list(doc.get("target_organs", []))
    unknown = [organ for organ in selected if organ not in doc["organ_prompt_bank"]]
    if unknown:
        raise SystemExit(f"Unknown target organs: {unknown[:20]}")
    plan = {
        "status": "dry_run" if args.dry_run else "pending",
        "target_config": str(target_path),
        "taxonomy": str(taxonomy_path),
        "output": str(output_path),
        "cache": str(cache_path),
        "base_url": args.base_url,
        "model": args.model or "resolve_from_vllm_models",
        "num_organs": len(selected),
        "categories": list(PROMPT_BANK_CATEGORIES),
        "prompts_per_category": args.prompts_per_category,
        "network_policy": "No request is made during --dry-run; generation only runs when explicitly invoked.",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    service_error: str | None = None
    try:
        model = resolve_model(args.base_url, args.model, args.timeout_sec)
    except Exception as exc:
        model = args.model or "unavailable"
        service_error = str(exc)
    cache = load_json(cache_path, default={}) if args.resume else {}
    if not isinstance(cache, dict):
        cache = {}
    candidate = copy.deepcopy(doc)
    bank = candidate["organ_prompt_bank"]
    audit: dict[str, Any] = {
        "model": model,
        "service_error": service_error,
        "organs": {},
        "started_unix": time.time(),
    }

    for index, organ in enumerate(selected, start=1):
        cached = cache.get(organ)
        if args.resume and isinstance(cached, dict) and cached.get("complete"):
            bank[organ] = cached["entry"]
            audit["organs"][organ] = cached.get("audit", {"status": "resumed"})
            continue
        entry = doc["organ_prompt_bank"][organ]
        hierarchy = taxonomy_context(organ, taxonomy)
        accepted = {category: [] for category in PROMPT_BANK_CATEGORIES}
        rejected: list[dict[str, Any]] = []
        raw_responses: list[str] = []
        attempts = 0
        last_error = service_error
        while not service_error and attempts < args.max_retries and any(
            len(accepted[category]) < args.prompts_per_category for category in PROMPT_BANK_CATEGORIES
        ):
            attempts += 1
            try:
                request = generation_request(
                    organ,
                    entry,
                    hierarchy,
                    args.prompts_per_category,
                    accepted,
                )
                response, raw = call_vllm(
                    base_url=args.base_url,
                    model=model,
                    user_prompt=request,
                    temperature=args.temperature,
                    timeout=args.timeout_sec,
                )
                raw_responses.append(raw)
                accepted, new_rejections = validate_response(
                    response,
                    organ=organ,
                    entry=entry,
                    hierarchy=hierarchy,
                    needed=args.prompts_per_category,
                    accepted=accepted,
                )
                rejected.extend(new_rejections)
            except Exception as exc:
                last_error = str(exc)
                rejected.append({"attempt": attempts, "reasons": ["request_or_parse_error"], "detail": last_error})
        complete = all(len(accepted[c]) >= args.prompts_per_category for c in PROMPT_BANK_CATEGORIES)
        merged = merge_generated_entry(
            entry,
            accepted,
            model=model,
            attempts=attempts,
            rejected=rejected,
            needed=args.prompts_per_category,
        )
        per_organ_audit = {
            "status": "success" if complete else "template_fallback",
            "attempts": attempts,
            "accepted_counts": {category: len(values) for category, values in accepted.items()},
            "rejected_count": len(rejected),
            "last_error": last_error,
        }
        bank[organ] = merged
        audit["organs"][organ] = per_organ_audit
        cache[organ] = {
            "complete": complete,
            "entry": merged,
            "audit": per_organ_audit,
            "raw_responses": raw_responses,
        }
        write_json_atomic(cache_path, cache)
        print(f"[{index}/{len(selected)}] {organ}: {per_organ_audit['status']}", flush=True)

    candidate["version"] = max(3, int(candidate.get("version", 0)))
    candidate["status"] = "hybrid_qwen_prompt_bank_candidate"
    candidate["prompt_generation"] = {
        "source": "qwen_freeform_plus_template_baseline",
        "model": model,
        "base_url": args.base_url,
        "prompts_per_category": args.prompts_per_category,
        "temperature": args.temperature,
        "active_config_unchanged": str(output_path) != str(target_path),
    }
    candidate["prompt_variants"] = {
        organ: flatten_prompt_bank_entry(entry)
        for organ, entry in candidate["organ_prompt_bank"].items()
    }
    counts = candidate.setdefault("counts", {})
    counts["prompt_bank_total_prompts"] = sum(len(values) for values in candidate["prompt_variants"].values())
    counts["prompt_bank_min_prompts_per_organ"] = min(
        (len(values) for values in candidate["prompt_variants"].values()),
        default=0,
    )
    audit["finished_unix"] = time.time()
    audit["summary"] = {
        "success": sum(item.get("status") == "success" for item in audit["organs"].values()),
        "template_fallback": sum(item.get("status") != "success" for item in audit["organs"].values()),
    }
    candidate["prompt_generation_audit"] = audit
    write_json_atomic(output_path, candidate)
    print(json.dumps({"status": "success", "output": str(output_path), **audit["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
