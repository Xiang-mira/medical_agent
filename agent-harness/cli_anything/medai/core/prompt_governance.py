"""Prompt taxonomy and validation contracts for CT prompt governance.

This module is intentionally CPU-only. It validates prompt text and provenance
metadata; it never calls Qwen, embedding models, inference, torch, CUDA, or a
network service.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


PROMPT_TAXONOMY: dict[str, dict[str, Any]] = {
    "AI Researcher Prompt": {
        "purpose": "Model training, controlled experiments, and reproducible evaluation.",
        "subclasses": {
            "simple_instruction": "Direct instruction such as 'segment the liver'.",
            "canonical_organ_name": "Canonical or controlled organ terminology such as 'segment the hepatic parenchyma'.",
            "task_specific_prompt": "Controlled task wording tied to the canonical organ token.",
        },
    },
    "Clinical Expert Prompt": {
        "purpose": "Medical-context prompts that describe anatomy and CT appearance.",
        "subclasses": {
            "anatomy_location_prompt": "Expected location, adjacent structures, or anatomic field of view.",
            "ct_appearance_prompt": "Density, boundary, contour, morphology, or CT visual appearance.",
            "synonym_clinical_naming_prompt": "Clinical names, abbreviations, and synonym forms.",
        },
    },
    "User Natural Prompt": {
        "purpose": "Real user phrasing after validation, never used as unchecked free text.",
        "subclasses": {
            "natural_language_request": "Polite or full-sentence request.",
            "informal_query": "Find/show/mark style query.",
            "simplified_description": "Short simplified wording such as 'outline the liver'.",
        },
    },
}


LEGACY_CATEGORY_TO_TAXONOMY: dict[str, tuple[str, str]] = {
    "canonical": ("AI Researcher Prompt", "simple_instruction"),
    "simple_instruction": ("AI Researcher Prompt", "simple_instruction"),
    "canonical_organ_name": ("AI Researcher Prompt", "canonical_organ_name"),
    "task_specific_prompt": ("AI Researcher Prompt", "task_specific_prompt"),
    "anatomical_location": ("Clinical Expert Prompt", "anatomy_location_prompt"),
    "anatomy_location_prompt": ("Clinical Expert Prompt", "anatomy_location_prompt"),
    "ct_appearance": ("Clinical Expert Prompt", "ct_appearance_prompt"),
    "ct_appearance_prompt": ("Clinical Expert Prompt", "ct_appearance_prompt"),
    "synonyms_medical_terms": ("Clinical Expert Prompt", "synonym_clinical_naming_prompt"),
    "synonym_clinical_naming_prompt": ("Clinical Expert Prompt", "synonym_clinical_naming_prompt"),
    "natural_language": ("User Natural Prompt", "natural_language_request"),
    "natural_language_request": ("User Natural Prompt", "natural_language_request"),
    "informal_query": ("User Natural Prompt", "informal_query"),
    "simplified_description": ("User Natural Prompt", "simplified_description"),
}


FREEFORM_REQUIRED_INPUT_FIELDS = [
    "organ_token",
    "canonical_name",
    "region",
    "ct_appearance",
    "synonyms",
    "laterality",
    "allowed_adjacent_anatomy",
    "ct_scan_range",
    "forbidden_targets",
]


FREEFORM_REQUIRED_OUTPUT_FIELDS = [
    "organ_token",
    "canonical_name",
    "prompt",
    "prompt_level",
    "prompt_subclass",
    "introduced_anatomy",
    "laterality",
    "ct_scan_range_assumption",
    "prompt_source",
]


FREEFORM_PROMPT_SAFETY_CONTRACT: dict[str, Any] = {
    "status": "design_contract_not_model_execution",
    "qwen_generation_allowed": "offline_or_explicit_replay_only",
    "required_input_fields": FREEFORM_REQUIRED_INPUT_FIELDS,
    "required_output_fields": FREEFORM_REQUIRED_OUTPUT_FIELDS,
    "required_output_schema": {
        "organ_token": "string",
        "canonical_name": "string",
        "prompt": "string",
        "prompt_level": "AI Researcher Prompt | Clinical Expert Prompt | User Natural Prompt",
        "prompt_subclass": "controlled subclass name",
        "introduced_anatomy": ["string"],
        "laterality": "left | right | bilateral | midline | none",
        "ct_scan_range_assumption": "string",
        "prompt_source": "fixed | template | qwen_generated | user_freeform | fallback_fixed",
    },
    "validation_steps": [
        "validate_json_schema",
        "validate_organ_identity_against_canonical_token",
        "validate_left_right_laterality",
        "reject_forbidden_parent_child_sibling_or_opposite_target",
        "reject_new_anatomy_not_in_allowed_adjacent_anatomy",
        "reject_prompt_outside_ct_scan_range",
        "reject_duplicate_or_near_duplicate_prompt",
        "validate_word_count_and_simple_prompt_length",
        "fallback_to_fixed_prompt_on_any_failure",
        "persist_prompt_source_validation_status_fallback_reason",
    ],
    "fallback_fields": ["prompt_source", "validation_status", "fallback_reason"],
    "hallucination_policy": "Free-form or Qwen-expanded prompts are never accepted unless identity, laterality, anatomy, and scan-range validation pass.",
}


SCAN_RANGE_KEYWORDS: dict[str, set[str]] = {
    "abdomen": {"liver", "hepatic", "pancreas", "spleen", "kidney", "renal", "stomach", "duodenum", "colon", "aorta", "vena", "portal", "gallbladder", "adrenal"},
    "pelvis": {"bladder", "prostate", "uterus", "ovary", "rectum", "pelvic", "iliac", "femur"},
    "thorax": {"lung", "heart", "cardiac", "mediastinum", "rib", "sternum", "aorta", "bronch", "trachea"},
    "head": {"brain", "skull", "orbit", "eye", "cerebellum", "ventricle", "cranial"},
    "neck": {"thyroid", "larynx", "pharynx", "carotid", "jugular", "hyoid", "cervical"},
}


def clean_prompt_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_phrase(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", clean_prompt_text(value).lower()))


def taxonomy_for_legacy_category(category: str) -> tuple[str, str]:
    return LEGACY_CATEGORY_TO_TAXONOMY.get(str(category), ("Clinical Expert Prompt", "synonym_clinical_naming_prompt"))


def prompt_level_subclasses() -> dict[str, set[str]]:
    return {level: set(info["subclasses"]) for level, info in PROMPT_TAXONOMY.items()}


def classify_prompt(prompt: str, canonical_prompt: str, organ: str, category: str | None = None) -> tuple[str, str]:
    if category:
        return taxonomy_for_legacy_category(category)
    text = clean_prompt_text(prompt)
    low = text.lower()
    canonical = clean_prompt_text(canonical_prompt).lower()
    organ_words = clean_prompt_text(str(organ).replace("_", " ")).lower()
    if low == canonical or low == f"segment the {organ_words}":
        return "AI Researcher Prompt", "simple_instruction"
    if any(token in low for token in ["specified by the canonical", "target organ", "canonical organ token"]):
        return "AI Researcher Prompt", "task_specific_prompt"
    if any(token in low for token in ["please", "delineate", "mask for", "voxels"]):
        return "User Natural Prompt", "natural_language_request"
    if any(token in low for token in ["adjacent", "inferior", "superior", "anterior", "posterior", "medial", "lateral", "field of view", "landmark", "location"]):
        return "Clinical Expert Prompt", "anatomy_location_prompt"
    if any(token in low for token in ["ct", "density", "boundary", "boundaries", "contour", "capsule", "morphology", "appearance", "soft-tissue", "solid"]):
        return "Clinical Expert Prompt", "ct_appearance_prompt"
    if low.startswith("segment the ") and low != canonical:
        return "AI Researcher Prompt", "canonical_organ_name"
    if low.startswith(("find ", "show ", "mark ", "identify ")):
        return "User Natural Prompt", "informal_query"
    if low.startswith(("outline ", "segment ")):
        return "User Natural Prompt", "simplified_description"
    return "Clinical Expert Prompt", "synonym_clinical_naming_prompt"


def identity_phrases_from_context(context: dict[str, Any]) -> list[str]:
    values = [
        context.get("organ_token"),
        context.get("canonical_name"),
        str(context.get("organ_token") or "").replace("_", " "),
        *(context.get("synonyms") or []),
    ]
    phrases = sorted({normalize_phrase(v) for v in values if normalize_phrase(v)}, key=len, reverse=True)
    return phrases


def infer_laterality(organ_token: str, text: str = "") -> str:
    token = str(organ_token or "").lower()
    normalized = normalize_phrase(text)
    if token.endswith("_left") or "_left_" in token or re.search(r"\bleft\b", normalized):
        return "left"
    if token.endswith("_right") or "_right_" in token or re.search(r"\bright\b", normalized):
        return "right"
    if re.search(r"\bbilateral\b|\bboth\b", normalized):
        return "bilateral"
    return "none"


def prompt_record(
    *,
    prompt: str,
    organ_token: str,
    canonical_name: str,
    prompt_source: str,
    category: str | None = None,
    introduced_anatomy: list[str] | None = None,
    laterality: str | None = None,
    ct_scan_range_assumption: str | None = None,
) -> dict[str, Any]:
    level, subclass = classify_prompt(prompt, f"segment the {canonical_name}", organ_token, category)
    return {
        "organ_token": organ_token,
        "canonical_name": canonical_name,
        "prompt": clean_prompt_text(prompt),
        "prompt_level": level,
        "prompt_subclass": subclass,
        "introduced_anatomy": introduced_anatomy or [],
        "laterality": laterality or infer_laterality(organ_token, prompt),
        "ct_scan_range_assumption": ct_scan_range_assumption or "",
        "prompt_source": prompt_source,
    }


def _validate_schema(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in FREEFORM_REQUIRED_OUTPUT_FIELDS:
        if field not in record:
            errors.append(f"schema_missing:{field}")
    if "introduced_anatomy" in record and not isinstance(record.get("introduced_anatomy"), list):
        errors.append("schema_introduced_anatomy_not_list")
    level = record.get("prompt_level")
    subclass = record.get("prompt_subclass")
    if level not in PROMPT_TAXONOMY:
        errors.append("schema_invalid_prompt_level")
    elif subclass not in PROMPT_TAXONOMY[level]["subclasses"]:
        errors.append("schema_invalid_prompt_subclass")
    return errors


def _scan_range_errors(text: str, assumption: str, context: dict[str, Any]) -> list[str]:
    scan_range = normalize_phrase(context.get("ct_scan_range"))
    if not scan_range:
        return []
    allowed_regions = {name for name in SCAN_RANGE_KEYWORDS if name in scan_range}
    if not allowed_regions:
        return []
    prompt_terms = set(normalize_phrase(f"{text} {assumption}").split())
    out_regions = [
        region for region, terms in SCAN_RANGE_KEYWORDS.items()
        if region not in allowed_regions and prompt_terms & terms
    ]
    return [f"ct_scan_range_out_of_scope:{region}" for region in sorted(out_regions)]


def validate_generated_prompt_record(
    record: dict[str, Any],
    context: dict[str, Any],
    *,
    existing_prompts: list[str] | tuple[str, ...] | None = None,
    fallback_prompt: str | None = None,
) -> dict[str, Any]:
    """Validate a model/user/free-form prompt record and attach fallback fields."""
    existing_prompts = list(existing_prompts or [])
    fallback_prompt = clean_prompt_text(fallback_prompt or f"segment the {context.get('canonical_name') or context.get('organ_token')}")
    errors = _validate_schema(record)
    prompt = clean_prompt_text(record.get("prompt"))
    normalized = normalize_phrase(prompt)
    context_token = str(context.get("organ_token") or "")
    output_token = str(record.get("organ_token") or "")
    if output_token and output_token != context_token:
        errors.append("organ_token_mismatch")
    if context.get("canonical_name") and normalize_phrase(record.get("canonical_name")) != normalize_phrase(context.get("canonical_name")):
        errors.append("canonical_name_mismatch")
    if not prompt:
        errors.append("empty_prompt")
    words = prompt.split()
    if len(words) < 3 or len(words) > 80:
        errors.append("word_count_out_of_range")
    if record.get("prompt_subclass") == "simple_instruction" and len(words) > 18:
        errors.append("simple_instruction_too_long")
    if not any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in identity_phrases_from_context(context)):
        errors.append("target_identity_not_explicit")
    expected_laterality = str(context.get("laterality") or infer_laterality(context_token)).lower()
    output_laterality = str(record.get("laterality") or infer_laterality(context_token, prompt)).lower()
    if expected_laterality in {"left", "right"}:
        opposite = "right" if expected_laterality == "left" else "left"
        if output_laterality == opposite or re.search(rf"\b{opposite}\b", normalized):
            errors.append(f"laterality_mismatch_expected_{expected_laterality}")
    for forbidden in context.get("forbidden_targets") or []:
        phrase = normalize_phrase(str(forbidden).replace("_", " "))
        if phrase and re.search(rf"\b{re.escape(phrase)}\b", normalized):
            errors.append(f"forbidden_target:{forbidden}")
            break
    allowed_anatomy = {normalize_phrase(x) for x in (context.get("allowed_adjacent_anatomy") or [])}
    allowed_anatomy |= {normalize_phrase(x) for x in identity_phrases_from_context(context)}
    for anatomy in record.get("introduced_anatomy") or []:
        phrase = normalize_phrase(anatomy)
        if phrase and phrase not in allowed_anatomy:
            errors.append(f"introduced_anatomy_not_allowed:{anatomy}")
            break
    errors.extend(_scan_range_errors(prompt, str(record.get("ct_scan_range_assumption") or ""), context))
    for prior in existing_prompts:
        prior_norm = normalize_phrase(prior)
        if prior_norm and (prior_norm == normalized or SequenceMatcher(None, prior_norm, normalized).ratio() > 0.96):
            errors.append("duplicate_or_near_duplicate")
            break
    accepted = not errors
    result = dict(record)
    result["prompt"] = prompt
    result["validation_status"] = "accepted" if accepted else "failed"
    result["validation_errors"] = errors
    result["fallback_reason"] = None if accepted else ";".join(errors)
    result["accepted_prompt"] = prompt if accepted else fallback_prompt
    if not accepted:
        result["prompt_source"] = "fallback_fixed"
        result["fallback_prompt"] = fallback_prompt
    return result


def build_prompt_validation_context(
    *,
    organ_token: str,
    canonical_name: str,
    region: str | None = None,
    ct_appearance: str | None = None,
    synonyms: list[str] | None = None,
    laterality: str | None = None,
    allowed_adjacent_anatomy: list[str] | None = None,
    ct_scan_range: str | None = None,
    forbidden_targets: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "organ_token": organ_token,
        "canonical_name": canonical_name,
        "region": region or "",
        "ct_appearance": ct_appearance or "",
        "synonyms": synonyms or [],
        "laterality": laterality or infer_laterality(organ_token),
        "allowed_adjacent_anatomy": allowed_adjacent_anatomy or [],
        "ct_scan_range": ct_scan_range or region or "",
        "forbidden_targets": forbidden_targets or [],
    }
