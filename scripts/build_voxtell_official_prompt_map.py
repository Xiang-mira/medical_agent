#!/usr/bin/env python3
"""Build an auditable 373-target adapter to VoxTell's published prompt bank."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = (
    "segment the ", "segment ", "outline the ", "outline ", "delineate the ", "delineate ",
    "identify the ", "identify ", "find the ", "find ", "create a mask for the ", "mark the ",
)


def anatomical_term(text: str) -> str:
    value = " ".join(str(text).strip().lower().split())
    for prefix in PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = re.sub(r"\s+on (this )?ct( volume)?[.;:].*$", "", value)
    return value.strip(" .;:")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-config", type=Path, default=ROOT / "configs/student_3d_prompt_target_organs.json")
    ap.add_argument("--official-labels", type=Path, default=ROOT / "checkpoints/VoxTell/embeddings/voxtell_v1.1/labels.json")
    ap.add_argument("--output", type=Path, default=ROOT / "configs/voxtell_official_prompt_map.json")
    args = ap.parse_args()
    target = json.loads(args.target_config.read_text(encoding="utf-8"))
    labels = json.loads(args.official_labels.read_text(encoding="utf-8"))
    index = {str(label).lower(): i for i, label in enumerate(labels)}
    mappings = []
    for organ in target["target_organs"]:
        configured = str((target.get("organ_to_prompt") or {}).get(organ, organ.replace("_", " ")))
        candidate_texts = [anatomical_term(configured), organ.replace("_", " ")]
        candidate_texts.extend(
            anatomical_term(x) for x in (target.get("prompt_variants") or {}).get(organ, [])
        )
        unique = []
        for candidate in candidate_texts:
            if candidate and candidate not in unique:
                unique.append(candidate)
        official = [x for x in unique if x in index]
        canonical = official[0] if official else anatomical_term(configured)
        mappings.append({
            "project_class": organ,
            "canonical_prompt": canonical,
            "canonical_bank_index": index.get(canonical),
            "approved_variants": [
                {"text": text, "bank_index": index[text]} for text in official[1:]
            ],
            "mapping_source": "official_exact_match" if official else "project_extension",
            "review_status": (
                "official_bank_exact_match"
                if official else "runtime_supported_canonical_anatomical_term"
            ),
            "manual_semantic_review_required": False,
            "encoding_source": (
                "official_precomputed_embedding_bank"
                if official else "official_qwen3_embedding_4b_instruction_wrapper"
            ),
            "mapping_provenance": (
                "VoxTell labels.json exact string match"
                if official else "project class canonical name; no medical-description rewrite"
            ),
        })
    extensions = [x["project_class"] for x in mappings if x["mapping_source"] == "project_extension"]
    payload = {
        "schema_version": "voxtell_official_prompt_map_v1",
        "official_bank_prompt_count": len(labels),
        "target_count": len(mappings),
        "official_exact_match_count": len(mappings) - len(extensions),
        "project_extension_count": len(extensions),
        "project_extensions": extensions,
        "policy": "Exact official-bank strings are reused; semantic-neighbor suggestions are never auto-approved.",
        "mappings": mappings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "mappings"}, indent=2, ensure_ascii=False))
    return 0 if len(mappings) == 373 else 1


if __name__ == "__main__":
    raise SystemExit(main())
