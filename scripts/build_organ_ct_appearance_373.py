#!/usr/bin/env python3
"""Build the 373-organ CT appearance knowledge base for LabelCritic.

This is intentionally CPU-only/offline: it reads the project's formal target
config and writes a JSON prompt knowledge base plus a small audit report.  It
does not call LabelCritic, a VLM server, CUDA, or any teacher model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.organ_prompt_bank import display_name  # noqa: E402


DEFAULT_TARGET_CONFIG = ROOT / "configs" / "student_3d_prompt_target_organs.json"
DEFAULT_OUTPUT = ROOT / "configs" / "organ_ct_appearance_373.json"
DEFAULT_AUDIT = ROOT / "outputs" / "organ_ct_appearance_373_audit.json"
DEFAULT_SEED = ROOT / "configs" / "organ_ct_appearance_373_seed_from_labelcritic.json"


LABELCRITIC_SEED_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "aorta": {
        "aliases": ["aorta", "aortic vessel"],
        "expected_location": "thoracoabdominal midline to left paraspinal course, extending from the mediastinum toward the abdomen/pelvis depending on scan range",
        "ct_appearance": "continuous tubular arterial structure following the expected aortic course on CT",
        "shape_and_continuity_prior": "should be a continuous tubular/curvilinear structure rather than scattered disconnected blobs",
        "neighbor_relations": ["spine", "heart", "diaphragm", "inferior vena cava"],
        "anatomical_constraints": [
            "should follow the expected aortic course near the spine",
            "should not be dominated by bowel, liver, spleen, or random soft-tissue regions",
        ],
        "common_failure_modes": ["fragmented vessel", "wrong vessel selected", "grossly implausible lateral position", "missing long aortic segment"],
        "rejection_rules": ["reject if the mask is mostly outside the expected vascular course", "reject if the mask is scattered rather than tubular"],
    },
    "descending aorta": {
        "aliases": ["descending aorta", "thoracic/abdominal descending aorta"],
        "expected_location": "posterior mediastinum and upper abdomen near the spine, usually left of midline",
        "ct_appearance": "continuous descending tubular arterial structure running near the spine",
        "shape_and_continuity_prior": "should be vertically continuous and tubular along the descending aortic course",
        "neighbor_relations": ["spine", "diaphragm", "mediastinum", "left lung"],
        "anatomical_constraints": ["should not start only at the lower chest/diaphragm if upper thorax is visible", "should remain close to the spinal axis"],
        "common_failure_modes": ["starts too low", "fragmented vessel", "confused with spine or other vessels"],
        "rejection_rules": ["reject if the overlay is not a plausible descending aortic tube", "reject if most voxels lie far from the paraspinal vascular course"],
    },
    "postcava": {
        "aliases": ["postcava", "inferior vena cava", "IVC"],
        "expected_location": "right paraspinal retroperitoneal region, anterior to vertebral bodies and usually right of the aorta",
        "ct_appearance": "long venous tubular structure coursing through the abdomen toward the liver/right atrium",
        "shape_and_continuity_prior": "should be a continuous tubular venous structure",
        "neighbor_relations": ["aorta", "liver", "right kidney", "renal veins"],
        "anatomical_constraints": ["should usually be right of the aorta", "should not include broad liver or bowel regions"],
        "common_failure_modes": ["confused with aorta", "fragmented vessel", "leakage into liver parenchyma"],
        "rejection_rules": ["reject if not tubular", "reject if mostly outside the caval/paraspinal course"],
    },
    "inferior_vena_cava": {
        "aliases": ["inferior vena cava", "IVC", "postcava"],
        "expected_location": "right paraspinal retroperitoneal region, anterior to vertebral bodies and usually right of the aorta",
        "ct_appearance": "long venous tubular structure coursing through the abdomen toward the liver/right atrium",
        "shape_and_continuity_prior": "should be a continuous tubular venous structure",
        "neighbor_relations": ["aorta", "liver", "right kidney", "renal veins"],
        "anatomical_constraints": ["should usually be right of the aorta", "should not include broad liver or bowel regions"],
        "common_failure_modes": ["confused with aorta", "fragmented vessel", "leakage into liver parenchyma"],
        "rejection_rules": ["reject if not tubular", "reject if mostly outside the caval/paraspinal course"],
    },
    "liver": {
        "aliases": ["liver", "hepatic parenchyma"],
        "expected_location": "right upper abdomen under the diaphragm and rib cage, often extending across midline",
        "ct_appearance": "large solid soft-tissue organ with smooth capsule and wedge-like contour on CT",
        "shape_and_continuity_prior": "should be one large contiguous solid organ rather than scattered islands",
        "neighbor_relations": ["diaphragm", "gallbladder", "stomach", "right kidney"],
        "anatomical_constraints": ["should not be located in the pelvis", "should not include stomach, spleen, bowel, or lung"],
        "common_failure_modes": ["leakage into stomach or spleen", "missing hepatic lobe", "pelvic false positive", "fragmented liver"],
        "rejection_rules": ["reject if the mask is mostly outside the upper abdomen", "reject if it does not resemble a large contiguous hepatic organ"],
    },
    "kidneys": {
        "aliases": ["kidneys", "renal organs"],
        "expected_location": "bilateral retroperitoneum lateral to the spine, around the lower ribs/upper abdomen",
        "ct_appearance": "paired bean-shaped retroperitoneal soft-tissue organs",
        "shape_and_continuity_prior": "should appear as one left and one right kidney when both are in field of view",
        "neighbor_relations": ["spine", "psoas muscles", "adrenal glands", "spleen", "liver"],
        "anatomical_constraints": ["should be near the paraspinal retroperitoneum", "should not be many random components"],
        "common_failure_modes": ["missing one kidney", "wrong number of components", "confusion with spleen/liver", "fragmentation"],
        "rejection_rules": ["reject if the mask is not kidney-shaped in the expected retroperitoneal region", "reject if dominated by scattered blobs"],
    },
    "kidney_left": {
        "aliases": ["left kidney", "left renal organ"],
        "expected_location": "left retroperitoneum lateral to the spine, inferior to the spleen",
        "ct_appearance": "bean-shaped retroperitoneal organ with renal parenchyma and sinus on CT",
        "shape_and_continuity_prior": "should be a single left-sided bean-shaped organ",
        "neighbor_relations": ["spleen", "left adrenal gland", "psoas muscle", "renal vessels"],
        "anatomical_constraints": ["must be left-sided", "should not include the right kidney or spleen"],
        "common_failure_modes": ["right-left swap", "confusion with spleen", "fragmentation", "missing renal pole"],
        "rejection_rules": ["reject if mostly right-sided", "reject if not in the left retroperitoneal kidney region"],
    },
    "kidney_right": {
        "aliases": ["right kidney", "right renal organ"],
        "expected_location": "right retroperitoneum lateral to the spine, inferior to the liver",
        "ct_appearance": "bean-shaped retroperitoneal organ with renal parenchyma and sinus on CT",
        "shape_and_continuity_prior": "should be a single right-sided bean-shaped organ",
        "neighbor_relations": ["liver", "right adrenal gland", "psoas muscle", "renal vessels"],
        "anatomical_constraints": ["must be right-sided", "should not include the left kidney or liver"],
        "common_failure_modes": ["right-left swap", "confusion with liver", "fragmentation", "missing renal pole"],
        "rejection_rules": ["reject if mostly left-sided", "reject if not in the right retroperitoneal kidney region"],
    },
    "spleen": {
        "aliases": ["spleen", "splenic parenchyma"],
        "expected_location": "left upper abdomen beneath the left hemidiaphragm and ribs",
        "ct_appearance": "oval or crescent-shaped homogeneous soft-tissue organ",
        "shape_and_continuity_prior": "should be one contiguous smooth solid organ",
        "neighbor_relations": ["stomach", "left kidney", "diaphragm", "left ribs"],
        "anatomical_constraints": ["should be left upper quadrant", "should not include liver, stomach lumen, or kidney"],
        "common_failure_modes": ["confusion with left kidney", "leakage into stomach", "fragmentation", "wrong-side location"],
        "rejection_rules": ["reject if mostly outside left upper abdomen", "reject if not a plausible contiguous splenic shape"],
    },
    "stomach": {
        "aliases": ["stomach", "gastric organ"],
        "expected_location": "left/central upper abdomen between esophagus and duodenum, below the diaphragm",
        "ct_appearance": "hollow J-shaped or sac-like gastrointestinal organ that may contain air, fluid, or contrast",
        "shape_and_continuity_prior": "should be a plausible connected stomach organ, not random disconnected dots",
        "neighbor_relations": ["liver", "spleen", "pancreas", "diaphragm"],
        "anatomical_constraints": ["should not be many disconnected components", "should not broadly include colon or liver"],
        "common_failure_modes": ["fragmented blobs", "confusion with bowel loops", "leakage into liver/spleen", "implausible pelvic location"],
        "rejection_rules": ["reject if mostly outside upper abdomen", "reject if dominated by random scattered components"],
    },
    "pancreas": {
        "aliases": ["pancreas", "pancreatic gland"],
        "expected_location": "upper retroperitoneal abdomen posterior to the stomach, from duodenal curve toward splenic hilum",
        "ct_appearance": "elongated lobulated soft-tissue gland with thicker head and thinner body/tail",
        "shape_and_continuity_prior": "should be a smooth elongated structure, usually connected and mostly horizontal/curved",
        "neighbor_relations": ["duodenum", "stomach", "spleen", "portal confluence"],
        "anatomical_constraints": ["should not be a large round abdominal organ", "should not fragment into random dots"],
        "common_failure_modes": ["missing tail", "confusion with bowel or vessel", "fragmentation", "oversegmentation into stomach/duodenum"],
        "rejection_rules": ["reject if not in the upper retroperitoneal pancreatic course", "reject if shape is grossly non-pancreatic"],
    },
    "gall_bladder": {
        "aliases": ["gallbladder", "gall bladder", "cholecystic sac"],
        "expected_location": "right upper abdomen along the inferior liver surface",
        "ct_appearance": "small pear-shaped fluid-density sac adjacent to the liver",
        "shape_and_continuity_prior": "should be a small single smooth sac-like structure",
        "neighbor_relations": ["liver", "duodenum", "bile duct", "right hepatic lobe"],
        "anatomical_constraints": ["should be near the inferior liver", "should not include large liver or bowel regions"],
        "common_failure_modes": ["leakage into liver", "confusion with bowel fluid", "fragmentation", "implausibly large mask"],
        "rejection_rules": ["reject if not near the gallbladder fossa/inferior liver", "reject if broad solid-organ leakage dominates"],
    },
}


def _category_for(organ: str) -> str:
    low = organ.lower()
    if any(k in low for k in ("artery", "vein", "aorta", "vessel", "cava", "portal")):
        return "vessel"
    if any(k in low for k in ("bowel", "colon", "intestine", "stomach", "duodenum", "esophagus", "airway", "bronch", "duct", "ureter")):
        return "tubular_or_luminal_structure"
    if any(k in low for k in ("bone", "femur", "tibia", "fibula", "humerus", "rib", "vertebra", "mandible", "clavicle", "sternum")):
        return "bone"
    if any(k in low for k in ("muscle", "scalene", "psoas", "gluteus", "rectus")):
        return "muscle"
    if any(k in low for k in ("segment", "lobe", "head", "body", "tail")):
        return "sub_organ_part"
    return "organ_or_structure"


def _laterality_for(organ: str) -> str | None:
    low = organ.lower()
    if low.endswith("_left") or "_left_" in low or low.startswith("left_"):
        return "left"
    if low.endswith("_right") or "_right_" in low or low.startswith("right_"):
        return "right"
    return None


def _load_seed_entries(seed_path: Path | None) -> dict[str, dict[str, Any]]:
    if not seed_path or not seed_path.exists():
        return {}
    try:
        doc = json.loads(seed_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for entry in doc.get("entries", []) or []:
        if isinstance(entry, dict) and entry.get("canonical_organ"):
            out[str(entry["canonical_organ"])] = entry
    return out


def _seed_for_organ(organ: str, seed_entries: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    aliases = {organ}
    if organ == "inferior_vena_cava":
        aliases.add("postcava")
    if organ == "postcava":
        aliases.add("inferior_vena_cava")
    if organ == "aorta":
        aliases.add("descending_aorta")
    for key in aliases:
        if key in seed_entries:
            return seed_entries[key]
    return None


def _clean_seed_source(source: str) -> str:
    return "labelcritic_seed" if source else "labelcritic_seed"


def _clean_list(values: Any, fallback: list[str]) -> list[str]:
    if isinstance(values, list):
        cleaned = [" ".join(str(v).split()) for v in values if str(v).strip()]
    elif values:
        cleaned = [" ".join(str(values).split())]
    else:
        cleaned = []
    out: list[str] = []
    seen: set[str] = set()
    for item in cleaned or fallback:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _infer_shape_prior(organ: str, ct_appearance: str) -> str:
    low = organ.lower()
    if any(k in low for k in ("artery", "vein", "aorta", "vessel", "cava", "portal")):
        return "The mask should follow a plausible tubular or branching vascular course and should not appear as random disconnected blobs."
    if any(k in low for k in ("bowel", "colon", "intestine", "stomach", "duodenum", "esophagus", "airway", "bronch")):
        return "The mask should follow a plausible hollow tubular or sac-like course with anatomically reasonable continuity."
    if any(k in low for k in ("bone", "femur", "tibia", "fibula", "humerus", "rib", "vertebra", "mandible")):
        return "The mask should follow the expected osseous structure with coherent cortical/trabecular shape rather than soft-tissue leakage."
    if any(k in low for k in ("muscle", "scalene", "psoas", "gluteus", "rectus")):
        return "The mask should follow a coherent muscle belly/fascial compartment and avoid adjacent organs or bones."
    if "soft-tissue structure with CT boundaries guided by adjacent anatomy" in ct_appearance:
        return "The mask should remain anatomically coherent for the named target and avoid scattered false positives."
    return "The mask should match the expected CT appearance, shape, location, and continuity of the named target."


def _common_failure_modes(organ: str) -> list[str]:
    low = organ.lower()
    modes = ["wrong organ or neighboring-structure leakage", "grossly implausible location", "oversegmentation outside expected boundaries"]
    if organ.endswith("_left") or "_left_" in organ or organ.endswith("_right") or "_right_" in organ:
        modes.append("left-right side confusion")
    if any(k in low for k in ("artery", "vein", "vessel", "aorta", "cava", "portal")):
        modes.extend(["fragmented vessel course", "confusion with adjacent vessel"])
    elif any(k in low for k in ("bowel", "colon", "intestine", "stomach", "duodenum", "airway", "bronch")):
        modes.extend(["disconnected scattered components", "leakage into adjacent loops or lumen-like structures"])
    else:
        modes.extend(["fragmented mask", "undersegmentation of a visible component"])
    return _clean_list(modes, [])


def _rejection_rules(organ: str) -> list[str]:
    rules = [
        "reject if the mask is mostly outside the expected anatomic region",
        "reject if the mask primarily marks a different organ or neighboring structure",
        "reject if the mask is dominated by scattered isolated false positives",
    ]
    if organ.endswith("_left") or "_left_" in organ:
        rules.append("reject if the mask is predominantly right-sided")
    if organ.endswith("_right") or "_right_" in organ:
        rules.append("reject if the mask is predominantly left-sided")
    return rules


def build_entry(organ: str, bank_entry: dict[str, Any], seed_entries: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    package_seed = _seed_for_organ(organ, seed_entries or {})
    seed = package_seed or LABELCRITIC_SEED_DESCRIPTIONS.get(organ)
    source = "labelcritic_seed" if seed else "generated_from_project_config"
    aliases = _clean_list((seed or {}).get("aliases") or bank_entry.get("aliases"), [display_name(organ)])
    landmarks = _clean_list((seed or {}).get("neighbor_relations") or bank_entry.get("landmarks"), ["nearby organs", "vessels", "bones", "soft-tissue planes"])
    expected_location = str((seed or {}).get("expected_location") or bank_entry.get("region") or "expected anatomic field of view on CT")
    ct_appearance = str((seed or {}).get("ct_appearance") or bank_entry.get("ct_appearance") or "soft-tissue structure with CT boundaries guided by adjacent anatomy")
    shape_prior = str((seed or {}).get("shape_and_continuity_prior") or _infer_shape_prior(organ, ct_appearance))
    constraints = _clean_list((seed or {}).get("anatomical_constraints"), [
        f"should be located in or near {expected_location}",
        "should respect adjacent anatomic boundaries and not leak into unrelated structures",
    ])
    failures = _clean_list((seed or {}).get("common_failure_modes"), _common_failure_modes(organ))
    rejection = _clean_list((seed or {}).get("rejection_rules"), _rejection_rules(organ))
    primary = aliases[0] if aliases else display_name(organ)
    return {
        "canonical_organ": organ,
        "canonical_id": organ,
        "display_name": bank_entry.get("display_name") or display_name(organ),
        "aliases": aliases,
        "category": str((seed or {}).get("category") or _category_for(organ)),
        "laterality": (seed or {}).get("laterality", _laterality_for(organ)),
        "expected_region": expected_location,
        "expected_location": expected_location,
        "landmarks": landmarks,
        "ct_appearance": ct_appearance,
        "shape_and_continuity_prior": shape_prior,
        "neighbor_relations": landmarks,
        "anatomical_constraints": constraints,
        "common_failure_modes": failures,
        "rejection_rules": rejection,
        "labelcritic_instruction": (
            f"Compare candidate masks for {primary} and select the candidate that best matches the target's CT appearance, "
            "expected location, shape/continuity, laterality when applicable, and exclusion boundaries. Reject candidates "
            "with wrong-organ leakage, gross mislocation, severe fragmentation, or implausible anatomy."
        ),
        "source": source,
        "requires_manual_review": source != "labelcritic_seed",
    }


def build(target_config: Path, seed_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    doc = json.loads(target_config.read_text(encoding="utf-8"))
    organs = [str(o) for o in doc.get("target_organs", [])]
    bank = doc.get("organ_prompt_bank", {}) or {}
    seed_entries = _load_seed_entries(seed_path)
    entries = {organ: build_entry(organ, bank.get(organ, {}), seed_entries) for organ in organs}
    required = {
        "canonical_organ",
        "expected_region",
        "ct_appearance",
        "expected_location",
        "common_failure_modes",
        "rejection_rules",
        "source",
        "requires_manual_review",
    }
    missing_fields = {
        organ: sorted(k for k in required if k not in entry or entry.get(k) in (None, "", []))
        for organ, entry in entries.items()
    }
    missing_fields = {k: v for k, v in missing_fields.items() if v}
    source_counts: dict[str, int] = {}
    for entry in entries.values():
        source_counts[str(entry.get("source"))] = source_counts.get(str(entry.get("source")), 0) + 1
    seed_organs = sorted([organ for organ, entry in entries.items() if entry.get("source") == "labelcritic_seed"])
    entry_list = [entries[organ] for organ in organs]
    audit = {
        "status": "success" if len(entries) == len(organs) and not missing_fields else "failed",
        "target_count": len(organs),
        "entry_count": len(entries),
        "missing_organs": sorted(set(organs) - set(entries)),
        "extra_organs": sorted(set(entries) - set(organs)),
        "missing_required_fields": missing_fields,
        "source_counts": source_counts,
        "labelcritic_seed_organs": seed_organs,
        "manual_review_count": sum(1 for entry in entries.values() if entry.get("requires_manual_review")),
        "cpu_only": True,
        "note": "373-organ prompt bank extension; non-seed organs are project drafts and require manual/expert review.",
    }
    out = {
        "version": "organ_ct_appearance_373.v1",
        "status": audit["status"],
        "schema_version": "labelcritic_organ_ct_appearance.v1",
        "target_source": str(target_config),
        "target_config": str(target_config),
        "seed_source": str(seed_path) if seed_path else None,
        "entry_count": len(entry_list),
        "entries": entry_list,
        "target_count": len(organs),
        "source_policy": {
            "labelcritic_seed": "Inherited/normalized from original LabelCritic organ-specific prompt concepts.",
            "project_prompt_bank": "Derived from the project's formal 373-organ prompt bank; project draft, not official LabelCritic 373 support.",
        },
        "organ_ct_appearance": entries,
        "audit": audit,
    }
    return out, audit


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CPU-only 373-organ CT appearance prompt bank for LabelCritic.")
    ap.add_argument("--target-config", "--student_targets", dest="target_config", default=str(DEFAULT_TARGET_CONFIG))
    ap.add_argument("--seed", default=str(DEFAULT_SEED))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    args = ap.parse_args()

    output, audit = build(Path(args.target_config).resolve(), Path(args.seed).resolve() if args.seed else None)
    out_path = Path(args.output)
    audit_path = Path(args.audit_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "target_count": audit["target_count"], "entry_count": audit["entry_count"], "output": str(out_path), "audit": str(audit_path)}, indent=2))
    if audit["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
