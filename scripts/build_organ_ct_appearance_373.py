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

from cli_anything.medai.core.organ_prompt_bank import (  # noqa: E402
    display_name,
    _infer_region_and_landmarks as infer_project_region_and_landmarks,
)


DEFAULT_TARGET_CONFIG = ROOT / "configs" / "student_3d_prompt_target_organs.json"
DEFAULT_OUTPUT = ROOT / "configs" / "organ_ct_appearance_373.json"
DEFAULT_AUDIT = ROOT / "outputs" / "organ_ct_appearance_373_audit.json"
DEFAULT_SEED = ROOT / "configs" / "organ_ct_appearance_373_seed_from_labelcritic.json"
DEFAULT_TAXONOMY = ROOT / "configs" / "organ_taxonomy.json"

CAPABILITY_LEVELS = {
    "runtime_supported",
    "official_benchmark_calibrated_7b",
    "class_agnostic_extrapolation",
    "runtime_only_unverified",
}


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


def _body_regions(location: str) -> list[str]:
    low = location.lower()
    regions = []
    rules = {
        "head_neck": ("head", "cran", "brain", "facial", "neck", "cervical"),
        "thorax": ("thorax", "thoracic", "mediast", "lung", "cardiac", "heart", "rib"),
        "abdomen": ("abdomen", "abdominal", "retroperitone", "hepatic", "liver", "pancrea"),
        "pelvis": ("pelvis", "pelvic", "prostate", "uter", "bladder"),
        "extremity": ("extremity", "appendicular", "femur", "tibia", "humer", "hand", "foot"),
    }
    for region, terms in rules.items():
        if any(term in low for term in terms):
            regions.append(region)
    return regions or ["unknown"]


def _morphology_for(organ: str, category: str, appearance: str) -> str:
    low = organ.lower()
    if organ == "colon":
        return (
            "Tubular, haustrated large-bowel structure forming a peripheral "
            "abdominopelvic frame; segmental visibility is acceptable, random "
            "isolated blobs are not."
        )
    if category == "vessel":
        return "Tubular or branching vascular structure following a plausible anatomic course."
    if category == "tubular_or_luminal_structure":
        return "Tubular, hollow, or sac-like structure with anatomically plausible segmental continuity."
    if category == "bone":
        return "Coherent osseous structure respecting the expected cortical and trabecular outline."
    if category == "muscle":
        return "Elongated or sheet-like muscle compartment with a coherent fascial course."
    if any(term in low for term in ("lesion", "tumor", "metasta", "node")):
        return "May be focal or multifocal; components must remain plausible for the named pathology and location."
    return appearance


def _symmetry_prior(organ: str, laterality: str | None) -> str:
    if laterality:
        opposite = "right" if laterality == "left" else "left"
        return (
            f"Laterality-specific target: foreground should be predominantly {laterality}-sided "
            f"and must not be the {opposite}-sided counterpart."
        )
    if organ in {"kidneys", "lungs", "adrenal_glands", "femurs"}:
        return "Paired target; both sides may be expected when fully covered, subject to anatomy and scan range."
    return "No mandatory bilateral symmetry; judge the named structure and scan coverage."


def _partial_fov_behavior(category: str) -> str:
    if category in {"vessel", "tubular_or_luminal_structure"}:
        return (
            "A structure may terminate at a scan boundary when partially covered; "
            "internal random breaks away from a boundary remain suspicious."
        )
    return (
        "Boundary truncation may be valid only when the target touches a physical scan "
        "boundary; missing internal portions must not be excused as partial field of view."
    )


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


def _region_specific_neighbors(expected_location: str, category: str) -> list[str]:
    low = expected_location.lower()
    if any(token in low for token in ("cran", "brain", "head", "facial", "neck")):
        base = ["skull or facial bones", "brain or upper aerodigestive tract", "regional vessels and muscles"]
    elif any(token in low for token in ("thorax", "chest", "mediast", "cardiac", "lung")):
        base = ["lungs", "mediastinum and heart", "ribs and thoracic spine"]
    elif any(token in low for token in ("pelvis", "pelvic", "lower abdomen")):
        base = ["urinary bladder", "rectum and pelvic viscera", "pelvic bones and regional vessels"]
    elif any(token in low for token in ("extrem", "arm", "leg", "hand", "foot")):
        base = ["adjacent named bone or joint", "regional muscle compartments", "regional vessels and subcutaneous tissue"]
    elif any(token in low for token in ("abdomen", "retroperitone", "upper quadrant")):
        base = ["liver, stomach, and bowel", "kidneys and retroperitoneum", "aorta, vena cava, and spine"]
    else:
        return []
    if category in {"artery", "vein", "vessel"}:
        base[0] = "parent and downstream vascular branches"
    elif category in {"bone", "vertebra", "rib"}:
        base[0] = "adjacent bones, joints, and cortical boundaries"
    elif category in {"muscle"}:
        base[0] = "adjacent muscle compartments and fascial planes"
    return base


def _extended_project_region(organ: str) -> tuple[str, list[str]]:
    low = organ.lower()
    rules = [
        (
            (
                "sulcus", "lobe", "gray_matter", "white_matter", "cerebrospinal",
                "auditory", "optic", "oral", "nasal", "cheek", "lip", "palate",
                "pituitary", "thalamus", "parotid", "submandibular", "scalp",
                "skull", "tongue", "pterygoid", "masseter", "temporalis",
                "inferior_rectus", "superior_rectus", "medial_rectus",
                "lateral_rectus", "inferior_oblique", "superior_oblique",
                "lacrimal", "zygomatic", "buccal", "digastric", "face",
                "levator_palpebrae", "mandible", "septum_pellucidum",
                "styloid", "subarachnoid",
            ),
            "head and craniofacial region",
            ["skull and facial bones", "brain and orbits", "upper aerodigestive tract and regional muscles"],
        ),
        (
            (
                "arytenoid", "cricoid", "cricopharyngeus", "pharyngeal",
                "glottis", "supraglottis", "sternocleidomastoid", "platysma",
                "prevertebral", "hard_palate", "soft_palate", "trachea",
            ),
            "neck and upper aerodigestive tract",
            ["larynx and pharynx", "cervical spine", "major neck vessels and muscles"],
        ),
        (
            (
                "breast", "mammary", "pericard", "pulmonary", "brachiocephalic",
                "subclavian", "rib", "costal", "sternum", "thoracic", "thymus",
                "mediastinal", "scapula", "trapezius", "levator_scapulae",
            ),
            "thorax or upper mediastinum",
            ["lungs and mediastinum", "heart and great vessels", "ribs, sternum, and thoracic spine"],
        ),
        (
            (
                "abdominal", "pancreatic", "renal", "mesenteric", "psoas",
                "iliopsoas", "rectus_abdominis", "cbd_stent", "colostomy",
                "small_bowel",
            ),
            "abdomen and retroperitoneum",
            ["solid abdominal organs", "bowel and abdominal wall", "aorta, vena cava, and spine"],
        ),
        (
            ("seminal", "uterocervix", "sacrum"),
            "pelvis and lower abdomen",
            ["pelvic viscera", "rectum and bladder", "sacrum and pelvic sidewalls"],
        ),
        (
            (
                "radius", "ulna", "patella", "tarsal", "toes", "phalanges",
                "autochthon",
            ),
            "appendicular skeleton or extremity field of view",
            ["adjacent bones and joints", "regional muscles", "subcutaneous tissue and vessels"],
        ),
    ]
    for tokens, region, neighbors in rules:
        if any(token in low for token in tokens):
            return region, neighbors
    if low.startswith("vertebrae_c"):
        return "neck and cervical spine", ["cervical spinal canal", "neck muscles and vessels", "adjacent cervical vertebrae"]
    if low.startswith("vertebrae_t"):
        return "thorax and thoracic spine", ["thoracic spinal canal", "ribs and paraspinal muscles", "adjacent thoracic vertebrae"]
    if low.startswith("vertebrae_l") or low.startswith("vertebrae_s"):
        return "abdomen/pelvis and lumbosacral spine", ["spinal canal", "psoas and paraspinal muscles", "adjacent lumbosacral vertebrae"]
    if low in {"spinal_canal", "spinal_cord"}:
        return "spinal axis within the scanned body region", ["vertebral bodies", "posterior elements", "paraspinal soft tissues"]
    return "expected anatomic field of view on CT", []


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


def build_entry(
    organ: str,
    bank_entry: dict[str, Any],
    seed_entries: dict[str, dict[str, Any]] | None = None,
    taxonomy_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package_seed = _seed_for_organ(organ, seed_entries or {})
    seed = package_seed or LABELCRITIC_SEED_DESCRIPTIONS.get(organ)
    source = "labelcritic_seed" if seed else "generated_from_project_config"
    aliases = _clean_list((seed or {}).get("aliases") or bank_entry.get("aliases"), [display_name(organ)])
    expected_location = str((seed or {}).get("expected_location") or bank_entry.get("region") or "expected anatomic field of view on CT")
    location_inferred = False
    if not seed and expected_location.strip().lower() == "expected anatomic field of view on ct":
        inferred_region, inferred_landmarks = infer_project_region_and_landmarks(organ)
        if inferred_region == "expected anatomic field of view on CT":
            inferred_region, inferred_landmarks = _extended_project_region(organ)
        if inferred_region != "expected anatomic field of view on CT":
            expected_location = inferred_region
            location_inferred = True
            configured_landmarks = _clean_list(bank_entry.get("landmarks"), [])
            if (
                not configured_landmarks
                or set(map(str.lower, configured_landmarks)) <= {
                    "nearby organs", "vessels", "bones", "soft-tissue planes"
                }
            ):
                bank_entry = {**bank_entry, "landmarks": inferred_landmarks}
    category = str((seed or {}).get("category") or _category_for(organ))
    landmark_source = (seed or {}).get("neighbor_relations") or bank_entry.get("landmarks")
    landmarks = _clean_list(
        landmark_source,
        _region_specific_neighbors(expected_location, category)
        or ["nearby organs", "vessels", "bones", "soft-tissue planes"],
    )
    ct_appearance = str((seed or {}).get("ct_appearance") or bank_entry.get("ct_appearance") or "soft-tissue structure with CT boundaries guided by adjacent anatomy")
    if not seed:
        ct_appearance = (
            f"{display_name(organ)} should be identified as the named {_category_for(organ).replace('_', ' ')} "
            f"within {expected_location}; use its relationship to {', '.join(landmarks[:3])} to distinguish it "
            f"from adjacent tissues. Base appearance prior: {ct_appearance}"
        )
    shape_prior = str((seed or {}).get("shape_and_continuity_prior") or _infer_shape_prior(organ, ct_appearance))
    constraints = _clean_list((seed or {}).get("anatomical_constraints"), [
        f"the candidate must specifically represent {display_name(organ)}, not merely a structure of the same broad type",
        f"should be located in or near {expected_location}",
        "should respect adjacent anatomic boundaries and not leak into unrelated structures",
    ])
    failures = _clean_list((seed or {}).get("common_failure_modes"), _common_failure_modes(organ))
    rejection = _clean_list((seed or {}).get("rejection_rules"), _rejection_rules(organ))
    if organ == "colon":
        shape_prior = (
            "The colon should follow a continuous or segmentally plausible tubular "
            "large-bowel course. Random disconnected blobs or an unexplained missing "
            "internal segment are not anatomically plausible."
        )
        constraints = [
            "follow the expected peripheral abdominal and pelvic large-bowel frame",
            "distinguish colon from small bowel, stomach, rectum, and abdominal wall",
            "allow boundary truncation only when the visible segment reaches a physical scan boundary",
        ]
        failures = [
            "random disconnected false-positive blobs",
            "missing internal colonic segment away from a scan boundary",
            "small-bowel or stomach inclusion",
            "abdominal-wall leakage",
            "gross oversegmentation outside the large-bowel course",
        ]
        rejection = [
            "reject if foreground is dominated by random disconnected blobs",
            "reject if the candidate primarily follows small bowel, stomach, or abdominal wall",
            "penalize an internal missing segment that cannot be explained by partial field of view",
            "penalize leakage outside the expected peripheral abdominopelvic colonic course",
        ]
    primary = aliases[0] if aliases else display_name(organ)
    laterality = (seed or {}).get("laterality", _laterality_for(organ))
    parent_structures = list((taxonomy_entry or {}).get("parent_ids") or [])
    generic_location = expected_location.strip().lower() in {
        "expected anatomic field of view on ct",
        "as described in original labelcritic prompt",
    }
    capability = (
        "runtime_supported"
        if seed
        else "runtime_only_unverified"
        if generic_location
        else "class_agnostic_extrapolation"
    )
    validation_reasons = []
    if generic_location and not seed:
        validation_reasons.append("generic_expected_location_requires_better_anatomic_source")
    if (not seed) and (not landmarks or set(map(str.lower, landmarks)) <= {
        "nearby organs", "vessels", "bones", "soft-tissue planes"
    }):
        validation_reasons.append("generic_adjacent_structures")
    validation_status = "eligible" if not validation_reasons else "runtime_only"
    return {
        "canonical_name": organ,
        "canonical_organ": organ,
        "canonical_id": organ,
        "display_name": bank_entry.get("display_name") or display_name(organ),
        "aliases": aliases,
        "category": category,
        "laterality": laterality,
        "parent_structures": parent_structures,
        "expected_body_regions": _body_regions(expected_location),
        "expected_region": expected_location,
        "expected_location": expected_location,
        "ct_location": expected_location,
        "landmarks": landmarks,
        "ct_appearance": ct_appearance,
        "morphology": _morphology_for(organ, category, ct_appearance),
        "shape_and_continuity_prior": shape_prior,
        "continuity_prior": shape_prior,
        "neighbor_relations": landmarks,
        "adjacent_structures": landmarks,
        "symmetry_prior": _symmetry_prior(organ, laterality),
        "partial_fov_behavior": _partial_fov_behavior(category),
        "anatomical_constraints": constraints,
        "common_failure_modes": failures,
        "rejection_rules": rejection,
        "penalty_rules": rejection,
        "labelcritic_instruction": (
            f"Compare candidate masks for {primary} and select the candidate that best matches the target's CT appearance, "
            "expected location, shape/continuity, laterality when applicable, and exclusion boundaries. Reject candidates "
            "with wrong-organ leakage, gross mislocation, severe fragmentation, or implausible anatomy."
        ),
        "source": source,
        "source_type": "official_labelcritic_seed" if seed else "project_extension",
        "provenance": {
            "description_source": (
                "official_labelcritic_prompt_seed"
                if seed
                else "project_373_prompt_bank"
            ),
            "taxonomy_source": str(DEFAULT_TAXONOMY),
            "field_policy": (
                "official seed text preserved; compatibility fields normalized"
                if seed
                else "structured project extension; not official LabelCritic or VoxTell content"
            ),
        },
        "field_provenance": {
            "canonical_name": "student_3d_prompt_target_organs.target_organs",
            "aliases": "official_labelcritic_seed" if seed else "student_3d_prompt_target_organs.organ_prompt_bank",
            "laterality": "official_labelcritic_seed" if seed and "laterality" in seed else "deterministic_name_rule",
            "parent_structures": "organ_taxonomy.json.parent_ids",
            "expected_body_regions": "derived_from_expected_location",
            "ct_location": (
                "official_labelcritic_seed" if seed
                else "project_organ_name_region_rule" if location_inferred
                else "student_3d_prompt_target_organs.organ_prompt_bank.region"
            ),
            "ct_appearance": "official_labelcritic_seed" if seed else "project_category_rule_plus_prompt_bank",
            "morphology": "official_labelcritic_seed" if seed else "project_category_rule",
            "adjacent_structures": (
                "official_labelcritic_seed" if seed
                else "student_3d_prompt_target_organs.organ_prompt_bank.landmarks"
                if landmark_source else "project_region_category_rule"
            ),
            "continuity_prior": "official_labelcritic_seed" if seed else "project_category_rule",
            "symmetry_prior": "deterministic_laterality_rule",
            "partial_fov_behavior": "project_category_rule",
            "common_failure_modes": "official_labelcritic_seed" if seed else "project_category_rule",
            "penalty_rules": "official_labelcritic_seed" if seed else "project_laterality_and_failure_rule",
        },
        "capability_level": capability,
        "validation_status": validation_status,
        "validation_reasons": validation_reasons,
        "formal_selection_eligible": validation_status == "eligible",
        "requires_manual_review": False,
        "automatic_failure_action": (
            "allow_official_benchmark_path"
            if seed else (
                "allow_class_agnostic_pairwise"
                if validation_status == "eligible"
                else "abstain_runtime_only_unverified"
            )
        ),
    }


def _cross_entry_validation(
    entries: dict[str, dict[str, Any]],
    taxonomy: dict[str, Any],
) -> None:
    """Apply conservative, machine-checkable consistency rules.

    These checks validate internal consistency, not medical accuracy.
    """
    names = set(entries)
    paired_bases: dict[str, set[str]] = {}
    for organ in names:
        if organ.endswith("_left"):
            paired_bases.setdefault(organ[:-len("_left")], set()).add("left")
        elif organ.endswith("_right"):
            paired_bases.setdefault(organ[:-len("_right")], set()).add("right")
    for organ, entry in entries.items():
        reasons = list(entry.get("validation_reasons") or [])
        laterality = str(entry.get("laterality") or "none")
        if organ.endswith("_left") and laterality != "left":
            reasons.append("laterality_name_mismatch")
        if organ.endswith("_right") and laterality != "right":
            reasons.append("laterality_name_mismatch")
        if laterality in {"left", "right"}:
            counterpart = (
                organ.replace("_left_", "_right_").removesuffix("_left") + ("_right" if organ.endswith("_left") else "")
                if laterality == "left"
                else organ.replace("_right_", "_left_").removesuffix("_right") + ("_left" if organ.endswith("_right") else "")
            )
            entry["counterpart_class"] = counterpart if counterpart in names else None
            entry["counterpart_in_label_space"] = counterpart in names
        unknown_parents = [
            parent for parent in entry.get("parent_structures", [])
            if parent not in names and parent not in taxonomy
        ]
        if unknown_parents:
            reasons.append("unknown_parent_structure")
        category = str(entry.get("category") or "")
        continuity = str(entry.get("continuity_prior") or "").lower()
        if category in {"artery", "vein", "vessel"} and not any(
            token in continuity for token in ("tubular", "branch", "course", "continuous")
        ):
            reasons.append("vascular_continuity_rule_missing")
        if any(token in organ for token in ("lesion", "tumor", "metast")) and (
            "single contiguous" in continuity or "single connected" in continuity
        ):
            reasons.append("multifocal_target_incorrectly_forced_single")
        partial = str(entry.get("partial_fov_behavior") or "").lower()
        if not any(token in partial for token in ("boundary", "truncat", "field of view")):
            reasons.append("partial_fov_rule_missing_boundary_basis")
        penalty = " ".join(map(str, entry.get("penalty_rules") or [])).lower()
        if "reject any anatomical variation" in penalty or "reject all variation" in penalty:
            reasons.append("normal_anatomic_variation_overrejected")
        field_provenance = entry.get("field_provenance") or {}
        required_provenance = {
            "canonical_name", "aliases", "laterality", "parent_structures",
            "expected_body_regions", "ct_location", "ct_appearance", "morphology",
            "adjacent_structures", "continuity_prior", "symmetry_prior",
            "partial_fov_behavior", "common_failure_modes", "penalty_rules",
        }
        if required_provenance - set(field_provenance):
            reasons.append("field_level_provenance_incomplete")
        reasons = list(dict.fromkeys(reasons))
        entry["validation_reasons"] = reasons
        if reasons:
            entry["validation_status"] = "runtime_only"
            entry["formal_selection_eligible"] = False
            entry["capability_level"] = (
                "runtime_supported"
                if entry.get("source") == "labelcritic_seed"
                else "runtime_only_unverified"
            )
            entry["automatic_failure_action"] = "abstain_runtime_only_unverified"


def build(
    target_config: Path,
    seed_path: Path | None = None,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    doc = json.loads(target_config.read_text(encoding="utf-8"))
    organs = [str(o) for o in doc.get("target_organs", [])]
    bank = doc.get("organ_prompt_bank", {}) or {}
    taxonomy_doc = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy = taxonomy_doc.get("organs", {}) or {}
    seed_entries = _load_seed_entries(seed_path)
    entries = {
        organ: build_entry(
            organ,
            bank.get(organ, {}),
            seed_entries,
            taxonomy.get(organ, {}),
        )
        for organ in organs
    }
    _cross_entry_validation(entries, taxonomy)
    required = {
        "canonical_name",
        "canonical_organ",
        "aliases",
        "parent_structures",
        "expected_body_regions",
        "ct_location",
        "ct_appearance",
        "morphology",
        "adjacent_structures",
        "continuity_prior",
        "symmetry_prior",
        "partial_fov_behavior",
        "common_failure_modes",
        "penalty_rules",
        "provenance",
        "source_type",
        "validation_status",
        "capability_level",
    }
    empty_allowed = {"parent_structures"}
    missing_fields = {
        organ: sorted(
            key
            for key in required
            if key not in entry
            or (
                key not in empty_allowed
                and entry.get(key) in (None, "", [])
            )
        )
        for organ, entry in entries.items()
    }
    missing_fields = {k: v for k, v in missing_fields.items() if v}
    source_counts: dict[str, int] = {}
    for entry in entries.values():
        source_counts[str(entry.get("source"))] = source_counts.get(str(entry.get("source")), 0) + 1
    seed_organs = sorted([organ for organ, entry in entries.items() if entry.get("source") == "labelcritic_seed"])
    invalid_capabilities = {
        organ: entry.get("capability_level")
        for organ, entry in entries.items()
        if entry.get("capability_level") not in CAPABILITY_LEVELS
    }
    formal_eligible = sorted(
        organ for organ, entry in entries.items() if entry.get("formal_selection_eligible")
    )
    runtime_only = sorted(set(organs) - set(formal_eligible))
    entry_list = [entries[organ] for organ in organs]
    audit = {
        "status": (
            "success"
            if len(entries) == len(organs) and not missing_fields and not invalid_capabilities
            else "failed"
        ),
        "target_count": len(organs),
        "entry_count": len(entries),
        "missing_organs": sorted(set(organs) - set(entries)),
        "extra_organs": sorted(set(entries) - set(organs)),
        "missing_required_fields": missing_fields,
        "invalid_capability_levels": invalid_capabilities,
        "source_counts": source_counts,
        "labelcritic_seed_organs": seed_organs,
        "formal_selection_eligible_count": len(formal_eligible),
        "runtime_only_unverified_count": len(runtime_only),
        "runtime_only_unverified_organs": runtime_only,
        "manual_review_count": sum(1 for entry in entries.values() if entry.get("requires_manual_review")),
        "cpu_only": True,
        "note": (
            "Schema completeness is not accuracy validation. Project extensions with "
            "generic anatomy remain runtime_only_unverified and are blocked from formal selection."
        ),
    }
    out = {
        "version": "organ_ct_appearance_373.v2",
        "status": audit["status"],
        "schema_version": "labelcritic_organ_ct_appearance.v2",
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
    ap.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    args = ap.parse_args()

    output, audit = build(
        Path(args.target_config).resolve(),
        Path(args.seed).resolve() if args.seed else None,
        Path(args.taxonomy).resolve(),
    )
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
