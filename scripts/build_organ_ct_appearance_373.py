#!/usr/bin/env python3
"""Build the 373-organ CT appearance knowledge base for LabelCritic.

This is intentionally CPU-only/offline: it reads the project's formal target
config and writes a JSON prompt knowledge base plus a small audit report.  It
does not call LabelCritic, a VLM server, CUDA, or any teacher model.
"""
from __future__ import annotations

import argparse
import csv
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
DEFAULT_DETAILED_AUDIT_JSON = ROOT / "configs" / "organ_ct_appearance_373_audit_detailed.json"
DEFAULT_DETAILED_AUDIT_CSV = ROOT / "configs" / "organ_ct_appearance_373_audit_detailed.csv"
DEFAULT_SEED = ROOT / "configs" / "organ_ct_appearance_373_seed_from_labelcritic.json"
DEFAULT_TAXONOMY = ROOT / "configs" / "organ_taxonomy.json"

PUBLIC_ANATOMY_REFERENCE_SOURCES = [
    "LabelCritic official prompt/style seed for comparative mask review",
    "TotalSegmentator public CT target taxonomy for common segmentation naming",
    "FMA anatomical hierarchy for parent/adjacent relationships",
    "RadLex/LOINC-RSNA radiology anatomy naming conventions",
]

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


BROAD_CATEGORY_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "blood": {
        "aliases": ["blood", "intravascular blood pool", "visible blood compartment"],
        "expected_location": "within vascular lumens, cardiac chambers, or venous sinuses that are covered by the CT field of view",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "scan-dependent intravascular or intracardiac content; on contrast CT it follows enhancing vascular lumen, while on noncontrast CT it should remain confined to plausible vessel or chamber spaces",
        "morphology": "Distributed lumen-filling compartment constrained by vessel or chamber boundaries rather than a free soft-tissue organ.",
        "shape_and_continuity_prior": "May be multipart across vessels and chambers, but each component must sit inside a plausible blood-containing lumen.",
        "neighbor_relations": ["arteries and veins", "cardiac chambers", "venous sinuses"],
        "anatomical_constraints": [
            "candidate foreground must remain inside plausible blood-containing spaces",
            "do not treat solid organ parenchyma or muscle as blood",
            "only compare candidates when both are intended to represent the same blood-compartment scope",
        ],
        "common_failure_modes": ["solid-organ leakage", "bone or calcification inclusion", "free-space blobs outside vessels", "artery/vein scope mismatch"],
        "rejection_rules": ["reject if the mask is mostly outside vascular or cardiac lumens", "reject if it broadly fills solid organs, muscle, fat, or bone"],
        "partial_fov_behavior": "Vascular components may truncate at scan boundaries; internal components must still remain within plausible lumens.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "body": {
        "aliases": ["body", "whole visible body envelope", "external body contour"],
        "expected_location": "entire visible patient body envelope within the CT field of view, excluding surrounding air and table when possible",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "outer patient contour containing all visible tissues and internal anatomy; boundary follows skin-air interface rather than a single organ",
        "morphology": "Broad connected external envelope whose scope is the visible patient body, not one internal structure.",
        "shape_and_continuity_prior": "Should be a coherent external contour across covered slices; holes or truncation are acceptable only when explained by scan boundaries or air spaces.",
        "neighbor_relations": ["skin surface", "subcutaneous fat", "scanner table and external air"],
        "anatomical_constraints": [
            "candidate semantics must match whole visible body envelope",
            "exclude external air and scanner table when distinguishable",
            "do not compare against single-organ masks as if they were body masks",
        ],
        "common_failure_modes": ["table inclusion", "air/background inclusion", "only trunk or only extremity segmented", "internal organ-only mask"],
        "rejection_rules": ["reject if the candidate is not an external body-envelope mask", "reject if dominated by scanner table or background air"],
        "partial_fov_behavior": "The body envelope may be truncated by any scan boundary; boundary truncation is expected and should not be penalized by itself.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "body_extremities": {
        "aliases": ["body extremities", "visible limb envelope", "appendicular body envelope"],
        "expected_location": "visible upper or lower extremity body envelope when limbs are included in the CT field of view",
        "expected_body_regions": ["extremity"],
        "ct_appearance": "external contour of covered limbs including skin, subcutaneous fat, muscle, vessels, and bone while excluding air/table",
        "morphology": "Long limb-envelope regions around appendicular skeleton and soft tissues; may be unilateral, bilateral, or partially covered.",
        "shape_and_continuity_prior": "Should follow coherent limb contours rather than internal bone-only or muscle-only regions.",
        "neighbor_relations": ["appendicular bones", "limb muscle compartments", "subcutaneous tissue and skin"],
        "anatomical_constraints": [
            "use only when target scope is the visible extremity envelope",
            "do not include trunk as extremities",
            "do not accept isolated femur/humerus/muscle masks as extremity envelope",
        ],
        "common_failure_modes": ["trunk leakage", "bone-only mask", "table/background inclusion", "missing visible limb segment"],
        "rejection_rules": ["reject if the mask primarily marks trunk or a single internal limb structure", "reject if external air/table dominates"],
        "partial_fov_behavior": "Limbs may enter or exit the scan at boundaries; truncation at physical boundaries is acceptable.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "body_trunc": {
        "aliases": ["body trunk", "trunk body envelope", "torso envelope"],
        "expected_location": "visible torso/trunk external body envelope across thorax, abdomen, and pelvis, depending on scan range",
        "expected_body_regions": ["thorax", "abdomen", "pelvis"],
        "ct_appearance": "external contour of the torso containing chest, abdominal, and pelvic soft tissues and skeleton while excluding air/table",
        "morphology": "Broad torso-envelope region bounded by skin surface and scan coverage.",
        "shape_and_continuity_prior": "Should form a coherent trunk contour; scan-boundary truncation is expected.",
        "neighbor_relations": ["skin surface", "thoracoabdominal wall", "spine, ribs, and pelvis"],
        "anatomical_constraints": [
            "candidate semantics must be torso envelope rather than whole-body, limb-only, or internal-organ scope",
            "exclude scanner table/background when distinguishable",
            "do not accept organ-only masks",
        ],
        "common_failure_modes": ["limb-only mask", "whole background inclusion", "organ-only mask", "table inclusion"],
        "rejection_rules": ["reject if not a trunk-envelope candidate", "reject if dominated by external table or air"],
        "partial_fov_behavior": "Torso envelope may truncate superiorly, inferiorly, or laterally at scan boundaries.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "bones": {
        "aliases": ["bones", "all visible bones", "osseous structures"],
        "expected_location": "all visible skeletal structures within the CT field of view",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "high-attenuation cortical bone with internal trabecular marrow spaces; includes visible axial or appendicular osseous anatomy according to target scope",
        "morphology": "Multipart skeletal mask following individual bone boundaries rather than soft-tissue compartments.",
        "shape_and_continuity_prior": "Multiple separated bones are expected, but components should match osseous density and anatomy.",
        "neighbor_relations": ["cortical bone boundaries", "joints", "adjacent muscles and marrow spaces"],
        "anatomical_constraints": [
            "candidate must represent visible osseous anatomy, not soft tissue",
            "preserve bone boundaries and avoid broad muscle or organ leakage",
            "only compare candidates with the same all-bones target scope",
        ],
        "common_failure_modes": ["calcified vessel inclusion as bone", "muscle or organ leakage", "missing major visible bones", "table/high-density artifact inclusion"],
        "rejection_rules": ["reject if foreground is dominated by non-osseous soft tissue", "reject if scanner table or external artifacts dominate"],
        "partial_fov_behavior": "Bones may be truncated at scan boundaries; visible internal bone segments should not disappear without explanation.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "compact_bone": {
        "aliases": ["compact bone", "cortical bone", "dense cortical osseous tissue"],
        "expected_location": "dense cortical shell of visible bones throughout the scanned region",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "very high-attenuation cortical bone forming thin dense outer shells and cortical plates",
        "morphology": "Thin dense osseous cortex following bone surfaces; should not fill all marrow as if it were whole bone.",
        "shape_and_continuity_prior": "Cortical components may be multipart but should trace coherent bone cortices.",
        "neighbor_relations": ["spongy bone", "bone marrow", "periosteal soft tissues"],
        "anatomical_constraints": [
            "prefer dense cortical boundaries over trabecular medullary spaces",
            "avoid calcified vessels, contrast, metal, and scanner table",
            "do not leak into muscle or solid organs",
        ],
        "common_failure_modes": ["spongy bone overfill", "metal/table artifact inclusion", "calcified vessel confusion", "soft-tissue leakage"],
        "rejection_rules": ["reject if not confined to dense osseous cortex", "reject if non-anatomic high-density artifacts dominate"],
        "partial_fov_behavior": "Cortical structures may truncate at scan boundaries; exposed bone edges at boundaries are acceptable.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "fat": {
        "aliases": ["fat", "adipose tissue", "visible fat compartment"],
        "expected_location": "low-attenuation adipose tissue compartments in subcutaneous, visceral, mediastinal, pelvic, or marrow-adjacent spaces depending on scan range",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "low-attenuation fat-density tissue that is darker than muscle and bounded by fascia, skin, organs, or vessels",
        "morphology": "Compartment-like adipose regions; may be distributed but should follow plausible fat spaces.",
        "shape_and_continuity_prior": "Can be multipart across compartments, but should not invade solid organs, vessels, bone cortex, or air.",
        "neighbor_relations": ["skin and subcutaneous tissues", "fascia and muscles", "visceral organs and mesentery"],
        "anatomical_constraints": [
            "distinguish fat from air, bowel gas, and lung",
            "do not include solid organs or muscle as adipose tissue",
            "candidate scope must match broad fat target rather than a named organ",
        ],
        "common_failure_modes": ["air/lung inclusion", "solid-organ leakage", "muscle inclusion", "subcutaneous-only versus visceral-scope mismatch"],
        "rejection_rules": ["reject if dominated by air, lung, bowel gas, muscle, or solid organ", "reject if candidate scope mismatches the target fat compartment"],
        "partial_fov_behavior": "Fat compartments may truncate at scan boundaries; internal compartment boundaries should remain plausible.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "gland_structure": {
        "aliases": ["gland structure", "visible glandular structures", "glandular tissue category"],
        "expected_location": "named or visible glandular structures such as salivary, endocrine, breast, pancreatic, prostate, or other glandular targets within the scan range",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis"],
        "ct_appearance": "soft-tissue glandular parenchyma whose exact attenuation and shape depend on the named gland and scan region",
        "morphology": "Category-level glandular tissue; not a single fixed organ shape.",
        "shape_and_continuity_prior": "Should follow the glandular structure intended by the candidate context and avoid unrelated soft tissue.",
        "neighbor_relations": ["regional ducts or vessels", "adjacent muscles", "nearby named organs"],
        "anatomical_constraints": [
            "only compare when both candidates target the same glandular scope",
            "do not treat any soft-tissue blob as glandular tissue",
            "use regional anatomy and aliases to resolve which gland is meant",
        ],
        "common_failure_modes": ["generic soft-tissue blob", "wrong gland or wrong body region", "muscle or lymph-node inclusion", "scope mismatch"],
        "rejection_rules": ["reject if the mask does not correspond to a plausible glandular target", "reject if candidates are not semantically comparable"],
        "partial_fov_behavior": "A glandular structure may be partially covered only at a scan boundary; internal missing parts remain suspicious.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "muscle": {
        "aliases": ["muscle", "skeletal muscle", "visible muscle compartments"],
        "expected_location": "skeletal muscle compartments visible in the scan, bounded by fascia and adjacent bones or organs",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "soft-tissue attenuation muscle, denser than fat and organized into fascial compartments",
        "morphology": "Compartmental muscle regions; can be multipart but should follow muscle belly and fascial planes.",
        "shape_and_continuity_prior": "Should respect fascial boundaries and avoid bone, fat, organs, and vessels.",
        "neighbor_relations": ["fascial planes", "adjacent bones", "subcutaneous and intermuscular fat"],
        "anatomical_constraints": [
            "candidate scope must match broad muscle tissue rather than one named muscle unless context says otherwise",
            "do not include solid organs, bone, or fat compartments",
            "preserve expected muscle compartment boundaries",
        ],
        "common_failure_modes": ["fat inclusion", "bone or organ leakage", "single named muscle confused with all muscle", "background/table inclusion"],
        "rejection_rules": ["reject if not muscle-density tissue in plausible compartments", "reject if scope mismatches broad muscle target"],
        "partial_fov_behavior": "Muscle compartments may truncate at scan boundaries; visible internal compartment continuity should be plausible.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "muscle_of_head": {
        "aliases": ["muscle of head", "head and facial muscles", "craniofacial muscle category"],
        "expected_location": "visible muscles of the head and face, including masticator, facial-expression, extraocular, tongue-associated, or scalp-related muscle groups when covered",
        "expected_body_regions": ["head_neck"],
        "ct_appearance": "soft-tissue attenuation craniofacial muscle groups adjacent to skull, mandible, orbit, tongue, or facial soft tissues",
        "morphology": "Multiple small head/face muscle compartments following regional fascial and bony boundaries.",
        "shape_and_continuity_prior": "May be multipart and bilateral; components should remain in craniofacial muscular locations.",
        "neighbor_relations": ["skull and facial bones", "orbits and mandible", "salivary glands and upper aerodigestive tract"],
        "anatomical_constraints": [
            "restrict to head and facial muscle compartments",
            "do not include salivary glands, brain, bone, or broad skin envelope",
            "allow bilateral/multipart muscles when visible",
        ],
        "common_failure_modes": ["salivary gland confusion", "bone or brain leakage", "neck muscle overextension", "skin/fat envelope inclusion"],
        "rejection_rules": ["reject if mostly outside craniofacial muscle regions", "reject if dominated by non-muscle structures"],
        "partial_fov_behavior": "Head muscles may truncate at superior/inferior scan boundaries; internal regional continuity should remain plausible.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "skin": {
        "aliases": ["skin", "cutaneous envelope", "body surface skin"],
        "expected_location": "thin superficial soft-tissue envelope at the external body surface wherever covered by CT",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "thin soft-tissue layer at the skin-air interface, superficial to subcutaneous fat",
        "morphology": "Very thin continuous or near-continuous surface shell, not a thick body mask.",
        "shape_and_continuity_prior": "Should trace the external body surface and remain much thinner than subcutaneous fat or whole-body envelope.",
        "neighbor_relations": ["external air", "subcutaneous adipose tissue", "body envelope"],
        "anatomical_constraints": [
            "foreground should be superficial at the skin surface",
            "do not fill subcutaneous fat, muscles, organs, or internal cavities",
            "exclude external air and table",
        ],
        "common_failure_modes": ["whole-body envelope mistaken for skin", "subcutaneous fat overfill", "external air/table inclusion", "internal organ leakage"],
        "rejection_rules": ["reject if the mask is thick body fill rather than a thin surface layer", "reject if internal organs or background dominate"],
        "partial_fov_behavior": "Skin surface may truncate at scan boundaries; surface continuity may be interrupted by FOV edges or contact regions.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "spongy_bone": {
        "aliases": ["spongy bone", "trabecular bone", "cancellous bone"],
        "expected_location": "trabecular/cancellous portions within visible bones, especially vertebral bodies, pelvis, ribs, sternum, and long-bone metaphyses when covered",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "intermediate-to-high attenuation trabecular marrow-containing bone inside cortical shells",
        "morphology": "Internal osseous trabecular compartments bounded by compact cortical bone.",
        "shape_and_continuity_prior": "Should remain inside bones and not extend into surrounding soft tissues or external artifacts.",
        "neighbor_relations": ["compact bone cortex", "bone marrow", "adjacent joints and soft tissues"],
        "anatomical_constraints": [
            "candidate must be internal cancellous/trabecular bone rather than cortical shell alone",
            "avoid soft tissue, calcified vessels, table, and metal artifacts",
            "scope must match visible spongy bone category",
        ],
        "common_failure_modes": ["compact bone-only mask", "soft-tissue leakage", "calcification/artifact inclusion", "missing major trabecular compartments"],
        "rejection_rules": ["reject if the mask is outside osseous boundaries", "reject if dominated by cortical-only or artifact-only signal"],
        "partial_fov_behavior": "Trabecular bone may truncate at scan boundaries only where the containing bone is physically cut off.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "subcutaneous_adipose_tissue": {
        "aliases": ["subcutaneous adipose tissue", "subcutaneous fat", "SAT"],
        "expected_location": "fat-density layer between skin surface and deep fascia/body wall throughout the visible body surface",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "low-attenuation fat-density tissue superficial to muscles and deep to skin",
        "morphology": "Sheet-like peripheral adipose compartment following the body contour.",
        "shape_and_continuity_prior": "Should wrap along the external body contour but stay outside deep muscle/visceral compartments.",
        "neighbor_relations": ["skin", "deep fascia and muscles", "visceral fat and body wall"],
        "anatomical_constraints": [
            "foreground should lie between skin and deep fascia",
            "reject visceral/mesenteric fat as subcutaneous when scope is SAT",
            "do not include muscle, bone, organs, lung, air, or table",
        ],
        "common_failure_modes": ["visceral fat inclusion", "air/lung inclusion", "muscle or organ leakage", "skin/body-envelope overfill"],
        "rejection_rules": ["reject if not in the peripheral subcutaneous fat layer", "reject if visceral fat or non-fat tissues dominate"],
        "partial_fov_behavior": "The subcutaneous layer may truncate at scan boundaries; local gaps should follow anatomy or contact regions.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "veins": {
        "aliases": ["veins", "venous structures", "visible venous system"],
        "expected_location": "visible venous tubular structures in head/neck, thorax, abdomen, pelvis, or extremities depending on scan coverage",
        "expected_body_regions": ["head_neck", "thorax", "abdomen", "pelvis", "extremity"],
        "ct_appearance": "tubular venous lumens following expected drainage pathways; enhancement depends on contrast timing",
        "morphology": "Branching or tubular venous structures, potentially multipart but anatomically connected by venous drainage patterns.",
        "shape_and_continuity_prior": "Should follow plausible venous courses and remain distinct from arteries, ducts, bowel, bone, and organs.",
        "neighbor_relations": ["companion arteries", "regional organs", "muscles and fascial planes"],
        "anatomical_constraints": [
            "candidate must represent venous structures with plausible venous course",
            "do not include arteries unless target scope explicitly allows mixed vessels",
            "do not accept broad organ or muscle leakage",
        ],
        "common_failure_modes": ["artery confusion", "organ leakage", "fragmented nonvascular blobs", "bowel/duct confusion"],
        "rejection_rules": ["reject if not tubular/branching in a plausible venous location", "reject if dominated by nonvascular tissue"],
        "partial_fov_behavior": "Veins may truncate at scan boundaries or contrast-limited segments; random isolated nonvascular blobs remain suspicious.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
    "venous_sinuses": {
        "aliases": ["venous sinuses", "dural venous sinuses", "cranial venous sinuses"],
        "expected_location": "intracranial dural venous sinus spaces along the skull/dura, such as sagittal, transverse, sigmoid, cavernous, or related sinuses when covered",
        "expected_body_regions": ["head_neck"],
        "ct_appearance": "venous sinus channels within dural reflections or skull base region; enhancement depends on contrast phase",
        "morphology": "Tubular or channel-like intracranial venous spaces following dural sinus anatomy.",
        "shape_and_continuity_prior": "Should follow dural venous sinus courses, not brain parenchyma, skull cortex, or scalp vessels.",
        "neighbor_relations": ["skull and dura", "brain parenchyma", "internal jugular/sigmoid venous drainage"],
        "anatomical_constraints": [
            "restrict to intracranial/skull-base venous sinus anatomy",
            "do not include arteries, bone, brain tissue, scalp, or neck veins outside the intended sinus course",
            "compare only when the head/skull base is in field of view",
        ],
        "common_failure_modes": ["arterial vessel confusion", "skull/bone inclusion", "brain parenchyma leakage", "neck vein scope mismatch"],
        "rejection_rules": ["reject if the mask is mostly outside dural venous sinus anatomy", "reject if bone or brain parenchyma dominates"],
        "partial_fov_behavior": "Venous sinuses may be absent outside head CT coverage; visible channels may truncate at scan boundaries.",
        "formal_selection_eligible": False,
        "automatic_failure_action": "withhold_or_audit_only",
    },
}


KEY_ORGAN_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "adrenal_gland_left": {
        "aliases": ["left adrenal gland", "left suprarenal gland"],
        "expected_location": "left upper retroperitoneum, superior and medial to the left kidney and near the diaphragmatic crus",
        "expected_body_regions": ["abdomen"],
        "ct_appearance": "small soft-tissue endocrine gland, often Y-, V-, or triangular-shaped, lower attenuation than vessels and separate from kidney, spleen, pancreas, and bowel",
        "morphology": "Small thin-limbed gland with a coherent adrenal contour; not a round mass, bowel loop, vessel, or broad retroperitoneal blob.",
        "shape_and_continuity_prior": "Should be a compact left-sided adrenal-shaped structure with no scattered components.",
        "neighbor_relations": ["left kidney upper pole", "aorta", "left diaphragmatic crus", "pancreatic tail", "spleen"],
        "anatomical_constraints": ["must be left-sided", "should lie above/medial to the left kidney", "should not include kidney, spleen, pancreas, bowel, or vessels"],
        "common_failure_modes": ["kidney upper-pole leakage", "pancreatic tail or splenic vessel confusion", "bowel loop inclusion", "left-right swap", "overly large retroperitoneal blob"],
        "rejection_rules": ["reject if mostly right-sided", "reject if larger than a plausible adrenal gland", "reject if the mask follows kidney, pancreas, spleen, bowel, or vessel rather than adrenal contour"],
        "formal_selection_eligible": True,
    },
    "adrenal_gland_right": {
        "aliases": ["right adrenal gland", "right suprarenal gland"],
        "expected_location": "right upper retroperitoneum, superior and medial to the right kidney, posterior to the liver and near the inferior vena cava",
        "expected_body_regions": ["abdomen"],
        "ct_appearance": "small soft-tissue endocrine gland, often linear, V-, or triangular-shaped, separate from liver, right kidney, inferior vena cava, and bowel",
        "morphology": "Small coherent right adrenal structure with thin limbs; not a vessel, kidney pole, liver edge, or bowel loop.",
        "shape_and_continuity_prior": "Should be a compact right-sided adrenal-shaped structure with no scattered components.",
        "neighbor_relations": ["right kidney upper pole", "liver", "inferior vena cava", "right diaphragmatic crus"],
        "anatomical_constraints": ["must be right-sided", "should lie above/medial to the right kidney", "should not include liver, kidney, vena cava, bowel, or pancreas"],
        "common_failure_modes": ["liver edge leakage", "right kidney upper-pole leakage", "inferior vena cava confusion", "bowel inclusion", "left-right swap"],
        "rejection_rules": ["reject if mostly left-sided", "reject if the mask follows liver/kidney/IVC/bowel rather than adrenal contour", "reject if volume is implausibly large for adrenal gland"],
        "formal_selection_eligible": True,
    },
    "duodenum": {
        "aliases": ["duodenum", "duodenal loop", "C-loop"],
        "expected_location": "upper abdomen, C-shaped course around the pancreatic head from gastric pylorus toward jejunum",
        "expected_body_regions": ["abdomen"],
        "ct_appearance": "hollow bowel segment with variable gas, fluid, or contrast; wall/lumen should follow the expected duodenal C-loop adjacent to the pancreatic head",
        "morphology": "C-shaped or segmentally tubular luminal structure, not random bowel blobs or a solid-organ mask.",
        "shape_and_continuity_prior": "Should follow a plausible continuous or segmental duodenal course; gaps are acceptable only when explained by luminal collapse or scan boundary.",
        "neighbor_relations": ["pancreatic head", "stomach/pylorus", "common bile duct", "right kidney", "inferior vena cava"],
        "anatomical_constraints": ["should remain in upper abdomen", "should not become colon, stomach body, small bowel loops far from pancreatic head, pancreas, or vessel"],
        "common_failure_modes": ["small-bowel loop confusion", "stomach inclusion", "pancreatic head leakage", "random disconnected bowel islands", "oversegmentation into colon"],
        "rejection_rules": ["reject if not near the pancreatic head/upper abdomen", "reject if dominated by distant bowel loops", "penalize scattered disconnected components away from duodenal course"],
        "formal_selection_eligible": True,
    },
    "small_bowel": {
        "aliases": ["small bowel", "small intestine", "jejunum and ileum"],
        "expected_location": "central abdomen and pelvis when covered, internal to the colonic frame and mesentery",
        "expected_body_regions": ["abdomen", "pelvis"],
        "ct_appearance": "multiple thin-walled bowel loops with variable gas, fluid, or contrast; folds may be visible and lumen caliber is smaller than colon",
        "morphology": "Multipart loop-like luminal structure; broad abdominal-wall, colon-only, stomach-only, or solid-organ masks are not acceptable.",
        "shape_and_continuity_prior": "May be discontinuous across collapsed loops, but components must remain plausible small-bowel loops within the mesenteric abdomen/pelvis.",
        "neighbor_relations": ["mesentery", "colon", "stomach", "abdominal wall", "pelvic bowel loops"],
        "anatomical_constraints": ["distinguish from colon by central loop pattern and smaller caliber", "exclude solid organs, muscle, vessels, and abdominal wall", "do not require one single connected component"],
        "common_failure_modes": ["colon confusion", "abdominal-wall leakage", "solid-organ inclusion", "random luminal islands outside bowel", "stomach inclusion"],
        "rejection_rules": ["reject if the candidate primarily follows colon or stomach", "reject if non-bowel tissue dominates", "penalize large disconnected blobs not shaped like bowel loops"],
        "formal_selection_eligible": True,
    },
    "bladder": {
        "aliases": ["urinary bladder", "bladder"],
        "expected_location": "midline anterior pelvis, inferior to bowel loops and superior/posterior to pubic symphysis when pelvis is covered",
        "expected_body_regions": ["pelvis"],
        "ct_appearance": "fluid-density hollow pelvic organ with smooth wall; size depends on filling",
        "morphology": "Single smooth round, oval, or collapsed pelvic sac-like structure.",
        "shape_and_continuity_prior": "Should be one coherent pelvic lumen/wall region when visible, not scattered pelvic blobs.",
        "neighbor_relations": ["pubic symphysis", "rectum", "uterus/prostate", "pelvic bowel loops"],
        "anatomical_constraints": ["must be pelvic/midline when present", "should not be selected in abdomen-only scans without pelvic coverage", "exclude bowel loops and uterus/prostate"],
        "common_failure_modes": ["bowel loop confusion", "uterus/prostate inclusion", "false positive in abdomen-only scan", "fragmented pelvic mask"],
        "rejection_rules": ["reject if pelvis is not in FOV", "reject if the mask is not a plausible pelvic bladder structure", "reject if bowel or reproductive organs dominate"],
        "formal_selection_eligible": True,
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
            seed = seed_entries[key]
            if (
                organ in LABELCRITIC_SEED_DESCRIPTIONS
                and str(seed.get("expected_location") or seed.get("expected_region") or "").strip().lower()
                in {"expected anatomic field of view on ct", "as described in original labelcritic prompt"}
            ):
                return LABELCRITIC_SEED_DESCRIPTIONS[organ]
            return seed
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
    project_desc = BROAD_CATEGORY_DESCRIPTIONS.get(organ) or KEY_ORGAN_DESCRIPTIONS.get(organ)
    project_desc_kind = (
        "project_broad_category_prompt"
        if organ in BROAD_CATEGORY_DESCRIPTIONS
        else "project_key_organ_prompt"
        if organ in KEY_ORGAN_DESCRIPTIONS
        else None
    )
    prior = seed or project_desc or {}
    source = (
        "labelcritic_seed"
        if seed
        else str(project_desc_kind)
        if project_desc
        else "generated_from_project_config"
    )
    aliases = _clean_list(prior.get("aliases") or bank_entry.get("aliases"), [display_name(organ)])
    expected_location = str(prior.get("expected_location") or prior.get("expected_region") or bank_entry.get("region") or "expected anatomic field of view on CT")
    location_inferred = False
    if not seed and not project_desc and expected_location.strip().lower() == "expected anatomic field of view on ct":
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
    category = str(prior.get("category") or _category_for(organ))
    landmark_source = prior.get("neighbor_relations") or bank_entry.get("landmarks")
    landmarks = _clean_list(
        landmark_source,
        _region_specific_neighbors(expected_location, category)
        or ["nearby organs", "vessels", "bones", "soft-tissue planes"],
    )
    ct_appearance = str(prior.get("ct_appearance") or bank_entry.get("ct_appearance") or "soft-tissue structure with CT boundaries guided by adjacent anatomy")
    if not seed and not project_desc:
        ct_appearance = (
            f"{display_name(organ)} should be identified as the named {_category_for(organ).replace('_', ' ')} "
            f"within {expected_location}; use its relationship to {', '.join(landmarks[:3])} to distinguish it "
            f"from adjacent tissues. Base appearance prior: {ct_appearance}"
        )
    shape_prior = str(prior.get("shape_and_continuity_prior") or _infer_shape_prior(organ, ct_appearance))
    constraints = _clean_list(prior.get("anatomical_constraints"), [
        f"the candidate must specifically represent {display_name(organ)}, not merely a structure of the same broad type",
        f"should be located in or near {expected_location}",
        "should respect adjacent anatomic boundaries and not leak into unrelated structures",
    ])
    failures = _clean_list(prior.get("common_failure_modes"), _common_failure_modes(organ))
    rejection = _clean_list(prior.get("rejection_rules"), _rejection_rules(organ))
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
    laterality = prior.get("laterality", _laterality_for(organ))
    parent_structures = list((taxonomy_entry or {}).get("parent_ids") or [])
    generic_location = expected_location.strip().lower() in {
        "expected anatomic field of view on ct",
        "as described in original labelcritic prompt",
    }
    capability = (
        "runtime_supported"
        if seed
        else "class_agnostic_extrapolation"
        if project_desc
        else "runtime_only_unverified"
        if generic_location
        else "class_agnostic_extrapolation"
    )
    validation_reasons = []
    if generic_location and not seed and not project_desc:
        validation_reasons.append("generic_expected_location_requires_better_anatomic_source")
    if (not seed) and (not project_desc) and (not landmarks or set(map(str.lower, landmarks)) <= {
        "nearby organs", "vessels", "bones", "soft-tissue planes"
    }):
        validation_reasons.append("generic_adjacent_structures")
    validation_status = str(project_desc.get("validation_status") or ("eligible" if not validation_reasons else "runtime_only")) if project_desc else ("eligible" if not validation_reasons else "runtime_only")
    formal_selection_eligible = (
        bool(project_desc.get("formal_selection_eligible"))
        if project_desc and "formal_selection_eligible" in project_desc
        else validation_status == "eligible"
    )
    automatic_failure_action = (
        str(project_desc.get("automatic_failure_action"))
        if project_desc and project_desc.get("automatic_failure_action")
        else (
            "allow_official_benchmark_path"
            if seed else (
                "allow_class_agnostic_pairwise"
                if validation_status == "eligible"
                else "abstain_runtime_only_unverified"
            )
        )
    )
    field_source = (
        "official_labelcritic_seed"
        if seed
        else str(project_desc_kind)
        if project_desc
        else None
    )
    return {
        "canonical_name": organ,
        "canonical_organ": organ,
        "canonical_id": organ,
        "display_name": bank_entry.get("display_name") or display_name(organ),
        "aliases": aliases,
        "category": category,
        "laterality": laterality,
        "parent_structures": parent_structures,
        "expected_body_regions": _clean_list(project_desc.get("expected_body_regions") if project_desc else None, _body_regions(expected_location)),
        "expected_region": expected_location,
        "expected_location": expected_location,
        "ct_location": expected_location,
        "landmarks": landmarks,
        "ct_appearance": ct_appearance,
        "morphology": str(prior.get("morphology") or _morphology_for(organ, category, ct_appearance)),
        "shape_and_continuity_prior": shape_prior,
        "continuity_prior": shape_prior,
        "neighbor_relations": landmarks,
        "adjacent_structures": landmarks,
        "symmetry_prior": str(prior.get("symmetry_prior") or _symmetry_prior(organ, laterality)),
        "partial_fov_behavior": str(prior.get("partial_fov_behavior") or _partial_fov_behavior(category)),
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
                else str(project_desc_kind)
                if project_desc
                else "project_373_prompt_bank"
            ),
            "taxonomy_source": str(DEFAULT_TAXONOMY),
            "field_policy": (
                "official seed text preserved; compatibility fields normalized"
                if seed
                else "project-authored broad-category prompt; audit-only unless explicitly promoted"
                if project_desc_kind == "project_broad_category_prompt"
                else "project-authored high-priority key-organ prompt"
                if project_desc
                else "structured project extension; not official LabelCritic or VoxTell content"
            ),
            "reference_sources": (
                ["LabelCritic official prompt/style seed for comparative mask review"]
                if seed
                else PUBLIC_ANATOMY_REFERENCE_SOURCES
            ),
        },
        "field_provenance": {
            "canonical_name": "student_3d_prompt_target_organs.target_organs",
            "aliases": field_source or "student_3d_prompt_target_organs.organ_prompt_bank",
            "laterality": field_source if field_source and "laterality" in prior else "deterministic_name_rule",
            "parent_structures": "organ_taxonomy.json.parent_ids",
            "expected_body_regions": "derived_from_expected_location",
            "ct_location": (
                field_source if field_source
                else "project_organ_name_region_rule" if location_inferred
                else "student_3d_prompt_target_organs.organ_prompt_bank.region"
            ),
            "ct_appearance": field_source or "project_category_rule_plus_prompt_bank",
            "morphology": field_source or "project_category_rule",
            "adjacent_structures": (
                field_source if field_source
                else "student_3d_prompt_target_organs.organ_prompt_bank.landmarks"
                if landmark_source else "project_region_category_rule"
            ),
            "continuity_prior": field_source or "project_category_rule",
            "symmetry_prior": field_source if field_source and "symmetry_prior" in prior else "deterministic_laterality_rule",
            "partial_fov_behavior": field_source or "project_category_rule",
            "common_failure_modes": field_source or "project_category_rule",
            "penalty_rules": field_source or "project_laterality_and_failure_rule",
        },
        "capability_level": capability,
        "validation_status": validation_status,
        "validation_reasons": validation_reasons,
        "formal_selection_eligible": formal_selection_eligible,
        "requires_manual_review": bool(project_desc.get("requires_manual_review", True)) if project_desc else False,
        "automatic_failure_action": automatic_failure_action,
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


def _detailed_prompt_audit(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    generic_locations = {
        "expected anatomic field of view on ct",
        "as described in original labelcritic prompt",
    }
    generic_neighbors = {"nearby organs", "vessels", "bones", "soft-tissue planes", "solid organs"}
    broad_organs = set(BROAD_CATEGORY_DESCRIPTIONS)
    key_organs = {
        "liver", "spleen", "pancreas", "kidney_left", "kidney_right", "aorta",
        "adrenal_gland_left", "adrenal_gland_right", "stomach", "duodenum",
        "colon", "small_bowel", "bladder",
    }
    rows: list[dict[str, Any]] = []
    for organ, entry in sorted(entries.items()):
        issues: list[str] = []
        regions = set(map(str, entry.get("expected_body_regions") or []))
        location = str(entry.get("ct_location") or "").strip()
        location_low = location.lower()
        adjacent = [str(x) for x in entry.get("adjacent_structures") or []]
        adjacent_low = {x.lower() for x in adjacent}
        formal = bool(entry.get("formal_selection_eligible"))
        text_blob = " ".join(
            str(entry.get(field) or "")
            for field in ("ct_location", "ct_appearance", "morphology", "continuity_prior")
        )
        if formal and location_low in generic_locations:
            issues.append("generic_location")
        if formal and (not adjacent or adjacent_low <= generic_neighbors):
            issues.append("generic_adjacent_structures")
        if organ in broad_organs and formal:
            issues.append("broad_category_treated_as_formal_organ")
        if organ in key_organs and not formal:
            issues.append("key_organ_not_formal_eligible")
        if organ in {"duodenum", "small_bowel", "colon", "pancreas", "stomach", "spleen", "adrenal_gland_left", "adrenal_gland_right", "kidney_left", "kidney_right", "aorta"}:
            if "abdomen" not in regions and "pelvis" not in regions:
                issues.append("wrong_body_region_for_abdominal_key_organ")
            if organ == "duodenum" and "head_neck" in regions:
                issues.append("duodenum_wrong_head_neck_region")
        if (organ.endswith("_left") or organ.endswith("_right")) and not entry.get("laterality"):
            issues.append("missing_laterality")
        if not entry.get("common_failure_modes"):
            issues.append("missing_failure_modes")
        if not entry.get("penalty_rules") and not entry.get("rejection_rules"):
            issues.append("missing_rejection_rules")
        if formal and len(text_blob) < 180:
            issues.append("prompt_too_short_or_generic")
        if len(text_blob) > 5000:
            issues.append("prompt_too_long_or_ambiguous")
        if not (entry.get("provenance") or {}).get("description_source"):
            issues.append("missing_source_provenance")
        rows.append({
            "organ": organ,
            "formal_selection_eligible": formal,
            "source": entry.get("source"),
            "source_type": entry.get("source_type"),
            "validation_status": entry.get("validation_status"),
            "capability_level": entry.get("capability_level"),
            "automatic_failure_action": entry.get("automatic_failure_action"),
            "expected_body_regions": sorted(regions),
            "ct_location": location,
            "issue_count": len(issues),
            "issues": issues,
        })
    return {
        "stage": "organ_ct_appearance_373_detailed_prompt_audit",
        "status": "success" if not [r for r in rows if r["issue_count"] > 0 and r["formal_selection_eligible"]] else "review",
        "detailed_audit_status": "success" if not [r for r in rows if r["issue_count"] > 0 and r["formal_selection_eligible"]] else "review",
        "target_count": len(rows),
        "entry_count": len(rows),
        "issue_count": sum(int(r["issue_count"]) for r in rows),
        "formal_issue_count": sum(int(r["issue_count"]) for r in rows if r["formal_selection_eligible"]),
        "detailed_formal_issue_count": sum(int(r["issue_count"]) for r in rows if r["formal_selection_eligible"]),
        "audit_only_count": sum(1 for r in rows if r["automatic_failure_action"] == "withhold_or_audit_only"),
        "rows": rows,
    }


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
    runtime_only = sorted(
        organ for organ, entry in entries.items()
        if entry.get("validation_status") == "runtime_only"
        or entry.get("capability_level") == "runtime_only_unverified"
    )
    audit_only = sorted(
        organ for organ, entry in entries.items()
        if entry.get("automatic_failure_action") in {
            "withhold_or_audit_only",
            "audit_only_withhold_from_formal_selection",
        }
    )
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
        "audit_only_count": len(audit_only),
        "audit_only_organs": audit_only,
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
    ap.add_argument("--detailed-audit-json", default=str(DEFAULT_DETAILED_AUDIT_JSON))
    ap.add_argument("--detailed-audit-csv", default=str(DEFAULT_DETAILED_AUDIT_CSV))
    args = ap.parse_args()

    output, audit = build(
        Path(args.target_config).resolve(),
        Path(args.seed).resolve() if args.seed else None,
        Path(args.taxonomy).resolve(),
    )
    out_path = Path(args.output)
    audit_path = Path(args.audit_output)
    detailed_json_path = Path(args.detailed_audit_json)
    detailed_csv_path = Path(args.detailed_audit_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_json_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    detailed = _detailed_prompt_audit(output.get("organ_ct_appearance", {}) or {})
    detailed_json_path.write_text(json.dumps(detailed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with detailed_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "organ", "formal_selection_eligible", "source", "source_type",
            "validation_status", "capability_level", "automatic_failure_action",
            "expected_body_regions", "ct_location", "issue_count", "issues",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detailed["rows"]:
            writer.writerow({
                **row,
                "expected_body_regions": json.dumps(row.get("expected_body_regions", []), ensure_ascii=False),
                "issues": json.dumps(row.get("issues", []), ensure_ascii=False),
            })
    print(json.dumps({
        "status": audit["status"],
        "target_count": audit["target_count"],
        "entry_count": audit["entry_count"],
        "output": str(out_path),
        "audit": str(audit_path),
        "detailed_audit_json": str(detailed_json_path),
        "detailed_audit_csv": str(detailed_csv_path),
        "detailed_audit_status": detailed["status"],
        "detailed_formal_issue_count": detailed["formal_issue_count"],
    }, indent=2))
    if audit["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
