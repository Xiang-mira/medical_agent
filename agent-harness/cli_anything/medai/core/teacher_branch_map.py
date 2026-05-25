"""Generate teacher_branch_map.yaml from class_checkpoint_map.xlsx.

This maps each organ to:
- Its VISTA3D label_id (for language-prompt inference)
- The best teacher model (for pseudo-label generation during E-step)
- The trainable scope for M-step continual fine-tuning

The map is used in two ways:
  Inference: text prompt → label_id → VISTA3D inference
  M-step:    teacher_model generates pseudo-label → VISTA3D continual fine-tune
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

# ── VISTA3D label_id mapping (from metadata.json channel_def) ────────────────
VISTA3D_LABEL_MAP: dict[str, int] = {
    "background": 0,
    "liver": 1,
    "kidney": 2,
    "spleen": 3,
    "pancreas": 4,
    "kidney_right": 5,
    "aorta": 6,
    "postcava": 7,
    "inferior_vena_cava": 7,
    "adrenal_gland_right": 8,
    "adrenal_gland_left": 9,
    "gall_bladder": 10,
    "gallbladder": 10,
    "esophagus": 11,
    "stomach": 12,
    "duodenum": 13,
    "kidney_left": 14,
    "bladder": 15,
    "prostate": 16,
    "uterus": 16,
    "portal_vein_and_splenic_vein": 17,
    "veins": 17,
    "rectum": 18,
    "intestine": 19,
    "small_bowel": 19,
    "lung": 20,
    "lung_left": 20,
    "lung_right": 20,
    "bone": 21,
    "brain": 22,
    "lung_tumor": 23,
    "pancreatic_pdac": 24,
    "pancreatic_cyst": 24,
    "pancreatic_pnet": 24,
    "hepatic_vessel": 25,
    "liver_hepatic_vein": 25,
    "colon": 62,
    "trachea": 57,
    "heart": 115,
    "spinal_cord": 121,
    "thyroid_gland": 126,
    "airway": 132,
    "airway_tree": 132,
    "vertebrae_l5": 33, "vertebrae_l4": 34, "vertebrae_l3": 35,
    "vertebrae_l2": 36, "vertebrae_l1": 37,
    "vertebrae_t12": 38, "vertebrae_t11": 39, "vertebrae_t10": 40,
    "vertebrae_t9": 41, "vertebrae_t8": 42, "vertebrae_t7": 43,
    "vertebrae_t6": 44, "vertebrae_t5": 45, "vertebrae_t4": 46,
    "vertebrae_t3": 47, "vertebrae_t2": 48, "vertebrae_t1": 49,
    "vertebrae_c7": 50, "vertebrae_c6": 51, "vertebrae_c5": 52,
    "vertebrae_c4": 53, "vertebrae_c3": 54, "vertebrae_c2": 55,
    "vertebrae_c1": 56, "vertebrae_s1": 127,
    "rib_left_1": 63, "rib_left_2": 64, "rib_left_3": 65,
    "rib_left_4": 66, "rib_left_5": 67, "rib_left_6": 68,
    "rib_left_7": 69, "rib_left_8": 70, "rib_left_9": 71,
    "rib_left_10": 72, "rib_left_11": 73, "rib_left_12": 74,
    "rib_right_1": 75, "rib_right_2": 76, "rib_right_3": 77,
    "rib_right_4": 78, "rib_right_5": 79, "rib_right_6": 80,
    "rib_right_7": 81, "rib_right_8": 82, "rib_right_9": 83,
    "rib_right_10": 84, "rib_right_11": 85, "rib_right_12": 86,
    "humerus_left": 87, "humerus_right": 88,
    "scapula_left": 89, "scapula_right": 90,
    "clavicula_left": 91, "clavicula_right": 92,
    "femur_left": 93, "femur_right": 94,
    "hip_left": 95, "hip_right": 96,
    "sacrum": 97,
    "gluteus_maximus_left": 98, "gluteus_maximus_right": 99,
    "gluteus_medius_left": 100, "gluteus_medius_right": 101,
    "gluteus_minimus_left": 102, "gluteus_minimus_right": 103,
    "autochthon_left": 104, "autochthon_right": 105,
    "iliopsoas_left": 106, "iliopsoas_right": 107,
    "atrial_appendage_left": 108,
    "brachiocephalic_trunk": 109,
    "brachiocephalic_vein_left": 110, "brachiocephalic_vein_right": 111,
    "common_carotid_artery_left": 112, "common_carotid_artery_right": 113,
    "costal_cartilages": 114,
    "skull": 120, "sternum": 122,
    "subclavian_artery_left": 123, "subclavian_artery_right": 124,
    "superior_vena_cava": 125,
    "bone_lesion": 128, "kidney_mass": 129,
    "liver_tumor": 130,
}

# Natural language aliases → canonical organ name
_TEXT_ALIASES: dict[str, str] = {
    "segment pancreas": "pancreas",
    "segment liver": "liver",
    "segment spleen": "spleen",
    "segment kidney": "kidney",
    "segment left kidney": "kidney_left",
    "segment right kidney": "kidney_right",
    "segment aorta": "aorta",
    "segment stomach": "stomach",
    "segment colon": "colon",
    "segment duodenum": "duodenum",
    "segment lung": "lung",
    "segment heart": "heart",
    "segment pancreatic tumor": "pancreatic_pdac",
    "segment tumor": "pancreatic_pdac",
    "segment gallbladder": "gall_bladder",
    "segment esophagus": "esophagus",
    "segment bladder": "bladder",
    "segment spinal cord": "spinal_cord",
    "segment trachea": "trachea",
    "segment thyroid": "thyroid_gland",
}

# Teacher model per organ (from class_checkpoint_map.xlsx analysis)
_TEACHER_MAP: dict[str, dict[str, Any]] = {
    "pancreas":            {"teacher": "epai_20250421", "fallback": ["vista3d", "vsmtrans"]},
    "pancreatic_pdac":     {"teacher": "epai_20250421", "fallback": ["vista3d"]},
    "pancreatic_cyst":     {"teacher": "epai_20250421", "fallback": []},
    "pancreatic_pnet":     {"teacher": "epai_20250421", "fallback": []},
    "liver":               {"teacher": "vsmtrans",      "fallback": ["totalsegmentator", "vista3d"]},
    "spleen":              {"teacher": "vsmtrans",      "fallback": ["totalsegmentator"]},
    "kidney_left":         {"teacher": "vsmtrans",      "fallback": ["totalsegmentator"]},
    "kidney_right":        {"teacher": "vsmtrans",      "fallback": ["totalsegmentator"]},
    "aorta":               {"teacher": "vista3d",       "fallback": ["totalsegmentator"]},
    "postcava":            {"teacher": "vsmtrans",      "fallback": ["totalsegmentator"]},
    "stomach":             {"teacher": "vsmtrans",      "fallback": ["totalsegmentator"]},
    "colon":               {"teacher": "vsmtrans",      "fallback": ["cads"]},
    "duodenum":            {"teacher": "epai_20250421", "fallback": ["vsmtrans"]},
    "gall_bladder":        {"teacher": "vsmtrans",      "fallback": ["totalsegmentator"]},
    "esophagus":           {"teacher": "vsmtrans",      "fallback": ["cads"]},
    "adrenal_gland_left":  {"teacher": "epai_20250421", "fallback": ["moose3_0"]},
    "adrenal_gland_right": {"teacher": "epai_20250421", "fallback": ["moose3_0"]},
    "lung":                {"teacher": "vista3d",       "fallback": ["totalsegmentator"]},
    "heart":               {"teacher": "vista3d",       "fallback": ["moose"]},
    "trachea":             {"teacher": "vista3d",       "fallback": ["cads"]},
    "airway_tree":         {"teacher": "atm",           "fallback": ["vista3d"]},
    "spinal_cord":         {"teacher": "vista3d",       "fallback": ["cads"]},
    "thyroid_gland":       {"teacher": "vista3d",       "fallback": ["cads"]},
    "bladder":             {"teacher": "vsmtrans",      "fallback": ["cads"]},
    "prostate":            {"teacher": "vsmtrans",      "fallback": ["cads"]},
    "hepatic_vessel":      {"teacher": "vsnet",         "fallback": ["vsmtrans"]},
    "sacrum":              {"teacher": "vista3d",       "fallback": ["moose3_0"]},
    "skull":               {"teacher": "vista3d",       "fallback": ["moose3_0"]},
    "sternum":             {"teacher": "vista3d",       "fallback": ["cads"]},
}
_DEFAULT_TEACHER: dict[str, Any] = {"teacher": "totalsegmentator", "fallback": ["vista3d"]}


def text_to_label_id(prompt: str) -> int | None:
    """Convert a natural language prompt or organ name to a VISTA3D label_id.

    Accepts natural language ("segment pancreas"), canonical names ("pancreas"),
    and aliases ("postcava", "intestine"). Returns None if not in VISTA3D vocabulary.
    """
    p = prompt.strip().lower()
    # Check full-phrase alias first
    if p in _TEXT_ALIASES:
        p = _TEXT_ALIASES[p]
    # Strip common prefixes
    for prefix in ("segment the ", "segment ", "find the ", "find ", "detect "):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    if p in VISTA3D_LABEL_MAP:
        return VISTA3D_LABEL_MAP[p]
    p_under = p.replace(" ", "_")
    if p_under in VISTA3D_LABEL_MAP:
        return VISTA3D_LABEL_MAP[p_under]
    return None


def texts_to_label_ids(prompts: list[str]) -> dict[str, int]:
    """Convert a list of organ names/prompts to {organ: label_id}.

    Organs not in VISTA3D vocabulary are silently skipped.
    """
    result: dict[str, int] = {}
    for p in prompts:
        lid = text_to_label_id(p)
        if lid is not None:
            canonical = p.strip().lower().replace(" ", "_")
            result[canonical] = lid
    return result


def get_teacher_for_organ(organ: str) -> dict[str, Any]:
    """Return teacher model info for a given organ."""
    return _TEACHER_MAP.get(organ.lower(), _DEFAULT_TEACHER)


def build_teacher_branch_map(output_yaml: str | Path) -> dict[str, Any]:
    """Generate teacher_branch_map.yaml.

    Each entry:
      vista3d_label_id: int        — VISTA3D class ID for inference
      teacher_model: str           — best teacher for pseudo-label generation
      fallback_teachers: list[str] — fallback if primary teacher fails
      update_mode: str             — continual learning update strategy
      trainable_scope: list[str]   — VISTA3D modules to update in M-step
      freeze_backbone: bool        — freeze SwinUNETR encoder during M-step
    """
    out = Path(output_yaml).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    branch_map: dict[str, Any] = {}
    for organ, label_id in sorted(VISTA3D_LABEL_MAP.items()):
        if organ == "background" or label_id == 0:
            continue
        teacher_info = _TEACHER_MAP.get(organ, _DEFAULT_TEACHER)
        branch_map[organ] = {
            "vista3d_label_id": label_id,
            "teacher_model": teacher_info["teacher"],
            "fallback_teachers": teacher_info.get("fallback", []),
            "update_mode": "class_continual",
            "trainable_scope": ["class_embedding", "point_head"],
            "freeze_backbone": True,
        }

    if yaml:
        out.write_text(yaml.safe_dump(branch_map, sort_keys=True, allow_unicode=True), encoding="utf-8")
    else:
        import json
        out.with_suffix(".json").write_text(json.dumps(branch_map, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "output": str(out),
        "num_organs": len(branch_map),
        "sample": dict(list(branch_map.items())[:3]),
    }
