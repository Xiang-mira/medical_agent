from __future__ import annotations

# Full ROI subset for TotalSegmentator (non-fast / full model).
SHAPEKIT_ABDOMEN_ROI = [
    "liver", "pancreas", "spleen", "stomach", "aorta", "inferior_vena_cava",
    "kidney_left", "kidney_right", "gallbladder", "colon", "duodenum", "esophagus",
    "adrenal_gland_left", "adrenal_gland_right", "bladder", "prostate",
    "femur_left", "femur_right", "small_bowel",
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
]

# Safe ROI subset for TotalSegmentator --fast mode (3mm low-res model).
# Pelvic/bowel organs (bladder, prostate, colon, small_bowel, femur, duodenum)
# are NOT in the fast model's class map and cause KeyError if requested.
SHAPEKIT_ABDOMEN_FAST_ROI = [
    "liver", "pancreas", "spleen", "stomach", "aorta", "inferior_vena_cava",
    "kidney_left", "kidney_right", "gallbladder", "esophagus",
    "adrenal_gland_left", "adrenal_gland_right",
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
]

TOTALSEG_TO_SHAPEKIT_RENAME = {
    "gallbladder": "gall_bladder",
    "inferior_vena_cava": "postcava",
    "small_bowel": "intestine",
}

LUNG_LEFT_PARTS = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
LUNG_RIGHT_PARTS = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]

SHAPEKIT_EXPECTED_ORGANS = [
    "aorta", "gall_bladder", "kidney_left", "kidney_right", "liver", "pancreas",
    "postcava", "spleen", "stomach", "adrenal_gland_left", "adrenal_gland_right",
    "bladder", "colon", "duodenum", "esophagus", "femur_left", "femur_right",
    "intestine", "lung_left", "lung_right", "prostate",
]

PRESETS = {
    "shapekit_abdomen": SHAPEKIT_ABDOMEN_ROI,
    "shapekit_abdomen_fast": SHAPEKIT_ABDOMEN_FAST_ROI,
    "none": [],
}


def parse_roi_subset(roi_preset: str = "shapekit_abdomen", roi_subset: str | None = None) -> list[str]:
    if roi_subset:
        return [x.strip() for x in roi_subset.replace(";", ",").split(",") if x.strip()]
    return list(PRESETS.get(roi_preset, SHAPEKIT_ABDOMEN_ROI))
