"""Report-supervised annotation verification.

The teacher said: 'use report to do supervision ... report already tells you whether
there is a tumor, how big it is ... you compare the AI label with the report, if they
match it is correct, if not it is wrong.'

This module compares AI-predicted tumor masks against radiology report content to flag
mismatches without requiring ROC/sensitivity-specificity analysis.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .json_utils import read_json, write_json


# Tumor keywords grouped by organ
_TUMOR_PATTERNS: dict[str, list[str]] = {
    "pancreas": ["pancrea", "pdac", "pnet", "cyst", "mass in pancrea", "pancreatic lesion", "pancreatic tumor", "pancreatic mass"],
    "liver": ["hepat", "hcc", "liver mass", "liver lesion", "liver tumor", "hepatocellular", "metasta"],
    "kidney": ["renal mass", "renal cell", "kidney tumor", "kidney lesion", "renal lesion"],
    "lung": ["lung nodule", "lung mass", "pulmonary", "lung lesion", "lung tumor"],
    "colon": ["colon", "colonic", "colorectal", "rectal mass"],
}

_NEGATION_RE = re.compile(
    r"\b(no|not|without|negative\s+for|absence\s+of|exclude[sd]?|ruled?\s+out|unremarkable|normal)\b.{0,40}$",
    re.I,
)

_SIZE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?))?)?\s*(mm|cm|millimeter|centimeter)",
    re.I,
)


def _term_affirmed(term: str, text: str) -> bool:
    for m in re.finditer(re.escape(term), text, re.I):
        prefix = text[max(0, m.start() - 60): m.start()]
        if not _NEGATION_RE.search(prefix):
            return True
    return False


def _extract_tumor_size_mm(text: str) -> float | None:
    """Extract the largest mentioned tumor dimension in mm."""
    best = 0.0
    for m in _SIZE_RE.finditer(text):
        dims = [float(m.group(i)) for i in (1, 2, 3) if m.group(i)]
        unit = m.group(4).lower()
        factor = 10.0 if unit.startswith("cm") or unit.startswith("centimeter") else 1.0
        val = max(dims) * factor
        if val > best:
            best = val
    return best if best > 0 else None


def _mask_volume_mm3(mask_path: Path) -> dict[str, Any]:
    if not mask_path.exists():
        return {"status": "missing", "voxels": 0, "volume_mm3": 0.0}
    try:
        import nibabel as nib
        import numpy as np
        img = nib.load(str(mask_path))
        arr = np.asanyarray(img.dataobj) > 0
        vox = int(arr.sum())
        zooms = img.header.get_zooms()[:3]
        vol = float(vox * zooms[0] * zooms[1] * zooms[2])
        return {"status": "success", "voxels": vox, "volume_mm3": round(vol, 2), "spacing_mm": [float(z) for z in zooms]}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def _estimate_diameter_mm(volume_mm3: float) -> float:
    """Approximate diameter assuming spherical tumor."""
    import math
    if volume_mm3 <= 0:
        return 0.0
    radius = (3 * volume_mm3 / (4 * math.pi)) ** (1.0 / 3.0)
    return round(2 * radius, 2)


def verify_tumor_with_report(
    report_path: str | Path | None,
    tumor_mask_path: str | Path | None,
    organ: str = "pancreas",
    clinical_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare AI tumor mask against report text.

    Returns a verdict: match / mismatch / uncertain.
    """
    result: dict[str, Any] = {"stage": "report_supervision", "organ": organ}

    # Read report
    report_text = ""
    if report_path and Path(report_path).exists():
        report_text = Path(report_path).read_text(encoding="utf-8", errors="ignore")
    result["report_path"] = str(report_path) if report_path else None
    result["report_available"] = bool(report_text)

    # Read clinical data
    clinical = read_json(clinical_path, {}) if clinical_path else {}
    result["clinical_path"] = str(clinical_path) if clinical_path else None

    # Analyze report for tumor mentions
    patterns = _TUMOR_PATTERNS.get(organ, [])
    affirmed_terms = [t for t in patterns if _term_affirmed(t, report_text.lower())]
    report_says_tumor = len(affirmed_terms) > 0
    report_tumor_size_mm = _extract_tumor_size_mm(report_text)

    result["report_analysis"] = {
        "affirmed_tumor_terms": affirmed_terms,
        "report_says_tumor": report_says_tumor,
        "extracted_size_mm": report_tumor_size_mm,
    }

    # Analyze mask
    mask_info = {"status": "missing", "voxels": 0}
    mask_has_tumor = False
    mask_diameter_mm = 0.0
    if tumor_mask_path:
        mp = Path(tumor_mask_path)
        mask_info = _mask_volume_mm3(mp)
        mask_has_tumor = mask_info.get("voxels", 0) > 0
        if mask_has_tumor and mask_info.get("volume_mm3", 0) > 0:
            mask_diameter_mm = _estimate_diameter_mm(mask_info["volume_mm3"])
    result["mask_analysis"] = {
        "mask_path": str(tumor_mask_path) if tumor_mask_path else None,
        "mask_has_tumor": mask_has_tumor,
        "estimated_diameter_mm": mask_diameter_mm,
        **mask_info,
    }

    # Compare
    if not report_text:
        verdict = "uncertain"
        reason = "No report available; cannot verify."
    elif report_says_tumor and mask_has_tumor:
        # Both agree tumor exists; check size consistency
        if report_tumor_size_mm and mask_diameter_mm > 0:
            ratio = mask_diameter_mm / report_tumor_size_mm
            if 0.3 < ratio < 3.0:
                verdict = "match"
                reason = f"Report mentions tumor (~{report_tumor_size_mm:.1f}mm), mask diameter ~{mask_diameter_mm:.1f}mm. Consistent."
            else:
                verdict = "size_mismatch"
                reason = f"Report tumor ~{report_tumor_size_mm:.1f}mm but mask diameter ~{mask_diameter_mm:.1f}mm. Ratio={ratio:.2f}. Needs review."
        else:
            verdict = "match"
            reason = "Both report and mask agree: tumor present."
    elif report_says_tumor and not mask_has_tumor:
        verdict = "mismatch_false_negative"
        reason = "Report says tumor present but mask is empty. AI missed the tumor."
    elif not report_says_tumor and mask_has_tumor:
        verdict = "mismatch_false_positive"
        reason = "Report does not mention tumor but mask is non-empty. Possible false positive."
    elif not report_says_tumor and not mask_has_tumor:
        verdict = "match"
        reason = "Both agree: no tumor."
    else:
        verdict = "uncertain"
        reason = "Could not determine."

    result["verdict"] = verdict
    result["reason"] = reason
    result["action"] = {
        "match": "accept",
        "size_mismatch": "send_to_vlm_or_human_review",
        "mismatch_false_negative": "flag_for_review_high_priority",
        "mismatch_false_positive": "send_to_vlm_review",
        "uncertain": "send_to_vlm_or_human_review",
    }.get(verdict, "uncertain")

    return result


def batch_verify_with_reports(
    case_list: list[dict[str, Any]],
    output_jsonl: str | Path,
    organ: str = "pancreas",
    tumor_mask_name: str = "pancreatic_lesion",
) -> dict[str, Any]:
    """Run report supervision across multiple cases."""
    out = Path(output_jsonl).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for case in case_list:
        case_id = case.get("case_id", "")
        report_path = case.get("report_path")
        ann_folder = case.get("annotation_folder", "")
        tumor_mask = Path(ann_folder) / f"{tumor_mask_name}.nii.gz" if ann_folder else None
        clinical_path = case.get("clinical_path")
        v = verify_tumor_with_report(report_path, tumor_mask, organ, clinical_path)
        v["case_id"] = case_id
        results.append(v)
        with out.open("a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    verdicts = [r["verdict"] for r in results]
    summary = {
        "stage": "report_supervision_batch",
        "status": "success",
        "num_cases": len(results),
        "output_jsonl": str(out),
        "verdict_counts": {v: verdicts.count(v) for v in set(verdicts)},
        "action_needed": len([r for r in results if r.get("action") != "accept"]),
    }
    return summary
