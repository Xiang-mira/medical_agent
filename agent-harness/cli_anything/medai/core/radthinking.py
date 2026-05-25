from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import nibabel as nib
except Exception:  # Keep non-image commands importable before optional medical deps are installed.
    np = None
    nib = None

from .json_utils import read_json, write_json


@dataclass
class RadThinkingScan:
    scan_id: str
    scan_dir: Path
    ct_image: Path | None
    report_path: Path | None
    clinical_path: Path | None
    def to_dict(self) -> dict:
        return {"scan_id": self.scan_id, "scan_dir": str(self.scan_dir), "ct_image": str(self.ct_image) if self.ct_image else None, "report_path": str(self.report_path) if self.report_path else None, "clinical_path": str(self.clinical_path) if self.clinical_path else None, "has_ct": bool(self.ct_image and self.ct_image.exists()), "has_report": bool(self.report_path and self.report_path.exists()), "has_clinical": bool(self.clinical_path and self.clinical_path.exists())}


def discover_radthinking_scans(patient_folder: str | Path) -> list[RadThinkingScan]:
    root = Path(patient_folder).resolve()
    scans_dir = root / "scans"
    if not scans_dir.exists():
        return []
    scans = []
    for d in sorted([p for p in scans_dir.iterdir() if p.is_dir()]):
        ct = d / "ct.nii.gz"
        if not ct.exists():
            cands = sorted([p for p in d.iterdir() if p.name.endswith(".nii.gz") or p.name.endswith(".nii")])
            ct = cands[0] if cands else None
        scans.append(RadThinkingScan(d.name, d, ct if ct and ct.exists() else None, d / "report.txt" if (d / "report.txt").exists() else None, d / "clinical.json" if (d / "clinical.json").exists() else None))
    return scans


def check_radthinking_patient(patient_folder: str | Path) -> dict:
    root = Path(patient_folder).resolve()
    scans = discover_radthinking_scans(root)
    meta = read_json(root / "metadata.json", {})
    patho = read_json(root / "pathology.json", {})
    missing_ct = [s.scan_id for s in scans if not s.ct_image]
    return {"status": "success" if root.exists() and scans and not missing_ct else "warning", "patient_folder": str(root), "patient_id": meta.get("patient_id", root.name), "num_scans": len(scans), "has_metadata_json": (root / "metadata.json").exists(), "has_pathology_json": (root / "pathology.json").exists(), "has_longitudinal_structure": len(scans) >= 2, "missing_ct_scan_ids": missing_ct, "scans": [s.to_dict() for s in scans], "radthinking_alignment": {"trace_steps": ["observation", "temporal_comparison", "clinical_context", "diagnostic_conclusion"], "note": "Layout checker only; it does not infer clinical diagnosis."}, "metadata_preview": meta, "pathology_preview": patho}


def _volume_cm3(mask_path: str | Path | None) -> dict:
    if mask_path is None:
        return {"status": "missing", "voxel_count": 0, "volume_cm3": 0.0}
    p = Path(mask_path)
    if not p.exists():
        return {"status": "missing", "mask": str(p), "voxel_count": 0, "volume_cm3": 0.0}
    if nib is None or np is None:
        return {"status": "failed", "reason": "numpy/nibabel not installed", "mask": str(p)}
    img = nib.load(str(p)); arr = np.asanyarray(img.dataobj) > 0
    voxels = int(arr.sum())
    zooms = img.header.get_zooms()[:3]
    vol = float(voxels * zooms[0] * zooms[1] * zooms[2] / 1000.0)
    return {"status": "success", "mask": str(p.resolve()), "voxel_count": voxels, "volume_cm3": round(vol, 6), "voxel_spacing_mm": [float(x) for x in zooms]}


def extract_observation(ct_image: str | Path | None, mask_path: str | Path | None, organ: str | None = None) -> dict:
    vol = _volume_cm3(mask_path)
    if vol.get("status") != "success" or vol.get("voxel_count", 0) == 0:
        return {"step": "observation", "status": "warning", "organ": organ, "message": "No non-empty mask; observation cannot be grounded.", "volume": vol}
    if nib is None or np is None:
        return {"step": "observation", "status": "failed", "organ": organ, "reason": "numpy/nibabel not installed"}
    mask_img = nib.load(str(mask_path)); mask = np.asanyarray(mask_img.dataobj) > 0
    coords = np.argwhere(mask)
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    centroid = coords.mean(axis=0)
    zooms = mask_img.header.get_zooms()[:3]
    bbox_mm = ((maxs - mins + 1) * np.array(zooms)).astype(float)
    hu_stats = None
    if ct_image and Path(ct_image).exists():
        try:
            ct = np.asanyarray(nib.load(str(ct_image)).dataobj)
            vals = ct[mask]
            hu_stats = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4), "min": round(float(vals.min()), 4), "max": round(float(vals.max()), 4)}
        except Exception as exc:
            hu_stats = {"status": "failed", "reason": str(exc)}
    return {"step": "observation", "status": "success", "organ": organ, "mask": str(Path(mask_path).resolve()), "volume_cm3": vol["volume_cm3"], "voxel_count": vol["voxel_count"], "bbox_index_min": mins.tolist(), "bbox_index_max": maxs.tolist(), "bbox_size_mm": [round(float(x), 4) for x in bbox_mm], "centroid_index": [round(float(x), 4) for x in centroid], "hu_statistics": hu_stats, "radthinking_alignment": "Step 1: mask-grounded observation: location, size, HU, morphology proxy."}


def compare_temporal_masks(previous_mask: str | Path | None, current_mask: str | Path | None, organ: str | None = None) -> dict:
    prev = _volume_cm3(previous_mask); cur = _volume_cm3(current_mask)
    pv, cv = float(prev.get("volume_cm3", 0.0)), float(cur.get("volume_cm3", 0.0))
    if pv == 0 and cv > 0:
        label, ratio = "NEW", None
    elif pv > 0 and cv == 0:
        label, ratio = "RESOLVED", 0.0
    elif pv == 0 and cv == 0:
        label, ratio = "NO_FINDING", None
    else:
        ratio = cv / pv
        if ratio > 1.2: label = "GROWING"
        elif ratio < 0.8: label = "SHRINKING"
        else: label = "STABLE"
    return {"step": "temporal_comparison", "status": "success", "organ": organ, "previous": prev, "current": cur, "volume_ratio": round(float(ratio), 6) if ratio is not None else None, "temporal_label": label, "rule": "NEW if previous empty/current present; RESOLVED if previous present/current empty; GROWING if ratio > 1.2; SHRINKING if ratio < 0.8; otherwise STABLE."}


def parse_report_sections(report_path: str | Path | None = None, clinical_path: str | Path | None = None, organ: str | None = None) -> dict:
    report_text = Path(report_path).read_text(encoding="utf-8", errors="ignore") if report_path and Path(report_path).exists() else ""
    clinical = read_json(clinical_path, {})
    sections = {"findings": "", "impression": "", "recommendation": ""}
    lower = report_text.lower()
    for key in sections:
        m = re.search(rf"{key}s?\s*:\s*(.*?)(?=\n\s*(findings?|impression|recommendations?)\s*:|\Z)", report_text, flags=re.I | re.S)
        if m: sections[key] = m.group(1).strip()
    suspicious_terms = ["suspicious", "recurrence", "growing", "new lesion", "malign", "cancer", "hcc", "li-rads", "mass"]
    # Issue 1 fix: align negation regex window with the 60-char prefix slice used in _term_affirmed
    _NEGATION_RE = re.compile(
        r"\b(no|not|without|negative\s+for|absence\s+of|exclude[sd]?|ruled?\s+out)\b.{0,60}$",
        re.I,
    )
    def _term_affirmed(term: str, text: str) -> bool:
        """Return True only if `term` appears in `text` WITHOUT a preceding negation."""
        for m in re.finditer(re.escape(term), text, re.I):
            prefix = text[max(0, m.start() - 60): m.start()]
            if not _NEGATION_RE.search(prefix):
                return True
        return False

    found_suspicious = [t for t in suspicious_terms if _term_affirmed(t, lower)]
    return {"step": "clinical_context", "status": "success", "organ": organ, "report_path": str(Path(report_path).resolve()) if report_path and Path(report_path).exists() else None, "clinical_path": str(Path(clinical_path).resolve()) if clinical_path and Path(clinical_path).exists() else None, "sections": sections, "clinical_variables": clinical, "suspicious_terms_found": found_suspicious, "radthinking_alignment": "Step 3: report sections + clinical variables. This prototype uses rule-based parsing; LLM extraction can be added later."}


def _build_trace_narrative(
    obs: dict,
    temp: dict,
    ctx: dict,
    conclusion: dict,
    organ: str | None,
    narrative_generator=None,  # Issue 4 fix: optional callable for LLM-generated prose
) -> dict:
    # Issue 4: if a real LLM generator is provided, delegate to it
    if narrative_generator is not None:
        return narrative_generator(obs, temp, ctx, conclusion, organ)
    organ_name = organ or "target structure"
    if obs.get("status") == "success":
        obs_sentence = (
            f"The {organ_name} mask is non-empty, with an estimated volume of "
            f"{obs.get('volume_cm3')} cm3 and a bounding-box size of "
            f"{obs.get('bbox_size_mm')} mm. The centroid in image index space is "
            f"{obs.get('centroid_index')}."
        )
        hu = obs.get("hu_statistics")
        if isinstance(hu, dict) and "mean" in hu:
            obs_sentence += f" Within the mask, the mean CT intensity is approximately {hu.get('mean')} HU."
    else:
        obs_sentence = f"No reliable non-empty {organ_name} mask is available, so image-grounded observation remains uncertain."

    temporal_label = temp.get("temporal_label", "UNKNOWN")
    if temporal_label == "NO_FINDING":
        temp_sentence = f"No prior or current non-empty {organ_name} mask is available for temporal comparison."
    elif temporal_label == "NEW":
        temp_sentence = f"The {organ_name} finding appears new relative to the prior mask because the previous mask is empty and the current mask is non-empty."
    elif temporal_label == "RESOLVED":
        temp_sentence = f"The {organ_name} finding appears resolved because the prior mask was non-empty while the current mask is empty."
    elif temp.get("volume_ratio") is not None:
        temp_sentence = f"The {organ_name} volume ratio is {temp.get('volume_ratio')}, which is categorized as {temporal_label.lower()} by the rule-based temporal check."
    else:
        temp_sentence = f"Temporal comparison for the {organ_name} is categorized as {temporal_label.lower()}."

    suspicious = ctx.get("suspicious_terms_found") or []
    if suspicious:
        ctx_sentence = f"The available report/clinical context contains potentially relevant terms: {', '.join(suspicious)}."
    elif ctx.get("report_path") or ctx.get("clinical_path"):
        ctx_sentence = "The available report/clinical context does not contain the configured suspicious terms after negation-aware parsing."
    else:
        ctx_sentence = "No report or clinical context file is available for this trace."

    if isinstance(conclusion, dict) and conclusion.get("status") == "not_provided":
        conclusion_sentence = "No pathology or follow-up conclusion is provided; the pipeline does not fabricate a diagnosis."
    elif conclusion:
        conclusion_sentence = f"The supplied conclusion metadata is preserved for supervision: {conclusion}."
    else:
        conclusion_sentence = "No diagnostic conclusion metadata is available."

    return {
        "step1_observations_text": obs_sentence,
        "step2_temporal_text": temp_sentence,
        "step3_clinical_context_text": ctx_sentence,
        "step4_conclusion_text": conclusion_sentence,
        "full_trace_text": " ".join([obs_sentence, temp_sentence, ctx_sentence, conclusion_sentence]),
        "generation_method": "deterministic_template_from_structured_trace",
        "scope_note": "Template narrative only; replace with VLM/LLM-generated radiologist-style prose when a validated model server is available.",
    }


def build_reasoning_trace(
    patient_folder: str | Path | None = None,
    scan_id: str | None = None,
    ct_image: str | Path | None = None,
    current_mask: str | Path | None = None,
    previous_mask: str | Path | None = None,
    organ: str | None = None,
    report_path: str | Path | None = None,
    clinical_path: str | Path | None = None,
    pathology_path: str | Path | None = None,
    output_json: str | Path | None = None,
    narrative_generator=None,  # Issue 4: optional LLM narrative callable
) -> dict:
    obs = extract_observation(ct_image, current_mask, organ)
    temp = compare_temporal_masks(previous_mask, current_mask, organ)
    ctx = parse_report_sections(report_path, clinical_path, organ)
    conclusion = read_json(pathology_path, {}) if pathology_path else {}
    if not conclusion:
        conclusion = {"status": "not_provided", "note": "No pathology/follow-up JSON was provided; this tool does not fabricate diagnosis."}

    # Issue 2 fix: multi-label complexity — collect ALL triggered labels instead of single priority chain
    label = temp.get("temporal_label")
    complexity_labels = []
    if label in {"NEW", "GROWING", "RESOLVED", "SHRINKING"}:
        complexity_labels.append("TEMPORAL")
    if ctx.get("suspicious_terms_found"):
        complexity_labels.append("INTEGRATIVE")
    if obs.get("status") == "success":
        complexity_labels.append("PERCEPTUAL")
    complexity = complexity_labels if complexity_labels else ["AMBIGUOUS"]

    narrative = _build_trace_narrative(obs, temp, ctx, conclusion, organ, narrative_generator=narrative_generator)

    # Issue 3 fix: propagate warning status from extract_observation to outer trace
    outer_status = "warning" if obs.get("status") == "warning" else "success"

    trace = {
        "status": outer_status,
        "patient_folder": str(Path(patient_folder).resolve()) if patient_folder else None,
        "scan_id": scan_id,
        "organ": organ,
        "radthinking_trace": {
            "observation": obs,
            "temporal_comparison": temp,
            "clinical_context": ctx,
            "diagnostic_conclusion": conclusion,
            "narrative": narrative,
        },
        "natural_language_trace": narrative,
        "complexity_level_prototype": complexity,
        "scope_note": "Prototype trace generator only. It structures available evidence and emits template prose, but does not infer clinical diagnosis.",
    }
    if output_json:
        trace["saved_to"] = write_json(output_json, trace)
    return trace


def build_patient_traces_from_outputs(patient_folder: str | Path, output_folder: str | Path, organ: str, output_json: str | Path | None = None) -> dict:
    root = Path(patient_folder).resolve(); out = Path(output_folder).resolve(); scans = discover_radthinking_scans(root)
    traces = []; prev = None
    for s in scans:
        mask_cands = list((out / s.scan_id).glob(f"**/segmentations/{organ}.nii.gz")) + list((out / "scan_outputs" / s.scan_id).glob(f"**/segmentations/{organ}.nii.gz"))
        cur = mask_cands[0] if mask_cands else None
        # Issue 5 fix: warn when no mask is found instead of silently passing None
        if cur is None:
            import logging
            logging.getLogger(__name__).warning(
                "build_patient_traces_from_outputs: no mask found for organ=%s scan=%s in %s",
                organ, s.scan_id, out,
            )
        tr = build_reasoning_trace(root, s.scan_id, s.ct_image, cur, prev, organ, s.report_path, s.clinical_path, root / "pathology.json" if (root / "pathology.json").exists() else None)
        traces.append(tr)
        if cur and Path(cur).exists(): prev = cur
    result = {"status": "success", "patient_id": root.name, "organ": organ, "num_traces": len(traces), "traces": traces}
    if output_json: result["saved_to"] = write_json(output_json, result)
    return result
