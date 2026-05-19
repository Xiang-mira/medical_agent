from __future__ import annotations
from pathlib import Path


def create_radthinking_trace_template(case_id: str, ct_image: str | None = None, segmentation_folder: str | None = None, report_path: str | None = None, prior_case_id: str | None = None) -> dict:
    masks = []
    if segmentation_folder and Path(segmentation_folder).exists():
        masks = sorted([p.name for p in Path(segmentation_folder).glob("*.nii.gz")])
    return {"case_id": case_id, "source": {"ct_image": ct_image, "segmentation_folder": segmentation_folder, "report_path": report_path, "prior_case_id": prior_case_id}, "radthinking_trace": {"observation": {"status": "template_only", "available_masks": masks}, "temporal_comparison": {"status": "template_only"}, "clinical_context": {"status": "template_only"}, "diagnostic_conclusion": {"status": "template_only", "warning": "Do not fabricate diagnosis."}}, "scope_note": "Template only; use trace-build for computable fields."}
