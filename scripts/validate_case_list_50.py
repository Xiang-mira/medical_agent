#!/usr/bin/env python3
"""Validate the final PanTS 50-case manifest before running real experiments."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

REQUIRED = ["case_id", "ct_path", "annotation_folder"]
TUMOR_NAMES = ["pancreatic_lesion.nii.gz", "pancreatic_pdac.nii.gz", "pancreatic_cyst.nii.gz", "pancreatic_pnet.nii.gz"]
CORE_ORGANS = ["pancreas", "liver", "spleen", "kidney_left", "kidney_right", "colon", "duodenum", "stomach", "aorta", "postcava"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_annotation_folder(row: dict[str, str]) -> tuple[Path, str]:
    """Return the best annotation folder and how it was resolved."""
    raw = (row.get("annotation_folder") or "").strip()
    if raw:
        return Path(raw).expanduser(), "manifest"
    cid = (row.get("case_id") or "").strip()
    if cid:
        candidate = PROJECT_ROOT / "data" / "PanTS" / "LabelTr" / cid / "segmentations"
        if candidate.exists():
            return candidate, "derived_from_case_id"
        candidate = PROJECT_ROOT / "third_party" / "PanTS-main" / "data" / "LabelTr" / cid / "segmentations"
        if candidate.exists():
            return candidate, "derived_from_case_id"
    return Path(raw).expanduser(), "missing"


def mask_nonempty(path: Path) -> tuple[bool, int | None]:
    try:
        import nibabel as nib
        import numpy as np
        voxels = int((np.asanyarray(nib.load(str(path)).dataobj) > 0).sum())
        return voxels > 0, voxels
    except Exception:
        return False, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--case-list", required=True)
    ap.add_argument("--require-num", type=int, default=50)
    ap.add_argument("--output-json", default=None)
    args=ap.parse_args()
    path=Path(args.case_list).resolve()
    rows=[]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        missing_cols=[c for c in REQUIRED if c not in fields]
        for row in reader:
            if any((v or "").strip() for v in row.values()): rows.append(row)
    issues=[]
    if missing_cols: issues.append({"level":"error","reason":"missing_required_columns","columns":missing_cols})
    if len(rows)!=args.require_num: issues.append({"level":"error","reason":"wrong_case_count","expected":args.require_num,"actual":len(rows)})
    validated=0
    resolved_annotations=0
    for row in rows:
        cid=row.get("case_id","")
        ct=Path(row.get("ct_path","")).expanduser()
        ann, ann_source = resolve_annotation_folder(row)
        case_errors=[]
        case_warnings=[]
        if ann_source == "derived_from_case_id":
            resolved_annotations += 1
            case_warnings.append("annotation_folder_derived_from_case_id")
        if not ct.exists(): case_errors.append("ct_missing")
        if not ann.exists(): case_errors.append("annotation_folder_missing")
        tumor_present=False
        if ann.exists():
            for name in TUMOR_NAMES:
                if (ann/name).exists():
                    try:
                        import nibabel as nib, numpy as np
                        tumor_present = int((np.asanyarray(nib.load(str(ann/name)).dataobj)>0).sum()) > 0
                    except Exception:
                        tumor_present = True
                    if tumor_present: break
        if not tumor_present: case_warnings.append("nonempty_tumor_mask_missing")
        for organ in CORE_ORGANS:
            organ_path = ann / f"{organ}.nii.gz"
            if ann.exists() and not organ_path.exists():
                case_warnings.append(f"core_organ_missing:{organ}")
            elif ann.exists():
                organ_nonempty, organ_voxels = mask_nonempty(organ_path)
                if not organ_nonempty:
                    case_warnings.append(f"core_organ_empty:{organ}")
        if case_errors:
            issues.append({"level":"error","case_id":cid,"issues":case_errors[:20],"annotation_source":ann_source})
        elif case_warnings:
            issues.append({"level":"warning","case_id":cid,"issues":case_warnings[:20],"annotation_source":ann_source})
            if not any(x.startswith("core_organ_missing:") or x.startswith("core_organ_empty:") or x == "nonempty_tumor_mask_missing" for x in case_warnings):
                validated += 1
        else:
            validated += 1
    result={
        "stage":"validate_case_list_50",
        "status":"success" if not [i for i in issues if i.get('level')=='error'] else "failed",
        "case_list":str(path),
        "num_rows":len(rows),
        "num_fully_validated_cases":validated,
        "num_annotation_folders_resolved_from_case_id":resolved_annotations,
        "issues":issues[:200],
    }
    if args.output_json:
        out=Path(args.output_json).resolve(); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
if __name__=="__main__": main()
