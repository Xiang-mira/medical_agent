from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import nibabel as nib
except Exception:  # Keep CLI importable before optional medical-image deps are installed.
    np = None
    nib = None


def pants_download_info() -> dict[str, Any]:
    """Return official PanTS/PanTSMini download information without downloading data."""
    return {
        "status": "info",
        "dataset": "PanTS / PanTSMini",
        "official_repo": "https://github.com/MrGiovanni/PanTS",
        "hf_dataset": "https://huggingface.co/datasets/BodyMaps/PanTSMini",
        "storage_note": "Official full-dataset script needs ~300 GB. For a 50-case run, use scripts/download_pants50_selective.py: peak ~44 GB (image tar + extracted), drops to ~15 GB after both tars are deleted. A 50 GB data disk is sufficient if phases are run sequentially.",
        "official_linux_commands": [
            "git clone https://github.com/MrGiovanni/PanTS.git",
            "cd PanTS",
            "bash download_PanTS_data.sh  # ~300GB storage",
            "bash download_PanTS_label.sh",
        ],
        "small_demo_strategy": {
            "recommended": True,
            "why": "The official image script downloads 1000-case tar blocks, not a one-case API. For laptop work, import any locally available PanTS case into the CLI with pants-import-case or pants-import-files.",
            "commands": [
                "python run_medai_cli.py --json pants-import-case --pants-root third_party/PanTS-main --case-id PanTS_00000001 --output-root data/pants_small",
                "python run_medai_cli.py --json pants-import-files --ct <path/to/ct.nii.gz> --label-folder <path/to/segmentations> --output-root data/pants_small --patient-id PanTS_00000001",
            ],
        },
        "expected_layout": {
            "metadata": "PanTS/data/metadata.xlsx",
            "images_train": "PanTS/data/ImageTr/PanTS_00000001/ct.nii.gz",
            "images_test": "PanTS/data/ImageTe/PanTS_00009001/ct.nii.gz",
            "reports_train": "PanTS/data/ReportTr/PanTS_00000001/report.pdf",
            "reports_test": "PanTS/data/ReportTe/PanTS_00009001/report.pdf",
            "labels_train": "PanTS/data/LabelTr/PanTS_00000001/segmentations/*.nii.gz",
            "labels_test": "PanTS/data/LabelTe/PanTS_00009001/segmentations/*.nii.gz",
        },
        "why_not_bundled": "The uploaded PanTS zip contains code/readme/download scripts but not the actual CT NIfTI dataset files.",
    }


def _data_root(pants_root: str | Path) -> Path:
    root = Path(pants_root).resolve()
    return root / "data" if (root / "data").exists() else root


def check_pants_dataset(pants_root: str | Path, max_cases: int = 10) -> dict[str, Any]:
    root = Path(pants_root).resolve()
    data_root = _data_root(root)
    summary: dict[str, Any] = {
        "status": "unknown",
        "pants_root": str(root),
        "data_root": str(data_root),
        "exists": root.exists(),
        "has_metadata": (data_root / "metadata.xlsx").exists(),
        "splits": {},
    }
    if not root.exists():
        summary["status"] = "failed"
        summary["reason"] = "PanTS root does not exist. Download the dataset first or point to a folder containing ImageTr/LabelTr."
        return summary
    for split_name, image_dir, label_dir, report_dir in [
        ("train", "ImageTr", "LabelTr", "ReportTr"),
        ("test", "ImageTe", "LabelTe", "ReportTe"),
    ]:
        img_root = data_root / image_dir
        lab_root = data_root / label_dir
        rep_root = data_root / report_dir
        image_cases = sorted([p for p in img_root.iterdir() if p.is_dir()]) if img_root.exists() else []
        label_cases = sorted([p for p in lab_root.iterdir() if p.is_dir()]) if lab_root.exists() else []
        report_cases = sorted([p for p in rep_root.iterdir() if p.is_dir()]) if rep_root.exists() else []
        samples = []
        for case in image_cases[:max_cases]:
            cid = case.name
            seg_dir = lab_root / cid / "segmentations"
            samples.append({
                "case_id": cid,
                "ct_exists": (case / "ct.nii.gz").exists(),
                "label_dir_exists": seg_dir.exists(),
                "num_label_masks": len(list(seg_dir.glob("*.nii.gz"))) if seg_dir.exists() else 0,
                "report_pdf_exists": (rep_root / cid / "report.pdf").exists(),
            })
        summary["splits"][split_name] = {
            "image_dir": str(img_root),
            "label_dir": str(lab_root),
            "report_dir": str(rep_root),
            "num_image_cases": len(image_cases),
            "num_label_cases": len(label_cases),
            "num_report_cases": len(report_cases),
            "sample_cases": samples,
        }
    has_any_images = any(v["num_image_cases"] > 0 for v in summary["splits"].values())
    has_any_labels = any(v["num_label_cases"] > 0 for v in summary["splits"].values())
    if has_any_images and has_any_labels:
        summary["status"] = "success"
    elif has_any_images:
        summary["status"] = "warning"
        summary["reason"] = "Images found but labels appear missing or not extracted."
    else:
        summary["status"] = "warning"
        summary["reason"] = "No downloaded ImageTr/ImageTe cases found. The repository may contain only scripts/readme."
    return summary


def find_pants_case(pants_root: str | Path, case_id: str, split: str = "auto") -> dict[str, Any]:
    root = Path(pants_root).resolve()
    data_root = _data_root(root)
    candidates = []
    split_defs = []
    if split in {"auto", "train"}:
        split_defs.append(("train", "ImageTr", "LabelTr", "ReportTr"))
    if split in {"auto", "test"}:
        split_defs.append(("test", "ImageTe", "LabelTe", "ReportTe"))
    for split_name, image_dir, label_dir, report_dir in split_defs:
        ct = data_root / image_dir / case_id / "ct.nii.gz"
        label_folder = data_root / label_dir / case_id / "segmentations"
        report_pdf = data_root / report_dir / case_id / "report.pdf"
        candidates.append({
            "split": split_name,
            "ct": ct,
            "label_folder": label_folder,
            "report_pdf": report_pdf,
            "ct_exists": ct.exists(),
            "label_folder_exists": label_folder.exists(),
            "num_label_masks": len(list(label_folder.glob("*.nii.gz"))) if label_folder.exists() else 0,
            "report_pdf_exists": report_pdf.exists(),
        })
    found = next((c for c in candidates if c["ct_exists"]), None)
    status = "success" if found else "failed"
    return {
        "status": status,
        "pants_root": str(root),
        "data_root": str(data_root),
        "case_id": case_id,
        "split_requested": split,
        "found": {k: str(v) if isinstance(v, Path) else v for k, v in found.items()} if found else None,
        "candidates": [{k: str(v) if isinstance(v, Path) else v for k, v in c.items()} for c in candidates],
        "reason": None if found else "Case CT not found. You may need to download/extract the relevant PanTS image tar block first.",
    }


def _copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def import_pants_case(
    pants_root: str | Path,
    case_id: str,
    output_root: str | Path,
    split: str = "auto",
    patient_id: str | None = None,
    scan_id: str | None = None,
    copy_labels: bool = True,
) -> dict[str, Any]:
    located = find_pants_case(pants_root, case_id, split)
    if located["status"] != "success":
        return located | {"operation": "pants-import-case"}
    found = located["found"]
    return import_pants_files(
        ct_path=found["ct"],
        label_folder=found["label_folder"] if found.get("label_folder_exists") else None,
        output_root=output_root,
        patient_id=patient_id or case_id,
        scan_id=scan_id or case_id,
        source_case_id=case_id,
        report_path=found["report_pdf"] if found.get("report_pdf_exists") else None,
        copy_labels=copy_labels,
        source_note=f"Imported from PanTS {found['split']} split.",
    ) | {"located_case": located}


def import_pants_files(
    ct_path: str | Path,
    label_folder: str | Path | None,
    output_root: str | Path,
    patient_id: str,
    scan_id: str | None = None,
    source_case_id: str | None = None,
    report_path: str | Path | None = None,
    copy_labels: bool = True,
    source_note: str = "Imported from user-provided PanTS-like files.",
) -> dict[str, Any]:
    ct = Path(ct_path).resolve()
    label_dir = Path(label_folder).resolve() if label_folder else None
    report = Path(report_path).resolve() if report_path else None
    scan_id = scan_id or source_case_id or patient_id
    patient_root = Path(output_root).resolve() / patient_id
    scan_dir = patient_root / "scans" / scan_id
    ref_dir = patient_root / "reference_labels" / scan_id / "segmentations"
    if not ct.exists():
        return {"status": "failed", "operation": "pants-import-files", "reason": "ct path does not exist", "ct": str(ct)}
    _copy_if_exists(ct, scan_dir / "ct.nii.gz")
    # We do not try to extract the PDF report text automatically here. Keep provenance and a placeholder text file.
    if report and report.exists() and report.suffix.lower() == ".txt":
        _copy_if_exists(report, scan_dir / "report.txt")
    else:
        _write_text(scan_dir / "report.txt", f"Report text not provided in TXT format. Source report: {str(report) if report else 'not provided'}\n")
    _write_json(scan_dir / "clinical.json", {
        "source_dataset": "PanTS",
        "source_case_id": source_case_id or patient_id,
        "scan_id": scan_id,
        "note": "Minimal clinical metadata placeholder. Replace with PanTS metadata.xlsx fields if needed.",
    })
    copied_masks = []
    if copy_labels and label_dir and label_dir.exists():
        ref_dir.mkdir(parents=True, exist_ok=True)
        for m in sorted(label_dir.glob("*.nii.gz")):
            shutil.copy2(m, ref_dir / m.name)
            copied_masks.append(m.name)
    _write_json(patient_root / "metadata.json", {
        "patient_id": patient_id,
        "source_dataset": "PanTS",
        "source_case_id": source_case_id or patient_id,
        "layout": "RadThinking-style one-scan patient folder imported from PanTS case.",
        "source_note": source_note,
    })
    _write_json(patient_root / "pathology.json", {
        "status": "not_provided",
        "source_dataset": "PanTS",
        "source_case_id": source_case_id or patient_id,
        "note": "PanTS segmentation labels do not automatically provide a pathology-confirmed RadThinking conclusion. Do not fabricate diagnosis.",
    })
    return {
        "status": "success",
        "operation": "pants-import-files",
        "patient_folder": str(patient_root),
        "scan_id": scan_id,
        "ct_copied_to": str(scan_dir / "ct.nii.gz"),
        "reference_label_folder": str(ref_dir) if copied_masks else None,
        "num_reference_masks": len(copied_masks),
        "sample_reference_masks": copied_masks[:30],
        "next_steps": [
            f"python run_medai_cli.py --json radthinking-check --patient-folder {patient_root}",
            f"python run_medai_cli.py --json agent-loop --patient-folder {patient_root} --output-folder outputs/{patient_id}_agent_loop --backend totalseg --postprocess shapekit --shapekit-root third_party/ShapeKit-main --fast --organ pancreas --expected-organs pancreas,liver,aorta,postcava",
        ],
    }


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    denom = int(pred_b.sum() + gt_b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_b, gt_b).sum() / denom)


def evaluate_segmentation_folder(pred_folder: str | Path, reference_label_folder: str | Path, organs: list[str] | None = None) -> dict[str, Any]:
    pred_root = Path(pred_folder).resolve()
    ref_root = Path(reference_label_folder).resolve()
    if not pred_root.exists():
        return {"status": "failed", "reason": "pred_folder does not exist", "pred_folder": str(pred_root)}
    if not ref_root.exists():
        return {"status": "failed", "reason": "reference_label_folder does not exist", "reference_label_folder": str(ref_root)}
    if organs:
        names = [o if o.endswith(".nii.gz") else f"{o}.nii.gz" for o in organs]
    else:
        names = sorted([p.name for p in ref_root.glob("*.nii.gz")])
    rows = []
    for name in names:
        p = pred_root / name
        r = ref_root / name
        row: dict[str, Any] = {"mask": name, "pred_exists": p.exists(), "reference_exists": r.exists()}
        if p.exists() and r.exists():
            try:
                pred = np.asanyarray(nib.load(str(p)).dataobj)
                ref = np.asanyarray(nib.load(str(r)).dataobj)
                if pred.shape != ref.shape:
                    row.update({"status": "failed", "reason": f"shape mismatch: pred={pred.shape}, ref={ref.shape}"})
                else:
                    row.update({"status": "success", "dice": round(dice_score(pred, ref), 6), "pred_voxels": int((pred > 0).sum()), "reference_voxels": int((ref > 0).sum())})
            except Exception as exc:
                row.update({"status": "failed", "reason": str(exc)})
        else:
            row["status"] = "missing"
        rows.append(row)
    valid = [x["dice"] for x in rows if x.get("status") == "success"]
    return {
        "status": "success" if valid else "warning",
        "pred_folder": str(pred_root),
        "reference_label_folder": str(ref_root),
        "num_evaluated": len(valid),
        "mean_dice": round(float(np.mean(valid)), 6) if valid else None,
        "results": rows,
        "scope_note": "Simple binary Dice per mask. This is a local sanity check, not the full PanTS benchmark protocol.",
    }
