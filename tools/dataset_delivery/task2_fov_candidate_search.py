#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from tools.dataset_delivery.task2_fov import collect_landmark_evidence, summarize_fov_evidence


DEFAULT_IMAGE_ROOT = Path("/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro")
DEFAULT_MASK_ROOT = Path("/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro")


def _read_case_ids(case_list: Path | None, image_root: Path) -> list[str]:
    if case_list:
        with case_list.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            return [str(row.get("case_id") or row.get("id") or "").strip() for row in rows if str(row.get("case_id") or row.get("id") or "").strip()]
        return [line.strip() for line in case_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(path.name for path in image_root.iterdir() if path.is_dir())


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ct_info(ct: Path) -> tuple[str, str]:
    try:
        import nibabel as nib

        img = nib.load(str(ct))
        return "x".join(str(int(x)) for x in img.shape[:3]), "x".join(f"{float(x):.6g}" for x in img.header.get_zooms()[:3])
    except Exception:
        return "", ""


def search_fov_candidates(
    *,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    mask_root: Path = DEFAULT_MASK_ROOT,
    output_root: Path,
    case_list: Path | None = None,
    progress_every: int = 250,
) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    case_ids = _read_case_ids(case_list, image_root)
    inventory_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    airway_rows: list[dict[str, Any]] = []
    thorax_rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids, start=1):
        ct = image_root / case_id / "ct.nii.gz"
        ref = mask_root / case_id / "segmentations"
        ct_shape, spacing = _ct_info(ct)
        landmarks = collect_landmark_evidence(ref, ct_path=ct)
        summary = summarize_fov_evidence(landmarks)
        target_group = []
        if summary["head_evidence"]:
            target_group.append("head")
        if summary["thorax_evidence"] or summary["partial_thorax_evidence"]:
            target_group.append("thorax")
        if summary["central_airway_evidence"]:
            target_group.append("central_airway")
        eligibility = "eligible" if target_group else "not_eligible"
        reason = ",".join(target_group) if target_group else "no_target_specific_fov_evidence"
        for name, evidence in sorted(landmarks.items()):
            inventory_rows.append({
                "case_id": case_id,
                "ct_path": str(ct),
                "ct_shape": ct_shape,
                "spacing": spacing,
                "landmark_name": name,
                "foreground_voxels": evidence.get("foreground_voxels", 0),
                "geometry_match": evidence.get("geometry_match", ""),
                "head_evidence": summary["head_evidence"],
                "thorax_evidence": summary["thorax_evidence"] or summary["partial_thorax_evidence"],
                "central_airway_evidence": summary["central_airway_evidence"],
                "candidate_target_group": ",".join(target_group),
                "eligibility": eligibility,
                "reason": reason,
            })
        case_row = {
            "case_id": case_id,
            "ct_path": str(ct),
            "annotation_folder": str(ref),
            "ct_shape": ct_shape,
            "spacing": spacing,
            "head_evidence": summary["head_evidence"],
            "thorax_evidence": summary["thorax_evidence"] or summary["partial_thorax_evidence"],
            "central_airway_evidence": summary["central_airway_evidence"],
            "candidate_target_group": ",".join(target_group),
            "eligibility": eligibility,
            "reason": reason,
            **summary,
        }
        if summary["head_evidence"]:
            head_rows.append(case_row)
        if summary["central_airway_evidence"]:
            airway_rows.append(case_row)
        if summary["thorax_evidence"] or summary["partial_thorax_evidence"]:
            thorax_rows.append(case_row)
        if progress_every > 0 and (index % progress_every == 0 or index == len(case_ids)):
            print(
                "[FOV_SEARCH] "
                f"scanned={index} total={len(case_ids)} "
                f"head_candidates={len(head_rows)} airway_candidates={len(airway_rows)} "
                f"elapsed_sec={int(time.time() - started)}",
                flush=True,
            )
    inventory_fields = [
        "case_id", "ct_path", "ct_shape", "spacing", "landmark_name",
        "foreground_voxels", "geometry_match", "head_evidence", "thorax_evidence",
        "central_airway_evidence", "candidate_target_group", "eligibility", "reason",
    ]
    candidate_fields = [
        "case_id", "ct_path", "annotation_folder", "ct_shape", "spacing",
        "head_evidence", "thorax_evidence", "central_airway_evidence",
        "candidate_target_group", "eligibility", "reason",
        "brain_foreground_voxels", "skull_foreground_voxels",
        "eyeball_left_foreground_voxels", "eyeball_right_foreground_voxels",
        "lung_left_foreground_voxels", "lung_right_foreground_voxels",
        "heart_foreground_voxels", "aorta_foreground_voxels",
        "central_airway_foreground_voxels",
    ]
    _write_csv(output_root / "candidate_inventory.csv", inventory_rows, inventory_fields)
    _write_csv(output_root / "head_candidates.csv", head_rows, candidate_fields)
    _write_csv(output_root / "airway_candidates.csv", airway_rows, candidate_fields)
    _write_csv(output_root / "thorax_candidates.csv", thorax_rows, candidate_fields)
    summary = {
        "status": "completed",
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "case_count": len(case_ids),
        "head_candidate_count": len(head_rows),
        "airway_candidate_count": len(airway_rows),
        "thorax_candidate_count": len(thorax_rows),
        "elapsed_sec": round(time.time() - started, 3),
        "progress_every": progress_every,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only Task2 target-specific FOV candidate search over AbdomenAtlasPro.")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, type=Path)
    parser.add_argument("--mask-root", default=DEFAULT_MASK_ROOT, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--progress-every", default=250, type=int)
    args = parser.parse_args()
    summary = search_fov_candidates(
        image_root=args.image_root,
        mask_root=args.mask_root,
        output_root=args.output_root,
        case_list=args.case_list,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
