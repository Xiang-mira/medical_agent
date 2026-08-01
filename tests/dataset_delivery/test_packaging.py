from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.dataset_delivery.delivery_lib import DeliveryError, package_dataset_overlay, sha256_file


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_packaging_only_copies_validated_masks_and_sha256(tmp_path: Path) -> None:
    src1 = tmp_path / "run/cases/BDMAP_0000001/raw/annotation_versions/BDMAP_0000001/updated/target_a.nii.gz"
    src2 = tmp_path / "run/cases/BDMAP_0000001/raw/annotation_versions/BDMAP_0000001/updated/target_b.nii.gz"
    src1.parent.mkdir(parents=True)
    src1.write_text("mask-a", encoding="utf-8")
    src2.write_text("mask-b", encoding="utf-8")
    ref = tmp_path / "official/BDMAP_0000001/segmentations"
    ref.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, [{"index": "0", "case_id": "BDMAP_0000001", "image_path": "/ct.nii.gz", "reference_mask_dir": str(ref)}])
    report = tmp_path / "validation.csv"
    write_csv(report, [
        {"case_id": "BDMAP_0000001", "target_name": "target_a", "mask_path": str(src1), "status": "passed", "reason": "validated"},
        {"case_id": "BDMAP_0000001", "target_name": "target_b", "mask_path": str(src2), "status": "failed", "reason": "mask_missing"},
    ])
    out = tmp_path / "delivery"
    package_dataset_overlay(tmp_path / "run", manifest, report, out)
    copied = out / "BDMAP_0000001/segmentations/target_a.nii.gz"
    assert copied.exists()
    assert not (out / "BDMAP_0000001/segmentations/target_b.nii.gz").exists()
    assert not copied.is_symlink()
    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    assert rows[0]["sha256"] == sha256_file(copied)


def test_packaging_conflict_stops_by_default(tmp_path: Path) -> None:
    src = tmp_path / "mask.nii.gz"
    src.write_text("mask", encoding="utf-8")
    ref = tmp_path / "official/BDMAP_0000001/segmentations"
    ref.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, [{"index": "0", "case_id": "BDMAP_0000001", "image_path": "/ct.nii.gz", "reference_mask_dir": str(ref)}])
    report = tmp_path / "validation.csv"
    write_csv(report, [
        {"case_id": "BDMAP_0000001", "target_name": "target_a", "mask_path": str(src), "status": "passed", "reason": "validated"},
        {"case_id": "BDMAP_0000001", "target_name": "target_a", "mask_path": str(src), "status": "passed", "reason": "validated"},
    ])
    with pytest.raises(DeliveryError):
        package_dataset_overlay(tmp_path / "run", manifest, report, tmp_path / "delivery")
