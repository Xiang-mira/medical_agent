from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.dataset_delivery.delivery_lib import DeliveryError, validate_100case_manifest
from tools.dataset_delivery.delivery_lib import validate_gap_resolution


def write_manifest(path: Path, rows: int = 100, duplicate: bool = False, bad_index: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "case_id", "image_path", "reference_mask_dir"])
        for i in range(rows):
            case = "BDMAP_DUP" if duplicate and i in {0, 1} else f"BDMAP_{i:07d}"
            idx = 42 if bad_index and i == 0 else i
            writer.writerow([idx, case, "/missing/ct.nii.gz", "/missing/segmentations"])


def test_100_row_manifest_passes_without_exists_check(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(path)
    assert validate_100case_manifest(path, check_exists=False)["status"] == "success"


@pytest.mark.parametrize("rows", [99, 101])
def test_99_or_101_rows_fail(tmp_path: Path, rows: int) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(path, rows=rows)
    with pytest.raises(DeliveryError):
        validate_100case_manifest(path, check_exists=False)


def test_duplicate_case_id_fails(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(path, duplicate=True)
    with pytest.raises(DeliveryError):
        validate_100case_manifest(path, check_exists=False)


def test_index_must_be_zero_to_99(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(path, bad_index=True)
    with pytest.raises(DeliveryError):
        validate_100case_manifest(path, check_exists=False)


def test_generate_and_rename_targets_cannot_overlap(tmp_path: Path) -> None:
    taxonomy = tmp_path / "taxonomy.json"
    names = ["target_a"] + [f"organ_{i:03d}" for i in range(372)]
    import json

    taxonomy.write_text(json.dumps({"target_organs": names}), encoding="utf-8")
    gap = tmp_path / "gap.csv"
    gap.write_text("target_name,dataset_name,resolution,status,reason,notes\ntarget_a,,generate,confirmed,test,\n", encoding="utf-8")
    rename = tmp_path / "rename.csv"
    rename.write_text("source_name,target_name,status,reason,notes\nsource_a,target_a,confirmed,test,\n", encoding="utf-8")
    with pytest.raises(DeliveryError):
        validate_gap_resolution(gap, taxonomy, rename)
