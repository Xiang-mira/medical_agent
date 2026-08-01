from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.dataset_delivery.delivery_lib import DeliveryError, run_teacher_case, validate_generated_masks

nib = pytest.importorskip("nibabel")
np = pytest.importorskip("numpy")


def write_img(path: Path, shape=(3, 3, 3), affine=None, value=1) -> None:
    affine = np.eye(4) if affine is None else affine
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.full(shape, value, dtype=np.uint8), affine), str(path))


def write_manifest(path: Path, ct: Path, ref: Path, case_id: str = "BDMAP_0000001") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "case_id", "image_path", "reference_mask_dir"])
        writer.writerow([0, case_id, ct, ref])
        for i in range(1, 100):
            writer.writerow([i, f"BDMAP_{i:07d}", ct, ref])


def write_gap_and_tax(tmp_path: Path, organ: str = "target_a") -> tuple[Path, Path]:
    taxonomy = tmp_path / "taxonomy.json"
    names = [organ] + [f"organ_{i:03d}" for i in range(372)]
    taxonomy.write_text(json.dumps({"target_organs": names}), encoding="utf-8")
    gap = tmp_path / "gap.csv"
    gap.write_text(f"target_name,dataset_name,resolution,status,reason,notes\n{organ},,generate,confirmed,test,\n", encoding="utf-8")
    return gap, taxonomy


def validation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_shape_mismatch_detected(tmp_path: Path) -> None:
    ct = tmp_path / "ct.nii.gz"
    ref = tmp_path / "ref"
    write_img(ct, shape=(3, 3, 3))
    write_img(tmp_path / "run/cases/BDMAP_0000001/raw/annotation_versions/BDMAP_0000001/updated/target_a.nii.gz", shape=(2, 3, 3))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ct, ref)
    gap, taxonomy = write_gap_and_tax(tmp_path)
    validate_generated_masks(tmp_path / "run", manifest, gap, taxonomy, tmp_path / "out")
    rows = validation_rows(tmp_path / "out/validation_report.csv")
    assert rows[0]["reason"] == "geometry_mismatch"


def test_affine_mismatch_detected(tmp_path: Path) -> None:
    ct = tmp_path / "ct.nii.gz"
    ref = tmp_path / "ref"
    write_img(ct)
    affine = np.eye(4)
    affine[0, 3] = 5
    write_img(tmp_path / "run/cases/BDMAP_0000001/raw/annotation_versions/BDMAP_0000001/updated/target_a.nii.gz", affine=affine)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ct, ref)
    gap, taxonomy = write_gap_and_tax(tmp_path)
    validate_generated_masks(tmp_path / "run", manifest, gap, taxonomy, tmp_path / "out")
    assert validation_rows(tmp_path / "out/validation_report.csv")[0]["reason"] == "geometry_mismatch"


def test_all_zero_mask_is_recorded_not_deleted(tmp_path: Path) -> None:
    ct = tmp_path / "ct.nii.gz"
    ref = tmp_path / "ref"
    mask = tmp_path / "run/cases/BDMAP_0000001/raw/annotation_versions/BDMAP_0000001/updated/target_a.nii.gz"
    write_img(ct)
    write_img(mask, value=0)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ct, ref)
    gap, taxonomy = write_gap_and_tax(tmp_path)
    validate_generated_masks(tmp_path / "run", manifest, gap, taxonomy, tmp_path / "out")
    row = validation_rows(tmp_path / "out/validation_report.csv")[0]
    assert row["status"] == "passed"
    assert row["is_all_zero"] == "True"
    assert mask.exists()


def test_unreadable_nifti_reported(tmp_path: Path) -> None:
    ct = tmp_path / "ct.nii.gz"
    ref = tmp_path / "ref"
    mask = tmp_path / "run/cases/BDMAP_0000001/raw/annotation_versions/BDMAP_0000001/updated/target_a.nii.gz"
    write_img(ct)
    mask.parent.mkdir(parents=True)
    mask.write_text("not nifti", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ct, ref)
    gap, taxonomy = write_gap_and_tax(tmp_path)
    validate_generated_masks(tmp_path / "run", manifest, gap, taxonomy, tmp_path / "out")
    assert validation_rows(tmp_path / "out/validation_report.csv")[0]["reason"].startswith("mask_unreadable:")


def test_no_real_nifti_output_not_validated(tmp_path: Path) -> None:
    script = tmp_path / "teacher.py"
    script.write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    ct = tmp_path / "ct.nii.gz"
    ref = tmp_path / "ref"
    write_img(ct)
    ref.mkdir()
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ct, ref)
    gap, taxonomy = write_gap_and_tax(tmp_path)
    with pytest.raises(DeliveryError):
        run_teacher_case(manifest, gap, taxonomy, tmp_path / "run", case_index=0, code_root=tmp_path, python="python", teacher_entry=script)
    status = json.loads((tmp_path / "run/cases/BDMAP_0000001/status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"

