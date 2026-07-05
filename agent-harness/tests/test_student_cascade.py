from __future__ import annotations

import json
import csv
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from cli_anything.medai.core.student_cascade import cascade_case, crop_image, paste_mask, roi_bounds

ROOT = Path(__file__).resolve().parents[2]


def save(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def test_crop_affine_and_paste_round_trip() -> None:
    affine = np.array([[2, 0, 0, -10], [0, 3, 0, 20], [0, 0, 4, 5], [0, 0, 0, 1]], dtype=float)
    data = np.arange(20 * 18 * 16, dtype=np.int32).reshape(20, 18, 16)
    image = nib.Nifti1Image(data, affine)
    roi = np.zeros(data.shape, bool)
    roi[3:10, 4:12, 2:9] = True
    bounds = roi_bounds(roi)
    cropped = crop_image(image, bounds)
    assert cropped.shape == (7, 8, 7)
    assert np.allclose(cropped.affine[:3, 3], nib.affines.apply_affine(affine, [3, 4, 2]))
    predicted = nib.Nifti1Image(np.ones(cropped.shape, np.uint8), cropped.affine)
    pasted = paste_mask(predicted, image, bounds)
    assert pasted.sum() == 7 * 8 * 7
    assert pasted[3:10, 4:12, 2:9].all()


def test_cascade_blocks_missing_parent_without_raw_fallback(tmp_path: Path) -> None:
    ct = tmp_path / "ct.nii.gz"
    save(ct, np.zeros((16, 16, 16), np.int16), np.eye(4))
    called = False

    def inference(_ct: Path, _organ: str, _out: Path) -> dict:
        nonlocal called
        called = True
        return {"status": "success"}

    result = cascade_case(
        case_id="case1",
        ct_path=ct,
        output_root=tmp_path / "cascade",
        inference=inference,
        teacher_parent_roots=[tmp_path / "missing_teacher"],
        student_parent_root=tmp_path / "raw_student",
        student_parent_allowlist=None,
        taxonomy_path=ROOT / "configs/organ_taxonomy.json",
        policy_path=ROOT / "configs/organ_postprocess_policy.yaml",
        organs=["pancreatic_duct"],
    )
    assert not called
    assert result["rows"][0]["status"] == "blocked"
    assert result["rows"][0]["reason"] == "missing_or_unreliable_parent"


def test_cascade_preserves_multiple_vessel_branches(tmp_path: Path) -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    ct = tmp_path / "ct.nii.gz"
    save(ct, np.zeros((30, 30, 30), np.int16), affine)
    teacher = tmp_path / "teacher" / "case1"
    liver = np.zeros((30, 30, 30), np.uint8)
    liver[5:25, 5:25, 5:25] = 1
    save(teacher / "liver.nii.gz", liver, affine)

    def inference(cropped_ct: Path, organ: str, out: Path) -> dict:
        image = nib.load(str(cropped_ct))
        pred = np.zeros(image.shape, np.uint8)
        pred[4:7, 4:7, 4:7] = 1
        pred[-7:-4, -7:-4, -7:-4] = 1
        save(out / f"{organ}.nii.gz", pred, image.affine)
        return {"status": "success"}

    result = cascade_case(
        case_id="case1",
        ct_path=ct,
        output_root=tmp_path / "cascade",
        inference=inference,
        teacher_parent_roots=[tmp_path / "teacher"],
        student_parent_root=None,
        student_parent_allowlist=None,
        taxonomy_path=ROOT / "configs/organ_taxonomy.json",
        policy_path=ROOT / "configs/organ_postprocess_policy.yaml",
        organs=["liver_portal_vein"],
    )
    row = result["rows"][0]
    assert row["status"] == "success"
    assert row["component_count_after"] == 2
    assert row["outside_roi_voxels"] == 0
    audit = json.loads((tmp_path / "cascade" / "case1" / "student_cascade_audit.json").read_text())
    assert audit["confidence_availability"] == "unavailable_binary_backend"


def test_allowlist_is_organ_level_and_fail_closed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    reference = tmp_path / "reference"
    case_list = tmp_path / "cases.csv"
    with case_list.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path"])
        writer.writeheader()
        for index in range(10):
            case = f"case{index}"
            writer.writerow({"case_id": case, "ct_path": "unused.nii.gz"})
            mask = np.zeros((8, 8, 8), np.uint8)
            mask[2:6, 2:6, 2:6] = 1
            save(raw / case / "liver.nii.gz", mask, np.eye(4))
            save(reference / case / "liver.nii.gz", mask, np.eye(4))
            save(raw / case / "abdominal_cavity.nii.gz", mask, np.eye(4))
            save(reference / case / "abdominal_cavity.nii.gz", mask, np.eye(4))
    gate = tmp_path / "gate"
    post = tmp_path / "post"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_student_round2_allowlist.py"),
            "--raw-root", str(raw),
            "--containment-root", str(tmp_path / "containment"),
            "--cascade-root", str(tmp_path / "cascade"),
            "--reference-root", str(reference),
            "--case-list", str(case_list),
            "--output-dir", str(gate),
            "--postprocessed-root", str(post),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((gate / "student_round2_organ_allowlist.json").read_text())
    decisions = {row["organ"]: row for row in report["organs"]}
    assert report["schema_version"] == 3
    assert decisions["liver"]["decision"] == "allow"
    assert decisions["liver"]["round2_route"] == "student_competition"
    assert decisions["abdominal_cavity"]["decision"] == "block"
    assert decisions["abdominal_cavity"]["round2_route"] == "teacher_or_fov_negative"
    assert (post / "case0" / "liver.nii.gz").is_file()
    assert not (post / "case0" / "abdominal_cavity.nii.gz").exists()
