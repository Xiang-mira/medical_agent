from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.student_postprocess import containment_rule_for_organ, process_mask, process_student_root


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _save(array: np.ndarray, path: Path, spacing=(1.0, 1.0, 1.0)) -> Path:
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(array.astype(np.uint8), affine), str(path))
    return path


@pytest.fixture()
def policy() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "configs/organ_postprocess_policy.yaml").read_text())


@pytest.fixture()
def taxonomy() -> dict:
    import json
    return json.loads((ROOT / "configs/organ_taxonomy.json").read_text())


def test_policy_infers_taxonomy_parent_rule(policy, taxonomy):
    rule = containment_rule_for_organ("pancreatic_duct", taxonomy, policy)
    assert rule.enabled is True
    assert rule.parents == ("pancreas",)
    assert rule.margin_mm == pytest.approx(20.0)

    colon = containment_rule_for_organ("colon", taxonomy, policy)
    assert colon.enabled is True
    assert colon.parents == ("abdominal_cavity",)



def test_disabled_or_missing_policy_never_clips(taxonomy):
    missing_policy_rule = containment_rule_for_organ("pancreatic_duct", taxonomy, {})
    assert missing_policy_rule.enabled is False
    assert missing_policy_rule.source == "student_containment_disabled"

    disabled_policy_rule = containment_rule_for_organ("pancreatic_duct", taxonomy, {"student_containment": {"enabled": False}})
    assert disabled_policy_rule.enabled is False
    assert disabled_policy_rule.source == "student_containment_disabled"

def test_roi_outside_blob_removed_and_inside_kept(tmp_path: Path, policy, taxonomy):
    case = "case001"
    raw_root = tmp_path / "student" / case
    parent_root = tmp_path / "teacher" / case / "updated"
    out = tmp_path / "post" / case / "pancreatic_duct.nii.gz"
    raw_root.mkdir(parents=True)
    parent_root.mkdir(parents=True)
    parent = np.zeros((50, 50, 50), dtype=np.uint8)
    parent[10:15, 10:15, 10:15] = 1
    student = np.zeros_like(parent)
    student[12:14, 12:14, 12:14] = 1
    student[42:44, 42:44, 42:44] = 1
    _save(parent, parent_root / "pancreas.nii.gz")
    src = _save(student, raw_root / "pancreatic_duct.nii.gz")
    row = process_mask(
        case_id=case,
        organ="pancreatic_duct",
        student_mask_path=src,
        output_mask_path=out,
        parent_roots=[tmp_path / "teacher"],
        taxonomy=taxonomy,
        policy=policy,
        roi_output_path=tmp_path / "roi.nii.gz",
    )
    arr = np.asanyarray(nib.load(str(out)).dataobj) > 0
    assert arr[12:14, 12:14, 12:14].sum() == 8
    assert arr[42:44, 42:44, 42:44].sum() == 0
    assert row["false_positive_voxels_removed"] >= 8
    assert Path(tmp_path / "roi.nii.gz").exists()


def test_dilation_uses_physical_spacing(tmp_path: Path, policy, taxonomy):
    case = "case001"
    parent_root = tmp_path / "teacher" / case / "updated"
    parent_root.mkdir(parents=True)
    parent = np.zeros((20, 20, 20), dtype=np.uint8)
    parent[10, 10, 10] = 1
    student = np.zeros_like(parent)
    # 6 voxels away on axis with 2mm spacing = 12mm, inside 20mm margin.
    student[16, 10, 10] = 1
    _save(parent, parent_root / "pancreas.nii.gz", spacing=(2, 1, 1))
    src = _save(student, tmp_path / "pancreatic_duct.nii.gz", spacing=(2, 1, 1))
    out = tmp_path / "out.nii.gz"
    process_mask(
        case_id=case,
        organ="pancreatic_duct",
        student_mask_path=src,
        output_mask_path=out,
        parent_roots=[tmp_path / "teacher"],
        taxonomy=taxonomy,
        policy=policy,
    )
    assert int(np.asanyarray(nib.load(str(out)).dataobj).sum()) == 1


def test_missing_parent_copies_raw_with_warning(tmp_path: Path, policy, taxonomy):
    student = np.zeros((10, 10, 10), dtype=np.uint8); student[1:3, 1:3, 1:3] = 1
    src = _save(student, tmp_path / "pancreatic_duct.nii.gz")
    out = tmp_path / "out.nii.gz"
    row = process_mask(
        case_id="case001",
        organ="pancreatic_duct",
        student_mask_path=src,
        output_mask_path=out,
        parent_roots=[tmp_path / "missing"],
        taxonomy=taxonomy,
        policy=policy,
    )
    assert row["status"] == "warning_missing_parent_roi_copied_raw"
    assert int(np.asanyarray(nib.load(str(out)).dataobj).sum()) == int(student.sum())


def test_large_organ_is_not_clipped(tmp_path: Path, policy, taxonomy):
    liver = np.zeros((20, 20, 20), dtype=np.uint8)
    liver[1:3, 1:3, 1:3] = 1
    liver[15:17, 15:17, 15:17] = 1
    src = _save(liver, tmp_path / "liver.nii.gz")
    out = tmp_path / "out.nii.gz"
    row = process_mask(
        case_id="case001",
        organ="liver",
        student_mask_path=src,
        output_mask_path=out,
        parent_roots=[tmp_path],
        taxonomy=taxonomy,
        policy=policy,
    )
    assert row["status"] == "copied_no_containment_rule"
    assert int(np.asanyarray(nib.load(str(out)).dataobj).sum()) == int(liver.sum())


def test_vessel_preserves_multiple_non_tiny_components_inside_roi(tmp_path: Path, policy, taxonomy):
    case = "case001"
    parent_root = tmp_path / "teacher" / case / "updated"
    parent_root.mkdir(parents=True)
    parent = np.zeros((40, 40, 40), dtype=np.uint8)
    parent[5:30, 5:30, 5:30] = 1
    vessel = np.zeros_like(parent)
    vessel[8:10, 8:10, 8:10] = 1
    vessel[20:22, 20:22, 20:22] = 1
    _save(parent, parent_root / "liver.nii.gz")
    src = _save(vessel, tmp_path / "liver_portal_vein.nii.gz")
    out = tmp_path / "out.nii.gz"
    row = process_mask(
        case_id=case,
        organ="liver_portal_vein",
        student_mask_path=src,
        output_mask_path=out,
        parent_roots=[tmp_path / "teacher"],
        taxonomy=taxonomy,
        policy=policy,
    )
    assert int(np.asanyarray(nib.load(str(out)).dataobj).sum()) == int(vessel.sum())
    assert row["connected_component_count_after"] == 2


def test_process_root_writes_summary_and_csv(tmp_path: Path):
    case = "case001"
    raw_case = tmp_path / "raw" / case
    parent_case = tmp_path / "teacher" / case / "updated"
    raw_case.mkdir(parents=True); parent_case.mkdir(parents=True)
    parent = np.zeros((12, 12, 12), dtype=np.uint8); parent[2:8, 2:8, 2:8] = 1
    duct = np.zeros_like(parent); duct[3:5, 3:5, 3:5] = 1
    _save(parent, parent_case / "pancreas.nii.gz")
    _save(duct, raw_case / "pancreatic_duct.nii.gz")
    summary = process_student_root(
        input_root=tmp_path / "raw",
        output_root=tmp_path / "post",
        parent_roots=[tmp_path / "teacher"],
        taxonomy_path=ROOT / "configs/organ_taxonomy.json",
        policy_path=ROOT / "configs/organ_postprocess_policy.yaml",
        organs=["pancreatic_duct"],
    )
    assert summary["processed_masks"] == 1
    assert (tmp_path / "post" / "student_containment_postprocess_per_mask.csv").exists()


def test_before_after_evaluator_outputs_required_fields(tmp_path: Path):
    case = "case001"
    for root in ["raw", "post", "teacher"]:
        (tmp_path / root / case).mkdir(parents=True)
    ref = np.zeros((10, 10, 10), dtype=np.uint8); ref[2:5, 2:5, 2:5] = 1
    raw = ref.copy(); raw[8:10, 8:10, 8:10] = 1
    post = ref.copy()
    _save(raw, tmp_path / "raw" / case / "pancreatic_duct.nii.gz")
    _save(post, tmp_path / "post" / case / "pancreatic_duct.nii.gz")
    _save(ref, tmp_path / "teacher" / case / "pancreatic_duct.nii.gz")
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/evaluate_student_postprocess.py"),
            "--raw-root", str(tmp_path / "raw"),
            "--post-root", str(tmp_path / "post"),
            "--teacher-root", str(tmp_path / "teacher"),
            "--output-dir", str(tmp_path / "eval"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "student_postprocess_before_after_evaluation" in completed.stdout
    rows = list(csv.DictReader((tmp_path / "eval" / "student_postprocess_before_after_per_case_organ.csv").open()))
    row = rows[0]
    for field in [
        "before_postprocessing_dice", "after_postprocessing_dice",
        "before_volume_ratio", "after_volume_ratio",
        "false_positive_voxels_removed",
        "connected_component_count_before", "connected_component_count_after",
        "largest_component_ratio_before", "largest_component_ratio_after",
    ]:
        assert field in row
    assert int(row["false_positive_voxels_removed"]) == 8
    assert float(row["after_postprocessing_dice"]) > float(row["before_postprocessing_dice"])


def test_before_after_evaluator_writes_empty_summary(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "post").mkdir()
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/evaluate_student_postprocess.py"),
            "--raw-root", str(tmp_path / "raw"),
            "--post-root", str(tmp_path / "post"),
            "--output-dir", str(tmp_path / "eval"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = tmp_path / "eval" / "student_postprocess_before_after_organ_summary.csv"
    assert summary.exists()
    header = summary.read_text(encoding="utf-8").splitlines()[0]
    assert "before_postprocessing_dice" in header


def test_before_after_evaluator_visual_copy_skips_empty_paths(tmp_path: Path):
    evaluator = load_script("evaluate_student_postprocess.py")
    raw = _save(np.zeros((3, 3, 3), dtype=np.uint8), tmp_path / "raw.nii.gz")
    post = _save(np.zeros((3, 3, 3), dtype=np.uint8), tmp_path / "post.nii.gz")

    evaluator.copy_review_masks(
        {
            "case_id": "case001",
            "organ": "brain",
            "raw_path": str(raw),
            "post_path": str(post),
            "reference_path": "",
            "parent_roi_path": "",
        },
        tmp_path / "visuals",
    )

    out = tmp_path / "visuals" / "case001" / "brain"
    assert (out / "raw_student.nii.gz").exists()
    assert (out / "postprocessed_student.nii.gz").exists()
    assert not (out / "reference.nii.gz").exists()
    assert not (out / "parent_organ_roi.nii.gz").exists()
