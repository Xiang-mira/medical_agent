from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def _save(array: np.ndarray, path: Path, *, affine: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4) if affine is None else affine), str(path))
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _case_roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "images", tmp_path / "masks"


def _write_ct(image_root: Path, case_id: str, shape: tuple[int, int, int] = (40, 40, 40)) -> None:
    _save(np.zeros(shape, dtype=np.int16), image_root / case_id / "ct.nii.gz")


def _mask(mask_root: Path, case_id: str, name: str, voxels: int | None = None, *, mismatch: bool = False) -> None:
    shape = (40, 40, 40)
    arr = np.zeros(shape, dtype=np.uint8)
    if voxels is None:
        arr[:] = 1
    elif voxels > 0:
        arr.flat[:voxels] = 1
    affine = np.diag([2.0, 1.0, 1.0, 1.0]) if mismatch else None
    _save(arr, mask_root / case_id / "segmentations" / f"{name}.nii.gz", affine=affine)


def _head_case(image_root: Path, mask_root: Path, case_id: str) -> None:
    _write_ct(image_root, case_id)
    _mask(mask_root, case_id, "brain", 12000)
    _mask(mask_root, case_id, "skull", 1200)
    _mask(mask_root, case_id, "eyeball_left", 1)
    _mask(mask_root, case_id, "eyeball_right", 1)


def _airway_case(image_root: Path, mask_root: Path, case_id: str, *, landmark: str = "airway") -> None:
    _write_ct(image_root, case_id)
    _mask(mask_root, case_id, "lung_left", 12000)
    _mask(mask_root, case_id, "lung_right", 12000)
    _mask(mask_root, case_id, landmark, 1)


def _pulmonary_case(image_root: Path, mask_root: Path, case_id: str) -> None:
    _write_ct(image_root, case_id)
    _mask(mask_root, case_id, "lung_left", 12000)
    _mask(mask_root, case_id, "lung_right", 12000)
    _mask(mask_root, case_id, "heart", 12000)
    _mask(mask_root, case_id, "aorta", 1200)


def test_target_specific_search_early_stops_when_all_targets_have_two_cases(tmp_path: Path, capsys):
    from tools.dataset_delivery.task2_fov_candidate_search import TARGETS, search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    for case_id in ("case_001", "case_002"):
        _head_case(image_root, mask_root, case_id)
        _airway_case(image_root, mask_root, case_id, landmark="bronchus")
    _head_case(image_root, mask_root, "case_003")

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=1,
        checkpoint_every=1,
    )

    assert summary["early_stopped"] is True
    assert summary["all_targets_covered"] is True
    assert summary["scanned_count"] == 2
    out = capsys.readouterr().out
    assert set(summary["target_counts"]) == set(TARGETS)
    assert all(count == 2 for count in summary["target_counts"].values())
    airway_rows = [row for row in _read_rows(tmp_path / "search" / "target_case_coverage.csv") if row["target_name"] == "airway_tree"]
    assert "bronchus" in airway_rows[0]["landmark_voxels"]
    assert "[FOV_SEARCH] scanned=1/2000" in out


def test_first2000_limit_stops_and_reports_uncovered_targets(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    for case_id in ("case_001", "case_002", "case_003", "case_004"):
        _write_ct(image_root, case_id)
        (mask_root / case_id / "segmentations").mkdir(parents=True)

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        scan_limit=3,
        progress_every=0,
        preseed_pulmonary=False,
    )

    assert summary["scanned_count"] == 3
    assert summary["early_stopped"] is False
    assert summary["all_targets_covered"] is False
    assert "brain_ventricle" in summary["uncovered_targets"]


def test_each_target_gets_two_cases_without_requiring_same_case_cover_everything(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    _head_case(image_root, mask_root, "case_head_a")
    _head_case(image_root, mask_root, "case_head_b")
    _airway_case(image_root, mask_root, "case_airway_a")
    _airway_case(image_root, mask_root, "case_airway_b")

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=0,
    )
    rows = _read_rows(tmp_path / "search" / "target_case_coverage.csv")
    target_to_cases: dict[str, set[str]] = {}
    for row in rows:
        target_to_cases.setdefault(row["target_name"], set()).add(row["case_id"])

    assert summary["all_targets_covered"] is True
    assert target_to_cases["brain_ventricle"] == {"case_head_a", "case_head_b"}
    assert target_to_cases["airway_wall"] == {"case_airway_a", "case_airway_b"}
    assert target_to_cases["lung_pulmonary_arteries"] == {"BDMAP_00000424", "BDMAP_00078156"}


def test_bronchus_only_central_airway_case_is_not_skipped(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    _airway_case(image_root, mask_root, "case_bronchus_a", landmark="bronchus")
    _airway_case(image_root, mask_root, "case_bronchus_b", landmark="bronchus")

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=0,
        preseed_pulmonary=False,
    )

    assert summary["target_counts"]["airway_tree"] == 2
    assert summary["target_counts"]["airway_wall"] == 2
    assert summary["target_counts"]["lung_pulmonary_arteries"] == 0


def test_pulmonary_and_central_airway_eligibility_are_separate(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    _pulmonary_case(image_root, mask_root, "case_pulm_a")
    _pulmonary_case(image_root, mask_root, "case_pulm_b")

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=0,
        preseed_pulmonary=False,
    )

    assert summary["target_counts"]["lung_pulmonary_arteries"] == 2
    assert summary["target_counts"]["lung_pulmonary_veins"] == 2
    assert summary["target_counts"]["airway_tree"] == 0
    assert summary["target_counts"]["airway_wall"] == 0


def test_known_append_cases_are_preseeded_only_for_pulmonary_targets(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=0,
    )

    assert summary["target_counts"]["lung_pulmonary_arteries"] == 2
    assert summary["target_counts"]["lung_pulmonary_veins"] == 2
    assert summary["target_counts"]["brain_ventricle"] == 0
    assert summary["target_counts"]["airway_wall"] == 0
    union = _read_rows(tmp_path / "search" / "selected_case_union.csv")
    assert {row["case_id"] for row in union} == {"BDMAP_00000424", "BDMAP_00078156"}


def test_checkpoint_resume_continues_without_restarting_from_zero(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    _head_case(image_root, mask_root, "case_001")
    _head_case(image_root, mask_root, "case_002")

    first = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        scan_limit=1,
        progress_every=0,
        checkpoint_every=1,
        preseed_pulmonary=False,
    )
    resumed = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        scan_limit=2,
        progress_every=0,
        checkpoint_every=1,
        preseed_pulmonary=False,
    )

    assert first["scanned_count"] == 1
    assert resumed["resume_used"] is True
    assert resumed["scanned_count"] == 2
    assert resumed["target_counts"]["brain_ventricle"] == 2


def test_sorted_order_is_deterministic_and_geometry_mismatch_cannot_qualify(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    _head_case(image_root, mask_root, "case_b")
    _head_case(image_root, mask_root, "case_a")
    _write_ct(image_root, "case_bad")
    _mask(mask_root, "case_bad", "brain", 12000, mismatch=True)
    _mask(mask_root, "case_bad", "skull", 1200, mismatch=True)

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=0,
        preseed_pulmonary=False,
    )
    rows = [row for row in _read_rows(tmp_path / "search" / "target_case_coverage.csv") if row["target_name"] == "brain_ventricle"]

    assert summary["target_counts"]["brain_ventricle"] == 2
    assert [row["case_id"] for row in rows] == ["case_a", "case_b"]
    assert "case_bad" not in {row["case_id"] for row in rows}


def test_append_qualified_cases_is_append_only_and_duplicate_safe(tmp_path: Path):
    from tools.dataset_delivery.task2_append_qualified_cases import append_qualified_cases

    base = tmp_path / "base.csv"
    base.write_text(
        "index,case_id,ct_path,annotation_folder\n"
        "0,case_existing,/base/ct.nii.gz,/base/ref\n",
        encoding="utf-8",
    )
    candidates = tmp_path / "selected_case_union.csv"
    candidates.write_text(
        "case_id,ct_path,annotation_folder,targets,evidence_types\n"
        "case_existing,/dup/ct.nii.gz,/dup/ref,brain_ventricle,landmark_mask_scan\n"
        "case_new,/new/ct.nii.gz,/new/ref,brain_ventricle,landmark_mask_scan\n",
        encoding="utf-8",
    )

    audit = append_qualified_cases(
        base_manifest=base,
        candidate_report=candidates,
        output_manifest=tmp_path / "out.csv",
        target_group="brain_ventricle",
    )

    rows = _read_rows(tmp_path / "out.csv")
    assert audit["appended_case_ids"] == ["case_new"]
    assert rows[0]["case_id"] == "case_existing"
    assert rows[1]["index"] == "1"
    assert rows[1]["case_id"] == "case_new"
