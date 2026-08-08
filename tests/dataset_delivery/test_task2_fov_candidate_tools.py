from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def _save(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fov_candidate_search_outputs_progress_and_candidate_reports(tmp_path: Path, capsys):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    shape = (30, 30, 30)
    small = np.zeros(shape, dtype=np.uint8)
    small[1:20, 1:20, 1:20] = 1
    large = np.ones(shape, dtype=np.uint8)
    for case_id in ["case_head", "case_thorax"]:
        _save(np.zeros(shape, dtype=np.int16), image_root / case_id / "ct.nii.gz")
    _save(large, mask_root / "case_head" / "segmentations" / "brain.nii.gz")
    _save(large, mask_root / "case_head" / "segmentations" / "skull.nii.gz")
    _save(small, mask_root / "case_head" / "segmentations" / "eyeball_left.nii.gz")
    _save(large, mask_root / "case_thorax" / "segmentations" / "lung_left.nii.gz")
    _save(large, mask_root / "case_thorax" / "segmentations" / "lung_right.nii.gz")
    _save(small, mask_root / "case_thorax" / "segmentations" / "heart.nii.gz")

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=1,
    )

    captured = capsys.readouterr().out
    assert "[FOV_SEARCH] scanned=1" in captured
    assert summary["case_count"] == 2
    assert _read_rows(tmp_path / "search" / "candidate_inventory.csv")
    assert _read_rows(tmp_path / "search" / "head_candidates.csv")[0]["case_id"] == "case_head"
    assert _read_rows(tmp_path / "search" / "thorax_candidates.csv")[0]["case_id"] == "case_thorax"


def test_append_qualified_cases_is_append_only_and_duplicate_safe(tmp_path: Path):
    from tools.dataset_delivery.task2_append_qualified_cases import append_qualified_cases

    base = tmp_path / "base.csv"
    base.write_text(
        "index,case_id,ct_path,annotation_folder\n"
        "0,case_existing,/base/ct.nii.gz,/base/ref\n",
        encoding="utf-8",
    )
    candidates = tmp_path / "head_candidates.csv"
    candidates.write_text(
        "case_id,ct_path,annotation_folder,candidate_target_group,reason\n"
        "case_existing,/dup/ct.nii.gz,/dup/ref,head,strong_head_evidence\n"
        "case_new,/new/ct.nii.gz,/new/ref,head,strong_head_evidence\n",
        encoding="utf-8",
    )

    audit = append_qualified_cases(
        base_manifest=base,
        candidate_report=candidates,
        output_manifest=tmp_path / "out.csv",
        target_group="head",
    )

    rows = _read_rows(tmp_path / "out.csv")
    assert audit["appended_case_ids"] == ["case_new"]
    assert rows[0]["case_id"] == "case_existing"
    assert rows[1]["index"] == "1"
    assert rows[1]["case_id"] == "case_new"
