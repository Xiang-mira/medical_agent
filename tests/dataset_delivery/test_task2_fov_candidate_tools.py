from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


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


def _corrupt_ct_case(image_root: Path, mask_root: Path, case_id: str) -> None:
    ct = image_root / case_id / "ct.nii.gz"
    ct.parent.mkdir(parents=True, exist_ok=True)
    ct.write_text("not a nifti\n", encoding="utf-8")
    (mask_root / case_id / "segmentations").mkdir(parents=True)


def _mkdir_roots(image_root: Path, mask_root: Path) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)


def test_nibabel_unavailable_runtime_preflight_fails_before_search_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import tools.dataset_delivery.task2_fov_candidate_search as search

    image_root, mask_root = _case_roots(tmp_path)
    _mkdir_roots(image_root, mask_root)

    def fail_if_called(_mask_root: Path) -> list[str]:
        raise AssertionError("search loop should not enumerate case directories when dependency preflight fails")

    real_import = search.importlib.import_module

    def fake_import(name: str):
        if name == "nibabel":
            raise ModuleNotFoundError("No module named 'nibabel'")
        return real_import(name)

    monkeypatch.setattr(search.importlib, "import_module", fake_import)
    monkeypatch.setattr(search, "_case_ids_from_mask_root", fail_if_called)

    summary = search.search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=0,
        preseed_pulmonary=False,
    )

    assert summary["status"] == "RUNTIME_DEPENDENCY_MISSING"
    assert summary["dependency"] == "nibabel"
    assert summary["directories_examined"] == 0
    assert summary["scanned_count"] == 0
    assert summary["runtime_failure_count"] == 1
    runtime = json.loads((tmp_path / "search" / "runtime_preflight.json").read_text(encoding="utf-8"))
    assert runtime["runtime_preflight"] == "FAILED"


def test_numpy_unavailable_runtime_preflight_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import tools.dataset_delivery.task2_fov_candidate_search as search

    image_root, mask_root = _case_roots(tmp_path)
    _mkdir_roots(image_root, mask_root)
    real_import = search.importlib.import_module

    def fake_import(name: str):
        if name == "numpy":
            raise ModuleNotFoundError("No module named 'numpy'")
        return real_import(name)

    monkeypatch.setattr(search.importlib, "import_module", fake_import)

    summary = search.search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=0,
        preseed_pulmonary=False,
    )

    assert summary["status"] == "RUNTIME_DEPENDENCY_MISSING"
    assert summary["dependency"] == "numpy"
    assert summary["scanned_count"] == 0


def test_dependency_missing_process_exit_is_nonzero_and_status_not_completed(tmp_path: Path):
    image_root, mask_root = _case_roots(tmp_path)
    _mkdir_roots(image_root, mask_root)
    output_root = tmp_path / "search"

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "tools/dataset_delivery/task2_fov_candidate_search.py",
            "--image-root",
            str(image_root),
            "--mask-root",
            str(mask_root),
            "--output-root",
            str(output_root),
            "--no-preseed",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    summary = json.loads((output_root / "search_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "RUNTIME_DEPENDENCY_MISSING"
    assert summary["status"] != "completed"
    assert summary["scanned_count"] == 0


def test_zero_valid_cases_is_failure_not_completed(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    image_root.mkdir(parents=True)
    for case_id in ("case_001", "case_002", "case_003"):
        (mask_root / case_id / "segmentations").mkdir(parents=True)

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        scan_limit=2,
        progress_every=0,
        preseed_pulmonary=False,
    )

    assert summary["status"] == "FAILED_NO_VALID_CASES_SCANNED"
    assert summary["scanned_count"] == 0
    assert summary["directories_examined"] == 3
    assert summary["missing_ct_count"] == 3


def test_single_corrupt_ct_records_case_failure_and_continues(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    _corrupt_ct_case(image_root, mask_root, "case_001")
    _head_case(image_root, mask_root, "case_002")

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        scan_limit=1,
        progress_every=0,
        preseed_pulmonary=False,
        consecutive_geometry_failure_limit=5,
        min_geometry_failures_for_rate=5,
    )

    assert summary["status"] == "completed_with_uncovered_targets"
    assert summary["scanned_count"] == 1
    assert summary["ct_geometry_failure_count"] == 1
    failures = _read_rows(tmp_path / "search" / "ct_geometry_failures.csv")
    assert failures[0]["status"] == "CT_GEOMETRY_FAILED"
    assert failures[0]["case_id"] == "case_001"


def test_many_ct_load_failures_trigger_systemic_guard(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    for index in range(3):
        _corrupt_ct_case(image_root, mask_root, f"case_{index:03d}")

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        progress_every=0,
        preseed_pulmonary=False,
        consecutive_geometry_failure_limit=3,
        min_geometry_failures_for_rate=99,
    )

    assert summary["status"] == "POSSIBLE_RUNTIME_OR_DATASET_CONFIGURATION_FAILURE"
    assert summary["scanned_count"] == 0
    assert summary["ct_geometry_failure_count"] == 3
    assert "consecutive CT geometry failures" in summary["failure_reason"]


def test_directory_traversal_guard_prevents_silent_full_library_walk(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    image_root.mkdir(parents=True)
    for index in range(10):
        (mask_root / f"case_{index:03d}" / "segmentations").mkdir(parents=True)

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        scan_limit=2000,
        max_directories_examined=5,
        progress_every=0,
        preseed_pulmonary=False,
    )

    assert summary["status"] == "POSSIBLE_RUNTIME_OR_DATASET_CONFIGURATION_FAILURE"
    assert summary["directories_examined"] == 5
    assert summary["next_mask_index"] == 5
    assert summary["scanned_count"] == 0
    assert "max_directories_examined" in summary["failure_reason"]


def test_runtime_manifest_records_actual_python_executable(tmp_path: Path):
    from tools.dataset_delivery.task2_fov_candidate_search import search_fov_candidates

    image_root, mask_root = _case_roots(tmp_path)
    _head_case(image_root, mask_root, "case_001")

    summary = search_fov_candidates(
        image_root=image_root,
        mask_root=mask_root,
        output_root=tmp_path / "search",
        scan_limit=1,
        progress_every=0,
        preseed_pulmonary=False,
    )
    runtime = json.loads((tmp_path / "search" / "runtime_preflight.json").read_text(encoding="utf-8"))

    assert summary["runtime"]["runtime_preflight"] == "READY"
    assert runtime["python_executable"] == sys.executable
    assert runtime["nibabel_version"]
    assert runtime["numpy_version"]


def test_fov_candidate_launcher_uses_explicit_python_and_compute_preflight():
    script = Path("scripts/task2/submit_fov_candidate_search.sh").read_text(encoding="utf-8")

    assert "TASK2_PYTHON_EXECUTABLE" in script
    assert "HPC_DEFAULT_PYTHON=\"/home/xhan74/envs/medical_agent/bin/python\"" in script
    assert "RUNTIME_PREFLIGHT_FAILED" in script
    assert "TASK2_FOV_COMPUTE_PREFLIGHT_OK" in script
    assert '"${TASK2_PYTHON_EXECUTABLE}" -c' in script
    assert "exec \"${TASK2_PYTHON_EXECUTABLE}\"" in script


def test_fov_candidate_launcher_wrong_python_blocks_before_sbatch(tmp_path: Path):
    result = subprocess.run(
        [
            "bash",
            "scripts/task2/submit_fov_candidate_search.sh",
            "--python-executable",
            str(tmp_path / "missing_python"),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "RUNTIME_PREFLIGHT_FAILED" in result.stderr
    assert not (tmp_path / "out" / "task2_fov_candidate_search.sbatch").exists()


def test_fov_candidate_launcher_dry_run_writes_compute_preflight_sbatch(tmp_path: Path):
    image_root, mask_root = _case_roots(tmp_path)
    _mkdir_roots(image_root, mask_root)
    output_root = tmp_path / "out"

    result = subprocess.run(
        [
            "bash",
            "scripts/task2/submit_fov_candidate_search.sh",
            "--python-executable",
            sys.executable,
            "--image-root",
            str(image_root),
            "--mask-root",
            str(mask_root),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    sbatch_path = output_root / "task2_fov_candidate_search.sbatch"

    assert result.returncode == 0
    assert sbatch_path.exists()
    sbatch = sbatch_path.read_text(encoding="utf-8")
    assert f"export TASK2_PYTHON_EXECUTABLE={sys.executable}" in sbatch
    assert "TASK2_FOV_COMPUTE_PREFLIGHT_OK" in sbatch
    assert "tools/dataset_delivery/task2_fov_candidate_search.py" in sbatch


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
    assert summary["status"] == "completed_with_uncovered_targets"
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
