from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deliverables/task1_rename_standalone_final/rename_anatomical_labels.py"
FINAL_DIR = ROOT / "deliverables/task1_rename_standalone_final"


def write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["current_name", "target_name", "issue_type", "action", "status", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            full_row = {field: "" for field in fields}
            full_row.update(row)
            writer.writerow(full_row)


def make_case(root: Path, case_id: str = "BDMAP_001", masks: list[str] | None = None) -> Path:
    seg = root / case_id / "segmentations"
    seg.mkdir(parents=True)
    for name in masks or []:
        (seg / f"{name}.nii.gz").write_text(name, encoding="utf-8")
    return root / case_id


def run_script(tmp_path: Path, data_root: Path, mapping: Path, *extra: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    report = tmp_path / "report.csv"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        str(SCRIPT),
        "--data-root",
        str(data_root),
        "--mapping-file",
        str(mapping),
        "--report",
        str(report),
        *extra,
    ]
    return subprocess.run(command, cwd=str(cwd or ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def report_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def statuses(path: Path) -> list[str]:
    return [row["status"] for row in report_rows(path)]


def base_rename(source: str = "source_a", target: str = "target_a") -> dict[str, str]:
    return {
        "current_name": source,
        "target_name": target,
        "issue_type": "synonym",
        "action": "rename",
        "status": "confirmed",
        "notes": "test",
    }


def test_dry_run_does_not_modify_files(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["source_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename()])

    result = run_script(tmp_path, data, mapping, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert (data / "BDMAP_001/segmentations/source_a.nii.gz").exists()
    assert not (data / "BDMAP_001/segmentations/target_a.nii.gz").exists()
    assert statuses(tmp_path / "report.csv") == ["would_rename"]


def test_apply_completes_safe_rename(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["source_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename()])

    result = run_script(tmp_path, data, mapping, "--apply")

    assert result.returncode == 0, result.stderr
    assert not (data / "BDMAP_001/segmentations/source_a.nii.gz").exists()
    assert (data / "BDMAP_001/segmentations/target_a.nii.gz").exists()
    assert statuses(tmp_path / "report.csv") == ["renamed"]


def test_source_and_target_existing_conflict(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["source_a", "target_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename()])

    result = run_script(tmp_path, data, mapping, "--apply")

    assert result.returncode == 0, result.stderr
    rows = report_rows(tmp_path / "report.csv")
    assert rows[0]["status"] == "conflict"
    assert rows[0]["reason"] == "source_and_target_exist"
    assert (data / "BDMAP_001/segmentations/source_a.nii.gz").exists()


def test_multiple_sources_to_same_target_conflict(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["source_a", "source_b"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename("source_a", "target_a"), base_rename("source_b", "target_a")])

    result = run_script(tmp_path, data, mapping, "--apply")

    assert result.returncode == 0, result.stderr
    rows = report_rows(tmp_path / "report.csv")
    assert [row["status"] for row in rows] == ["conflict", "conflict"]
    assert {row["reason"] for row in rows} == {"multiple_sources_to_same_target_in_case"}
    assert (data / "BDMAP_001/segmentations/source_a.nii.gz").exists()
    assert (data / "BDMAP_001/segmentations/source_b.nii.gz").exists()


def test_source_and_target_both_missing_is_reported(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["other"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename()])

    result = run_script(tmp_path, data, mapping, "--dry-run")

    assert result.returncode == 0, result.stderr
    rows = report_rows(tmp_path / "report.csv")
    assert rows[0]["status"] == "source_missing"
    assert rows[0]["reason"] == "source_and_target_missing"


def test_source_missing_but_target_exists_is_already_normalized(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["target_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename()])

    result = run_script(tmp_path, data, mapping, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert statuses(tmp_path / "report.csv") == ["already_normalized"]


def test_generate_manual_review_and_blocked_rows_do_not_execute(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["source_a", "manual_source", "blocked_source"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(
        mapping,
        [
            {
                "current_name": "",
                "target_name": "generated_target",
                "issue_type": "missing_label",
                "action": "generate",
                "status": "confirmed",
                "notes": "test",
            },
            {
                "current_name": "manual_source",
                "target_name": "manual_target",
                "issue_type": "synonym",
                "action": "manual_review",
                "status": "needs_review",
                "notes": "test",
            },
            {
                "current_name": "blocked_source",
                "target_name": "blocked_target",
                "issue_type": "synonym",
                "action": "rename",
                "status": "blocked",
                "notes": "test",
            },
        ],
    )

    result = run_script(tmp_path, data, mapping, "--apply")

    assert result.returncode == 0, result.stderr
    assert not (data / "BDMAP_001/segmentations/generated_target.nii.gz").exists()
    assert (data / "BDMAP_001/segmentations/manual_source.nii.gz").exists()
    assert (data / "BDMAP_001/segmentations/blocked_source.nii.gz").exists()
    assert statuses(tmp_path / "report.csv") == ["skipped", "skipped", "skipped"]


def test_left_to_right_mapping_is_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["left_kidney"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename("left_kidney", "kidney_right")])

    result = run_script(tmp_path, data, mapping, "--dry-run", "--strict")

    assert result.returncode != 0
    rows = report_rows(tmp_path / "report.csv")
    assert rows[0]["status"] == "invalid_mapping"
    assert rows[0]["reason"] == "laterality_mismatch"


def test_right_to_left_mapping_is_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["right_kidney"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename("right_kidney", "kidney_left")])

    result = run_script(tmp_path, data, mapping, "--dry-run", "--strict")

    assert result.returncode != 0
    rows = report_rows(tmp_path / "report.csv")
    assert rows[0]["status"] == "invalid_mapping"
    assert rows[0]["reason"] == "laterality_mismatch"


def test_same_current_name_to_multiple_targets_fails_strict(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["source_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename("source_a", "target_a"), base_rename("source_a", "target_b")])

    result = run_script(tmp_path, data, mapping, "--dry-run", "--strict")

    assert result.returncode != 0
    assert "one_current_name_to_multiple_rename_targets" in result.stdout


def test_existing_target_is_not_overwritten(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["source_a", "target_a"])
    target = data / "BDMAP_001/segmentations/target_a.nii.gz"
    target.write_text("keep me", encoding="utf-8")
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename()])

    result = run_script(tmp_path, data, mapping, "--apply")

    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == "keep me"


def test_unlisted_files_are_not_modified(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["source_a", "ct", "other"])
    other = data / "BDMAP_001/segmentations/other.nii.gz"
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename()])

    result = run_script(tmp_path, data, mapping, "--apply")

    assert result.returncode == 0, result.stderr
    assert other.exists()
    assert (data / "BDMAP_001/segmentations/ct.nii.gz").exists()


def test_script_runs_without_pythonpath_and_from_outside_repo(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_case(data, masks=["source_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [base_rename()])
    outside = tmp_path / "outside"
    outside.mkdir()

    result = run_script(tmp_path, data, mapping, "--dry-run", cwd=outside)

    assert result.returncode == 0, result.stderr
    assert statuses(tmp_path / "report.csv") == ["would_rename"]


def test_final_delivery_directory_contains_only_two_files() -> None:
    files = sorted(path.name for path in FINAL_DIR.iterdir() if path.is_file())

    assert files == ["rename_anatomical_labels.py", "taxonomy_373_mapping.csv"]
