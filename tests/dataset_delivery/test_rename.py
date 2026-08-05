from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.dataset_delivery.delivery_lib import DeliveryError, apply_rename, validate_rename_mapping


def write_taxonomy(path: Path, names: list[str]) -> None:
    import json

    path.write_text(json.dumps({"target_organs": names}), encoding="utf-8")


def write_mapping(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_name", "target_name", "status", "reason", "notes"])
        for source, target, status in rows:
            writer.writerow([source, target, status, "test", ""])


def write_alias_groups(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alias_group", "target_name", "source_name", "status", "evidence"])
        for group, target, source in rows:
            writer.writerow([group, target, source, "confirmed", "test alias"])


def make_case(root: Path, case_id: str, masks: list[str]) -> Path:
    seg = root / case_id / "segmentations"
    seg.mkdir(parents=True)
    for name in masks:
        (seg / f"{name}.nii.gz").write_text(name, encoding="utf-8")
    return seg


@pytest.fixture()
def taxonomy(tmp_path: Path) -> Path:
    names = ["target_a", "target_b", "target_c", "artery_left", "artery_right"]
    names.extend(f"organ_{i:03d}" for i in range(368))
    path = tmp_path / "taxonomy.json"
    write_taxonomy(path, names)
    return path


def statuses(report: Path) -> list[str]:
    with report.open(encoding="utf-8", newline="") as handle:
        return [row["status"] for row in csv.DictReader(handle)]


def test_rename_default_dry_run_does_not_modify(tmp_path: Path, taxonomy: Path) -> None:
    make_case(tmp_path / "data", "BDMAP_001", ["source_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [("source_a", "target_a", "confirmed")])
    report = tmp_path / "report.csv"
    apply_rename(tmp_path / "data", mapping, taxonomy, report)
    assert (tmp_path / "data/BDMAP_001/segmentations/source_a.nii.gz").exists()
    assert "would_rename" in statuses(report)


def test_rename_apply_changes_filename(tmp_path: Path, taxonomy: Path) -> None:
    make_case(tmp_path / "data", "BDMAP_001", ["source_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [("source_a", "target_a", "confirmed")])
    apply_rename(tmp_path / "data", mapping, taxonomy, tmp_path / "report.csv", apply=True, output_data_root=tmp_path / "out_data")
    assert (tmp_path / "data/BDMAP_001/segmentations/source_a.nii.gz").exists()
    assert not (tmp_path / "data/BDMAP_001/segmentations/target_a.nii.gz").exists()
    assert not (tmp_path / "out_data/BDMAP_001/segmentations/source_a.nii.gz").exists()
    assert (tmp_path / "out_data/BDMAP_001/segmentations/target_a.nii.gz").exists()


def test_target_exists_never_overwrites(tmp_path: Path, taxonomy: Path) -> None:
    make_case(tmp_path / "data", "BDMAP_001", ["source_a", "target_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [("source_a", "target_a", "confirmed")])
    report = tmp_path / "report.csv"
    apply_rename(tmp_path / "data", mapping, taxonomy, report, apply=True, output_data_root=tmp_path / "out_data")
    assert statuses(report) == ["conflict"]
    assert (tmp_path / "data/BDMAP_001/segmentations/source_a.nii.gz").exists()
    assert (tmp_path / "out_data/BDMAP_001/segmentations/source_a.nii.gz").exists()


def test_repeat_apply_reports_already_applied(tmp_path: Path, taxonomy: Path) -> None:
    make_case(tmp_path / "data", "BDMAP_001", ["source_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [("source_a", "target_a", "confirmed")])
    apply_rename(tmp_path / "data", mapping, taxonomy, tmp_path / "first.csv", apply=True, output_data_root=tmp_path / "out_data")
    apply_rename(tmp_path / "data", mapping, taxonomy, tmp_path / "second.csv", apply=True, output_data_root=tmp_path / "out_data")
    assert statuses(tmp_path / "second.csv") == ["already_normalized"]


def test_one_to_many_mapping_rejected(tmp_path: Path, taxonomy: Path) -> None:
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [("source_a", "target_a", "confirmed"), ("source_a", "target_b", "confirmed")])
    with pytest.raises(DeliveryError):
        validate_rename_mapping(mapping, taxonomy)


def test_many_to_one_same_case_blocked(tmp_path: Path, taxonomy: Path) -> None:
    make_case(tmp_path / "data", "BDMAP_001", ["source_a", "source_b"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [("source_a", "target_a", "confirmed"), ("source_b", "target_a", "confirmed")])
    aliases = tmp_path / "aliases.csv"
    write_alias_groups(aliases, [("target_a_aliases", "target_a", "source_a"), ("target_a_aliases", "target_a", "source_b")])
    report = tmp_path / "report.csv"
    apply_rename(tmp_path / "data", mapping, taxonomy, report, apply=True, alias_groups=aliases, output_data_root=tmp_path / "out_data")
    assert statuses(report) == ["conflict", "conflict"]


def test_pending_review_does_not_execute(tmp_path: Path, taxonomy: Path) -> None:
    make_case(tmp_path / "data", "BDMAP_001", ["source_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [("source_a", "target_a", "pending_review")])
    report = tmp_path / "report.csv"
    summary = apply_rename(tmp_path / "data", mapping, taxonomy, report, apply=True, output_data_root=tmp_path / "out_data")
    assert summary["status_counts"]["skipped_pending_review"] == 1
    assert (tmp_path / "data/BDMAP_001/segmentations/source_a.nii.gz").exists()


def test_apply_requires_output_data_root(tmp_path: Path, taxonomy: Path) -> None:
    make_case(tmp_path / "data", "BDMAP_001", ["source_a"])
    mapping = tmp_path / "mapping.csv"
    write_mapping(mapping, [("source_a", "target_a", "confirmed")])
    with pytest.raises(DeliveryError):
        apply_rename(tmp_path / "data", mapping, taxonomy, tmp_path / "report.csv", apply=True)
