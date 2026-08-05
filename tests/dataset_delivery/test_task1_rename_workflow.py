from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.dataset_delivery.compare_task1_mapping import compare_mappings
from tools.dataset_delivery.delivery_lib import DeliveryError, validate_rename_mapping
from tools.dataset_delivery.verify_task1_hpc import verify_task1_hpc


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def taxonomy(path: Path, extra: list[str] | None = None) -> Path:
    names = list(dict.fromkeys((extra or []) + [f"organ_{i:03d}" for i in range(500)]))[:373]
    path.write_text(json.dumps({"target_organs": names}), encoding="utf-8")
    return path


def mapping(path: Path, rows: list[dict[str, str]]) -> Path:
    return write_csv(path, ["source_name", "target_name", "status", "reason", "notes"], rows)


def boundary(path: Path, rows: list[dict[str, str]]) -> Path:
    fields = [
        "source_name",
        "target_name",
        "same_anatomy",
        "same_laterality",
        "same_granularity",
        "requires_voxel_change",
        "classification",
        "status",
        "reason",
        "evidence",
        "alias_group",
    ]
    return write_csv(path, fields, rows)


def aliases(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    return write_csv(
        path,
        ["alias_group", "target_name", "source_name", "status", "evidence"],
        [{"alias_group": group, "target_name": target, "source_name": source, "status": "confirmed", "evidence": "test"} for group, target, source in rows],
    )


def task2(path: Path, targets: list[str]) -> Path:
    return write_csv(
        path,
        ["target_name", "classification", "status", "reason", "evidence"],
        [{"target_name": target, "classification": "task2_generate", "status": "confirmed", "reason": "fixed", "evidence": "test"} for target in targets],
    )


def confirmed_boundary(source: str, target: str, *, evidence: str = "manual exact equivalence") -> dict[str, str]:
    return {
        "source_name": source,
        "target_name": target,
        "same_anatomy": "true",
        "same_laterality": "true",
        "same_granularity": "true",
        "requires_voxel_change": "false",
        "classification": "task1_rename",
        "status": "confirmed",
        "reason": "test",
        "evidence": evidence,
        "alias_group": "",
    }


def pending_boundary(source: str, target: str) -> dict[str, str]:
    row = confirmed_boundary(source, target)
    row.update({
        "same_anatomy": "false",
        "same_laterality": "false",
        "same_granularity": "false",
        "classification": "boundary_review",
        "status": "pending_review",
        "reason": "insufficient_evidence",
        "evidence": "needs human review",
    })
    return row


def make_case(data_root: Path, case_id: str, names: list[str]) -> Path:
    seg = data_root / case_id / "segmentations"
    seg.mkdir(parents=True, exist_ok=True)
    for name in names:
        (seg / f"{name}.nii.gz").write_text(name, encoding="utf-8")
    return seg


def manifest(path: Path, data_root: Path, case_id: str = "BDMAP_001") -> Path:
    return write_csv(
        path,
        ["index", "case_id", "reference_mask_dir"],
        [{"index": "0", "case_id": case_id, "reference_mask_dir": str(data_root / case_id / "segmentations")}],
    )


def base_files(tmp_path: Path, *, source: str = "source_a", target: str = "target_a"):
    tax = taxonomy(tmp_path / "taxonomy.json", [target, "task2_target"])
    map_path = mapping(tmp_path / "mapping.csv", [{"source_name": source, "target_name": target, "status": "confirmed", "reason": "test", "notes": ""}])
    boundary_path = boundary(tmp_path / "task_boundary_classification.csv", [confirmed_boundary(source, target)])
    task2_path = task2(tmp_path / "task2.csv", ["task2_target"])
    alias_path = aliases(tmp_path / "aliases.csv", [])
    return tax, map_path, boundary_path, task2_path, alias_path


def test_hpc_dry_run_does_not_modify_files(tmp_path: Path) -> None:
    tax, map_path, _boundary, task2_path, alias_path = base_files(tmp_path)
    data = tmp_path / "data"
    make_case(data, "BDMAP_001", ["source_a"])
    result = verify_task1_hpc(
        data_root=data,
        case_manifest=manifest(tmp_path / "cases.csv", data),
        mapping=map_path,
        taxonomy=tax,
        task2_targets=task2_path,
        alias_groups=alias_path,
        output_root=tmp_path / "out",
    )
    assert result["read_only"] is True
    assert result["status_counts"]["would_rename"] == 1
    assert (data / "BDMAP_001/segmentations/source_a.nii.gz").exists()
    assert not (data / "BDMAP_001/segmentations/target_a.nii.gz").exists()


def test_confirmed_exact_alias_can_apply_to_output_copy(tmp_path: Path) -> None:
    tax, map_path, _boundary, task2_path, alias_path = base_files(tmp_path)
    data = tmp_path / "data"
    make_case(data, "BDMAP_001", ["source_a"])
    verify_task1_hpc(
        data_root=data,
        case_manifest=manifest(tmp_path / "cases.csv", data),
        mapping=map_path,
        taxonomy=tax,
        task2_targets=task2_path,
        alias_groups=alias_path,
        output_root=tmp_path / "out",
        apply=True,
        output_data_root=tmp_path / "out_data",
    )
    assert (data / "BDMAP_001/segmentations/source_a.nii.gz").exists()
    assert (tmp_path / "out_data/BDMAP_001/segmentations/target_a.nii.gz").exists()


def test_pending_review_and_task2_rows_do_not_execute(tmp_path: Path) -> None:
    tax = taxonomy(tmp_path / "taxonomy.json", ["target_a", "task2_target"])
    map_path = mapping(
        tmp_path / "mapping.csv",
        [
            {"source_name": "pending_source", "target_name": "target_a", "status": "pending_review", "reason": "review", "notes": ""},
            {"source_name": "task2_source", "target_name": "task2_target", "status": "pending_review", "reason": "task2", "notes": ""},
        ],
    )
    boundary(tmp_path / "task_boundary_classification.csv", [pending_boundary("pending_source", "target_a"), pending_boundary("task2_source", "task2_target")])
    task2_path = task2(tmp_path / "task2.csv", ["task2_target"])
    alias_path = aliases(tmp_path / "aliases.csv", [])
    data = tmp_path / "data"
    make_case(data, "BDMAP_001", ["pending_source", "task2_source"])
    result = verify_task1_hpc(
        data_root=data,
        case_manifest=manifest(tmp_path / "cases.csv", data),
        mapping=map_path,
        taxonomy=tax,
        task2_targets=task2_path,
        alias_groups=alias_path,
        output_root=tmp_path / "out",
    )
    assert result["status_counts"]["skipped_pending_review"] == 1
    assert result["status_counts"]["task2_overlap"] == 1
    assert not (data / "BDMAP_001/segmentations/target_a.nii.gz").exists()
    assert not (data / "BDMAP_001/segmentations/task2_target.nii.gz").exists()


def test_target_exists_conflict_never_overwrites(tmp_path: Path) -> None:
    tax, map_path, _boundary, task2_path, alias_path = base_files(tmp_path)
    data = tmp_path / "data"
    make_case(data, "BDMAP_001", ["source_a", "target_a"])
    target = data / "BDMAP_001/segmentations/target_a.nii.gz"
    target.write_text("keep", encoding="utf-8")
    result = verify_task1_hpc(
        data_root=data,
        case_manifest=manifest(tmp_path / "cases.csv", data),
        mapping=map_path,
        taxonomy=tax,
        task2_targets=task2_path,
        alias_groups=alias_path,
        output_root=tmp_path / "out",
    )
    assert result["status_counts"]["conflict"] == 1
    assert target.read_text(encoding="utf-8") == "keep"


def test_static_validation_rejects_one_to_many_and_undeclared_many_to_one(tmp_path: Path) -> None:
    tax = taxonomy(tmp_path / "taxonomy.json", ["target_a", "target_b"])
    task2_path = task2(tmp_path / "task2.csv", [])
    alias_path = aliases(tmp_path / "aliases.csv", [])
    one_to_many = mapping(
        tmp_path / "one_to_many.csv",
        [
            {"source_name": "source_a", "target_name": "target_a", "status": "confirmed", "reason": "test", "notes": ""},
            {"source_name": "source_a", "target_name": "target_b", "status": "confirmed", "reason": "test", "notes": ""},
        ],
    )
    boundary(tmp_path / "task_boundary_classification.csv", [confirmed_boundary("source_a", "target_a"), confirmed_boundary("source_a", "target_b")])
    with pytest.raises(DeliveryError):
        validate_rename_mapping(one_to_many, tax, task2_targets=task2_path, alias_groups=alias_path, boundary_classification=tmp_path / "task_boundary_classification.csv")
    many_to_one = mapping(
        tmp_path / "many_to_one.csv",
        [
            {"source_name": "source_a", "target_name": "target_a", "status": "confirmed", "reason": "test", "notes": ""},
            {"source_name": "source_b", "target_name": "target_a", "status": "confirmed", "reason": "test", "notes": ""},
        ],
    )
    boundary(tmp_path / "task_boundary_classification.csv", [confirmed_boundary("source_a", "target_a"), confirmed_boundary("source_b", "target_a")])
    with pytest.raises(DeliveryError):
        validate_rename_mapping(many_to_one, tax, task2_targets=task2_path, alias_groups=alias_path, boundary_classification=tmp_path / "task_boundary_classification.csv")


def test_alias_group_allows_static_many_to_one_but_coexistence_conflicts(tmp_path: Path) -> None:
    tax = taxonomy(tmp_path / "taxonomy.json", ["target_a"])
    map_path = mapping(
        tmp_path / "mapping.csv",
        [
            {"source_name": "source_a", "target_name": "target_a", "status": "confirmed", "reason": "test", "notes": ""},
            {"source_name": "source_b", "target_name": "target_a", "status": "confirmed", "reason": "test", "notes": ""},
        ],
    )
    boundary(tmp_path / "task_boundary_classification.csv", [confirmed_boundary("source_a", "target_a"), confirmed_boundary("source_b", "target_a")])
    task2_path = task2(tmp_path / "task2.csv", [])
    alias_path = aliases(tmp_path / "aliases.csv", [("target_a_aliases", "target_a", "source_a"), ("target_a_aliases", "target_a", "source_b")])
    validate_rename_mapping(map_path, tax, task2_targets=task2_path, alias_groups=alias_path, boundary_classification=tmp_path / "task_boundary_classification.csv")
    data = tmp_path / "data"
    make_case(data, "BDMAP_001", ["source_a", "source_b"])
    result = verify_task1_hpc(
        data_root=data,
        case_manifest=manifest(tmp_path / "cases.csv", data),
        mapping=map_path,
        taxonomy=tax,
        task2_targets=task2_path,
        alias_groups=alias_path,
        output_root=tmp_path / "out",
    )
    assert result["status_counts"]["conflict"] == 2
    assert result["status_counts"]["alias_group_coexistence"] == 1


def test_task2_overlap_invalid_taxonomy_missing_evidence_and_boundary_mismatch_fail(tmp_path: Path) -> None:
    tax = taxonomy(tmp_path / "taxonomy.json", ["target_a", "task2_target"])
    task2_path = task2(tmp_path / "task2.csv", ["task2_target"])
    alias_path = aliases(tmp_path / "aliases.csv", [])
    overlap = mapping(tmp_path / "overlap.csv", [{"source_name": "source_a", "target_name": "task2_target", "status": "confirmed", "reason": "test", "notes": ""}])
    boundary(tmp_path / "task_boundary_classification.csv", [confirmed_boundary("source_a", "task2_target")])
    with pytest.raises(DeliveryError):
        validate_rename_mapping(overlap, tax, task2_targets=task2_path, alias_groups=alias_path, boundary_classification=tmp_path / "task_boundary_classification.csv")
    invalid = mapping(tmp_path / "invalid.csv", [{"source_name": "source_a", "target_name": "not_in_taxonomy", "status": "confirmed", "reason": "test", "notes": ""}])
    boundary(tmp_path / "task_boundary_classification.csv", [confirmed_boundary("source_a", "not_in_taxonomy")])
    with pytest.raises(DeliveryError):
        validate_rename_mapping(invalid, tax, task2_targets=task2_path, alias_groups=alias_path, boundary_classification=tmp_path / "task_boundary_classification.csv")
    missing_evidence = mapping(tmp_path / "missing_evidence.csv", [{"source_name": "source_a", "target_name": "target_a", "status": "confirmed", "reason": "test", "notes": ""}])
    boundary(tmp_path / "task_boundary_classification.csv", [confirmed_boundary("source_a", "target_a", evidence="")])
    with pytest.raises(DeliveryError):
        validate_rename_mapping(missing_evidence, tax, task2_targets=task2_path, alias_groups=alias_path, boundary_classification=tmp_path / "task_boundary_classification.csv")
    mismatch = mapping(tmp_path / "mismatch.csv", [{"source_name": "source_a", "target_name": "target_a", "status": "confirmed", "reason": "test", "notes": ""}])
    boundary(tmp_path / "task_boundary_classification.csv", [pending_boundary("source_a", "target_a")])
    with pytest.raises(DeliveryError):
        validate_rename_mapping(mismatch, tax, task2_targets=task2_path, alias_groups=alias_path, boundary_classification=tmp_path / "task_boundary_classification.csv")


def test_mapping_copy_differences_are_reported(tmp_path: Path) -> None:
    base = mapping(tmp_path / "base.csv", [{"source_name": "source_a", "target_name": "target_a", "status": "confirmed", "reason": "a", "notes": ""}])
    other = mapping(tmp_path / "other.csv", [{"source_name": "source_a", "target_name": "target_a", "status": "pending_review", "reason": "b", "notes": ""}])
    result = compare_mappings(base, [other])
    assert result["status"] == "different"
    assert result["comparisons"][0]["status_changes"]
    assert result["comparisons"][0]["reason_notes_changes"]


def test_python_310_compatibility_compile_subset() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "tools/dataset_delivery/verify_task1_hpc.py",
            "tools/dataset_delivery/compare_task1_mapping.py",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
