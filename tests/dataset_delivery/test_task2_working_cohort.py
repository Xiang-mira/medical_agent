from __future__ import annotations

import csv
from pathlib import Path


def _write_base_manifest(path: Path, case_ids: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "case_id", "ct_path", "annotation_folder"])
        writer.writeheader()
        for index, case_id in enumerate(case_ids):
            writer.writerow({
                "index": index,
                "case_id": case_id,
                "ct_path": f"/base/{case_id}/ct.nii.gz",
                "annotation_folder": f"/base/{case_id}/segmentations",
            })
    return path


def _touch_case_roots(root: Path, case_id: str) -> None:
    ct = root / "images" / case_id / "ct.nii.gz"
    ref = root / "masks" / case_id / "segmentations"
    ct.parent.mkdir(parents=True, exist_ok=True)
    ref.mkdir(parents=True, exist_ok=True)
    ct.write_bytes(b"ct")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_working_cohort_preserves_base_100_and_appends_two_cases(tmp_path: Path):
    from tools.dataset_delivery.task2_working_cohort import build_working_cohort

    base_ids = [f"BDMAP_{idx:08d}" for idx in range(100)]
    base = _write_base_manifest(tmp_path / "cases_100_manifest.csv", base_ids)
    append = tmp_path / "append.csv"
    append.write_text(
        "case_id,reason\n"
        "BDMAP_00000424,fov_extension_thorax_candidate\n"
        "BDMAP_00078156,fov_extension_thorax_candidate\n",
        encoding="utf-8",
    )
    _touch_case_roots(tmp_path, "BDMAP_00000424")
    _touch_case_roots(tmp_path, "BDMAP_00078156")

    audit = build_working_cohort(
        base_manifest=base,
        append_cases=append,
        output_manifest=tmp_path / "cases_102_manifest.csv",
        audit_json=tmp_path / "audit.json",
        audit_csv=tmp_path / "audit.csv",
        image_root=tmp_path / "images",
        mask_root=tmp_path / "masks",
    )

    rows = _read_rows(tmp_path / "cases_102_manifest.csv")
    assert audit["status"] == "READY"
    assert audit["base_count"] == 100
    assert audit["appended_count"] == 2
    assert audit["final_count"] == 102
    assert [row["case_id"] for row in rows[:100]] == base_ids
    assert rows[100]["index"] == "100"
    assert rows[100]["case_id"] == "BDMAP_00000424"
    assert rows[101]["index"] == "101"
    assert rows[101]["case_id"] == "BDMAP_00078156"


def test_working_cohort_is_duplicate_safe_and_deterministic(tmp_path: Path):
    from tools.dataset_delivery.task2_working_cohort import build_working_cohort

    base = _write_base_manifest(tmp_path / "base.csv", ["BDMAP_00000424", "BDMAP_00000002"])
    append = tmp_path / "append.csv"
    append.write_text(
        "case_id,reason\n"
        "BDMAP_00000424,already_present\n"
        "BDMAP_00078156,fov_extension_thorax_candidate\n",
        encoding="utf-8",
    )
    _touch_case_roots(tmp_path, "BDMAP_00078156")

    kwargs = {
        "base_manifest": base,
        "append_cases": append,
        "image_root": tmp_path / "images",
        "mask_root": tmp_path / "masks",
    }
    first = build_working_cohort(output_manifest=tmp_path / "out1.csv", **kwargs)
    second = build_working_cohort(output_manifest=tmp_path / "out2.csv", **kwargs)

    assert first["appended_case_ids"] == ["BDMAP_00078156"]
    assert "BDMAP_00000424" in first["duplicate_skipped"]
    assert (tmp_path / "out1.csv").read_text(encoding="utf-8") == (tmp_path / "out2.csv").read_text(encoding="utf-8")


def test_working_cohort_future_append_preserves_existing_indices(tmp_path: Path):
    from tools.dataset_delivery.task2_working_cohort import build_working_cohort

    base = _write_base_manifest(tmp_path / "cases_102_manifest.csv", [f"BDMAP_{idx:08d}" for idx in range(102)])
    append = tmp_path / "append.csv"
    append.write_text("case_id,reason\nBDMAP_00090000,new_head_candidate\n", encoding="utf-8")
    _touch_case_roots(tmp_path, "BDMAP_00090000")

    audit = build_working_cohort(
        base_manifest=base,
        append_cases=append,
        output_manifest=tmp_path / "cases_103_manifest.csv",
        image_root=tmp_path / "images",
        mask_root=tmp_path / "masks",
    )
    rows = _read_rows(tmp_path / "cases_103_manifest.csv")

    assert audit["final_count"] == 103
    assert rows[102]["index"] == "102"
    assert rows[102]["case_id"] == "BDMAP_00090000"
