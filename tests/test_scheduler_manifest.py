from __future__ import annotations

import csv
import json

import pytest

from scheduler.manifest import assert_strict_no_gt_manifest, validate_split, validate_target_mapping
from scheduler.utils import SchedulerError


def _write_manifest(path, case_ids, extra=None):
    fields = ["case_id", "ct_path"] + list((extra or {}).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_id in case_ids:
            row = {"case_id": case_id, "ct_path": f"/images/{case_id}/ct.nii.gz"}
            row.update(extra or {})
            writer.writerow(row)


def test_strict_manifest_rejects_gt_columns(tmp_path):
    path = tmp_path / "bad.csv"
    _write_manifest(path, ["BDMAP_00000001"], {"gt_path": "/mask_only/x/segmentations/liver.nii.gz"})
    with pytest.raises(SchedulerError):
        assert_strict_no_gt_manifest(path)


def test_split_requires_20_20_10_and_no_overlap(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    reserve = tmp_path / "reserve.csv"
    _write_manifest(train, [f"BDMAP_{i:08d}" for i in range(1, 21)])
    _write_manifest(test, [f"BDMAP_{i:08d}" for i in range(21, 41)])
    _write_manifest(reserve, [f"BDMAP_{i:08d}" for i in range(41, 51)])
    report = validate_split(train, test, reserve)
    assert len(report["splits"]["train"]["case_ids"]) == 20


def test_mapping_validation_rejects_unresolved_pilot_target(tmp_path):
    full = ["a", "b"]
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({
        "targets": [
            {"target_name": "a", "mapping_status": "direct", "participates_in_pilot338": True},
            {"target_name": "b", "mapping_status": "unverified_skip", "participates_in_pilot338": True},
        ]
    }), encoding="utf-8")
    with pytest.raises(SchedulerError):
        validate_target_mapping(path, full, pilot_target_count=2)
