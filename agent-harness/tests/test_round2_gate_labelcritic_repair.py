from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_pancreas_head_abdomen_prior_is_suspicious_and_withheld(tmp_path: Path) -> None:
    mod = _load_script("audit_negative_absent_manifest_source")
    manifest = tmp_path / "manifest.json"
    diagnosis = tmp_path / "negative.csv"
    appearance = tmp_path / "appearance.json"
    _write_json(
        manifest,
        {
            "items": [
                {
                    "case_id": "case_1",
                    "organ": "pancreas_head",
                    "target_type": "negative_absent",
                    "fov_status": "out_of_fov",
                    "fov_evidence": ["dataset_prior:PanTS_abdomen_only_not_373_whole_body"],
                    "negative_source": "case_373_expected_absent",
                }
            ]
        },
    )
    _write_json(
        appearance,
        {
            "entries": [
                {
                    "canonical_id": "pancreas_head",
                    "expected_body_regions": ["abdomen"],
                    "ct_location": "abdomen and retroperitoneum",
                    "formal_selection_eligible": True,
                }
            ]
        },
    )
    _write_csv(
        diagnosis,
        [
            {
                "case_id": "case_1",
                "organ": "pancreas_head",
                "fov_status": "out_of_fov",
                "fov_evidence": json.dumps(["dataset_prior:PanTS_abdomen_only_not_373_whole_body"]),
                "postprocess_status": "copied_no_containment_rule",
                "containment_enabled": "False",
            }
        ],
    )
    payload, rows = mod.build_audit(manifest, diagnosis, appearance)
    assert rows[0]["source_classification"] == "suspicious_abdominal_negative_should_withhold"
    assert rows[0]["withheld_required"] is True
    assert payload["suspicious_abdominal_negative_count"] == 1
    assert payload["suspicious_abdominal_all_withheld_required"] is True


def test_brain_out_of_fov_stays_true_negative(tmp_path: Path) -> None:
    mod = _load_script("audit_negative_absent_manifest_source")
    manifest = tmp_path / "manifest.json"
    diagnosis = tmp_path / "negative.csv"
    appearance = tmp_path / "appearance.json"
    _write_json(
        manifest,
        {"items": [{"case_id": "case_1", "organ": "brain", "target_type": "negative_absent"}]},
    )
    _write_json(
        appearance,
        {
            "entries": [
                {
                    "canonical_id": "brain",
                    "expected_body_regions": ["head_neck"],
                    "ct_location": "head and craniofacial region",
                }
            ]
        },
    )
    _write_csv(
        diagnosis,
        [
            {
                "case_id": "case_1",
                "organ": "brain",
                "fov_status": "out_of_fov",
                "fov_evidence": json.dumps(["dataset_prior:PanTS_abdomen_only_not_373_whole_body"]),
                "postprocess_status": "warning_empty_parent_roi_copied_raw",
                "containment_enabled": "True",
            }
        ],
    )
    _, rows = mod.build_audit(manifest, diagnosis, appearance)
    assert rows[0]["source_classification"] == "true_out_of_fov_negative"
    assert rows[0]["withheld_required"] is False


def test_master_gate_blocks_when_labelcritic_missing_or_failed(tmp_path: Path) -> None:
    mod = _load_script("build_round2_formal_master_gate")
    trainset = tmp_path / "trainset.json"
    negative_safe = tmp_path / "negative_safe.json"
    negative_audit = tmp_path / "negative_audit.json"
    reselection = tmp_path / "reselection.json"
    _write_json(
        trainset,
        {"status": "passed", "round2_progression_allowed": True, "summary": {"negative_false_positive_count": 0}},
    )
    _write_json(negative_safe, {"status": "success", "pre_suppression_nonempty_negative_count": 10})
    _write_json(
        negative_audit,
        {
            "status": "passed",
            "suspicious_abdominal_negative_count": 0,
            "suspicious_abdominal_all_withheld_required": True,
        },
    )
    _write_json(reselection, {"status": "ready_for_cached_reselection"})
    gate = mod.build_master_gate(trainset, negative_safe, negative_audit, tmp_path / "missing.json", reselection)
    assert gate["status"] == "blocked"
    assert "missing_labelcritic_benchmark_summary" in gate["block_reasons"]

    failed_labelcritic = tmp_path / "labelcritic.json"
    _write_json(failed_labelcritic, {"status": "failed", "rows": [{"pass": False}]})
    gate = mod.build_master_gate(trainset, negative_safe, negative_audit, failed_labelcritic, reselection)
    assert gate["status"] == "blocked"
    assert "labelcritic_known_better_benchmark_failed" in gate["block_reasons"]


def test_master_gate_blocks_unhandled_suspicious_abdominal_negatives(tmp_path: Path) -> None:
    mod = _load_script("build_round2_formal_master_gate")
    trainset = tmp_path / "trainset.json"
    negative_safe = tmp_path / "negative_safe.json"
    negative_audit = tmp_path / "negative_audit.json"
    labelcritic = tmp_path / "labelcritic.json"
    reselection = tmp_path / "reselection.json"
    _write_json(
        trainset,
        {"status": "passed", "round2_progression_allowed": True, "summary": {"negative_false_positive_count": 0}},
    )
    _write_json(negative_safe, {"status": "success", "pre_suppression_nonempty_negative_count": 10})
    _write_json(
        negative_audit,
        {
            "status": "passed",
            "suspicious_abdominal_negative_count": 2,
            "suspicious_abdominal_all_withheld_required": False,
        },
    )
    _write_json(labelcritic, {"status": "passed", "rows": [{"pass": True}]})
    _write_json(reselection, {"status": "ready_for_cached_reselection"})
    gate = mod.build_master_gate(trainset, negative_safe, negative_audit, labelcritic, reselection)
    assert gate["status"] == "blocked"
    assert "suspicious_abdominal_negatives_handled" in gate["block_reasons"]
    text = json.dumps(gate)
    assert "ground_truth" not in text.lower()
    assert "student_vs_gt" not in text.lower()
    assert "teacher_vs_gt" not in text.lower()
    assert "metric_target=GT" not in text
