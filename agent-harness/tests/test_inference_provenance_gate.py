from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.run_em_training as em
from scripts.run_repaired_round2_continuation import validate_case_scope


def _write_case_list(path: Path, case_ids: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path"])
        writer.writeheader()
        for case_id in case_ids:
            writer.writerow({"case_id": case_id, "ct_path": f"/tmp/{case_id}.nii.gz"})
    return path


def _write_valid_prediction_set(
    root: Path,
    *,
    case_ids: list[str],
    organs: list[str],
    provenance: dict,
) -> None:
    for case_id in case_ids:
        case_dir = root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for organ in organs:
            (case_dir / f"{organ}.nii.gz").touch()
        result = {
            **provenance,
            "case_id": case_id,
            "status": "partial_success",
            "failed_organs": [],
            "num_masks": len(organs),
        }
        (case_dir / "voxtell_student_result.json").write_text(json.dumps(result), encoding="utf-8")


def test_clean_10_by_373_prediction_contract_passes(tmp_path: Path) -> None:
    cases = [f"case{i:02d}" for i in range(10)]
    organs = [f"organ{i:03d}" for i in range(373)]
    provenance = {
        "run_id": "round2-checkpoint-cases-targets",
        "checkpoint_sha256": "a" * 64,
        "case_list_sha256": "b" * 64,
        "target_config_sha256": "c" * 64,
        "expected_case_ids": cases,
        "case_list_duplicate_case_ids": [],
    }
    prediction_root = tmp_path / "predictions"
    _write_valid_prediction_set(prediction_root, case_ids=cases, organs=organs, provenance=provenance)

    audit = em.audit_prompt_student_predictions(
        2,
        prediction_root=prediction_root,
        provenance=provenance,
        expected_organs=organs,
    )

    assert audit["status"] == "passed"
    assert audit["actual_num_cases"] == 10
    assert audit["expected_num_masks_per_case"] == 373


def test_unknown_or_mixed_checkpoint_predictions_fail(tmp_path: Path) -> None:
    cases = ["case01", "case02"]
    organs = ["liver", "spleen"]
    provenance = {
        "run_id": "expected-run",
        "checkpoint_sha256": "a" * 64,
        "case_list_sha256": "b" * 64,
        "target_config_sha256": "c" * 64,
        "expected_case_ids": cases,
        "case_list_duplicate_case_ids": [],
    }
    prediction_root = tmp_path / "predictions"
    _write_valid_prediction_set(prediction_root, case_ids=cases, organs=organs, provenance=provenance)
    mixed_path = prediction_root / "case01" / "voxtell_student_result.json"
    mixed = json.loads(mixed_path.read_text(encoding="utf-8"))
    mixed["checkpoint_sha256"] = "d" * 64
    mixed_path.write_text(json.dumps(mixed), encoding="utf-8")
    unknown_path = prediction_root / "case02" / "voxtell_student_result.json"
    unknown = json.loads(unknown_path.read_text(encoding="utf-8"))
    unknown.pop("run_id")
    unknown_path.write_text(json.dumps(unknown), encoding="utf-8")

    audit = em.audit_prompt_student_predictions(
        2,
        prediction_root=prediction_root,
        provenance=provenance,
        expected_organs=organs,
    )

    assert audit["status"] == "failed"
    assert audit["mixed_checkpoint_cases"] == ["case01"]
    assert audit["unknown_provenance_cases"] == ["case02"]


def test_incomplete_staging_contract_fails_without_touching_canonical(tmp_path: Path) -> None:
    provenance = {
        "run_id": "run",
        "checkpoint_sha256": "a" * 64,
        "case_list_sha256": "b" * 64,
        "target_config_sha256": "c" * 64,
        "expected_case_ids": ["case01"],
        "case_list_duplicate_case_ids": [],
    }
    canonical = tmp_path / "student_predictions"
    canonical.mkdir()
    marker = canonical / "published.marker"
    marker.write_text("old", encoding="utf-8")
    staging = tmp_path / "student_predictions_staging"
    _write_valid_prediction_set(staging, case_ids=["case01"], organs=["liver"], provenance=provenance)

    audit = em.audit_prompt_student_predictions(
        2,
        prediction_root=staging,
        provenance=provenance,
        expected_organs=["liver", "spleen"],
    )

    assert audit["status"] == "failed"
    assert marker.read_text(encoding="utf-8") == "old"


def test_promotion_is_forced_blocked_without_passed_inference_audit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path)
    checkpoint = tmp_path / "model" / "fold_0" / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    mstep = {
        "inference_checkpoint": str(checkpoint),
        "inference_model_dir": str(checkpoint.parents[1]),
        "eligible_for_next_round_prompt_student": True,
    }

    entry = em.record_checkpoint_promotion(
        2,
        status="promoted",
        mstep_result=mstep,
        inference_audit={"status": "failed"},
    )

    assert entry["status"] == "competition_blocked"
    assert entry["reason"] == "inference_provenance_audit_not_passed"


def test_case_scope_rejects_50_case_list_for_10_case_manifest(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "run"
    monkeypatch.setattr(em, "OUTPUT_ROOT", output_root)
    ten_cases = [f"case{i:02d}" for i in range(10)]
    fifty_cases = [f"case{i:02d}" for i in range(50)]
    manifest = output_root / "round2" / "mstep" / "voxtell_prompt_student_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"items": [{"case_id": case_id} for case_id in ten_cases]}), encoding="utf-8")
    estep = output_root / "round2" / "estep" / "annotation_versions"
    for case_id in ten_cases:
        (estep / case_id).mkdir(parents=True)
    case_list = _write_case_list(tmp_path / "cases50.csv", fifty_cases)

    audit = validate_case_scope(case_list, manifest)

    assert audit["status"] == "failed"
    assert "expected_exactly_10_cases:50" in audit["failures"]
    assert "manifest_case_scope_mismatch" in audit["failures"]


def test_round1_reference_falls_back_to_formal_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(em, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path / "outputs" / "current")
    formal = (
        tmp_path
        / "outputs"
        / "formal_round1_final_20260627"
        / "round1"
        / "estep"
        / "annotation_versions"
        / "case01"
    )
    formal.mkdir(parents=True)

    resolved = em.resolve_round_reference_root(1)

    assert resolved == formal.parent.resolve()


def test_materialize_strict_78_mask_competition_root(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "outputs" / "run"
    case_ids = [f"case{i:02d}" for i in range(10)]
    case_list = _write_case_list(tmp_path / "cases.csv", case_ids)
    monkeypatch.setattr(em, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(em, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(em, "CASE_LIST", case_list)
    raw_root = output_root / "round2" / "student_predictions"
    destination = output_root / "round2" / "student_predictions_round3_competition"
    rows = []
    for index in range(78):
        case_id = case_ids[index % len(case_ids)]
        organ = f"organ{index:03d}"
        source = raw_root / case_id / f"{organ}.nii.gz"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        rows.append({
            "case_id": case_id,
            "organ": organ,
            "route": "student_competition",
            "source_path": str(source),
        })
    allowlist = output_root / "round2" / "metrics" / "student_round3_gate" / "student_round2_organ_allowlist.json"
    allowlist.parent.mkdir(parents=True)
    allowlist.write_text(json.dumps({
        "schema_version": 3,
        "status": "success",
        "postprocessed_root": str(destination),
        "allowed_case_organ_count": 78,
        "allowed_case_organs": rows,
    }), encoding="utf-8")

    first = em.materialize_student_competition_root(3, allowlist_path=allowlist)
    (destination / case_ids[0] / "unexpected.nii.gz").touch()
    second = em.materialize_student_competition_root(3, allowlist_path=allowlist)

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert second["materialized_masks"] == 78
    assert len([path for path in destination.iterdir() if path.is_dir()]) == 10
    assert len(list(destination.glob("*/*.nii.gz"))) == 78
    assert not (destination / case_ids[0] / "unexpected.nii.gz").exists()
    assert all(path.is_symlink() for path in destination.glob("*/*.nii.gz"))
