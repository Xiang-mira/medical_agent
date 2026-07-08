from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

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


def _write_nii(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    array[1:3, 1:3, 1:3] = 1
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


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


def test_postprocessed_prediction_audit_ignores_metadata_dirs(tmp_path: Path) -> None:
    cases = ["case01"]
    organs = ["liver"]
    provenance = {
        "run_id": "run",
        "checkpoint_sha256": "a" * 64,
        "case_list_sha256": "b" * 64,
        "target_config_sha256": "c" * 64,
        "expected_case_ids": cases,
        "case_list_duplicate_case_ids": [],
    }
    prediction_root = tmp_path / "student_predictions_postprocessed"
    _write_valid_prediction_set(prediction_root, case_ids=cases, organs=organs, provenance=provenance)
    (prediction_root / "parent_rois" / "case01").mkdir(parents=True)

    audit = em.audit_prompt_student_predictions(
        1,
        prediction_root=prediction_root,
        provenance=provenance,
        expected_organs=organs,
    )

    assert audit["status"] == "passed"
    assert audit["actual_case_ids"] == ["case01"]


def test_student_postprocess_preserves_exact_target_collision_names(tmp_path: Path) -> None:
    from cli_anything.medai.core.student_postprocess import process_student_root

    input_root = tmp_path / "student_predictions"
    _write_nii(input_root / "case01" / "inferior_vena_cava.nii.gz")
    _write_nii(input_root / "case01" / "postcava.nii.gz")

    output_root = tmp_path / "student_predictions_postprocessed"
    summary = process_student_root(
        input_root=input_root,
        output_root=output_root,
        parent_roots=[input_root],
        taxonomy_path=ROOT / "configs" / "organ_taxonomy.json",
        policy_path=ROOT / "configs" / "organ_postprocess_policy.yaml",
        case_list=None,
        organs=["inferior_vena_cava", "postcava"],
        overwrite=True,
    )

    assert summary["status"] == "success"
    assert (output_root / "case01" / "inferior_vena_cava.nii.gz").is_file()
    assert (output_root / "case01" / "postcava.nii.gz").is_file()


def test_next_round_student_candidate_validation_requires_complete_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "outputs"
    case_list = _write_case_list(tmp_path / "cases.csv", ["case01"])
    target_config = tmp_path / "targets.json"
    organs = ["inferior_vena_cava", "postcava"]
    target_config.write_text(json.dumps({"target_organs": organs}), encoding="utf-8")
    monkeypatch.setattr(em, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(em, "CASE_LIST", case_list)
    monkeypatch.setattr(em, "PROMPT_TARGET_CONFIG", target_config)
    monkeypatch.setenv("MEDAI_USE_LEGACY_STUDENT_COMPETITION_ALLOWLIST", "0")

    raw_root = output_root / "round1" / "student_predictions"
    post_root = output_root / "round1" / "student_predictions_postprocessed"
    case_provenance = em.case_list_provenance()
    provenance = {
        "run_id": "round1-ckpt-cases-targets",
        "checkpoint_sha256": "a" * 64,
        "case_list_sha256": case_provenance["sha256"],
        "target_config_sha256": em._sha256_file(target_config),
        "expected_case_ids": case_provenance["case_ids"],
        "case_list_duplicate_case_ids": [],
    }
    _write_valid_prediction_set(raw_root, case_ids=["case01"], organs=organs, provenance=provenance)
    _write_valid_prediction_set(post_root, case_ids=["case01"], organs=organs, provenance=provenance)
    (raw_root / "inference_audit.json").write_text(
        json.dumps({"status": "passed", "failures": [], "provenance": provenance}),
        encoding="utf-8",
    )
    (raw_root / "student_inference_summary.json").write_text(
        json.dumps({
            "status": "success",
            "eligible_for_competition": True,
            "checkpoint_eligible_for_next_round": True,
            **provenance,
        }),
        encoding="utf-8",
    )
    (post_root / "organ_type_postprocess_summary.json").write_text(
        json.dumps({
            "status": "success",
            "input_root": str(raw_root),
            "output_root": str(post_root),
            "processed_masks": 2,
        }),
        encoding="utf-8",
    )
    with (post_root / "student_containment_postprocess_per_mask.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "organ", "output_path", "status"])
        writer.writeheader()
        for organ in organs:
            writer.writerow({
                "case_id": "case01",
                "organ": organ,
                "output_path": str(post_root / "case01" / f"{organ}.nii.gz"),
                "status": "postprocessed_no_voxels_removed",
            })
    (post_root / "parent_rois").mkdir()

    root, audit = em._validated_student_prediction_root_for_next_round(2)
    assert root == post_root
    assert audit["status"] == "success"

    with (post_root / "student_containment_postprocess_per_mask.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "organ", "output_path", "status"])
        writer.writeheader()
        for organ in organs:
            writer.writerow({
                "case_id": "case01",
                "organ": organ,
                "output_path": str(post_root / "case01" / f"{organ}.nii.gz"),
                "status": "warning_missing_parent_roi_copied_raw" if organ == "postcava" else "postprocessed",
            })
    _, high_risk_failed = em._validated_student_prediction_root_for_next_round(2)
    assert high_risk_failed["status"] == "success"
    assert high_risk_failed["high_risk_student_postprocess_blocked_from_labelcritic"] == [
        "case01/postcava:warning_missing_parent_roi_copied_raw"
    ]

    with (post_root / "student_containment_postprocess_per_mask.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "organ", "output_path", "status"])
        writer.writeheader()
        for organ in organs:
            writer.writerow({
                "case_id": "case01",
                "organ": organ,
                "output_path": str(post_root / "case01" / f"{organ}.nii.gz"),
                "status": "postprocessed",
            })
    (post_root / "case01" / "postcava.nii.gz").unlink()
    _, failed = em._validated_student_prediction_root_for_next_round(2)
    assert failed["status"] == "failed"
    assert any("allowlisted_masks_missing" in reason for reason in failed["reasons"])


def test_high_risk_student_postprocess_warning_blocks_labelcritic(tmp_path: Path) -> None:
    from cli_anything.medai.core.multimodel_loop import (
        _CaseMaskCache,
        _evaluate_candidate_for_organ,
    )
    from cli_anything.medai.core.organ_taxonomy import load_taxonomy

    ct = _write_nii(tmp_path / "ct.nii.gz")
    seg_dir = tmp_path / "student_prev"
    _write_nii(seg_dir / "postcava.nii.gz")

    row = _evaluate_candidate_for_organ(
        ct=ct,
        organ="postcava",
        model_key="student_prev",
        seg_dir=seg_dir,
        raw_seg_dir=seg_dir,
        alias_config={},
        shapekit_report={
            "status": "success",
            "per_mask_status_by_organ": {
                "postcava": {"status": "warning_missing_parent_roi_copied_raw"}
            },
        },
        current_ref=None,
        current_ref_exists=False,
        vlm_threshold=0.5,
        case_id="case01",
        mask_cache=_CaseMaskCache(),
        taxonomy=load_taxonomy(ROOT / "configs" / "organ_taxonomy.json"),
        presence_context={},
    )

    assert row["candidate_qc_status"] == "fail"
    assert row["eligible_for_labelcritic"] is False
    assert "high_risk_postprocess_required" in row["candidate_qc_flags"]


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
    assert "manifest_case_scope_mismatch" in audit["failures"]
    assert "estep_case_scope_mismatch" in audit["failures"]
    assert "manifest_case_scope_mismatch" in audit["failures"]


def test_round1_reference_requires_explicit_frozen_root(tmp_path: Path, monkeypatch) -> None:
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
    monkeypatch.setenv("MEDAI_ROUND_REFERENCE_ROOT", str(formal.parent))

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
