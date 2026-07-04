from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.repair_round2_gate_from_existing_estep import repair_round2_gate


def _save_nii(path: Path, data: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = np.zeros((8, 8, 8), dtype=np.uint8)
        data[2:5, 2:5, 2:5] = 1
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _write_repair_case(
    run_root: Path,
    *,
    case_id: str = "case001",
    organ: str = "spleen",
    expected_presence: str = "expected_present",
    fov_status: str = "fully_visible",
    candidate_model: str = "round_prev_selected",
    candidate_flags: list[str] | None = None,
    candidate_qc_status: str = "pass",
    selection_method: str = "label_critic_audit_only",
) -> tuple[Path, Path, Path]:
    ct = _save_nii(run_root / "ct" / f"{case_id}.nii.gz", np.zeros((8, 8, 8), dtype=np.int16))
    candidate = _save_nii(run_root / "candidate" / case_id / f"{organ}.nii.gz")
    meta_dir = run_root / "round2" / "estep" / "annotation_versions" / case_id
    row = {
        "case_id": case_id,
        "ct_path": str(ct),
        "organ": organ,
        "expected_presence": expected_presence,
        "fov_status": fov_status,
        "candidate_models": [candidate_model],
        "candidate_count": 1,
        "candidate_predictions": [
            {
                "model": candidate_model,
                "prediction": str(candidate),
                "pre_shapekit_prediction": str(candidate),
                "candidate_qc_status": candidate_qc_status,
                "candidate_qc_score": 1.0,
                "candidate_qc_flags": candidate_flags or [],
                "candidate_shapekit_status": "success",
                "identity_status": "valid",
                "identity_mismatch_reasons": [],
            }
        ],
        "selection_method": selection_method,
        "selection_status": "review_required",
        "selected_model": None,
        "source_model": None,
        "selected_prediction": None,
        "grade": "D",
        "target_type": "rejected",
        "quality_flags": ["labelcritic_audit_only"],
        "review_flags": ["automatic_abstention"],
        "identity_status": "missing",
        "identity_mismatch_reasons": ["no_selected_candidate"],
    }
    doc = {
        "case_id": case_id,
        "ct_path": str(ct),
        "selection_rows": [row],
        "selected_organs": [],
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "selection_metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    return ct, candidate, meta_dir / "selection_metadata.json"


def test_trusted_round_prev_selected_can_be_carried_forward(tmp_path: Path) -> None:
    _, _, meta = _write_repair_case(tmp_path)

    report = repair_round2_gate(tmp_path, tmp_path / "round1_selected", apply=True)

    assert report["repaired_count"] == 1
    doc = json.loads(meta.read_text(encoding="utf-8"))
    row = doc["selection_rows"][0]
    assert row["selection_method"] == "round_prev_selected_carry_forward"
    assert row["selected_model"] == "round_prev_selected"
    assert row["grade"] == "B"
    assert row["target_type"] == "positive_hard"
    assert row["training_weight"] == 0.7
    assert row["distillation_eligible"] is True
    assert row["should_enter_student_training"] is True
    assert row["publication_status"] == "accepted_carry_forward"
    assert Path(row["final_mask"]).is_file()
    assert doc["selected_organs"][0]["organ"] == "spleen"
    assert (tmp_path / "round2" / "estep" / "repair_backups" / "carry_forward_v1" / "case001" / "selection_metadata.json").is_file()


def test_student_prev_is_not_a_carry_forward_source(tmp_path: Path) -> None:
    _, _, meta = _write_repair_case(tmp_path, candidate_model="student_prev")

    report = repair_round2_gate(tmp_path, tmp_path / "round1_selected", apply=True)

    assert report["repaired_count"] == 0
    row = json.loads(meta.read_text(encoding="utf-8"))["selection_rows"][0]
    assert row["selection_method"] == "label_critic_audit_only"
    assert row["selected_model"] is None


def test_audit_only_labelcritic_winner_without_round_prev_is_not_formalized(tmp_path: Path) -> None:
    _, _, meta = _write_repair_case(
        tmp_path,
        candidate_model="teacher_a",
        selection_method="label_critic_audit_only",
    )

    report = repair_round2_gate(tmp_path, tmp_path / "round1_selected", apply=True)

    assert report["repaired_count"] == 0
    row = json.loads(meta.read_text(encoding="utf-8"))["selection_rows"][0]
    assert row["selection_method"] == "label_critic_audit_only"
    assert row["selected_model"] is None


@pytest.mark.parametrize(
    "flag",
    ["zero_volume_mask", "geometry_mismatch", "postprocess_failed", "many_connected_components"],
)
def test_hard_flags_block_carry_forward(tmp_path: Path, flag: str) -> None:
    _, _, meta = _write_repair_case(tmp_path, candidate_flags=[flag])

    report = repair_round2_gate(tmp_path, tmp_path / "round1_selected", apply=True)

    assert report["repaired_count"] == 0
    row = json.loads(meta.read_text(encoding="utf-8"))["selection_rows"][0]
    assert row["selection_status"] == "review_required"


@pytest.mark.parametrize(
    ("expected_presence", "fov_status"),
    [
        ("unknown", "fully_visible"),
        ("expected_present", "out_of_fov"),
    ],
)
def test_unknown_presence_or_out_of_fov_blocks_carry_forward(
    tmp_path: Path,
    expected_presence: str,
    fov_status: str,
) -> None:
    _, _, meta = _write_repair_case(
        tmp_path,
        expected_presence=expected_presence,
        fov_status=fov_status,
    )

    report = repair_round2_gate(tmp_path, tmp_path / "round1_selected", apply=True)

    assert report["repaired_count"] == 0
    row = json.loads(meta.read_text(encoding="utf-8"))["selection_rows"][0]
    assert row["selected_model"] is None


def _write_gate_case(root: Path, round_idx: int, case_id: str, organ: str, *, selected: bool) -> None:
    case_dir = root / f"round{round_idx}" / "estep" / "annotation_versions" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    final_mask = case_dir / "updated" / f"{organ}.nii.gz"
    if selected:
        final_mask.parent.mkdir(parents=True, exist_ok=True)
        final_mask.write_bytes(b"mask")
    row = {
        "case_id": case_id,
        "organ": organ,
        "expected_presence": "expected_present",
        "candidate_models": ["teacher_a", "teacher_b"],
        "candidate_count": 2,
        "independent_family_count": 2,
        "candidate_predictions": [],
        "labelcritic_compare_used": False,
        "labelcritic_compare_skipped_reason": "not_executed",
    }
    selected_organs = []
    if selected:
        selected_organs.append({
            "case_id": case_id,
            "organ": organ,
            "selection_status": "selected",
            "selected_model": "teacher_a",
            "grade": "A",
            "target_type": "positive_hard",
            "final_mask": str(final_mask),
            "publication_status": "accepted",
        })
        row["candidate_predictions"] = [{"candidate_shapekit_status": "success"}]
    doc = {
        "case_id": case_id,
        "quality_contract_version": "estep_quality_contract_v3",
        "fov_policy_version": "fov_appearance_regions_v4",
        "selection_rows": [row],
        "selected_organs": selected_organs,
    }
    (case_dir / "selection_metadata.json").write_text(json.dumps(doc), encoding="utf-8")


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"training_gate_summary": {"included_grade_counts": {"A": 1, "B": 0}}}),
        encoding="utf-8",
    )


def test_gate_records_single_shapekit_and_labelcritic_failures_without_blocking_when_cohort_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_em_training as em

    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path)
    for idx in range(10):
        _write_gate_case(tmp_path, 2, f"case{idx}", "adrenal_gland_left", selected=idx != 0)
    manifest = tmp_path / "round2" / "mstep" / "voxtell_prompt_student_manifest.json"
    _write_manifest(manifest)

    gate = em.formal_estep_gate(2, {"status": "success", "num_cases": 10}, manifest)

    assert gate["status"] == "success"
    assert gate["shapekit_coverage_failures"]
    assert gate["labelcritic_coverage_failures"]
    assert gate["blocking_shapekit_coverage_failures"] == []
    assert gate["blocking_labelcritic_coverage_failures"] == []
    assert (tmp_path / "round2" / "estep" / "formal_gate.json").is_file()
    summary = json.loads((tmp_path / "round2" / "round_summary.json").read_text(encoding="utf-8"))
    assert summary["estep_formal_gate"]["status"] == "success"


def test_gate_blocks_when_key_organ_cohort_coverage_fails_and_persists_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_em_training as em

    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path)
    for idx in range(10):
        _write_gate_case(tmp_path, 2, f"case{idx}", "spleen", selected=idx < 7)
    manifest = tmp_path / "round2" / "mstep" / "voxtell_prompt_student_manifest.json"
    _write_manifest(manifest)

    gate = em.formal_estep_gate(2, {"status": "success", "num_cases": 10}, manifest)

    assert gate["status"] == "failed"
    assert gate["reason"] == "cohort_key_organ_coverage_failed"
    assert gate["cohort_failed_key_organs"] == ["spleen"]
    assert gate["blocking_shapekit_coverage_failures"]
    assert (tmp_path / "round2" / "estep" / "formal_gate.json").is_file()
