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
    round_root = path.parents[1]
    ann_root = round_root / "estep" / "annotation_versions"
    items = []
    for mask in sorted(ann_root.glob("*/updated/*.nii.gz")):
        items.append({
            "case_id": mask.parents[1].name,
            "organ": mask.name[:-7],
            "supervision_type": "positive",
            "target_type": "positive_hard",
            "grade": "A",
            "training_weight": 1.0,
            "distillation_eligible": True,
            "mask_path": str(mask),
        })
    grade_counts = {"A": len(items), "B": 0, "C": 0, "D": 0}
    path.write_text(
        json.dumps({
            "items": items,
            "grade_counts": grade_counts,
            "training_gate_summary": {"included_grade_counts": grade_counts},
        }),
        encoding="utf-8",
    )


def _write_target_config(path: Path, organs: list[str]) -> Path:
    path.write_text(
        json.dumps({
            "target_organs": organs,
            "organ_to_student_id": {organ: idx + 1 for idx, organ in enumerate(organs)},
            "organ_to_prompt": {organ: organ.replace("_", " ") for organ in organs},
        }),
        encoding="utf-8",
    )
    return path


def _write_case_list(path: Path, case_id: str, ct: Path) -> Path:
    path.write_text(f"case_id,ct_path\n{case_id},{ct}\n", encoding="utf-8")
    return path


def _write_voxtell_case(
    root: Path,
    *,
    case_id: str,
    ct: Path,
    organ: str,
    mask_data: np.ndarray | None = None,
    selected_model: str = "teacher_a",
    grade: str = "A",
    target_type: str = "hard",
    qc_status: str = "pass",
    dataset_role: str = "pseudo_label",
    ground_truth_status: str = "machine_generated_candidate",
) -> Path:
    case_dir = root / case_id
    updated = case_dir / "updated"
    mask = _save_nii(updated / f"{organ}.nii.gz", mask_data)
    selected = {
        "case_id": case_id,
        "ct_path": str(ct),
        "organ": organ,
        "selection_status": "selected",
        "selected_model": selected_model,
        "source_model": selected_model,
        "grade": grade,
        "target_type": target_type,
        "training_weight": 1.0 if grade in {"A", "B"} else 0.5,
        "distillation_eligible": True,
        "final_mask": str(mask),
        "mask_path": str(mask),
        "scoring_schema_version": "autolabel_core_v2",
        "selected_candidate_qc_status": qc_status,
        "selected_candidate_qc_flags": [],
        "shapekit_status": "success",
        "dataset_role": dataset_role,
        "ground_truth_status": ground_truth_status,
    }
    doc = {
        "case_id": case_id,
        "quality_contract_version": "estep_quality_contract_v3",
        "fov_policy_version": "fov_appearance_regions_v4",
        "selected_organs": [selected],
        "selection_rows": [selected],
    }
    (case_dir / "selection_metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    return mask


def test_negative_absent_does_not_pollute_positive_manifest_counts(tmp_path: Path) -> None:
    from cli_anything.medai.core.voxtell_student import VoxTellStudent

    organs = ["liver", "brain"]
    target_config = _write_target_config(tmp_path / "targets.json", organs)
    cases_root = tmp_path / "cases"
    case_id = "case001"
    ct = _save_nii(tmp_path / "ct.nii.gz", np.zeros((8, 8, 8), dtype=np.int16))
    _write_case_list(tmp_path / "cases.csv", case_id, ct)
    _write_voxtell_case(cases_root, case_id=case_id, ct=ct, organ="liver")
    zero = np.zeros((8, 8, 8), dtype=np.uint8)
    zero_mask = _save_nii(cases_root / case_id / "negative_targets" / "zero_mask.nii.gz", zero)
    doc_path = cases_root / case_id / "selection_metadata.json"
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    absent = {
        **doc["selected_organs"][0],
        "organ": "brain",
        "target_type": "negative_absent",
        "grade": "A",
        "training_weight": 0.1,
        "final_mask": str(zero_mask),
        "mask_path": str(zero_mask),
        "supervision_type": "negative",
        "distillation_role": "negative",
        "zero_mask_role": "negative_absent_target_mask",
    }
    doc["selected_organs"].append(absent)
    doc["selection_rows"].append(absent)
    doc_path.write_text(json.dumps(doc), encoding="utf-8")

    manifest = VoxTellStudent(model_dir=tmp_path, target_config=target_config).build_training_manifest(
        cases_root=cases_root,
        output_manifest=tmp_path / "manifest.json",
        case_list=tmp_path / "cases.csv",
        require_images=True,
    )

    assert manifest["num_positive_items"] == 1
    assert manifest["num_negative_items"] == 1
    assert manifest["grade_counts"] == {"A": 1, "B": 0, "C": 0, "D": 0}
    assert manifest["all_grade_counts"]["A"] == 2
    gate_summary = manifest["training_gate_summary"]
    assert gate_summary["included_grade_counts"] == {"A": 1, "B": 0, "C": 0, "D": 0}
    assert gate_summary["ab_nonzero"] == 1
    assert gate_summary["negative_absent_excluded_from_positive_quota"] is True


def test_cumulative_manifest_replays_only_safe_promoted_positive_rows(tmp_path: Path) -> None:
    from cli_anything.medai.core.voxtell_student import VoxTellStudent

    organs = ["liver", "spleen", "pancreas", "kidney_left"]
    target_config = _write_target_config(tmp_path / "targets.json", organs)
    case_id = "case001"
    ct = _save_nii(tmp_path / "ct.nii.gz", np.zeros((8, 8, 8), dtype=np.int16))
    _write_case_list(tmp_path / "cases.csv", case_id, ct)
    current_root = tmp_path / "round2_cases"
    _write_voxtell_case(current_root, case_id=case_id, ct=ct, organ="liver")

    replay_root = tmp_path / "replay_masks"
    spleen = _save_nii(replay_root / "spleen.nii.gz")
    kidney = _save_nii(replay_root / "kidney_left.nii.gz")
    empty_pancreas = _save_nii(replay_root / "pancreas.nii.gz", np.zeros((8, 8, 8), dtype=np.uint8))
    historical = {
        "items": [
            {
                "case_id": case_id,
                "organ": "spleen",
                "image": str(ct),
                "ct_path": str(ct),
                "mask": str(spleen),
                "mask_path": str(spleen),
                "supervision_type": "positive",
                "target_type": "positive_hard",
                "grade": "A",
                "training_weight": 1.0,
                "distillation_eligible": True,
                "selected_model": "teacher_a",
                "source_model": "teacher_a",
                "dataset_role": "pseudo_label",
                "ground_truth_status": "machine_generated_candidate",
                "selected_candidate_qc_status": "pass",
            },
            {
                "case_id": case_id,
                "organ": "kidney_left",
                "image": str(ct),
                "mask": str(kidney),
                "mask_path": str(kidney),
                "supervision_type": "positive",
                "target_type": "positive_hard",
                "grade": "A",
                "training_weight": 1.0,
                "distillation_eligible": True,
                "selected_model": "student_prev",
                "source_model": "student_prev",
                "dataset_role": "pseudo_label",
                "ground_truth_status": "machine_generated_candidate",
                "selected_candidate_qc_status": "pass",
            },
            {
                "case_id": case_id,
                "organ": "pancreas",
                "image": str(ct),
                "mask": str(empty_pancreas),
                "mask_path": str(empty_pancreas),
                "supervision_type": "positive",
                "target_type": "positive_hard",
                "grade": "A",
                "training_weight": 1.0,
                "distillation_eligible": True,
                "selected_model": "teacher_a",
                "source_model": "teacher_a",
                "dataset_role": "pseudo_label",
                "ground_truth_status": "machine_generated_candidate",
                "selected_candidate_qc_status": "pass",
            },
        ],
    }
    replay_manifest = tmp_path / "round1_manifest.json"
    replay_manifest.write_text(json.dumps(historical), encoding="utf-8")

    manifest = VoxTellStudent(model_dir=tmp_path, target_config=target_config).build_training_manifest(
        cases_root=current_root,
        output_manifest=tmp_path / "manifest.json",
        case_list=tmp_path / "cases.csv",
        require_images=True,
        historical_replay_manifests=[replay_manifest],
    )

    organs_in_manifest = {row["organ"] for row in manifest["items"] if row.get("supervision_type") == "positive"}
    assert "liver" in organs_in_manifest
    assert "spleen" in organs_in_manifest
    assert "kidney_left" not in organs_in_manifest
    assert "pancreas" not in organs_in_manifest
    replay_rows = [row for row in manifest["items"] if row.get("historical_replay")]
    assert [row["organ"] for row in replay_rows] == ["spleen"]
    cumulative = manifest["cumulative_manifest_summary"]
    assert cumulative["num_replay_positive_items"] == 1
    assert "spleen" in cumulative["recovered_historical_trainable_positive_organs"]
    excluded_reasons = {row["reason"] for row in cumulative["replay_excluded_rows"]}
    assert "student_prediction_not_replay_memory" in excluded_reasons
    assert "mask_missing" not in excluded_reasons
    assert "empty_mask_not_replay_memory" in excluded_reasons


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


def test_gate_uses_manifest_trainable_positive_not_negative_absent_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_em_training as em

    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path)
    case_dir = tmp_path / "round2" / "estep" / "annotation_versions" / "case001"
    case_dir.mkdir(parents=True, exist_ok=True)
    zero = _save_nii(case_dir / "updated" / "brain.nii.gz", np.zeros((8, 8, 8), dtype=np.uint8))
    doc = {
        "case_id": "case001",
        "quality_contract_version": "estep_quality_contract_v3",
        "fov_policy_version": "fov_appearance_regions_v4",
        "selection_rows": [{
            "case_id": "case001",
            "organ": "brain",
            "expected_presence": "expected_absent",
            "target_type": "negative_absent",
        }],
        "selected_organs": [{
            "case_id": "case001",
            "organ": "brain",
            "selection_status": "selected",
            "selected_model": "scan_coverage_absent",
            "grade": "A",
            "target_type": "negative_absent",
            "supervision_type": "negative",
            "final_mask": str(zero),
            "publication_status": "accepted_absent_negative",
        }],
    }
    (case_dir / "selection_metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    manifest = tmp_path / "round2" / "mstep" / "voxtell_prompt_student_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "items": [{
            "case_id": "case001",
            "organ": "brain",
            "supervision_type": "negative",
            "target_type": "negative_absent",
            "grade": "A",
            "training_weight": 0.1,
            "distillation_eligible": True,
            "mask_path": str(zero),
        }],
        "grade_counts": {"A": 0, "B": 0, "C": 0, "D": 0},
        "training_gate_summary": {
            "included_grade_counts": {"A": 0, "B": 0, "C": 0, "D": 0},
            "ab_nonzero": 0,
            "negative_absent_excluded_from_positive_quota": True,
        },
    }), encoding="utf-8")

    gate = em.formal_estep_gate(2, {"status": "success", "num_cases": 1}, manifest)

    assert gate["status"] == "failed"
    assert gate["reason"] == "A_plus_B_zero"
    assert gate["ab_nonzero"] == 0
    assert gate["manifest_trainable_positive_summary"]["num_trainable_positive_items"] == 0


def test_gate_blocks_historical_replay_deficit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_em_training as em

    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path)
    _write_gate_case(tmp_path, 2, "case001", "spleen", selected=True)
    manifest = tmp_path / "round2" / "mstep" / "voxtell_prompt_student_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    mask = tmp_path / "round2" / "estep" / "annotation_versions" / "case001" / "updated" / "spleen.nii.gz"
    manifest.write_text(json.dumps({
        "items": [{
            "case_id": "case001",
            "organ": "spleen",
            "supervision_type": "positive",
            "target_type": "positive_hard",
            "grade": "A",
            "training_weight": 1.0,
            "distillation_eligible": True,
            "mask_path": str(mask),
        }],
        "training_gate_summary": {"included_grade_counts": {"A": 1, "B": 0, "C": 0, "D": 0}},
        "cumulative_manifest_summary": {
            "missing_historical_trainable_positive_organs": ["pancreas"],
        },
    }), encoding="utf-8")

    gate = em.formal_estep_gate(2, {"status": "success", "num_cases": 1}, manifest)

    assert gate["status"] == "failed"
    assert gate["historical_replay_deficits"] == ["pancreas"]
    assert gate["reason"] in {"trainer_positive_coverage_failed", "historical_replay_deficit"}


def test_round2_retention_audit_requires_nonzero_finite_loss(tmp_path: Path) -> None:
    import scripts.run_em_training as em

    model_dir = tmp_path / "model"
    (model_dir / "fold_0").mkdir(parents=True)
    (model_dir / "plans.json").write_text("{}", encoding="utf-8")
    (model_dir / "fold_0" / "checkpoint_final.pth").write_bytes(b"checkpoint")
    train_result = {
        "official_retention_weight": 0.2,
        "mean_retention_loss": 0.031,
    }
    audit = em.build_retention_audit(
        round_idx=2,
        source_model_dir=model_dir,
        train_result=train_result,
        required=True,
    )
    assert audit["status"] == "passed"
    assert audit["retention_loss_nonzero_finite"] is True

    failed = em.build_retention_audit(
        round_idx=2,
        source_model_dir=model_dir,
        train_result={**train_result, "mean_retention_loss": 0.0},
        required=True,
    )
    assert failed["status"] == "failed"
    assert "mean_retention_loss_not_nonzero_finite" in failed["failures"]


def test_soft_consensus_repair_creates_c_soft_only_from_safe_candidates(tmp_path: Path) -> None:
    from scripts.repair_soft_consensus_positive_targets import repair_soft_consensus_targets

    ann = tmp_path / "round2" / "estep" / "annotation_versions"
    case_dir = ann / "case001"
    case_dir.mkdir(parents=True)
    ct = _save_nii(tmp_path / "ct.nii.gz", np.zeros((8, 8, 8), dtype=np.int16))
    mask_a = _save_nii(tmp_path / "cand_a.nii.gz")
    data_b = np.zeros((8, 8, 8), dtype=np.uint8)
    data_b[3:6, 3:6, 3:6] = 1
    mask_b = _save_nii(tmp_path / "cand_b.nii.gz", data_b)
    mask_bad = _save_nii(tmp_path / "student_prev.nii.gz")
    row = {
        "case_id": "case001",
        "ct_path": str(ct),
        "organ": "small_bowel",
        "expected_presence": "expected_present",
        "selection_status": "review_required",
        "selection_method": "label_critic_audit_only",
        "selected_model": None,
        "grade": "C",
        "target_type": "positive_soft",
        "training_weight": 0.0,
        "distillation_eligible": False,
        "labelcritic_compare_used": True,
        "review_flags": ["automatic_abstention"],
        "scoring_schema_version": "autolabel_core_v2",
        "candidate_predictions": [
            {
                "model": "teacher_a",
                "prediction": str(mask_a),
                "candidate_qc_status": "pass",
                "candidate_qc_flags": [],
                "candidate_shapekit_status": "success",
                "identity_status": "valid",
            },
            {
                "model": "teacher_b",
                "prediction": str(mask_b),
                "candidate_qc_status": "pass",
                "candidate_qc_flags": [],
                "candidate_shapekit_status": "success",
                "identity_status": "valid",
            },
            {
                "model": "student_prev",
                "prediction": str(mask_bad),
                "candidate_qc_status": "pass",
                "candidate_qc_flags": [],
                "candidate_shapekit_status": "success",
                "identity_status": "valid",
            },
        ],
    }
    (case_dir / "selection_metadata.json").write_text(json.dumps({
        "case_id": "case001",
        "selection_rows": [row],
        "selected_organs": [],
    }), encoding="utf-8")

    report = repair_soft_consensus_targets(ann, apply=True)

    assert report["repaired_count"] == 1
    doc = json.loads((case_dir / "selection_metadata.json").read_text(encoding="utf-8"))
    selected = doc["selected_organs"][0]
    assert selected["grade"] == "C"
    assert selected["target_type"] == "positive_soft"
    assert selected["training_weight"] == 0.1
    assert selected["distillation_eligible"] is True
    assert selected["selected_model"] == "soft_consensus_qc_pass_candidates"
    assert selected["soft_consensus_source_models"] == ["teacher_a", "teacher_b"]
    assert Path(selected["probability_mask_path"]).is_file()
    assert Path(selected["final_mask"]).is_file()
