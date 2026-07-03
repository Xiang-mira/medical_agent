from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_vendor_lock_matches() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from audit_labelcritic_vendor import audit

    result = audit(
        ROOT / "configs" / "labelcritic_vendor_lock.json",
        ROOT / "third_party" / "LabelCritic-main",
    )
    assert result["status"] == "passed"


def test_description_v2_and_colon_rules() -> None:
    doc = json.loads((ROOT / "configs" / "organ_ct_appearance_373.json").read_text())
    assert doc["entry_count"] == 373
    colon = doc["organ_ct_appearance"]["colon"]
    required = {
        "canonical_name", "aliases", "laterality", "parent_structures",
        "expected_body_regions", "ct_location", "ct_appearance", "morphology",
        "adjacent_structures", "continuity_prior", "symmetry_prior",
        "partial_fov_behavior", "common_failure_modes", "penalty_rules",
        "provenance", "source_type", "validation_status",
    }
    assert required <= set(colon)
    assert len(colon["field_provenance"]) >= 14
    text = json.dumps(colon).lower()
    for phrase in ("tubular", "random disconnected", "small bowel", "stomach", "abdominal wall"):
        assert phrase in text


def test_short_abdominal_ct_does_not_claim_full_thorax(tmp_path: Path) -> None:
    from cli_anything.medai.core.scan_coverage import infer_scan_coverage

    data = np.zeros((32, 32, 30), dtype=np.float32)
    data[8:24, 8:24, :] = 40
    data[12:20, 12:20, :8] = -900
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    path = tmp_path / "short_abdomen.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    result = infer_scan_coverage(path)
    assert result["body_length_mm"] <= 150.0
    assert result["has_thorax_coverage"] is False
    if result["thorax_air_slices"] >= 5:
        assert result["has_partial_thorax_coverage"] is True


def test_four_state_fov_never_uses_missing_teacher() -> None:
    from cli_anything.medai.core.multimodel_loop import _fov_status_for_organ

    context = {
        "has_region_evidence": True,
        "has_abdomen_coverage": True,
        "has_pelvis_coverage": False,
        "has_thorax_coverage": False,
        "has_partial_thorax_coverage": True,
        "has_head_coverage": False,
        "has_extremity_coverage": False,
    }
    assert _fov_status_for_organ("liver", context) == "fully_visible"
    assert _fov_status_for_organ("lung_left", context) == "partially_visible"
    assert _fov_status_for_organ("brain", context) == "out_of_fov"
    assert _fov_status_for_organ("bladder", context) == "unknown"


def test_materialized_case_has_exactly_373_fail_closed_records(tmp_path: Path) -> None:
    from cli_anything.medai.core.multimodel_loop import _materialize_case_373_targets

    organs = json.loads(
        (ROOT / "configs" / "student_3d_prompt_target_organs.json").read_text()
    )["target_organs"]
    ct = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), np.eye(4)), ct)
    rows: list[dict] = []
    metadata: list[dict] = []
    summary = _materialize_case_373_targets(
        case_id="synthetic_contract_case",
        ct=ct,
        organs=organs,
        case_updated=tmp_path / "updated",
        selection_rows=rows,
        selected_metadata=metadata,
        presence_context={
            "has_region_evidence": True,
            "has_abdomen_coverage": True,
            "has_pelvis_coverage": False,
            "has_thorax_coverage": False,
            "has_partial_thorax_coverage": True,
            "has_head_coverage": False,
            "has_extremity_coverage": False,
            "coverage_evidence": ["synthetic_abdomen_contract_test"],
        },
    )
    assert summary["complete_case_373"] is True
    assert summary["target_identity_holds"] is True
    assert len(rows) == len({row["organ"] for row in rows}) == 373
    assert set(summary["target_type_counts"]) <= {
        "positive_hard", "positive_soft", "negative_absent",
        "unresolved_visible", "partial_fov", "rejected",
    }
    for row in rows:
        if row["target_type"] in {"unresolved_visible", "partial_fov"}:
            assert row["training_weight"] == 0
            assert row["zero_mask_role"] == "io_placeholder_not_training_target"
        if row["target_type"] == "negative_absent":
            assert row["absence_confidence"] == "high"
            assert row["fov_status"] == "out_of_fov"


def test_near_identical_selection_is_order_invariant_and_never_fusion(
    tmp_path: Path, monkeypatch
) -> None:
    from cli_anything.medai.core import multimodel_loop as loop

    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[4:12, 4:12, 4:12] = 1
    paths = {}
    for name in ("fusion", "teacher_b", "teacher_a"):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(mask, np.eye(4)), path)
        paths[name] = path
    candidates = [
        {
            "model": "fusion_consensus",
            "prediction": str(paths["fusion"]),
            "is_fusion": True,
            "eligible_for_labelcritic": True,
        },
        {
            "model": "teacher_b",
            "prediction": str(paths["teacher_b"]),
            "eligible_for_labelcritic": True,
        },
        {
            "model": "teacher_a",
            "prediction": str(paths["teacher_a"]),
            "eligible_for_labelcritic": True,
        },
    ]
    called = []

    def fail_compare(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("near-identical teacher masks must not invoke VLM")

    monkeypatch.setattr(loop, "run_labelcritic_compare", fail_compare)
    winners = []
    for order in (candidates, list(reversed(candidates)), [candidates[1], candidates[0], candidates[2]]):
        selected, evidence = loop._select_candidate(
            ct=tmp_path / "ct.nii.gz",
            organ="liver",
            candidates=[dict(row) for row in order],
            out=tmp_path / "out",
            case_id="case",
            enable_critic=True,
            critic_backend="labelcritic",
            critic_base_url="http://localhost",
            critic_port=8000,
            timeout_sec=30,
            dry_run=False,
            labelcritic_options={},
        )
        winners.append(selected["model"])
        assert evidence["selection_method"] == "geometric_teacher_consensus"
        assert evidence["winner_is_original_teacher"] is True
        assert evidence["excluded_fusion_candidates"] == ["fusion_consensus"]
    assert winners == ["teacher_a", "teacher_a", "teacher_a"]
    assert called == []


def test_geometric_consensus_ignores_outlier_and_family_metadata(tmp_path: Path) -> None:
    from cli_anything.medai.core import multimodel_loop as loop

    ct = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.int16), np.eye(4)), ct)
    base = np.zeros((16, 16, 16), dtype=np.uint8)
    base[4:12, 4:12, 4:12] = 1
    outlier = np.zeros_like(base)
    outlier[1:5, 1:5, 1:5] = 1
    paths = {}
    for name, arr in {"teacher_a": base, "teacher_b": base, "teacher_c": outlier}.items():
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
        paths[name] = path

    def candidates(family: str) -> list[dict]:
        return [
            {
                "model": name,
                "prediction": str(paths[name]),
                "eligible_for_labelcritic": True,
                "identity_status": "valid",
                "candidate_qc_status": "pass",
                "evidence_family": family,
            }
            for name in ("teacher_c", "teacher_b", "teacher_a")
        ]

    selected_same, evidence_same = loop._select_candidate(
        ct=ct, organ="liver", candidates=candidates("same_family"),
        out=tmp_path / "out_same", case_id="case", enable_critic=True,
        critic_backend="labelcritic", critic_base_url="http://localhost",
        critic_port=8000, timeout_sec=30, dry_run=False,
        labelcritic_options={},
    )
    selected_changed, evidence_changed = loop._select_candidate(
        ct=ct, organ="liver", candidates=[
            {**row, "evidence_family": f"changed_{idx}"}
            for idx, row in enumerate(candidates("same_family"))
        ],
        out=tmp_path / "out_changed", case_id="case", enable_critic=True,
        critic_backend="labelcritic", critic_base_url="http://localhost",
        critic_port=8000, timeout_sec=30, dry_run=False,
        labelcritic_options={},
    )
    assert selected_same["model"] == selected_changed["model"] == "teacher_a"
    assert evidence_same["selection_method"] == "geometric_teacher_consensus"
    assert evidence_changed["selection_method"] == "geometric_teacher_consensus"
    assert evidence_same["should_enter_student_training"] is True
    assert evidence_changed["should_enter_student_training"] is True
    assert evidence_same["family_role"] == "audit_only_not_used_for_selection_or_training_gate"


def test_geometric_consensus_abstains_on_tied_largest_clusters(tmp_path: Path) -> None:
    from cli_anything.medai.core.multimodel_loop import _geometric_teacher_consensus_selection

    arr1 = np.zeros((16, 16, 16), dtype=np.uint8)
    arr1[2:6, 2:6, 2:6] = 1
    arr2 = np.zeros_like(arr1)
    arr2[10:14, 10:14, 10:14] = 1
    paths = {}
    for name, arr in {"teacher_a": arr1, "teacher_b": arr1, "teacher_c": arr2, "teacher_d": arr2}.items():
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
        paths[name] = path
    selected, evidence = _geometric_teacher_consensus_selection(
        [
            {"model": name, "prediction": str(paths[name]), "identity_status": "valid", "candidate_qc_status": "pass"}
            for name in ("teacher_a", "teacher_b", "teacher_c", "teacher_d")
        ],
        near_identical_dice=0.95,
    )
    assert selected is None
    assert evidence["geometric_consensus_status"] == "abstain_tied_largest_clusters"


def test_labelcritic_audit_only_runs_pairwise_but_cannot_train(tmp_path: Path, monkeypatch) -> None:
    from cli_anything.medai.core import multimodel_loop as loop

    ct = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.int16), np.eye(4)), ct)
    a = np.zeros((16, 16, 16), dtype=np.uint8); a[2:6, 2:6, 2:6] = 1
    b = np.zeros_like(a); b[10:14, 10:14, 10:14] = 1
    pa = tmp_path / "teacher_a.nii.gz"; pb = tmp_path / "teacher_b.nii.gz"
    nib.save(nib.Nifti1Image(a, np.eye(4)), pa)
    nib.save(nib.Nifti1Image(b, np.eye(4)), pb)
    calls = []

    def fake_compare(ct_image, mask_a, mask_b, organ, output_json, **kwargs):
        calls.append((Path(mask_a).name, Path(mask_b).name, Path(output_json).name, kwargs.get("candidate_context")))
        reverse = str(output_json).endswith("_reverse.json")
        return {
            "status": "success",
            "decision": {"winner": "b" if reverse else "a"},
            "output_json": str(output_json),
        }

    monkeypatch.setattr(loop, "run_labelcritic_compare", fake_compare)
    selected, evidence = loop._select_candidate(
        ct=ct,
        organ="liver",
        candidates=[
            {"model": "teacher_a", "prediction": str(pa), "eligible_for_labelcritic": True, "identity_status": "valid", "candidate_qc_status": "pass"},
            {"model": "teacher_b", "prediction": str(pb), "eligible_for_labelcritic": True, "identity_status": "valid", "candidate_qc_status": "pass"},
        ],
        out=tmp_path / "out",
        case_id="case",
        enable_critic=True,
        critic_backend="labelcritic",
        critic_base_url="http://localhost",
        critic_port=8000,
        timeout_sec=30,
        dry_run=False,
        labelcritic_options={},
    )
    assert selected is None
    assert evidence["selection_method"] == "label_critic_audit_only"
    assert evidence["formal_winner"] is None
    assert evidence["audit_winner"] == "teacher_a"
    assert evidence["should_enter_student_training"] is False
    assert len(calls) == 2
    assert all("teacher_a" not in json.dumps(call[3]) for call in calls)


def test_anonymous_candidate_context_contains_geometry_but_no_teacher_identity() -> None:
    from cli_anything.medai.core.multimodel_loop import _candidate_prompt_context

    context = _candidate_prompt_context({
        "candidate_id": "cand_a",
        "model": "secret_teacher",
        "evidence_family": "secret_family",
        "prediction": "/secret/path.nii.gz",
        "candidate_qc_status": "pass",
        "candidate_qc_flags": [],
        "fov_status": "partially_visible",
        "candidate_qc": {
            "mask_voxels": 12,
            "mask_volume_mm3": 24.0,
            "bbox_voxel": {"min": [0, 1, 2], "max": [3, 4, 5]},
            "connected_components": 1,
            "boundary_contacts": {"axis2_max": True},
            "centroid_ras_mm": [1.0, 2.0, 3.0],
            "centroid_laterality": "left",
            "truncation_suspected": True,
        },
    })
    assert context["volume_mm3"] == 24.0
    assert context["fov_status"] == "partially_visible"
    serialized = json.dumps(context)
    for forbidden in ("secret_teacher", "secret_family", "/secret/path"):
        assert forbidden not in serialized


def test_scan_coverage_records_physical_ras_bounds(tmp_path: Path) -> None:
    from cli_anything.medai.core.scan_coverage import infer_scan_coverage

    path = tmp_path / "ct.nii.gz"
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((5, 6, 7), dtype=np.int16), affine), path)
    result = infer_scan_coverage(path)
    assert result["orientation_ras"] == ["R", "A", "S"]
    assert result["ras_bounds_mm"]["min"] == [0.0, 0.0, 0.0]
    assert result["ras_bounds_mm"]["max"] == [8.0, 15.0, 24.0]
    assert result["inferior_superior_bounds_mm"] == [0.0, 24.0]


def test_audit_only_positive_cannot_leak_into_training(tmp_path: Path) -> None:
    from cli_anything.medai.core.multimodel_loop import _materialize_case_373_targets

    ct = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), ct)
    rows = [{
        "case_id": "case",
        "organ": "liver",
        "target_type": "hard",
        "selection_method": "label_critic_audit_only",
        "selection_status": "review_required",
        "training_weight": 1.0,
        "should_enter_student_training": False,
    }]
    selected = [dict(rows[0])]
    _materialize_case_373_targets(
        case_id="case", ct=ct, organs=["liver"], case_updated=tmp_path / "updated",
        selection_rows=rows, selected_metadata=selected,
        presence_context={"has_region_evidence": True, "has_abdomen_coverage": True},
    )
    assert rows[0]["target_type"] == "positive_hard"
    assert rows[0]["training_weight"] == 0
    assert rows[0]["distillation_eligible"] is False
    assert rows[0]["training_block_reason"] == "selection_not_formally_eligible_under_current_gate"
