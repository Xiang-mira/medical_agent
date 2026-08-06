from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_HARNESS = REPO_ROOT / "agent-harness"
if str(AGENT_HARNESS) not in sys.path:
    sys.path.insert(0, str(AGENT_HARNESS))


CADS15_TARGETS = [
    "blood",
    "cerebrospinal_fluid",
    "common_iliac_artery_left",
    "common_iliac_artery_right",
    "common_iliac_vein_left",
    "common_iliac_vein_right",
    "compact_bone",
    "eyeball",
    "face",
    "gland_structure",
    "gray_matter",
    "muscle_of_head",
    "scalp",
    "spongy_bone",
    "white_matter",
]


def _save(array: np.ndarray, path: Path, affine: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4) if affine is None else affine), str(path))
    return path


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_cads15_contract_has_unique_verified_routes(tmp_path: Path):
    from tools.dataset_delivery.cads15_contract_audit import audit_contract

    predictor = _write_executable(tmp_path / "bin" / "nnUNetv2_predict")
    report = audit_contract(
        checkpoint_root_arg=REPO_ROOT / "checkpoints",
        predictor_arg=str(predictor),
        require_runtime_files=True,
    )

    assert report["status"] == "READY_FOR_HPC_SMOKE"
    assert report["verified_count"] == 15
    rows = {row["canonical_id"]: row for row in report["rows"]}
    assert sorted(rows) == CADS15_TARGETS
    assert rows["spongy_bone"]["model"] == "cads557"
    assert rows["spongy_bone"]["source_label"] == "spongy bone"
    assert rows["spongy_bone"]["source_label_id_dataset_json"] == 7
    assert rows["compact_bone"]["source_label_id_dataset_json"] == 6
    assert rows["common_iliac_artery_left"]["source_label"] == "iliac_artery_left"
    assert rows["common_iliac_artery_right"]["source_label"] == "iliac_artery_right"
    assert rows["common_iliac_vein_left"]["source_label"] == "iliac_vena_left"
    assert rows["common_iliac_vein_right"]["source_label"] == "iliac_vena_right"
    assert rows["gland_structure"]["model"] == "cads559"
    assert rows["gland_structure"]["source_label"] == "glands"


def test_cads15_semantic_guards_block_swaps():
    from tools.dataset_delivery.cads15_contract_audit import _semantic_guard

    assert "left_target_mapped_to_right_source" in _semantic_guard("common_iliac_artery_left", "iliac_artery_right")
    assert "right_target_mapped_to_left_source" in _semantic_guard("common_iliac_vein_right", "iliac_vena_left")
    assert "artery_target_mapped_to_vein_source" in _semantic_guard("common_iliac_artery_left", "iliac_vena_left")
    assert "vein_target_mapped_to_artery_source" in _semantic_guard("common_iliac_vein_left", "iliac_artery_left")
    assert "compact_bone_mapped_to_spongy_bone" in _semantic_guard("compact_bone", "spongy bone")
    assert "spongy_bone_mapped_to_compact_bone" in _semantic_guard("spongy_bone", "compact bone")


def test_spongy_bone_grade_d_is_delivery_review_not_training(tmp_path: Path):
    from cli_anything.medai.core.multimodel_loop import (
        _delivery_hard_validation,
        _delivery_status_for_selected_label,
        _distillation_gate_for_selected_label,
    )

    ct = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask_path = _save(mask, tmp_path / "spongy_bone.nii.gz")
    meta = {
        "organ": "spongy_bone",
        "grade": "D",
        "training_weight": 0.0,
        "review_flags": ["single_teacher_no_pairwise_comparison"],
        "quality_flags": ["many_connected_components"],
        "identity_status": "valid",
        "fov_status": "fully_visible",
    }

    hard = _delivery_hard_validation(mask_path, ct, meta)

    assert hard["valid"] is True
    assert _delivery_status_for_selected_label(meta, hard) == "delivered_for_review"
    gate = _distillation_gate_for_selected_label(meta)
    assert gate["eligible"] is False
    assert gate["reason"] == "grade_D_or_zero_weight"


def test_many_connected_components_is_review_flag_not_hard_reject(tmp_path: Path):
    from cli_anything.medai.core.multimodel_loop import _delivery_hard_validation, _delivery_status_for_selected_label

    ct = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[3, 3, 3] = 1
    mask_path = _save(mask, tmp_path / "compact_bone.nii.gz")
    meta = {
        "organ": "compact_bone",
        "grade": "B",
        "training_weight": 1.0,
        "quality_flags": ["many_connected_components"],
        "identity_status": "valid",
        "fov_status": "fully_visible",
    }

    hard = _delivery_hard_validation(mask_path, ct, meta)

    assert hard["valid"] is True
    assert _delivery_status_for_selected_label(meta, hard) == "delivered_for_review"


def _write_selection_case(root: Path, case_id: str, ct: Path, rows: list[dict[str, object]]) -> None:
    case_root = root / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "selection_metadata.json").write_text(
        json.dumps({"ct_path": str(ct), "selected_organs": rows}),
        encoding="utf-8",
    )


def test_final_delivery_counts_rebuild_from_final_publication_layer(tmp_path: Path):
    from cli_anything.medai.core.multimodel_loop import _build_final_delivery_rows

    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), tmp_path / "ct.nii.gz")
    updated_root = tmp_path / "annotation_versions"
    case_id = "case_001"
    rows = []
    for target in CADS15_TARGETS:
        rows.append({
            "organ": target,
            "delivery_status": "delivered",
            "identity_status": "valid",
            "fov_status": "fully_visible",
            "grade": "A",
            "training_weight": 1.0,
        })
    _write_selection_case(updated_root, case_id, ct, rows)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    for target in CADS15_TARGETS[:-1]:
        _save(mask, updated_root / case_id / "updated" / f"{target}.nii.gz")
    _save(mask, updated_root / case_id / "raw_predictions" / f"{CADS15_TARGETS[-1]}.nii.gz")

    final_rows, counts = _build_final_delivery_rows(
        cases=[{"case_id": case_id, "ct_path": str(ct)}],
        organs=CADS15_TARGETS,
        updated_root=updated_root,
    )

    assert counts["requested_target_count"] == 15
    assert counts["delivered_count"] == 14
    assert counts["missing_delivery_count"] == 1
    assert [row for row in final_rows if row["organ"] == CADS15_TARGETS[-1]][0]["final_status"] == "failed"


def test_rejected_and_negative_zero_masks_do_not_count_as_positive_delivery(tmp_path: Path):
    from cli_anything.medai.core.multimodel_loop import _build_final_delivery_rows

    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), tmp_path / "ct.nii.gz")
    updated_root = tmp_path / "annotation_versions"
    case_id = "case_001"
    rows = [
        {"organ": "blood", "delivery_status": "rejected", "publication_status": "rejected_but_recorded", "identity_status": "valid", "fov_status": "fully_visible"},
        {"organ": "compact_bone", "target_type": "negative_absent", "delivery_status": "", "identity_status": "valid", "fov_status": "out_of_fov"},
    ]
    _write_selection_case(updated_root, case_id, ct, rows)
    _save(np.zeros((3, 3, 3), dtype=np.uint8), updated_root / case_id / "updated" / "compact_bone.nii.gz")
    _save(np.ones((3, 3, 3), dtype=np.uint8), updated_root / case_id / "rejected" / "blood.nii.gz")

    _, counts = _build_final_delivery_rows(
        cases=[{"case_id": case_id, "ct_path": str(ct)}],
        organs=["blood", "compact_bone"],
        updated_root=updated_root,
    )

    assert counts["delivered_count"] == 0
    assert counts["delivered_for_review_count"] == 0
    assert counts["rejected_count"] == 1
    assert counts["confirmed_absent_count"] == 1


def _write_case_with_coverage(tmp_path: Path, case_id: str, coverage: str) -> tuple[Path, Path]:
    case_root = tmp_path / case_id
    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), case_root / "ct.nii.gz")
    (case_root / "scan_coverage.json").write_text(
        json.dumps({"scan_coverage": [coverage], "coverage_regions": [coverage]}),
        encoding="utf-8",
    )
    ref = case_root / "ref"
    ref.mkdir()
    return ct, ref


def test_cads15_smoke_panel_selects_multiple_positive_in_fov_cases(tmp_path: Path):
    from tools.dataset_delivery.cads15_smoke_panel import build_smoke_panel

    head_targets = [
        "blood", "cerebrospinal_fluid", "compact_bone", "eyeball", "face",
        "gland_structure", "gray_matter", "muscle_of_head", "scalp",
        "spongy_bone", "white_matter",
    ]
    pelvis_targets = [
        "common_iliac_artery_left", "common_iliac_artery_right",
        "common_iliac_vein_left", "common_iliac_vein_right",
    ]
    ct_head, ref_head = _write_case_with_coverage(tmp_path, "case_head", "head_neck")
    ct_pelvis, ref_pelvis = _write_case_with_coverage(tmp_path, "case_pelvis", "abdomen")
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = 1
    for target in head_targets:
        _save(arr, ref_head / f"{target}.nii.gz")
    for target in pelvis_targets:
        _save(arr, ref_pelvis / f"{target}.nii.gz")
    _save(arr, ref_pelvis / "white_matter.nii.gz")
    manifest = tmp_path / "cases.csv"
    manifest.write_text(
        f"case_id,ct_path,annotation_folder\ncase_head,{ct_head},{ref_head}\ncase_pelvis,{ct_pelvis},{ref_pelvis}\n",
        encoding="utf-8",
    )

    panel = build_smoke_panel(case_manifest=manifest, output_root=tmp_path / "panel")

    assert panel["status"] == "READY_FOR_HPC_SMOKE"
    assert len(panel["cases"]) == 2
    assert not panel["uncovered_targets"]
    selected = {case["case_id"]: set(case["targets"]) for case in panel["cases"]}
    assert "white_matter" in selected["case_head"]
    assert "white_matter" not in selected["case_pelvis"]


def _write_panel_run(
    root: Path,
    *,
    case_id: str,
    targets: list[str],
    delivered_for_review: set[str] | None = None,
) -> None:
    from tools.dataset_delivery.cads15_smoke_launcher import _models_for_targets

    delivered_for_review = delivered_for_review or set()
    case_root = root / "cads15" / case_id
    run_out = case_root / "run_loop"
    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), case_root / "ct.nii.gz")
    (case_root / "selected_case_manifest.csv").write_text(
        f"case_id,ct_path,annotation_folder\n{case_id},{ct},{case_root / 'ref'}\n",
        encoding="utf-8",
    )
    models = _models_for_targets(targets)
    plan_path = run_out / "annotation_versions" / case_id / "case_execution_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"ct_path": str(ct), "teacher_run_list": models}), encoding="utf-8")
    summary = {"status": "success", "strict_delivery_failure_count": 0, "strict_delivery_failures": []}
    (run_out / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for model in models:
        path = run_out / "cases" / case_id / "raw_predictions" / model / case_id / "inference_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "success", "return_code": 0, "model_key": model}), encoding="utf-8")
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = 1
    final_rows = []
    for target in targets:
        _save(arr, run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz")
        status = "delivered_for_review" if target in delivered_for_review else "delivered"
        final_rows.append({
            "case_id": case_id,
            "organ": target,
            "final_status": status,
            "delivery_status": status,
            "mask_path": str(run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz"),
        })
    (run_out / "final_delivery_status.json").write_text(
        json.dumps({"status": "success", "rows": final_rows}),
        encoding="utf-8",
    )


def test_cads15_panel_validator_requires_all_targets_by_name(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    case_a_targets = CADS15_TARGETS[:8]
    case_b_targets = CADS15_TARGETS[8:-1]
    panel = {
        "status": "READY_FOR_HPC_SMOKE",
        "cases": [
            {"case_id": "case_a", "targets": case_a_targets},
            {"case_id": "case_b", "targets": case_b_targets},
        ],
    }
    panel_path = tmp_path / "preflight" / "cads15_smoke_case_panel.json"
    panel_path.parent.mkdir(parents=True)
    panel_path.write_text(json.dumps(panel), encoding="utf-8")
    _write_panel_run(tmp_path, case_id="case_a", targets=case_a_targets)
    _write_panel_run(tmp_path, case_id="case_b", targets=case_b_targets)

    report = validate_smoke_root(smoke_root=tmp_path, groups=["cads15"], panel_json=panel_path)

    assert report["status"] == "VALIDATION_FAILED"
    assert report["cads15_summary"]["TARGETS_WITH_POSITIVE_SMOKE"] == 14
    assert "white_matter" in report["groups"][0]["failed_targets"]


def test_cads15_panel_validator_accepts_delivered_for_review(tmp_path: Path):
    from tools.dataset_delivery.task2_smoke_validator import validate_smoke_root

    case_a_targets = CADS15_TARGETS[:8]
    case_b_targets = CADS15_TARGETS[8:]
    panel = {
        "status": "READY_FOR_HPC_SMOKE",
        "cases": [
            {"case_id": "case_a", "targets": case_a_targets},
            {"case_id": "case_b", "targets": case_b_targets},
        ],
    }
    panel_path = tmp_path / "preflight" / "cads15_smoke_case_panel.json"
    panel_path.parent.mkdir(parents=True)
    panel_path.write_text(json.dumps(panel), encoding="utf-8")
    _write_panel_run(tmp_path, case_id="case_a", targets=case_a_targets)
    _write_panel_run(tmp_path, case_id="case_b", targets=case_b_targets, delivered_for_review={"spongy_bone"})

    report = validate_smoke_root(smoke_root=tmp_path, groups=["cads15"], panel_json=panel_path)

    assert report["status"] == "PASSED"
    assert report["cads15_summary"]["CADS15_SMOKE_STATUS"] == "PASSED"
    assert report["cads15_summary"]["TARGETS_WITH_POSITIVE_SMOKE"] == 15
    assert report["cads15_summary"]["TARGETS_DELIVERED"] + report["cads15_summary"]["TARGETS_DELIVERED_FOR_REVIEW"] == 15
