from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np


EXPECTED_CADS_15 = [
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

EXPECTED_COMPLETED_7 = [
    "cerebrospinal_fluid",
    "eyeball",
    "face",
    "gray_matter",
    "muscle_of_head",
    "scalp",
    "white_matter",
]

EXPECTED_REMAINING_8 = [
    "blood",
    "common_iliac_artery_left",
    "common_iliac_artery_right",
    "common_iliac_vein_left",
    "common_iliac_vein_right",
    "compact_bone",
    "gland_structure",
    "spongy_bone",
]


def _save(array: np.ndarray, path: Path, affine: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4) if affine is None else affine), str(path))
    return path


def _load_nnunet_wrapper_module():
    path = Path("scripts/nnunetv2_predict_and_split.py").resolve()
    spec = importlib.util.spec_from_file_location("nnunetv2_predict_and_split_cads_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cads_15_targets_are_recomputed_from_current_task2_and_registry():
    from tools.dataset_delivery import audit_cads_remaining_targets as audit

    targets = audit.compute_cads_targets()

    assert targets == EXPECTED_CADS_15
    assert not set(["brain", "trachea", "brainstem", "oral_cavity", "larynx"]) & set(targets)


def test_cads_completed_7_and_remaining_8_difference_is_stable():
    from tools.dataset_delivery import audit_cads_remaining_targets as audit

    all_targets = audit.compute_cads_targets()
    completed = audit.completed_targets_for_cads(all_targets)
    remaining = audit.remaining_targets_for_cads(all_targets, completed)

    assert completed == EXPECTED_COMPLETED_7
    assert remaining == EXPECTED_REMAINING_8


def test_cads_status_audit_writes_required_readonly_reports(tmp_path: Path):
    from tools.dataset_delivery import audit_cads_remaining_targets as audit

    summary = audit.build_cads_status(
        output_root=tmp_path,
        case_manifest=tmp_path / "missing_manifest.csv",
        existing_cads_root=tmp_path / "missing_history",
        smoke_roots=[tmp_path / "no_smokes"],
    )

    assert summary["read_only"] is True
    assert summary["source_data_mutation"] is False
    assert summary["canonical_config_mutation"] is False
    assert summary["cads_all_targets"] == EXPECTED_CADS_15
    assert summary["cads_completed_targets"] == EXPECTED_COMPLETED_7
    assert summary["cads_remaining_targets"] == EXPECTED_REMAINING_8
    assert set(summary["stale_candidates_outside_current_task2"]) == {
        "brain",
        "trachea",
        "brainstem",
        "oral_cavity",
        "larynx",
    }
    for name in [
        "cads_all_targets.csv",
        "cads_completed_targets.csv",
        "cads_remaining_targets.csv",
        "cads_target_status.json",
        "cads_target_status.md",
    ]:
        assert (tmp_path / name).exists()


def test_cads_source_label_mapping_uses_existing_alias_config():
    from cli_anything.medai.core.model_registry import load_registry
    from tools.dataset_delivery import audit_cads_remaining_targets as audit

    registry = load_registry(Path("configs/model_registry.yaml"))
    alias_config = json.loads(Path("configs/model_label_aliases.json").read_text(encoding="utf-8"))

    assert audit.source_label_for_target(
        target="common_iliac_vein_left",
        model_key="cads553",
        registry=registry,
        alias_config=alias_config,
    ) == "iliac_vena_left"
    assert audit.source_label_for_target(
        target="compact_bone",
        model_key="cads557",
        registry=registry,
        alias_config=alias_config,
    ) == "compact bone"
    assert audit.source_label_for_target(
        target="gland_structure",
        model_key="cads559",
        registry=registry,
        alias_config=alias_config,
    ) == "glands"


def test_nnunet_split_labelmap_writes_canonical_cads_alias_names(tmp_path: Path):
    wrapper = _load_nnunet_wrapper_module()
    dataset_json = tmp_path / "dataset.json"
    dataset_json.write_text(
        json.dumps({
            "labels": {
                "background": 0,
                "iliac_vena_left": 1,
                "compact bone": 2,
                "glands": 3,
                "blood": 4,
            }
        }),
        encoding="utf-8",
    )
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    arr[1, 1, 1] = 1
    arr[2, 2, 2] = 2
    arr[3, 3, 3] = 3
    arr[4, 4, 4] = 4
    combined = _save(arr, tmp_path / "combined.nii.gz")

    result = wrapper.split_labelmap(
        combined,
        dataset_json,
        tmp_path / "segmentations",
        requested_organs=["common_iliac_vein_left", "compact_bone", "gland_structure", "blood"],
    )

    assert result["num_written"] == 4
    assert (tmp_path / "segmentations" / "common_iliac_vein_left.nii.gz").exists()
    assert (tmp_path / "segmentations" / "compact_bone.nii.gz").exists()
    assert (tmp_path / "segmentations" / "gland_structure.nii.gz").exists()
    assert (tmp_path / "segmentations" / "blood.nii.gz").exists()
    assert {row["source_label"] for row in result["written_masks"]} == {
        "iliac_vena_left",
        "compact bone",
        "glands",
        "blood",
    }


def test_registered_infer_expected_outputs_resolve_cads_aliases():
    from cli_anything.medai.core.model_registry import load_registry
    from cli_anything.medai.core import registered_infer as ri

    registry = load_registry(Path("configs/model_registry.yaml"))
    cads553 = {**registry["models"]["cads553"], "model_key": "cads553"}
    cads557 = {**registry["models"]["cads557"], "model_key": "cads557"}
    cads559 = {**registry["models"]["cads559"], "model_key": "cads559"}

    assert ri._expected_output_organs(
        cads553,
        {"requested_organs": ["common_iliac_artery_left", "common_iliac_vein_right"]},
        project_root=Path.cwd(),
    ) == ["common_iliac_artery_left", "common_iliac_vein_right"]
    assert ri._expected_output_organs(
        cads557,
        {"requested_organs": ["compact_bone", "spongy_bone", "blood"]},
        project_root=Path.cwd(),
    ) == ["compact_bone", "spongy_bone", "blood"]
    assert ri._expected_output_organs(
        cads559,
        {"requested_organs": ["gland_structure"]},
        project_root=Path.cwd(),
    ) == ["gland_structure"]


def test_validate_nifti_mask_detects_missing_empty_and_geometry_mismatch(tmp_path: Path):
    from tools.dataset_delivery import audit_cads_remaining_targets as audit

    ct = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
    assert audit.validate_nifti_mask(tmp_path / "missing.nii.gz", ct)["reason"] == "missing_mask"

    empty = _save(np.zeros((4, 4, 4), dtype=np.uint8), tmp_path / "empty.nii.gz")
    empty_result = audit.validate_nifti_mask(empty, ct)
    assert empty_result["valid"] is False
    assert empty_result["reason"] == "empty_mask"

    shifted = np.eye(4)
    shifted[0, 3] = 2
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    geom = _save(mask, tmp_path / "geom.nii.gz", affine=shifted)
    geom_result = audit.validate_nifti_mask(geom, ct)
    assert geom_result["valid"] is False
    assert geom_result["reason"] == "geometry_mismatch"


def _write_fake_smoke_run(
    root: Path,
    *,
    targets: list[str],
    target_model_map: dict[str, str],
    missing_target: str | None = None,
) -> Path:
    case_id = "case_001"
    run_out = root / "run_loop"
    ct = _save(np.zeros((4, 4, 4), dtype=np.int16), root / "case" / "ct.nii.gz")
    plan = {
        "ct_path": str(ct),
        "teacher_run_list": sorted(set(target_model_map.values())),
        "per_organ": {target: {"primary_teacher": target_model_map[target]} for target in targets},
    }
    (run_out / "annotation_versions" / case_id).mkdir(parents=True)
    (run_out / "annotation_versions" / case_id / "case_execution_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (run_out / "run_summary.json").write_text(
        json.dumps({"status": "success", "strict_delivery_failure_count": 0}),
        encoding="utf-8",
    )
    for model in sorted(set(target_model_map.values())):
        summary = {"status": "success", "return_code": 0, "model_key": model}
        path = run_out / "cases" / case_id / "raw_predictions" / model / case_id / "inference_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[1:3, 1:3, 1:3] = 1
    for target in targets:
        if target != missing_target:
            _save(arr, run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz")
    return run_out


def test_cads_smoke_verdict_fails_when_one_target_missing(tmp_path: Path):
    from tools.dataset_delivery import audit_cads_remaining_targets as audit

    targets = ["blood", "compact_bone"]
    target_model_map = {"blood": "cads557", "compact_bone": "cads557"}
    run_out = _write_fake_smoke_run(tmp_path, targets=targets, target_model_map=target_model_map, missing_target="compact_bone")

    verdict = audit.validate_cads_smoke(
        run_out=run_out,
        output_root=tmp_path / "verdict",
        case_id="case_001",
        targets=targets,
        target_model_map=target_model_map,
        run_return_code=0,
    )

    assert verdict["status"] == "failed"
    assert verdict["passed_targets"] == ["blood"]
    assert verdict["failed_targets"] == ["compact_bone"]
    assert any("compact_bone:missing_mask" in failure for failure in verdict["failures"])


def test_cads_smoke_verdict_passes_when_all_targets_valid(tmp_path: Path):
    from tools.dataset_delivery import audit_cads_remaining_targets as audit

    targets = ["blood", "compact_bone"]
    target_model_map = {"blood": "cads557", "compact_bone": "cads557"}
    run_out = _write_fake_smoke_run(tmp_path, targets=targets, target_model_map=target_model_map)

    verdict = audit.validate_cads_smoke(
        run_out=run_out,
        output_root=tmp_path / "verdict",
        case_id="case_001",
        targets=targets,
        target_model_map=target_model_map,
        run_return_code=0,
    )

    assert verdict["status"] == "passed"
    assert verdict["passed_targets"] == targets
    assert verdict["failed_targets"] == []


def test_cads_target_set_fingerprint_is_order_stable_and_changes_for_old_cache():
    from tools.dataset_delivery import audit_cads_remaining_targets as audit

    left = audit.target_set_fingerprint(["compact_bone", "blood", "blood"])
    right = audit.target_set_fingerprint(["blood", "compact_bone"])
    old_cache = audit.target_set_fingerprint(["blood"])

    assert left == right
    assert left != old_cache
