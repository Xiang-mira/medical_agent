from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _save(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_case_roots(image_root: Path, mask_root: Path, case_id: str) -> tuple[Path, Path]:
    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), image_root / case_id / "ct.nii.gz")
    ann = mask_root / case_id / "segmentations"
    ann.mkdir(parents=True, exist_ok=True)
    return ct, ann


def _write_manifest(path: Path, case_ids: list[str], image_root: Path, mask_root: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "case_id", "ct_path", "annotation_folder"])
        writer.writeheader()
        for index, case_id in enumerate(case_ids):
            ct, ann = _write_case_roots(image_root, mask_root, case_id)
            writer.writerow({"index": index, "case_id": case_id, "ct_path": ct, "annotation_folder": ann})
    return path


def _write_legacy_base_manifest_without_annotation_folder(path: Path, case_ids: list[str], image_root: Path, mask_root: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "case_id", "image_path", "reference_mask_dir"])
        writer.writeheader()
        for index, case_id in enumerate(case_ids):
            ct, ann = _write_case_roots(image_root, mask_root, case_id)
            writer.writerow({"index": index, "case_id": case_id, "image_path": ct, "reference_mask_dir": ann})
    return path


def _append_cases(path: Path) -> Path:
    path.write_text(
        "case_id,reason\n"
        "BDMAP_00000001,formal_103case_required_append\n"
        "BDMAP_00000002,formal_103case_required_append\n"
        "BDMAP_00000006,formal_103case_required_append\n",
        encoding="utf-8",
    )
    return path


def _formal_manifest(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    from tools.dataset_delivery.task2_formal_manifest import build_formal_manifest

    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    base_ids = [f"BDMAP_BASE_{index:05d}" for index in range(100)]
    base = _write_manifest(tmp_path / "cases_100_manifest.csv", base_ids, image_root, mask_root)
    for case_id in ("BDMAP_00000001", "BDMAP_00000002", "BDMAP_00000006"):
        _write_case_roots(image_root, mask_root, case_id)
    append = _append_cases(tmp_path / "formal_append_cases.csv")
    out = tmp_path / "cases_103_manifest.csv"
    build_formal_manifest(
        base_manifest=base,
        append_cases=append,
        output_manifest=out,
        audit_json=tmp_path / "cases_103_manifest.audit.json",
        image_root=image_root,
        mask_root=mask_root,
    )
    return base, out, image_root, mask_root, append


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_executable_body(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_formal_manifest_builds_exact_103_and_rejects_duplicate_append(tmp_path: Path):
    from tools.dataset_delivery.delivery_lib import DeliveryError
    from tools.dataset_delivery.task2_formal_manifest import build_formal_manifest

    base, manifest, _, _, append = _formal_manifest(tmp_path)
    rows = _read_rows(manifest)

    assert len(rows) == 103
    assert len({row["case_id"] for row in rows}) == 103
    assert [row["case_id"] for row in rows[-3:]] == ["BDMAP_00000001", "BDMAP_00000002", "BDMAP_00000006"]

    duplicate_base = tmp_path / "duplicate_base.csv"
    image_root = tmp_path / "dup_images"
    mask_root = tmp_path / "dup_masks"
    duplicate_ids = ["BDMAP_00000001", *[f"BDMAP_DUP_{index:05d}" for index in range(99)]]
    _write_manifest(duplicate_base, duplicate_ids, image_root, mask_root)
    for case_id in ("BDMAP_00000002", "BDMAP_00000006"):
        _write_case_roots(image_root, mask_root, case_id)
    with pytest.raises(DeliveryError, match="actual_unique_count"):
        build_formal_manifest(
            base_manifest=duplicate_base,
            append_cases=append,
            output_manifest=tmp_path / "bad_cases_103_manifest.csv",
            image_root=image_root,
            mask_root=mask_root,
        )


def test_formal_manifest_materializes_annotation_folder_for_legacy_base_rows(tmp_path: Path):
    from tools.dataset_delivery.task2_formal_manifest import (
        FORMAL_APPEND_CASE_IDS,
        build_formal_manifest,
        validate_formal_manifest,
    )

    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    base_ids = [f"BDMAP_BASE_{index:05d}" for index in range(100)]
    base = _write_legacy_base_manifest_without_annotation_folder(
        tmp_path / "cases_100_legacy_manifest.csv",
        base_ids,
        image_root,
        mask_root,
    )
    for case_id in FORMAL_APPEND_CASE_IDS:
        _write_case_roots(image_root, mask_root, case_id)
    append = _append_cases(tmp_path / "formal_append_cases.csv")
    out = tmp_path / "cases_103_manifest.csv"

    audit = build_formal_manifest(
        base_manifest=base,
        append_cases=append,
        output_manifest=out,
        image_root=image_root,
        mask_root=mask_root,
    )
    rows = _read_rows(out)
    validation = validate_formal_manifest(manifest=out, base_manifest=base)

    assert audit["status"] == "READY"
    assert audit["base_count"] == 100
    assert audit["append_count"] == 3
    assert audit["final_count"] == 103
    assert audit["final_unique_count"] == 103
    assert audit["base_case_ids"] == base_ids
    assert audit["appended_case_ids"] == list(FORMAL_APPEND_CASE_IDS)
    assert validation["rows"] == 103
    assert validation["unique_case_count"] == 103
    assert [row["case_id"] for row in rows[:100]] == base_ids
    assert [row["case_id"] for row in rows[-3:]] == list(FORMAL_APPEND_CASE_IDS)
    assert all("annotation_folder" in row and row["annotation_folder"] for row in rows)
    assert all(Path(row["annotation_folder"]).is_dir() for row in rows)
    assert all("ct_path" in row and Path(row["ct_path"]).is_file() for row in rows)
    assert rows[0]["annotation_folder"] == rows[0]["reference_mask_dir"]


def test_formal_group_target_mapping_is_exact_22_and_excludes_totalsegmentator():
    from tools.dataset_delivery.task2_formal_manifest import FORMAL_MODEL_TARGETS, FORMAL_TARGETS

    assert len(FORMAL_TARGETS) == 22
    assert "brain_ventricle" not in FORMAL_TARGETS
    assert set(FORMAL_MODEL_TARGETS) == {"cads", "atm", "airrc", "unest"}
    assert FORMAL_MODEL_TARGETS["atm"] == ["airway_tree"]
    assert FORMAL_MODEL_TARGETS["airrc"] == ["airway_wall", "lung_pulmonary_arteries", "lung_pulmonary_veins"]


def test_formal_launcher_plans_teacher_tasks_without_target_reference_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery import task2_formal_launcher as launcher

    base, manifest, _, _, _ = _formal_manifest(tmp_path)
    monkeypatch.setenv("LABELCRITIC_BASE_URL", "http://labelcritic-node")
    monkeypatch.setenv("LABELCRITIC_PORT", "8123")
    monkeypatch.setenv("LABELCRITIC_MODEL_ID", "Qwen/Qwen2-VL-72B-Instruct-AWQ")
    monkeypatch.setattr(launcher, "build_preflight", lambda **kwargs: {"status": "READY", "blocked_checks": [], "runtime_manifest": {}})
    monkeypatch.setattr(launcher, "preflight_sbatch_script", lambda **kwargs: {"status": "READY", "reason": "test"})
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    summary = launcher.build_formal_plan(
        output_root=tmp_path / "formal",
        case_manifest=manifest,
        base_manifest=base,
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=checkpoint_root,
        nnunet_predict_executable=_write_executable(tmp_path / "bin" / "nnUNetv2_predict"),
        unest_python_executable=_write_executable(tmp_path / "venv" / "python"),
        dry_run=True,
        allow_dirty_tracked=True,
        run_slurm_test_only=False,
    )

    assert summary["status"] == "READY"
    assert summary["case_count"] == 103
    assert summary["target_count"] == 22
    assert summary["task_count"] == 103 * 4
    assert (tmp_path / "formal" / "cads" / "cads_task_manifest.csv").exists()
    sbatch_text = (tmp_path / "formal" / "slurm" / "cads_task2_array.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --export=ALL" in sbatch_text
    assert "export LABELCRITIC_BASE_URL=http://labelcritic-node" in sbatch_text
    assert "export LABELCRITIC_PORT=8123" in sbatch_text
    assert "export LABELCRITIC_MODEL_ID=Qwen/Qwen2-VL-72B-Instruct-AWQ" in sbatch_text


def test_formal_launcher_preserves_python_venv_symlink_execution_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery import task2_formal_launcher as launcher

    base, manifest, _, _, _ = _formal_manifest(tmp_path)
    monkeypatch.setattr(launcher, "build_preflight", lambda **kwargs: {"status": "READY", "blocked_checks": [], "runtime_manifest": {}})
    monkeypatch.setattr(launcher, "preflight_sbatch_script", lambda **kwargs: {"status": "READY", "reason": "test"})
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    base_unest = _write_executable_body(tmp_path / "base" / "python3.11", "#!/usr/bin/env bash\nexit 0\n")
    unest_middle = tmp_path / "unest_env" / "bin" / "python3.11"
    unest_python = tmp_path / "unest_env" / "bin" / "python"
    unest_middle.parent.mkdir(parents=True)
    unest_middle.symlink_to(base_unest)
    unest_python.symlink_to("python3.11")
    base_outer = _write_executable_body(tmp_path / "base_outer" / "python3.11", "#!/usr/bin/env bash\nexit 0\n")
    outer_python = tmp_path / "outer_env" / "bin" / "python"
    outer_python.parent.mkdir(parents=True)
    outer_python.symlink_to(base_outer)

    summary = launcher.build_formal_plan(
        output_root=tmp_path / "formal",
        case_manifest=manifest,
        base_manifest=base,
        code_root=REPO_ROOT,
        python=outer_python,
        checkpoint_root=checkpoint_root,
        nnunet_predict_executable=_write_executable(tmp_path / "bin" / "nnUNetv2_predict"),
        unest_python_executable=unest_python,
        groups=["unest"],
        dry_run=True,
        allow_dirty_tracked=True,
        run_slurm_test_only=False,
    )

    assert summary["status"] == "READY"
    assert summary["runtime_python_execution_path"] == str(outer_python.absolute())
    assert summary["runtime_python_binary_realpath"] == str(base_outer)
    assert summary["unest_python_execution_path"] == str(unest_python.absolute())
    assert summary["unest_python_binary_realpath"] == str(base_unest)
    rows = _read_rows(Path(summary["groups"]["unest"]["task_manifest"]))
    assert rows
    assert {row["unest_python_executable"] for row in rows} == {str(unest_python.absolute())}
    assert {row["unest_python_execution_path"] for row in rows} == {str(unest_python.absolute())}
    assert {row["unest_python_binary_realpath"] for row in rows} == {str(base_unest)}
    assert {row["runtime_python"] for row in rows} == {str(outer_python.absolute())}
    assert {row["runtime_python_binary_realpath"] for row in rows} == {str(base_outer)}


def test_formal_resume_skips_existing_valid_group_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.dataset_delivery import task2_formal_launcher as launcher

    base, manifest, _, _, _ = _formal_manifest(tmp_path)
    rows = _read_rows(manifest)
    case_id = rows[0]["case_id"]
    ct = Path(rows[0]["ct_path"])
    root = tmp_path / "formal"
    run_out = root / "cases" / case_id / "atm" / "run_loop"
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask_path = _save(mask, run_out / "annotation_versions" / case_id / "updated" / "airway_tree.nii.gz")
    plan_path = run_out / "annotation_versions" / case_id / "case_execution_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"ct_path": str(ct), "teacher_run_list": ["atm"], "per_organ": {"airway_tree": {"fov_status": "fully_visible"}}}), encoding="utf-8")
    (run_out / "run_summary.json").write_text(
        json.dumps({"status": "success", "teacher_inference_models": ["atm"], "teacher_inference_count": 1, "inference_success_count": 1}),
        encoding="utf-8",
    )
    (run_out / "final_delivery_status.json").write_text(
        json.dumps({"rows": [{"case_id": case_id, "organ": "airway_tree", "final_status": "delivered_for_review", "delivery_status": "delivered_for_review", "fov_status": "fully_visible", "mask_path": str(mask_path)}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "build_preflight", lambda **kwargs: {"status": "READY", "blocked_checks": [], "runtime_manifest": {}})
    monkeypatch.setattr(launcher, "preflight_sbatch_script", lambda **kwargs: {"status": "READY", "reason": "test"})
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    summary = launcher.build_formal_plan(
        output_root=root,
        case_manifest=manifest,
        base_manifest=base,
        code_root=REPO_ROOT,
        python=Path(sys.executable),
        checkpoint_root=checkpoint_root,
        nnunet_predict_executable=_write_executable(tmp_path / "bin" / "nnUNetv2_predict"),
        unest_python_executable=_write_executable(tmp_path / "venv" / "python"),
        resume=True,
        dry_run=True,
        allow_dirty_tracked=True,
        run_slurm_test_only=False,
    )

    assert summary["status"] == "READY"
    assert summary["task_count"] == 103 * 4 - 1


def test_formal_validator_distinguishes_completion_and_failure_statuses(tmp_path: Path):
    from tools.dataset_delivery.task2_formal_validator import build_case_target_table

    base, manifest, _, _, _ = _formal_manifest(tmp_path)
    rows = _read_rows(manifest)
    case_id = rows[0]["case_id"]
    ct = Path(rows[0]["ct_path"])
    root = tmp_path / "formal"
    run_out = root / "cases" / case_id / "cads" / "run_loop"
    plan = {
        "ct_path": str(ct),
        "teacher_run_list": ["cads553", "cads557", "cads559"],
        "per_organ": {target: {"fov_status": "fully_visible"} for target in [
            "blood",
            "cerebrospinal_fluid",
            "compact_bone",
            "eyeball",
            "face",
        ]},
    }
    plan["per_organ"]["compact_bone"]["fov_status"] = "out_of_fov"
    plan_path = run_out / "annotation_versions" / case_id / "case_execution_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    (run_out / "run_summary.json").write_text(
        json.dumps({"status": "success", "teacher_inference_models": ["cads557"], "teacher_inference_count": 1, "inference_success_count": 1}),
        encoding="utf-8",
    )
    valid = np.zeros((3, 3, 3), dtype=np.uint8)
    valid[1, 1, 1] = 1
    valid_path = _save(valid, run_out / "annotation_versions" / case_id / "updated" / "blood.nii.gz")
    zero_path = _save(np.zeros((3, 3, 3), dtype=np.uint8), run_out / "annotation_versions" / case_id / "updated" / "face.nii.gz")
    (run_out / "final_delivery_status.json").write_text(
        json.dumps({
            "rows": [
                {"case_id": case_id, "organ": "blood", "final_status": "delivered_for_review", "delivery_status": "delivered_for_review", "fov_status": "fully_visible", "mask_path": str(valid_path)},
                {"case_id": case_id, "organ": "cerebrospinal_fluid", "final_status": "confirmed_absent", "delivery_status": "confirmed_absent", "fov_status": "fully_visible"},
                {"case_id": case_id, "organ": "compact_bone", "final_status": "out_of_fov", "delivery_status": "out_of_fov", "fov_status": "out_of_fov"},
                {"case_id": case_id, "organ": "face", "final_status": "delivered_for_review", "delivery_status": "delivered_for_review", "fov_status": "fully_visible", "mask_path": str(zero_path)},
            ]
        }),
        encoding="utf-8",
    )

    report = build_case_target_table(output_root=root, case_manifest=manifest, base_manifest=base)
    by_target = {
        row["target_name"]: row
        for row in report["rows"]
        if row["case_id"] == case_id and row["model_group"] == "cads"
    }

    assert report["summary"]["row_count"] == 103 * 22
    assert by_target["blood"]["final_status"] == "generated_valid_mask"
    assert by_target["cerebrospinal_fluid"]["final_status"] == "confirmed_absent"
    assert by_target["compact_bone"]["final_status"] == "out_of_fov"
    assert by_target["eyeball"]["final_status"] == "expected_present_but_missing"
    assert by_target["face"]["final_status"] == "validation_failed"
    assert report["summary"]["strict_delivery_failure_count"] == (
        report["summary"]["expected_present_but_missing"]
        + report["summary"]["runtime_failed"]
        + report["summary"]["validation_failed"]
    )


def test_formal_validator_keeps_not_applicable_and_runtime_failed_separate(tmp_path: Path):
    from tools.dataset_delivery.task2_formal_validator import validate_case_group_run

    base, manifest, _, _, _ = _formal_manifest(tmp_path)
    rows = _read_rows(manifest)
    case_id = rows[0]["case_id"]
    ct = Path(rows[0]["ct_path"])
    root = tmp_path / "formal"
    run_out = root / "cases" / case_id / "cads" / "run_loop"
    plan_path = run_out / "annotation_versions" / case_id / "case_execution_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps({
            "ct_path": str(ct),
            "teacher_run_list": ["cads557"],
            "per_organ": {"blood": {"fov_status": "fully_visible"}},
        }),
        encoding="utf-8",
    )
    (run_out / "task_state.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run_out / "run_summary.json").write_text(
        json.dumps({"status": "success", "teacher_inference_models": ["cads557"], "teacher_inference_count": 1, "inference_success_count": 1}),
        encoding="utf-8",
    )
    (run_out / "final_delivery_status.json").write_text(json.dumps({"rows": []}), encoding="utf-8")

    not_applicable = validate_case_group_run(
        output_root=root,
        case_id=case_id,
        group="cads",
        target="airway_tree",
        ct_path=ct,
    )
    runtime_failed = validate_case_group_run(
        output_root=tmp_path / "missing",
        case_id=case_id,
        group="cads",
        target="blood",
        ct_path=ct,
    )

    assert not_applicable["final_status"] == "not_applicable"
    assert runtime_failed["final_status"] == "runtime_failed"
