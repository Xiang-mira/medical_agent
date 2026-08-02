from __future__ import annotations

import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def _save(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def test_registered_infer_fails_when_only_auxiliary_masks_exist(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import registered_infer as ri

    registry_path = tmp_path / "configs" / "model_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        """
models:
  atm:
    name: ATM
    enabled: true
    runner: command_template
    covered_organs: [airway_tree]
    command_template: python fake.py
""",
        encoding="utf-8",
    )
    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")

    def fake_run(command, **kwargs):
        seg = tmp_path / "out" / "case" / "segmentations"
        _save(np.zeros((4, 4, 4), dtype=np.uint8), seg / "image.nii.gz")
        _save(np.zeros((4, 4, 4), dtype=np.uint8), seg / "zero_mask.nii.gz")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ri.subprocess, "run", fake_run)
    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "atm",
        registry_path=registry_path,
        extra_context={"requested_organs": ["airway_tree"]},
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "expected_mask_missing"
    assert result["formal_mask_count"] == 0
    assert result["auxiliary_outputs"] == ["image.nii.gz", "zero_mask.nii.gz"]


def test_registered_infer_resolves_task2_nnunet_command(tmp_path: Path):
    from cli_anything.medai.core import registered_infer as ri

    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "atm",
        registry_path=Path("configs/model_registry.yaml"),
        case_id="case",
        dry_run=True,
        extra_context={"requested_organs": ["airway_tree"]},
    )

    assert result["status"] == "dry_run"
    assert result["resolved_python"] == "/home/xhan74/envs/medical_agent/bin/python"
    assert "/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict" in result["command"]
    assert "--sitecustomize-path" in result["command"]
    assert "/home/xhan74/nnunet_torch22_compat/sitecustomize.py" in result["command"]


def test_registered_infer_unest_python_env_overrides_registry(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import registered_infer as ri

    monkeypatch.setenv("MEDAI_UNEST_PYTHON", "/opt/unest/bin/python")
    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "unest",
        registry_path=Path("configs/model_registry.yaml"),
        case_id="case",
        dry_run=True,
        extra_context={"requested_organs": ["kidney_cortex"]},
    )

    assert result["status"] == "dry_run"
    assert '--python-executable "/opt/unest/bin/python"' in result["command"]


def test_registered_infer_fails_when_airrc_expected_output_is_missing(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import registered_infer as ri

    registry_path = tmp_path / "configs" / "model_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        """
models:
  airrc:
    name: AirRC
    enabled: true
    runner: command_template
    covered_organs: [airway_wall, lung_pulmonary_arteries, lung_pulmonary_veins]
    command_template: python fake.py
""",
        encoding="utf-8",
    )
    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")

    def fake_run(command, **kwargs):
        seg = tmp_path / "out" / "case" / "segmentations"
        _save(np.ones((4, 4, 4), dtype=np.uint8), seg / "airway_wall.nii.gz")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ri.subprocess, "run", fake_run)
    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "airrc",
        registry_path=registry_path,
        extra_context={"requested_organs": ["airway_wall", "lung_pulmonary_arteries", "lung_pulmonary_veins"]},
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "expected_mask_missing"
    assert result["missing_expected_outputs"] == [
        "lung_pulmonary_arteries.nii.gz",
        "lung_pulmonary_veins.nii.gz",
    ]


def test_external_kidney_parent_roi_builds_union_from_left_right(tmp_path: Path):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
    ref = tmp_path / "segmentations"
    left = np.zeros((8, 8, 8), dtype=np.uint8)
    right = np.zeros((8, 8, 8), dtype=np.uint8)
    left[1:3, 1:3, 1:3] = 1
    right[5:7, 5:7, 5:7] = 1
    _save(left, ref / "kidney_left.nii.gz")
    _save(right, ref / "kidney_right.nii.gz")
    taxonomy = json.loads(Path("configs/organ_taxonomy.json").read_text(encoding="utf-8"))

    parent_masks, records = loop._resolve_external_parent_masks(
        ref_dir=ref,
        requested_organs=["kidney_cortex", "kidney_medulla"],
        taxonomy=taxonomy,
        ct=ct,
        case_out=tmp_path / "case",
    )

    assert set(parent_masks) == {"kidney"}
    assert records[0]["status"] == "built_union_parent"
    merged = np.asanyarray(nib.load(str(parent_masks["kidney"])).dataobj)
    assert int(merged.sum()) == int(left.sum() + right.sum())


def test_unest_child_reports_missing_kidney_parent(tmp_path: Path):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
    taxonomy = json.loads(Path("configs/organ_taxonomy.json").read_text(encoding="utf-8"))
    result = loop._run_hierarchical_case_inference(
        ct=ct,
        case_id="case_001",
        case_raw=tmp_path / "raw",
        case_out=tmp_path / "case",
        registry_path=Path("configs/model_registry.yaml"),
        execution_plan={
            "per_organ": {
                "kidney_cortex": {"primary_teacher": "unest", "backup_teachers": [], "competition_teachers": []},
                "kidney": {"primary_teacher": None, "backup_teachers": [], "competition_teachers": []},
            }
        },
        requested_organs=["kidney_cortex"],
        taxonomy=taxonomy,
        alias_config={"models": {}},
        timeout_sec=30,
        device="cpu",
        dry_run=False,
        margin_mm=2,
    )

    assert any(
        item.get("organ") == "kidney_cortex"
        and item.get("status") == "blocked_by_parent"
        and item.get("missing_parents") == ["kidney"]
        for item in result["blocked"]
    )


def test_atm_child_runs_full_volume_without_airway_parent(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
    taxonomy = json.loads(Path("configs/organ_taxonomy.json").read_text(encoding="utf-8"))
    calls = []

    def fake_registered(image, output, model, **kwargs):
        requested = list((kwargs.get("extra_context") or {}).get("requested_organs", []))
        calls.append({"model": model, "requested": requested})
        seg = Path(output) / kwargs["case_id"] / "segmentations"
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:5, 2:5, 2:5] = 1
        for organ in requested:
            _save(arr, seg / f"{organ}.nii.gz")
        return {"status": "success", "model_key": model, "segmentation_output": str(seg), "num_masks": len(requested)}

    monkeypatch.setattr(loop, "run_registered_model", fake_registered)
    result = loop._run_hierarchical_case_inference(
        ct=ct,
        case_id="case_001",
        case_raw=tmp_path / "raw",
        case_out=tmp_path / "case",
        registry_path=Path("configs/model_registry.yaml"),
        execution_plan={
            "per_organ": {
                "airway_tree": {"primary_teacher": "atm", "backup_teachers": [], "competition_teachers": []},
                "airway": {"primary_teacher": None, "backup_teachers": [], "competition_teachers": []},
            }
        },
        requested_organs=["airway_tree"],
        taxonomy=taxonomy,
        alias_config={"models": {}},
        timeout_sec=30,
        device="cpu",
        dry_run=False,
        margin_mm=2,
    )

    assert calls == [{"model": "atm", "requested": ["airway_tree"]}]
    assert (result["model_seg_dirs"]["atm"] / "airway_tree.nii.gz").exists()
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert not any(item.get("organ") == "airway_tree" and item.get("status") == "blocked_by_parent" for item in manifest["blocked"])


def test_strict_run_loop_fails_when_requested_teacher_pruned_to_empty_plan(tmp_path: Path):
    from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop

    cases = tmp_path / "cases.csv"
    cases.write_text(
        "case_id,ct_path,annotation_folder,body_region\n"
        f"case_001,{tmp_path / 'missing_ct.nii.gz'},{tmp_path / 'ref'},abdomen\n",
        encoding="utf-8",
    )
    result = run_multimodel_annotation_loop(
        cases,
        tmp_path / "out",
        models=["atm"],
        organs=["airway_tree"],
        registry_path=Path("configs/model_registry.yaml"),
        enable_shapekit=False,
        enable_critic=False,
        dry_run=True,
        strict_delivery_targets=True,
    )

    assert result["status"] == "failed"
    assert result["strict_delivery_failure_count"] >= 1
    assert {row["reason"] for row in result["strict_delivery_failures"]} >= {
        "requested_organs_pruned_by_fov",
        "requested_teacher_not_scheduled",
    }


def test_task2_scope_contains_blocked_totalsegmentator(tmp_path: Path):
    from tools.dataset_delivery.task2_audit import build_scope

    rows = build_scope(tmp_path)
    assert len(rows) == 23
    total = [row for row in rows if row["model_group"] == "TotalSegmentator"]
    assert len(total) == 1
    assert total[0]["execution_status"] == "blocked"
