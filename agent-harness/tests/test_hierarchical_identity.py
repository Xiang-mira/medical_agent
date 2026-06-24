from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.static_audit
def test_updated_workbook_builds_valid_hierarchy():
    from cli_anything.medai.core.model_registry import parse_checkpoint_map
    from cli_anything.medai.core.organ_taxonomy import build_taxonomy

    workbook = ROOT / "configs/class_checkpoint_map_updates.xlsx"
    parsed = parse_checkpoint_map(workbook)
    taxonomy = build_taxonomy(parsed["organs"], workbook)
    assert parsed["num_organs"] == 384
    assert taxonomy["validation"]["status"] == "success"
    assert taxonomy["organs"]["liver"]["hierarchy_role"] == "major"
    assert taxonomy["organs"]["liver_segment_1"]["parent_ids"] == ["liver"]
    assert taxonomy["organs"]["pancreas_head"]["parent_ids"] == ["pancreas"]
    assert taxonomy["organs"]["prosthetic_breast_implant"]["parent_ids"] == ["breast_left", "breast_right"]


@pytest.mark.static_audit
def test_whole_suborgan_and_vessel_remain_distinct():
    taxonomy = json.loads((ROOT / "configs/organ_taxonomy.json").read_text())
    organs = taxonomy["organs"]
    assert organs["liver"]["comparison_family"] == "whole_organ"
    assert organs["liver_segment_1"]["comparison_family"] == "sub_organ"
    assert organs["pancreas_body"]["comparison_family"] == "sub_organ"
    assert organs["pancreas_head"]["canonical_id"] != organs["pancreas"]["canonical_id"]
    assert organs["pancreas_tail"]["comparison_family"] == "sub_organ"
    assert organs["liver_portal_vein"]["comparison_family"] == "vessel_or_small_structure"
    assert organs["aorta"]["hierarchy_role"] == "major"
    assert organs["aorta"]["comparison_family"] == "vessel_or_small_structure"


@pytest.mark.cpu_unit
@pytest.mark.parametrize("source,target", [
    ("liver_segment_1", "liver"),
    ("pancreas_head", "pancreas"),
    ("pancreas_body", "pancreas"),
    ("pancreas_tail", "pancreas"),
    ("portal_vein_and_splenic_vein", "veins"),
])
def test_cross_identity_contract_is_rejected(source: str, target: str):
    from cli_anything.medai.core.organ_taxonomy import identity_contract

    taxonomy = json.loads((ROOT / "configs/organ_taxonomy.json").read_text())
    contract = identity_contract(
        taxonomy, target, source, source,
        mapping_type="forbidden_coarsening", mapping_source="regression_test",
    )
    assert contract["identity_status"] == "identity_mismatch"
    assert "canonical_id_mismatch" in contract["identity_mismatch_reasons"]


@pytest.mark.static_audit
def test_coarse_portal_vein_aliases_are_removed():
    aliases = json.loads((ROOT / "configs/model_label_aliases.json").read_text())
    forbidden = {tuple(pair) for pair in aliases["forbidden_coarsening"]}
    assert ("portal_vein_and_splenic_vein", "veins") in forbidden
    registry_text = (ROOT / "configs/model_registry.yaml").read_text()
    assert "portal_vein_and_splenic_vein: veins" not in registry_text


def _save(array: np.ndarray, path: Path, spacing=(1.0, 1.0, 1.0)) -> Path:
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(array, affine), str(path))
    return path


@pytest.mark.cpu_nifti_integration
def test_roi_margin_uses_physical_mm_and_restores_full_geometry(tmp_path: Path):
    from cli_anything.medai.core.hierarchical_roi import crop_nifti, mask_bbox, restore_mask_to_full

    ct = _save(np.zeros((40, 50, 60), dtype=np.int16), tmp_path / "ct.nii.gz", spacing=(2.0, 1.0, 5.0))
    mask_array = np.zeros((40, 50, 60), dtype=np.uint8)
    mask_array[20:22, 25:27, 30:32] = 1
    mask = _save(mask_array, tmp_path / "parent.nii.gz", spacing=(2.0, 1.0, 5.0))
    bbox = mask_bbox(mask, ct, margin_mm=20.0)
    assert bbox is not None
    assert bbox.start == (10, 5, 26)
    assert bbox.stop == (32, 47, 36)
    crop = tmp_path / "crop.nii.gz"
    crop_nifti(ct, bbox, crop)
    prediction = np.ones(tuple(b - a for a, b in zip(bbox.start, bbox.stop)), dtype=np.uint8)
    pred = _save(prediction, tmp_path / "pred.nii.gz", spacing=(2.0, 1.0, 5.0))
    # Give the prediction the crop affine, as a real model wrapper should.
    crop_img = nib.load(str(crop))
    nib.save(nib.Nifti1Image(prediction, crop_img.affine), str(pred))
    restored = tmp_path / "restored.nii.gz"
    restore_mask_to_full(pred, crop, ct, bbox, restored)
    restored_img = nib.load(str(restored))
    assert restored_img.shape == (40, 50, 60)
    assert np.allclose(restored_img.affine, nib.load(str(ct)).affine)
    assert int(np.asanyarray(restored_img.dataobj).sum()) == prediction.size


@pytest.mark.cpu_nifti_integration
def test_roi_planner_blocks_child_when_parent_missing(tmp_path: Path):
    from cli_anything.medai.core.hierarchical_roi import plan_roi_tasks

    ct = _save(np.zeros((20, 20, 20), dtype=np.int16), tmp_path / "ct.nii.gz")
    tasks, blocked = plan_roi_tasks(
        ct_path=ct,
        child_routes=[{"organ": "liver_segment_1", "parent_ids": ["liver"], "model": "teacher"}],
        parent_masks={},
    )
    assert tasks == []
    assert blocked[0]["status"] == "blocked_by_parent"


@pytest.mark.cpu_unit
def test_candidate_qc_distinguishes_missing_unreadable_and_zero_volume(tmp_path: Path):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
    missing = loop._compute_candidate_qc(ct=ct, mask=tmp_path / "missing.nii.gz", organ="liver")
    assert missing["flags"] == ["missing_file"]
    assert missing["mask_availability"] == "missing_file"

    bad = tmp_path / "bad.nii.gz"
    bad.write_text("not nifti", encoding="utf-8")
    unreadable = loop._compute_candidate_qc(ct=ct, mask=bad, organ="liver")
    assert "unreadable_mask" in unreadable["flags"]
    assert unreadable["mask_availability"] == "unreadable_mask"

    zero = _save(np.zeros((8, 8, 8), dtype=np.uint8), tmp_path / "zero.nii.gz")
    zqc = loop._compute_candidate_qc(ct=ct, mask=zero, organ="liver")
    assert "zero_volume_mask" in zqc["flags"]
    assert "empty_mask" not in zqc["flags"]
    assert zqc["mask_availability"] == "zero_volume_mask"
    assert zqc["status"] == "review"


@pytest.mark.cpu_unit
def test_abcd_grade_does_not_promote_common_whole_organ_from_qc_only():
    from cli_anything.medai.core.auto_fine_label import build_label_passport

    passport = build_label_passport({
        "case_id": "case_001",
        "organ": "liver",
        "mask_path": "/tmp/liver.nii.gz",
        "identity_status": "valid",
        "comparison_family": "whole_organ",
        "selected_candidate_qc_status": "pass",
        "selection_method": "label_critic_inconclusive",
        "selection_status": "fallback",
        "review_flags": ["selection_fallback", "labelcritic_uncertain", "shapekit_fallback"],
        "quality_flags": ["fallback_selection", "labelcritic_uncertain", "shapekit_fallback"],
        "shapekit_status": "unsupported_target",
    })
    assert passport["grade"] == "D"
    assert passport["training_weight"] == 0.0


@pytest.mark.cpu_unit
def test_abcd_grade_allows_d_for_insufficient_evidence():
    from cli_anything.medai.core.auto_fine_label import compute_reliability

    result = compute_reliability({
        "organ": "femur_left",
        "mask_path": "/tmp/femur_left.nii.gz",
        "identity_status": "valid",
        "selected_candidate_qc_status": "review",
        "selection_method": "label_critic_fallback",
        "selected_pseudo_consistency_dice": 0.1,
        "review_flags": ["selection_fallback", "labelcritic_uncertain", "shapekit_fallback"],
        "quality_flags": ["fallback_selection", "low_pseudo_consistency_dice"],
    })
    assert result["grade"] == "D"
    assert result["training_weight"] == 0.0


@pytest.mark.cpu_unit
def test_fusion_gate_is_conservative_qc_pass_same_identity_only(tmp_path: Path):
    from cli_anything.medai.core import multimodel_loop as loop

    m1 = _save(np.ones((8, 8, 8), dtype=np.uint8), tmp_path / "m1.nii.gz")
    m2 = _save(np.ones((8, 8, 8), dtype=np.uint8), tmp_path / "m2.nii.gz")
    base = {
        "requested_canonical_id": "liver",
        "comparison_family": "whole_organ",
        "identity_status": "valid",
        "candidate_qc_status": "pass",
        "candidate_qc_flags": [],
    }
    should, gate = loop._should_fuse_candidates([
        {**base, "model": "a", "prediction": str(m1)},
        {**base, "model": "b", "prediction": str(m2)},
    ])
    assert should is True
    assert gate["reason"] == "high_candidate_agreement_allows_conservative_fusion"

    should, gate = loop._should_fuse_candidates([
        {**base, "model": "a", "prediction": str(m1), "candidate_qc_flags": ["zero_volume_mask"]},
        {**base, "model": "b", "prediction": str(m2)},
    ])
    assert should is False
    assert gate["reason"] == "candidate_qc_not_all_pass_blocks_fusion"


@pytest.mark.cpu_nifti_integration
def test_multiple_children_share_crop_but_keep_independent_masks(tmp_path: Path):
    from cli_anything.medai.core.hierarchical_roi import execute_roi_tasks, plan_roi_tasks

    ct = _save(np.zeros((30, 30, 30), dtype=np.int16), tmp_path / "ct.nii.gz")
    parent_array = np.zeros((30, 30, 30), dtype=np.uint8)
    parent_array[10:20, 10:20, 10:20] = 1
    parent = _save(parent_array, tmp_path / "liver.nii.gz")
    routes = [
        {"organ": "liver_segment_1", "parent_ids": ["liver"], "model": "teacher"},
        {"organ": "liver_segment_2", "parent_ids": ["liver"], "model": "teacher"},
    ]
    tasks, blocked = plan_roi_tasks(ct_path=ct, child_routes=routes, parent_masks={"liver": parent}, margin_mm=2)
    assert not blocked and len(tasks) == 1
    merged = tmp_path / "merged"

    def fake_run(image: Path, output: Path, model: str, organs: list[str], task_id: str):
        crop = nib.load(str(image))
        seg = output / task_id / "segmentations"
        seg.mkdir(parents=True)
        for index, organ in enumerate(organs, start=1):
            arr = np.zeros(crop.shape, dtype=np.uint8)
            arr[index:index + 2, index:index + 2, index:index + 2] = 1
            nib.save(nib.Nifti1Image(arr, crop.affine), str(seg / f"{organ}.nii.gz"))
        return {"status": "success", "segmentation_output": str(seg)}

    execute_roi_tasks(
        ct_path=ct, tasks=tasks, work_root=tmp_path / "work",
        merged_model_dirs={"teacher": merged}, run_model=fake_run,
    )
    first = np.asanyarray(nib.load(str(merged / "liver_segment_1.nii.gz")).dataobj)
    second = np.asanyarray(nib.load(str(merged / "liver_segment_2.nii.gz")).dataobj)
    assert first.shape == second.shape == (30, 30, 30)
    assert not np.array_equal(first, second)


@pytest.mark.cpu_nifti_integration
def test_different_parents_are_not_merged_by_default(tmp_path: Path):
    from cli_anything.medai.core.hierarchical_roi import plan_roi_tasks

    ct = _save(np.zeros((40, 40, 40), dtype=np.int16), tmp_path / "ct.nii.gz")
    liver = np.zeros((40, 40, 40), dtype=np.uint8)
    pancreas = np.zeros((40, 40, 40), dtype=np.uint8)
    liver[5:15, 5:15, 5:15] = 1
    pancreas[20:25, 20:25, 20:25] = 1
    liver_path = _save(liver, tmp_path / "liver.nii.gz")
    pancreas_path = _save(pancreas, tmp_path / "pancreas.nii.gz")
    tasks, blocked = plan_roi_tasks(
        ct_path=ct,
        child_routes=[
            {"organ": "liver_segment_1", "parent_ids": ["liver"], "model": "teacher"},
            {"organ": "pancreas_head", "parent_ids": ["pancreas"], "model": "teacher"},
        ],
        parent_masks={"liver": liver_path, "pancreas": pancreas_path},
        margin_mm=2,
    )
    assert not blocked
    assert len(tasks) == 2
    assert {tuple(task["parents"]) for task in tasks} == {("liver",), ("pancreas",)}


@pytest.mark.cpu_nifti_integration
def test_cross_parent_merge_rejects_child_outside_own_support(tmp_path: Path):
    from cli_anything.medai.core.hierarchical_roi import execute_roi_tasks, plan_roi_tasks

    ct = _save(np.zeros((40, 40, 40), dtype=np.int16), tmp_path / "ct.nii.gz")
    liver = np.zeros((40, 40, 40), dtype=np.uint8)
    pancreas = np.zeros((40, 40, 40), dtype=np.uint8)
    liver[5:20, 5:20, 5:20] = 1
    pancreas[15:25, 15:25, 15:25] = 1
    tasks, blocked = plan_roi_tasks(
        ct_path=ct,
        child_routes=[
            {"organ": "liver_segment_1", "parent_ids": ["liver"], "model": "teacher"},
            {"organ": "pancreas_head", "parent_ids": ["pancreas"], "model": "teacher"},
        ],
        parent_masks={
            "liver": _save(liver, tmp_path / "liver.nii.gz"),
            "pancreas": _save(pancreas, tmp_path / "pancreas.nii.gz"),
        },
        margin_mm=2,
        allow_cross_parent_merge=True,
    )
    assert not blocked and len(tasks) == 1
    assert set(tasks[0]["organ_support_bboxes"]) == {"liver_segment_1", "pancreas_head"}

    def fake_run(image: Path, output: Path, model: str, organs: list[str], task_id: str):
        crop = nib.load(str(image))
        seg = output / task_id / "segmentations"
        seg.mkdir(parents=True)
        for organ in organs:
            arr = np.zeros(crop.shape, dtype=np.uint8)
            if organ == "liver_segment_1":
                arr[4:8, 4:8, 4:8] = 1
            else:
                arr[0:2, 0:2, 0:2] = 1
            nib.save(nib.Nifti1Image(arr, crop.affine), str(seg / f"{organ}.nii.gz"))
        return {"status": "success", "segmentation_output": str(seg)}

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "identity_provenance.json").write_text(json.dumps({
        "organs": {"pancreas_head": {"mapping_source": "stale_previous_run"}}
    }))
    results = execute_roi_tasks(
        ct_path=ct,
        tasks=tasks,
        work_root=tmp_path / "work",
        merged_model_dirs={"teacher": merged},
        run_model=fake_run,
    )
    records = {row["organ"]: row for row in results[0]["restored_masks"]}
    assert records["liver_segment_1"]["status"] == "accepted_parent_support"
    assert records["pancreas_head"]["status"] == "rejected_parent_support"
    assert (merged / "liver_segment_1.nii.gz").exists()
    assert not (merged / "pancreas_head.nii.gz").exists()
    provenance = json.loads((merged / "identity_provenance.json").read_text())
    assert "pancreas_head" not in provenance["organs"]


@pytest.mark.cpu_nifti_integration
def test_hierarchical_case_runs_major_then_child_roi(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((32, 32, 32), dtype=np.int16), tmp_path / "ct.nii.gz")
    taxonomy = json.loads((ROOT / "configs/organ_taxonomy.json").read_text())
    calls: list[dict] = []

    def fake_registered(image, output, model, **kwargs):
        requested = list((kwargs.get("extra_context") or {}).get("requested_organs", []))
        calls.append({"model": model, "image": str(image), "requested": requested})
        case_id = kwargs["case_id"]
        seg = Path(output) / case_id / "segmentations"
        seg.mkdir(parents=True, exist_ok=True)
        reference = nib.load(str(image))
        for organ in requested:
            arr = np.zeros(reference.shape, dtype=np.uint8)
            if organ == "liver":
                arr[8:24, 8:24, 8:24] = 1
            else:
                arr[3:7, 3:7, 3:7] = 1
            nib.save(nib.Nifti1Image(arr, reference.affine), str(seg / f"{organ}.nii.gz"))
        return {"status": "success", "model_key": model, "segmentation_output": str(seg), "num_masks": len(requested)}

    monkeypatch.setattr(loop, "run_registered_model", fake_registered)
    execution_plan = {
        "per_organ": {
            "liver": {"primary_teacher": "major_teacher", "backup_teachers": []},
            "liver_segment_1": {"primary_teacher": "child_teacher", "backup_teachers": ["unused_backup"]},
        }
    }
    result = loop._run_hierarchical_case_inference(
        ct=ct, case_id="case_001", case_raw=tmp_path / "raw", case_out=tmp_path / "case",
        registry_path=tmp_path / "registry.yaml", execution_plan=execution_plan,
        requested_organs=["liver", "liver_segment_1"], taxonomy=taxonomy,
        alias_config={"models": {}}, timeout_sec=30, device="cpu", dry_run=False, margin_mm=2,
    )
    assert [call["model"] for call in calls] == ["major_teacher", "child_teacher"]
    assert nib.load(str(result["model_seg_dirs"]["major_teacher"] / "liver.nii.gz")).shape == (32, 32, 32)
    assert nib.load(str(result["model_seg_dirs"]["child_teacher"] / "liver_segment_1.nii.gz")).shape == (32, 32, 32)
    provenance = json.loads((result["model_seg_dirs"]["child_teacher"] / "identity_provenance.json").read_text())
    assert provenance["organs"]["liver_segment_1"]["resolved_canonical_id"] == "liver_segment_1"
    assert not result["blocked"]
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["child_resolution"][0]["attempts"][1] == {
        "model": "unused_backup", "attempted": False, "reason": "not_needed_after_prior_success"
    }
    cached = loop._run_hierarchical_case_inference(
        ct=ct, case_id="case_001", case_raw=tmp_path / "raw", case_out=tmp_path / "case",
        registry_path=tmp_path / "registry.yaml", execution_plan=execution_plan,
        requested_organs=["liver", "liver_segment_1"], taxonomy=taxonomy,
        alias_config={"models": {}}, timeout_sec=30, device="cpu", dry_run=False, margin_mm=2,
    )
    assert len(calls) == 2
    assert cached["inference_results"][0]["cache_status"] == "reused_hierarchical_roi_cache"
    assert manifest["requested_organs"] == ["liver", "liver_segment_1"]
    assert manifest["hierarchical_plan_cache_key"]["requested_organs"] == ["liver", "liver_segment_1"]
    plan_changed = loop._run_hierarchical_case_inference(
        ct=ct, case_id="case_001", case_raw=tmp_path / "raw", case_out=tmp_path / "case",
        registry_path=tmp_path / "registry.yaml", execution_plan=execution_plan,
        requested_organs=["liver"], taxonomy=taxonomy,
        alias_config={"models": {}}, timeout_sec=30, device="cpu", dry_run=False, margin_mm=2,
    )
    assert len(calls) == 3
    assert plan_changed["inference_results"][0].get("cache_status") != "reused_hierarchical_roi_cache"


@pytest.mark.static_audit
def test_production_entrypoints_default_to_hierarchical_roi():
    run_em = (ROOT / "scripts/run_em_training.py").read_text(encoding="utf-8")
    cli = (ROOT / "agent-harness/cli_anything/medai/medai_cli.py").read_text(encoding="utf-8")
    assert 'teacher_inference_mode=TEACHER_INFERENCE_MODE if round_idx == 1 else "full_volume"' not in run_em
    assert "teacher_inference_mode=TEACHER_INFERENCE_MODE" in run_em
    assert '@click.option("--teacher-inference-mode", type=click.Choice(["full_volume", "hierarchical_roi"]), default="hierarchical_roi"' in cli


@pytest.mark.cpu_nifti_integration
def test_outer_resume_does_not_skip_when_hierarchical_plan_changes(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((32, 32, 32), dtype=np.int16), tmp_path / "ct.nii.gz")
    case_list = tmp_path / "cases.csv"
    case_list.write_text(f"case_id,ct_path,annotation_folder\ncase_001,{ct},\n", encoding="utf-8")
    out = tmp_path / "out"
    calls: list[str] = []

    def fake_registered(image, output, model, **kwargs):
        calls.append(model)
        requested = list((kwargs.get("extra_context") or {}).get("requested_organs", []))
        seg = Path(output) / kwargs["case_id"] / "segmentations"
        seg.mkdir(parents=True, exist_ok=True)
        reference = nib.load(str(image))
        for organ in requested:
            arr = np.zeros(reference.shape, dtype=np.uint8)
            if organ == "liver":
                arr[8:24, 8:24, 8:24] = 1
            else:
                arr[3:7, 3:7, 3:7] = 1
            nib.save(nib.Nifti1Image(arr, reference.affine), str(seg / f"{organ}.nii.gz"))
        return {"status": "success", "model_key": model, "segmentation_output": str(seg), "num_masks": len(requested)}

    monkeypatch.setattr(loop, "run_registered_model", fake_registered)
    monkeypatch.setattr(loop, "run_shapekit", lambda *args, **kwargs: {"status": "skipped_debug_only"})
    monkeypatch.setattr(loop, "load_registry", lambda registry_path: {"models": {
        "major_teacher": {},
        "child_teacher_old": {},
        "child_teacher_new": {},
    }})
    monkeypatch.setattr(loop, "candidate_models_for_organs", lambda registry, organs, include_mock=False: {
        "liver": ["major_teacher"],
        "liver_segment_1": ["child_teacher_old", "child_teacher_new"],
    })
    branch_map = {
        "liver": {"teacher_model": "major_teacher", "fallback_teachers": []},
        "liver_segment_1": {"teacher_model": "child_teacher_old", "fallback_teachers": []},
    }
    monkeypatch.setattr(loop, "_load_teacher_branch_map", lambda root: branch_map)

    first = loop.run_multimodel_annotation_loop(
        case_list=case_list,
        output_folder=out,
        models=["major_teacher", "child_teacher_old", "child_teacher_new"],
        organs=["liver_segment_1"],
        enable_critic=False,
        enable_shapekit=False,
        dry_run=False,
        resume=True,
        timeout_sec=30,
        teacher_inference_mode="hierarchical_roi",
        enable_fusion=False,
        enable_auto_arbitration=False,
        candidate_mode="route_pruned_with_competition",
    )
    assert first["status"] == "success"
    assert calls == ["major_teacher", "child_teacher_old"]

    branch_map["liver_segment_1"] = {"teacher_model": "child_teacher_new", "fallback_teachers": []}
    second = loop.run_multimodel_annotation_loop(
        case_list=case_list,
        output_folder=out,
        models=["major_teacher", "child_teacher_old", "child_teacher_new"],
        organs=["liver_segment_1"],
        enable_critic=False,
        enable_shapekit=False,
        dry_run=False,
        resume=True,
        timeout_sec=30,
        teacher_inference_mode="hierarchical_roi",
        enable_fusion=False,
        enable_auto_arbitration=False,
        candidate_mode="route_pruned_with_competition",
    )
    assert second["status"] == "success"
    assert calls == ["major_teacher", "child_teacher_old", "major_teacher", "child_teacher_new"]
    assert second["resume_audit"][0]["reason"] == "hierarchical_roi_plan_cache_mismatch"
    manifest = json.loads((out / "cases" / "case_001" / "hierarchical_inference_plan.json").read_text())
    assert manifest["hierarchical_plan_cache_key"]["execution_plan"]["per_organ"]["liver_segment_1"]["primary_teacher"] == "child_teacher_new"


@pytest.mark.cpu_nifti_integration
def test_hierarchical_reuses_legacy_major_parent_cache_but_reruns_child_roi(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((32, 32, 32), dtype=np.int16), tmp_path / "ct.nii.gz")
    case_raw = tmp_path / "case" / "raw_predictions"
    legacy_major = case_raw / "major_teacher" / "case_001" / "segmentations"
    legacy_major.mkdir(parents=True)
    liver = np.zeros((32, 32, 32), dtype=np.uint8)
    liver[8:24, 8:24, 8:24] = 1
    _save(liver, legacy_major / "liver.nii.gz")
    _save(np.ones((32, 32, 32), dtype=np.uint8), legacy_major / "liver_segment_1.nii.gz")
    calls: list[tuple[str, list[str], str]] = []

    def fake_registered(image, output, model, **kwargs):
        requested = list((kwargs.get("extra_context") or {}).get("requested_organs", []))
        scope = str((kwargs.get("extra_context") or {}).get("inference_scope"))
        calls.append((model, requested, scope))
        seg = Path(output) / kwargs["case_id"] / "segmentations"
        seg.mkdir(parents=True, exist_ok=True)
        reference = nib.load(str(image))
        for organ in requested:
            arr = np.zeros(reference.shape, dtype=np.uint8)
            arr[2:6, 2:6, 2:6] = 1
            nib.save(nib.Nifti1Image(arr, reference.affine), str(seg / f"{organ}.nii.gz"))
        return {"status": "success", "model_key": model, "segmentation_output": str(seg), "num_masks": len(requested)}

    monkeypatch.setattr(loop, "run_registered_model", fake_registered)
    taxonomy = json.loads((ROOT / "configs/organ_taxonomy.json").read_text())
    execution_plan = {"per_organ": {
        "liver": {"primary_teacher": "major_teacher", "backup_teachers": []},
        "liver_segment_1": {"primary_teacher": "child_teacher", "backup_teachers": []},
    }}
    result = loop._run_hierarchical_case_inference(
        ct=ct,
        case_id="case_001",
        case_raw=case_raw,
        case_out=tmp_path / "case",
        registry_path=tmp_path / "registry.yaml",
        execution_plan=execution_plan,
        requested_organs=["liver", "liver_segment_1"],
        taxonomy=taxonomy,
        alias_config={"models": {}},
        timeout_sec=30,
        device="cpu",
        dry_run=False,
        margin_mm=2,
    )
    assert calls == [("child_teacher", ["liver_segment_1"], "child_roi")]
    assert any(r.get("cache_status") == "reused_legacy_full_volume_major_parent_cache" for r in result["inference_results"])
    assert (result["model_seg_dirs"]["child_teacher"] / "liver_segment_1.nii.gz").exists()


@pytest.mark.cpu_nifti_integration
def test_preseeded_major_parent_is_reused_inside_hierarchical_pipeline(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((32, 32, 32), dtype=np.int16), tmp_path / "ct.nii.gz")
    preseed = tmp_path / "old_round1" / "segmentations"
    preseed.mkdir(parents=True)
    liver = np.zeros((32, 32, 32), dtype=np.uint8)
    liver[8:24, 8:24, 8:24] = 1
    _save(liver, preseed / "liver.nii.gz")
    calls: list[tuple[str, str]] = []

    def fake_registered(image, output, model, **kwargs):
        scope = str((kwargs.get("extra_context") or {}).get("inference_scope"))
        calls.append((model, scope))
        seg = Path(output) / kwargs["case_id"] / "segmentations"
        seg.mkdir(parents=True, exist_ok=True)
        reference = nib.load(str(image))
        arr = np.zeros(reference.shape, dtype=np.uint8)
        arr[2:6, 2:6, 2:6] = 1
        nib.save(nib.Nifti1Image(arr, reference.affine), str(seg / "liver_segment_1.nii.gz"))
        return {"status": "success", "model_key": model, "segmentation_output": str(seg), "num_masks": 1}

    monkeypatch.setattr(loop, "run_registered_model", fake_registered)
    taxonomy = json.loads((ROOT / "configs/organ_taxonomy.json").read_text())
    result = loop._run_hierarchical_case_inference(
        ct=ct,
        case_id="case_001",
        case_raw=tmp_path / "new" / "raw_predictions",
        case_out=tmp_path / "new",
        registry_path=tmp_path / "registry.yaml",
        execution_plan={"per_organ": {
            "liver": {"primary_teacher": "uncached_primary", "backup_teachers": ["major_teacher"]},
            "liver_segment_1": {"primary_teacher": "child_teacher", "backup_teachers": []},
        }},
        requested_organs=["liver", "liver_segment_1"],
        taxonomy=taxonomy,
        alias_config={"models": {}},
        timeout_sec=30,
        device="cpu",
        dry_run=False,
        margin_mm=2,
        preseeded_seg_dirs={"major_teacher": preseed},
    )
    assert calls == [("child_teacher", "child_roi")]
    assert any(r.get("cache_status") == "reused_preseeded_major_parent_cache" for r in result["inference_results"])


def test_preseed_resolver_prefers_binary_segmentations_over_combined_label_root(tmp_path: Path):
    from cli_anything.medai.core.multimodel_loop import _resolve_preseeded_case_dir

    teacher = tmp_path / "teacher"
    case_root = teacher / "case_001"
    segmentations = case_root / "segmentations"
    segmentations.mkdir(parents=True)
    (case_root / "combined_labels.nii.gz").write_bytes(b"combined")
    (segmentations / "liver.nii.gz").write_bytes(b"binary")
    assert _resolve_preseeded_case_dir(teacher, "case_001") == segmentations


@pytest.mark.cpu_nifti_integration
def test_hierarchical_keeps_display_case_major_target_from_normalized_taxonomy(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
    preseed = tmp_path / "old" / "segmentations"
    preseed.mkdir(parents=True)
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[4:12, 4:12, 4:12] = 1
    _save(mask, preseed / "vertebrae_L2.nii.gz")
    monkeypatch.setattr(loop, "run_registered_model", lambda *args, **kwargs: pytest.fail("fresh inference must not run"))
    taxonomy = json.loads((ROOT / "configs/organ_taxonomy.json").read_text())
    result = loop._run_hierarchical_case_inference(
        ct=ct,
        case_id="case_001",
        case_raw=tmp_path / "new" / "raw_predictions",
        case_out=tmp_path / "new",
        registry_path=tmp_path / "registry.yaml",
        execution_plan={"per_organ": {
            "vertebrae_L2": {"primary_teacher": "vertebra_teacher", "backup_teachers": []},
        }},
        requested_organs=["vertebrae_L2"],
        taxonomy=taxonomy,
        alias_config={"models": {}},
        timeout_sec=30,
        device="cpu",
        dry_run=False,
        margin_mm=2,
        preseeded_seg_dirs={"vertebra_teacher": preseed},
    )
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["major_organs"] == ["vertebrae_L2"]
    assert manifest["major_resolution"][0]["status"] == "resolved"


@pytest.mark.cpu_nifti_integration
def test_completed_zero_preseed_is_not_rerun_as_fresh_full_volume(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((16, 16, 16), dtype=np.int16), tmp_path / "ct.nii.gz")
    preseed = tmp_path / "old" / "case_001"
    preseed.mkdir(parents=True)
    (preseed / "combined_labels.nii.gz").write_bytes(b"completed-empty-placeholder")
    (preseed / "inference_summary.json").write_text(json.dumps({"return_code": 0, "timed_out": False, "num_masks": 0}))
    monkeypatch.setattr(loop, "run_registered_model", lambda *args, **kwargs: pytest.fail("completed zero-output cache must not rerun"))
    taxonomy = json.loads((ROOT / "configs/organ_taxonomy.json").read_text())
    result = loop._run_hierarchical_case_inference(
        ct=ct, case_id="case_001", case_raw=tmp_path / "new" / "raw_predictions", case_out=tmp_path / "new",
        registry_path=tmp_path / "registry.yaml",
        execution_plan={"per_organ": {"brain": {"primary_teacher": "brain_teacher", "backup_teachers": []}}},
        requested_organs=["brain"], taxonomy=taxonomy, alias_config={"models": {}}, timeout_sec=30,
        device="cpu", dry_run=False, margin_mm=2, preseeded_seg_dirs={"brain_teacher": preseed},
    )
    assert any(r.get("cache_status") == "reused_preseeded_completed_zero_output" for r in result["inference_results"])
    assert any(item.get("organ") == "brain" and item.get("status") == "major_unresolved" for item in result["blocked"])


@pytest.mark.cpu_nifti_integration
def test_cross_parent_support_failure_retries_only_failed_child(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((32, 32, 32), dtype=np.int16), tmp_path / "ct.nii.gz")
    taxonomy = json.loads((ROOT / "configs/organ_taxonomy.json").read_text())
    calls: list[tuple[str, list[str]]] = []

    def fake_registered(image, output, model, **kwargs):
        requested = list((kwargs.get("extra_context") or {}).get("requested_organs", []))
        calls.append((model, requested))
        seg = Path(output) / kwargs["case_id"] / "segmentations"
        seg.mkdir(parents=True, exist_ok=True)
        reference = nib.load(str(image))
        for organ in requested:
            arr = np.zeros(reference.shape, dtype=np.uint8)
            if organ == "liver":
                arr[4:20, 4:20, 4:20] = 1
            elif organ == "pancreas":
                arr[14:26, 14:26, 14:26] = 1
            elif len(requested) > 1 and organ == "pancreas_head":
                arr[0:3, 0:3, 0:3] = 1
            else:
                center = tuple(max(1, size // 2) for size in reference.shape)
                arr[center[0] - 2:center[0] + 2, center[1] - 2:center[1] + 2, center[2] - 2:center[2] + 2] = 1
            nib.save(nib.Nifti1Image(arr, reference.affine), str(seg / f"{organ}.nii.gz"))
        return {"status": "success", "model_key": model, "segmentation_output": str(seg)}

    monkeypatch.setattr(loop, "run_registered_model", fake_registered)
    execution_plan = {"per_organ": {
        "liver": {"primary_teacher": "major_teacher", "backup_teachers": []},
        "pancreas": {"primary_teacher": "major_teacher", "backup_teachers": []},
        "liver_segment_1": {"primary_teacher": "child_teacher", "backup_teachers": []},
        "pancreas_head": {"primary_teacher": "child_teacher", "backup_teachers": []},
    }}
    result = loop._run_hierarchical_case_inference(
        ct=ct,
        case_id="case_001",
        case_raw=tmp_path / "raw",
        case_out=tmp_path / "case",
        registry_path=tmp_path / "registry.yaml",
        execution_plan=execution_plan,
        requested_organs=["liver", "pancreas", "liver_segment_1", "pancreas_head"],
        taxonomy=taxonomy,
        alias_config={"models": {}},
        timeout_sec=30,
        device="cpu",
        dry_run=False,
        margin_mm=2,
    )
    assert calls == [
        ("major_teacher", ["liver", "pancreas"]),
        ("child_teacher", ["liver_segment_1", "pancreas_head"]),
        ("child_teacher", ["pancreas_head"]),
    ]
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["roi_tasks"][0]["restored_masks"][1]["status"] == "rejected_parent_support"
    assert manifest["primary_parent_retry_tasks"][0]["organs"] == ["pancreas_head"]
    assert all(row["status"] == "resolved" for row in manifest["child_resolution"])


@pytest.mark.cpu_nifti_integration
def test_hierarchical_major_hard_qc_triggers_backup(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((20, 20, 20), dtype=np.int16), tmp_path / "ct.nii.gz")
    taxonomy = json.loads((ROOT / "configs/organ_taxonomy.json").read_text())
    calls: list[str] = []

    def fake_registered(image, output, model, **kwargs):
        calls.append(model)
        seg = Path(output) / kwargs["case_id"] / "segmentations"
        seg.mkdir(parents=True, exist_ok=True)
        if model == "bad_teacher":
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)), str(seg / "liver.nii.gz"))
        else:
            reference = nib.load(str(image))
            array = np.zeros(reference.shape, dtype=np.uint8)
            array[5:15, 5:15, 5:15] = 1
            nib.save(nib.Nifti1Image(array, reference.affine), str(seg / "liver.nii.gz"))
        return {"status": "success", "model_key": model, "segmentation_output": str(seg)}

    monkeypatch.setattr(loop, "run_registered_model", fake_registered)
    result = loop._run_hierarchical_case_inference(
        ct=ct, case_id="case_001", case_raw=tmp_path / "raw", case_out=tmp_path / "case",
        registry_path=tmp_path / "registry.yaml",
        execution_plan={"per_organ": {"liver": {"primary_teacher": "bad_teacher", "backup_teachers": ["good_teacher"]}}},
        requested_organs=["liver"], taxonomy=taxonomy, alias_config={"models": {}},
        timeout_sec=30, device="cpu", dry_run=False, margin_mm=20,
    )
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert calls == ["bad_teacher", "good_teacher"]
    assert [item["status"] for item in manifest["major_resolution"]] == ["unusable", "resolved"]
    assert manifest["major_resolution"][0]["hierarchy_qc"]["status"] == "fail"


@pytest.mark.cpu_unit
def test_gpu_guard_blocks_foreign_compute_process(monkeypatch):
    from cli_anything.medai.core import resource_guard

    class Result:
        def __init__(self, stdout: str):
            self.stdout, self.stderr, self.returncode = stdout, "", 0

    outputs = iter([
        Result("0, NVIDIA A100, 81920, 18000, 63920, 100\n"),
        Result("GPU-id, 1234, other-project-python, 18000\n"),
    ])
    monkeypatch.setattr(resource_guard.subprocess, "run", lambda *args, **kwargs: next(outputs))
    result = resource_guard.inspect_gpu_resources(samples=1)
    assert result["status"] == "skipped_resource_busy"
    assert "foreign_compute_process_present" in result["busy_reasons"]


@pytest.mark.cpu_unit
def test_mstep_manifest_quarantines_identity_mismatch(tmp_path: Path):
    from cli_anything.medai.core.mstep_runner import build_training_manifest

    root = tmp_path / "dataset"
    for case_id, status, resolved in (
        ("valid_case", "valid", "liver_segment_1"),
        ("bad_case", "identity_mismatch", "liver"),
    ):
        case = root / case_id
        seg = case / "segmentations"
        seg.mkdir(parents=True)
        _save(np.ones((4, 4, 4), dtype=np.uint8), seg / "liver_segment_1.nii.gz")
        (case / "selection_metadata.json").write_text(json.dumps({
            "selected_organs": [{
                "case_id": case_id,
                "organ": "liver_segment_1",
                "requested_canonical_id": "liver_segment_1",
                "resolved_canonical_id": resolved,
                    "identity_status": status,
                    "identity_mismatch_reasons": [] if status == "valid" else ["canonical_id_mismatch"],
                    "grade": "B",
                    "training_weight": 0.5,
                    "scoring_schema_version": "autolabel_core_v2",
            }]
        }))
    output = tmp_path / "training_manifest.json"
    result = build_training_manifest(root, output)
    rows = json.loads(output.read_text())
    exclusions = json.loads(Path(result["identity_exclusions"]).read_text())
    assert [(row["case_id"], row["organ"]) for row in rows] == [("valid_case", "liver_segment_1")]
    assert [(row["case_id"], row["identity_status"]) for row in exclusions] == [("bad_case", "identity_mismatch")]


@pytest.mark.cpu_nifti_integration
def test_approved_union_requires_all_components(tmp_path: Path):
    from cli_anything.medai.core import multimodel_loop as loop

    seg = tmp_path / "segmentations"
    seg.mkdir()
    first = np.zeros((8, 8, 8), dtype=np.uint8)
    second = np.zeros((8, 8, 8), dtype=np.uint8)
    first[1:3, 1:3, 1:3] = 1
    second[5:7, 5:7, 5:7] = 1
    _save(first, seg / "lung_arteries.nii.gz")
    aliases = {
        "models": {"teacher": {
            "local_to_global": {"lung_arteries": "lung_vessels", "lung_veins": "lung_vessels"},
            "mapping_types": {"lung_arteries": "approved_union", "lung_veins": "approved_union"},
        }}
    }
    missing, mode = loop._candidate_mask_path(seg, "lung_vessels", "teacher", aliases)
    assert not missing.exists()
    assert mode == "missing_approved_union_components"
    _save(second, seg / "lung_veins.nii.gz")
    union, mode = loop._candidate_mask_path(seg, "lung_vessels", "teacher", aliases)
    assert mode == "approved_union:lung_arteries+lung_veins"
    assert int(np.asanyarray(nib.load(str(union)).dataobj).sum()) == 16


@pytest.mark.cpu_nifti_integration
def test_label_merger_materializes_approved_union(tmp_path: Path):
    from cli_anything.medai.core import label_merger

    seg = tmp_path / "segmentations"
    seg.mkdir()
    a = np.zeros((6, 6, 6), dtype=np.uint8)
    b = np.zeros((6, 6, 6), dtype=np.uint8)
    a[1:2, 1:2, 1:2] = 1
    b[4:5, 4:5, 4:5] = 1
    _save(a, seg / "lung_arteries.nii.gz")
    aliases = {
        "local_to_global": {"lung_arteries": "lung_vessels", "lung_veins": "lung_vessels"},
        "mapping_types": {"lung_arteries": "approved_union", "lung_veins": "approved_union"},
    }
    incomplete = label_merger._materialize_approved_union(seg, "lung_vessels", aliases)
    assert incomplete["status"] == "incomplete"
    _save(b, seg / "lung_veins.nii.gz")
    complete = label_merger._materialize_approved_union(seg, "lung_vessels", aliases)
    assert complete["status"] == "success"
    assert int(np.asanyarray(nib.load(complete["output"]).dataobj).sum()) == 2
