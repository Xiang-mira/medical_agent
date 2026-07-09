from __future__ import annotations

import csv
import json
import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def load_script(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_student_masks_can_live_directly_in_case_directory(tmp_path):
    protocol = load_script("run_blind10_protocol.py")
    case = tmp_path / "student_only" / "case_001"
    case.mkdir(parents=True)
    (case / "liver.nii.gz").touch()

    assert protocol.find_student_segmentations(tmp_path / "student_only", "case_001") == case


def test_evaluator_accepts_direct_case_directory(tmp_path):
    evaluator = load_script("evaluate_blind_segmentation.py")
    case = tmp_path / "student_only" / "case_001"
    case.mkdir(parents=True)
    (case / "liver.nii.gz").touch()

    assert evaluator.prediction_dir(tmp_path / "student_only", "case_001") == case


def write_nib(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), np.eye(4)), str(path))


def write_sitk(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(arr.astype(np.uint8)), str(path))


def test_blind_metric_target_reflects_reference_kind():
    evaluator = load_script("evaluate_blind_segmentation.py")

    assert evaluator.metric_target_from_reference_kind("expert_gt")[0] == "GT"
    assert evaluator.metric_target_from_reference_kind("historical_pseudo")[0] == "pseudo-label"
    assert evaluator.metric_target_from_reference_kind("absent_negative")[0] == "all-zero target"


def test_blind_per_organ_metrics_include_metric_target_and_fp_voxels(tmp_path, monkeypatch):
    evaluator = load_script("evaluate_blind_segmentation.py")
    case_list = tmp_path / "cases.csv"
    ct = tmp_path / "ct.nii.gz"
    write_nib(ct, np.zeros((4, 4, 4), dtype=np.uint8))
    case_list.write_text(f"case_id,ct_path\ncase_001,{ct}\n", encoding="utf-8")
    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    pred[1, 1, 1] = 1
    ref = np.zeros((4, 4, 4), dtype=np.uint8)
    write_nib(tmp_path / "pred" / "case_001" / "liver.nii.gz", pred)
    write_nib(tmp_path / "ref" / "case_001" / "segmentations" / "liver.nii.gz", ref)
    out = tmp_path / "out"

    monkeypatch.setattr(sys, "argv", [
        "evaluate_blind_segmentation.py",
        "--case-list", str(case_list),
        "--prediction-root", str(tmp_path / "pred"),
        "--reference-root", str(tmp_path / "ref"),
        "--output-dir", str(out),
        "--reference-kind", "expert_gt",
        "--bootstrap-samples", "1",
    ])
    assert evaluator.main() == 0
    with (out / "per_organ_metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["metric_target"] == "GT"
    assert rows[0]["metric_interpretation"] == "real_gt_segmentation_performance"
    assert rows[0]["false_positive_voxels"] == "1"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["metric_target"] == "GT"
    assert summary["false_positive_voxels"] == 1


def test_segmentation_chain_writes_long_metric_targets(tmp_path, monkeypatch):
    evaluator = load_script("evaluate_segmentation_chain.py")
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[1:3, 1:3, 1:3] = 1
    write_sitk(tmp_path / "student" / "case_001" / "liver.nii.gz", arr)
    write_sitk(tmp_path / "teacher" / "case_001" / "updated" / "liver.nii.gz", arr)
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"target_organs": ["liver"]}), encoding="utf-8")
    out = tmp_path / "chain"

    monkeypatch.setattr(sys, "argv", [
        "evaluate_segmentation_chain.py",
        "--student-root", str(tmp_path / "student"),
        "--teacher-root", str(tmp_path / "teacher"),
        "--target-config", str(targets),
        "--output-dir", str(out),
    ])
    assert evaluator.main() == 0
    with (out / "evaluation_chain_metrics_long.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    targets_seen = {row["metric_target"] for row in rows}
    comparisons = {row["metric_comparison"] for row in rows}
    assert targets_seen == {"selected_pseudo_label"}
    assert "student_candidate_vs_selected_pseudo_label" in comparisons


def test_formal_em_pipeline_has_no_gt_entrypoint() -> None:
    source = (Path(__file__).resolve().parents[2] / "scripts" / "run_em_training.py").read_text(encoding="utf-8")
    assert "MEDAI_GT_ROOT" not in source
    assert "--gt-root" not in source
    assert "student_vs_gt.csv" not in source
    assert "teacher_vs_gt.csv" not in source


def test_student_shapekit_materializes_flat_candidate_root(tmp_path, monkeypatch):
    runner = load_script("run_em_training.py")
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[1:3, 1:3, 1:3] = 1
    write_nib(tmp_path / "outputs" / "round1" / "student_predictions" / "case_001" / "liver.nii.gz", arr)
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"target_organs": ["liver"]}), encoding="utf-8")

    from cli_anything.medai.core import shapekit_runner

    def fake_run_shapekit(shapekit_root, input_folder, output_folder, log_folder, **kwargs):
        src = Path(input_folder) / "case_001" / "segmentations" / "liver.nii.gz"
        dst = Path(output_folder) / "case_001" / "segmentations" / "liver.nii.gz"
        dst.parent.mkdir(parents=True)
        shutil.copy2(src, dst)
        return {"status": "success", "reason": None}

    import shutil
    monkeypatch.setattr(shapekit_runner, "run_shapekit", fake_run_shapekit)
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(runner, "PROMPT_TARGET_CONFIG", targets)
    monkeypatch.setattr(runner, "ENABLE_SHAPEKIT", True)

    summary = runner.apply_round_student_shapekit(1)
    assert summary["status"] == "success"
    assert summary["processed_by_shapekit"] == 1
    assert (tmp_path / "outputs" / "round1" / "student_predictions_shapekit" / "case_001" / "liver.nii.gz").is_file()
    with (tmp_path / "outputs" / "round1" / "student_predictions_shapekit" / "student_shapekit_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["student_shapekit_status"] == "success"


def test_compute_round_metrics_marks_student_dice_as_pseudo_label(tmp_path, monkeypatch):
    runner = load_script("run_em_training.py")
    case_list = tmp_path / "cases.csv"
    case_list.write_text("case_id,ct_path\ncase_001,/tmp/ct.nii.gz\n", encoding="utf-8")
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[1:3, 1:3, 1:3] = 1
    write_nib(tmp_path / "outputs" / "round2" / "student_predictions" / "case_001" / "liver.nii.gz", arr)
    write_nib(tmp_path / "outputs" / "round1" / "estep" / "annotation_versions" / "case_001" / "updated" / "liver.nii.gz", arr)
    monkeypatch.setattr(runner, "CASE_LIST", case_list)
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "outputs")

    result = runner.compute_round_metrics(2, reference_round=1)
    assert result["metric_target"] == "selected_pseudo_label"
    with (tmp_path / "outputs" / "round2" / "metrics" / "student_dice_per_organ.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["metric_target"] == "selected_pseudo_label"
    assert rows[0]["metric_comparison"] == "student_vs_selected_pseudo_label"


def test_run_em_training_preserves_exact_373_target_space():
    runner = load_script("run_em_training.py")

    targets = runner.load_student_target_organs()
    audit = runner.student_target_space_audit()
    assert len(targets) == 373
    assert "inferior_vena_cava" in targets
    assert "postcava" in targets
    assert audit["exact_target_count"] == 373
    assert audit["canonical_unique_count"] == 372
    assert audit["canonical_collisions"]["inferior_vena_cava"] == ["inferior_vena_cava", "postcava"]


def test_run_em_training_dry_run_does_not_launch_estep(tmp_path, monkeypatch):
    runner = load_script("run_em_training.py")
    called = {"estep": False}

    def fake_run_estep(_round_idx):
        called["estep"] = True
        raise AssertionError("dry-run must not launch E-step")

    monkeypatch.setattr(runner, "run_estep", fake_run_estep)
    monkeypatch.setattr(runner, "CASE_LIST", tmp_path / "missing.csv")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(runner, "LOG_FILE", tmp_path / "outputs" / "training.log")
    result = runner.dry_run_preflight(tmp_path / "preflight.json")

    assert result["would_launch_em"] is False
    assert result["target_space"]["exact_target_count"] == 373
    assert called["estep"] is False
    assert (tmp_path / "preflight.json").exists()


def test_repair_plan_audit_target_space_success(tmp_path, monkeypatch):
    auditor = load_script("audit_repair_plan.py")
    out = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", [
        "audit_repair_plan.py",
        "--output", str(out),
    ])

    assert auditor.main() in {0, 1}
    report = json.loads(out.read_text())
    assert report["target_space"]["exact_target_count"] == 373
    assert report["target_space"]["appearance_entry_count"] == 373
    assert report["target_space"]["appearance_map_count"] == 373
    assert report["target_space"]["status"] == "success"


def test_runtime_sampler_does_not_derive_crop_negative_for_container_organs():
    trainer = load_script("train_voxtell_prompt_student.py")
    rows = [
        {
            "case_id": "case_001",
            "organ": "abdominal_cavity",
            "supervision_type": "positive",
            "training_weight": 1.0,
        },
        {
            "case_id": "case_001",
            "organ": "liver",
            "supervision_type": "positive",
            "training_weight": 1.0,
        },
    ]
    pools = trainer.build_sample_pools(rows)

    derived_organs = {row["organ"] for row in pools["derived_crop_negative"]}
    assert "abdominal_cavity" not in derived_organs
    assert "liver" in derived_organs


def test_student_inference_counts_empty_absent_masks_as_layout_success():
    runner = load_script("run_student_infer_then_round2.py")

    result = {
        "status": "partial_success",
        "num_masks": 5,
        "num_empty_masks": 1,
        "empty_organs": ["brain"],
        "num_failed_organs": 0,
    }
    assert runner.inference_case_acceptable(result, expected_organs=5)
    result["num_failed_organs"] = 1
    assert not runner.inference_case_acceptable(result, expected_organs=5)


def test_student_inference_resume_requires_result_and_every_exact_mask(tmp_path):
    runner = load_script("run_student_infer_then_round2.py")
    organs = ["liver", "spleen"]
    result = {
        "status": "partial_success",
        "num_masks": 2,
        "num_failed_organs": 0,
    }
    (tmp_path / "voxtell_student_result.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "liver.nii.gz").touch()
    assert not runner.existing_case_complete(tmp_path, organs)
    (tmp_path / "spleen.nii.gz").touch()
    assert runner.existing_case_complete(tmp_path, organs)


def test_voxtell_baseline_regression_gate_detects_semantic_expansion(tmp_path):
    audit = load_script("audit_voxtell_baseline_regression.py")
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    reference = np.zeros((4, 4, 4), dtype=np.uint8)
    reference[1:3, 1:3, 1:3] = 1
    empty = np.zeros_like(reference)
    expanded = np.ones_like(reference)
    write_nib(baseline / "liver.nii.gz", reference)
    write_nib(candidate / "liver.nii.gz", reference)
    write_nib(baseline / "brain.nii.gz", empty)
    write_nib(candidate / "brain.nii.gz", expanded)
    ref_path = tmp_path / "liver_ref.nii.gz"
    zero_path = tmp_path / "zero.nii.gz"
    write_nib(ref_path, reference)
    write_nib(zero_path, empty)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [
        {"case_id": "case", "organ": "liver", "mask": str(ref_path)},
        {"case_id": "case", "organ": "brain", "mask": str(zero_path)},
    ]}), encoding="utf-8")

    result = audit.audit(
        baseline_root=baseline,
        candidate_root=candidate,
        manifest_path=manifest,
        case_id="case",
        positive_organs=["liver"],
        absent_organs=["brain"],
        max_positive_dice_drop=0.02,
        min_positive_volume_ratio=0.8,
        max_positive_volume_ratio=1.25,
        max_absent_volume_ratio=1.1,
        absent_volume_slack_ml=0.0001,
    )

    assert result["status"] == "failed"
    assert result["failures"] == [{"organ": "brain", "reason": "absent_volume_regression"}]


def test_voxtell_baseline_regression_accepts_shared_empty_positive(tmp_path):
    audit = load_script("audit_voxtell_baseline_regression.py")
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    empty = np.zeros((4, 4, 4), dtype=np.uint8)
    write_nib(baseline / "spleen.nii.gz", empty)
    write_nib(candidate / "spleen.nii.gz", empty)
    ref_path = tmp_path / "spleen_ref.nii.gz"
    write_nib(ref_path, empty)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [
        {"case_id": "case", "organ": "spleen", "mask": str(ref_path)},
    ]}), encoding="utf-8")

    result = audit.audit(
        baseline_root=baseline,
        candidate_root=candidate,
        manifest_path=manifest,
        case_id="case",
        positive_organs=["spleen"],
        absent_organs=[],
        max_positive_dice_drop=0.02,
        min_positive_volume_ratio=0.8,
        max_positive_volume_ratio=1.25,
        max_absent_volume_ratio=1.1,
        absent_volume_slack_ml=0.1,
    )

    assert result["status"] == "success"
    assert result["rows"][0]["volume_ratio"] == 1.0
