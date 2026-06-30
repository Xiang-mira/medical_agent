from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent-harness"))


def _target_config(tmp_path: Path) -> Path:
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({
        "target_organs": ["liver"],
        "organ_to_prompt": {"liver": "segment the liver"},
    }), encoding="utf-8")
    return path


def test_voxtell_student_dry_run_defaults_to_official_python_api(tmp_path: Path) -> None:
    from cli_anything.medai.core.voxtell_student import VoxTellStudent

    student = VoxTellStudent(
        model_dir=tmp_path / "model",
        device="cpu",
        target_config=_target_config(tmp_path),
        text_encoding_model=tmp_path / "qwen",
    )
    result = student.segment(tmp_path / "ct.nii.gz", tmp_path / "out", prompts=["liver"], dry_run=True)
    assert result["backend"] == "official_python_api"
    assert result["voxtell_source_mode"] == "official_vendor_via_project_adapter"
    assert result["vendor_audit"]["official_repo"] == "https://github.com/MIC-DKFZ/VoxTell"
    assert result["command"][0] == "official_python_api"
    assert "--text-encoding-model" in result["command"]
    assert result["expected_masks"]["liver"].endswith("liver.nii.gz")


def test_voxtell_student_official_cli_dry_run_uses_only_official_cli_args(tmp_path: Path) -> None:
    from cli_anything.medai.core.voxtell_student import VoxTellStudent

    student = VoxTellStudent(
        model_dir=tmp_path / "model",
        device="cpu",
        target_config=_target_config(tmp_path),
        text_encoding_model=tmp_path / "qwen",
        backend="official_cli",
    )
    result = student.segment(tmp_path / "ct.nii.gz", tmp_path / "out", prompts=["liver"], dry_run=True)
    assert result["backend"] == "official_cli"
    assert result["command"][:3] == [sys.executable, "-m", "voxtell.inference.predict_from_raw_data"]
    assert "--text-encoding-model" not in result["command"]


def test_voxtell_vendor_audit_script_reports_clean_vendor_metadata() -> None:
    from scripts.audit_voxtell_vendor import build_audit

    audit = build_audit()
    assert audit["official_repo"] == "https://github.com/MIC-DKFZ/VoxTell"
    assert audit["commit"]
    assert audit["dirty"] is False


def test_voxtell_mstep_mode_defaults_to_advisor_project_student_profile(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_em_training as em

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "num_items": 1,
        "num_cases": 1,
        "num_distillation_eligible_items": 1,
        "items": [{"training_weight": 1.0, "distillation_eligible": True}],
    }), encoding="utf-8")
    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(em, "load_student_target_organs", lambda: ["liver"])
    monkeypatch.setattr(em, "VOXTELL_TRAIN_CMD", "")
    monkeypatch.setattr(sys, "argv", ["run_em_training.py"])
    monkeypatch.delenv("MEDAI_VOXTELL_MSTEP_MODE", raising=False)
    monkeypatch.delenv("MEDAI_ENABLE_VOXTELL_TRAINING", raising=False)
    monkeypatch.delenv("MEDAI_EXPERIMENT_PROFILE", raising=False)

    result = em.run_prompt_student_mstep(1, manifest)
    assert result["status"] == "failed"
    assert result["canonical_training_backend"] == "project_voxtell_prompt_distillation_student"
    assert result["training_status"] == "failed_project_distillation_command_disabled"
    audit = json.loads((tmp_path / "outputs" / "round1" / "mstep" / "run_mode_audit.json").read_text(encoding="utf-8"))
    assert audit["experiment_profile"] == "advisor_aligned_default"
    assert audit["gpu_training_launched"] is False


def test_voxtell_manifest_only_requires_explicit_allowance(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_em_training as em

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "num_items": 1,
        "num_cases": 1,
        "num_distillation_eligible_items": 1,
        "items": [{"training_weight": 1.0, "distillation_eligible": True}],
    }), encoding="utf-8")
    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(em, "load_student_target_organs", lambda: ["liver"])
    monkeypatch.setattr(sys, "argv", ["run_em_training.py", "--voxtell-mstep-mode", "manifest_only"])
    monkeypatch.delenv("MEDAI_ALLOW_MANIFEST_ONLY", raising=False)

    blocked = em.run_prompt_student_mstep(1, manifest)
    assert blocked["training_status"] == "failed_manifest_only_not_allowed_for_formal_training"

    monkeypatch.setattr(sys, "argv", ["run_em_training.py", "--voxtell-mstep-mode", "manifest_only", "--allow-manifest-only"])
    allowed = em.run_prompt_student_mstep(2, manifest)
    assert allowed["status"] == "manifest_ready"
    assert allowed["training_status"] == "manifest_ready_manifest_only"


def test_legacy_official_voxtell_finetune_mode_is_rejected(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_em_training as em

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "num_items": 1,
        "num_cases": 1,
        "num_distillation_eligible_items": 1,
        "items": [{"training_weight": 1.0, "distillation_eligible": True}],
    }), encoding="utf-8")
    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(em, "load_student_target_organs", lambda: ["liver"])
    monkeypatch.setattr(sys, "argv", ["run_em_training.py", "--voxtell-mstep-mode", "official_voxtell_finetune"])

    result = em.run_prompt_student_mstep(1, manifest)
    assert result["status"] == "failed"
    assert "official_voxtell_nnunet_encoder_baseline" in result["reason"]


def test_manifest_to_nnunet_converter_includes_only_ab_hard_and_reports_overlap(tmp_path: Path) -> None:
    import numpy as np
    import nibabel as nib
    from cli_anything.medai.core.voxtell_nnunet_encoder import convert_manifest_to_nnunet_dataset

    target_config = tmp_path / "targets.json"
    target_config.write_text(json.dumps({"target_organs": ["liver", "spleen", "pancreas"]}), encoding="utf-8")
    image = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), str(image))

    def mask(path: Path, slices) -> str:
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[slices] = 1
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
        return str(path)

    liver = mask(tmp_path / "liver.nii.gz", np.s_[0:3, 0:3, 0:3])
    spleen = mask(tmp_path / "spleen.nii.gz", np.s_[1:4, 1:4, 1:4])
    pancreas = mask(tmp_path / "pancreas.nii.gz", np.s_[0:2, 0:2, 0:2])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [
        {"case_id": "case001", "organ": "liver", "image": str(image), "mask": liver, "grade": "A", "target_type": "hard", "training_weight": 1.0, "distillation_eligible": True, "scoring_schema_version": "autolabel_core_v2"},
        {"case_id": "case001", "organ": "spleen", "image": str(image), "mask": spleen, "grade": "B", "target_type": "hard", "training_weight": 0.5, "distillation_eligible": True, "scoring_schema_version": "autolabel_core_v2"},
        {"case_id": "case001", "organ": "pancreas", "image": str(image), "mask": pancreas, "grade": "C", "target_type": "soft", "training_weight": 0.1, "distillation_eligible": True, "scoring_schema_version": "autolabel_core_v2"},
    ]}), encoding="utf-8")

    result = convert_manifest_to_nnunet_dataset(manifest, tmp_path / "nnunet_out", 997, "Tiny", target_config)
    ds = Path(result["dataset_folder"])
    assert result["status"] == "success"
    dataset_json = json.loads((ds / "dataset.json").read_text(encoding="utf-8"))
    assert dataset_json["labels"] == {"background": 0, "liver": 1, "spleen": 2}
    label_mapping = json.loads((ds / "label_mapping.json").read_text(encoding="utf-8"))
    assert label_mapping["nnunet_label_ids"] == {"liver": 1, "spleen": 2}
    assert label_mapping["project_frozen_target_ids"] == {"liver": 1, "spleen": 2}
    assert label_mapping["labels"][0] == {"organ": "liver", "nnunet_label_id": 1, "project_frozen_target_id": 1, "target_config_order": 0}
    exclusions = json.loads((tmp_path / "nnunet_out" / "training_exclusions.json").read_text(encoding="utf-8"))
    assert {row["organ"] for row in exclusions} == {"pancreas"}
    conflicts = (tmp_path / "nnunet_out" / "overlap_conflicts.csv").read_text(encoding="utf-8")
    assert "new_nnunet_label_id" in conflicts and "new_project_frozen_target_id" in conflicts
    assert "liver" in conflicts and "spleen" in conflicts


def test_official_encoder_mode_preflight_fails_without_fallback(tmp_path: Path, monkeypatch) -> None:
    import numpy as np
    import nibabel as nib
    import scripts.run_em_training as em

    image = tmp_path / "ct.nii.gz"
    mask = tmp_path / "liver.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), str(image))
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)), str(mask))
    target_config = tmp_path / "targets.json"
    target_config.write_text(json.dumps({"target_organs": ["liver"]}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "num_items": 1,
        "num_cases": 1,
        "num_distillation_eligible_items": 1,
        "items": [{"case_id": "case001", "organ": "liver", "image": str(image), "mask": str(mask), "grade": "A", "target_type": "hard", "training_weight": 1.0, "distillation_eligible": True, "scoring_schema_version": "autolabel_core_v2"}],
    }), encoding="utf-8")
    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(em, "PROMPT_TARGET_CONFIG", target_config)
    monkeypatch.setattr(em, "VOXTELL_MODEL_DIR", tmp_path / "missing_voxtell_model")
    monkeypatch.setattr(em, "load_student_target_organs", lambda: ["liver"])
    monkeypatch.setattr(em, "VOXTELL_TRAIN_CMD", "should-not-run")
    monkeypatch.setattr(sys, "argv", ["run_em_training.py", "--voxtell-mstep-mode", "official_voxtell_nnunet_encoder_finetune"])
    monkeypatch.delenv("MEDAI_VOXTELL_MSTEP_MODE", raising=False)

    result = em.run_prompt_student_mstep(1, manifest)
    assert result["status"] == "failed"
    assert result["training_status"] == "failed_official_voxtell_nnunet_baseline_requires_explicit_mode"
    assert result["canonical_training_backend"] == "official_voxtell_nnunet_encoder_baseline"
    assert result["is_official_voxtell_encoder_transfer"] is True
    assert result["is_prompt_conditioned_student"] is False
    assert result["eligible_for_next_round_prompt_student"] is False
    assert result.get("trainer_command") != "should-not-run"
    audit = json.loads((tmp_path / "outputs" / "round1" / "mstep" / "run_mode_audit.json").read_text(encoding="utf-8"))
    assert audit["gpu_training_launched"] is False
    assert audit["explicit_baseline_mode"] is False
