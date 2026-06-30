from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent-harness"))

def test_backend_capabilities_and_default_profile() -> None:
    from cli_anything.medai.core.backend_capabilities import (
        OFFICIAL_NNUNET_BASELINE,
        OFFICIAL_VOXTELL_PRETRAINED,
        PROJECT_PROMPT_STUDENT,
        backend_capability,
        profile_runtime_policy,
    )

    policy = profile_runtime_policy("advisor_aligned_default")
    assert policy["mstep_backend"] == PROJECT_PROMPT_STUDENT
    official = policy["official_voxtell_pretrained"]
    assert official["source_name"] == OFFICIAL_VOXTELL_PRETRAINED
    assert official["official_voxtell_mode"] == "baseline_only"
    assert official["eligible_as_teacher_candidate"] is True
    assert official["active_as_teacher_candidate"] is False
    assert official["allow_selection_by_autolabelcore"] is False
    assert official["used_for_selected_pseudo_label"] is False
    assert official["used_for_training_manifest"] is False

    enhanced = profile_runtime_policy("enhanced_candidate_pool")["official_voxtell_pretrained"]
    assert enhanced["official_voxtell_mode"] == "candidate"
    assert enhanced["active_as_teacher_candidate"] is True
    assert enhanced["allow_selection_by_autolabelcore"] is True

    project = backend_capability(PROJECT_PROMPT_STUDENT)
    assert project["main_mstep_allowed"] is True
    assert project["prompt_conditioned"] is True
    baseline = backend_capability(OFFICIAL_NNUNET_BASELINE)
    assert baseline["main_mstep_allowed"] is False
    assert baseline["eligible_for_next_round_prompt_student"] is False
    assert baseline["baseline_only"] is True


def test_enhanced_candidate_pool_injects_official_voxtell_and_nonempty_competition(monkeypatch) -> None:
    from cli_anything.medai.core.multimodel_loop import _build_case_execution_plan

    monkeypatch.setenv("MEDAI_EXPERIMENT_PROFILE", "enhanced_candidate_pool")
    registry = {
        "models": {
            "cads551": {"evidence_family": "cads"},
            "totalsegmentator": {"evidence_family": "totalsegmentator"},
        }
    }
    plan = _build_case_execution_plan(
        registry=registry,
        project_root=ROOT,
        organs=["liver"],
        requested_models=["cads551", "totalsegmentator"],
        preseeded_model_dirs=None,
        candidate_mode="route_pruned_with_competition",
    )
    route = plan["per_organ"]["liver"]
    assert route["backup_teachers"] or route["competition_teachers"]
    assert "official_voxtell_pretrained" in route["eligible_teachers"]


def test_official_voxtell_pretrained_adapter_baseline_only_dry_run(tmp_path: Path) -> None:
    from cli_anything.medai.core.voxtell_official_predictor import OfficialVoxTellPretrainedAdapter

    target_config = tmp_path / "targets.json"
    target_config.write_text(json.dumps({
        "target_organs": ["liver"],
        "organ_to_prompt": {"liver": "segment the liver"},
    }), encoding="utf-8")
    adapter = OfficialVoxTellPretrainedAdapter(
        model_dir=tmp_path / "model",
        target_config=target_config,
        text_encoding_model=tmp_path / "qwen",
        mode="baseline_only",
        device="cpu",
    )
    result = adapter.segment(tmp_path / "ct.nii.gz", tmp_path / "out", prompts=["liver"], dry_run=True)
    assert result["provider"] == "official_voxtell_pretrained"
    assert result["official_voxtell_mode"] == "baseline_only"
    assert result["allow_selection_by_autolabelcore"] is False
    assert result["active_as_teacher_candidate"] is False
    assert result["used_for_selected_pseudo_label"] is False
    assert result["used_for_training_manifest"] is False
    assert result["is_project_student"] is False


def test_selected_label_review_packets_mark_high_risk_and_block_manifest(tmp_path: Path) -> None:
    from scripts.build_selected_label_review_packets import build_review_packets

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [
        {
            "case_id": "case001",
            "organ": "liver",
            "prompt": "segment the liver",
            "mask_path": str(tmp_path / "liver.nii.gz"),
            "selected_provider": "student_prev",
            "evidence_confidence": 0.6,
            "winner_margin": 0.01,
            "candidate_count": 3,
            "labelcritic_summary": "warning: uncertain",
        }
    ]}), encoding="utf-8")
    summary = build_review_packets(manifest, tmp_path / "review")
    assert summary["num_review_items"] == 1
    queue = json.loads((tmp_path / "review" / "review_queue.json").read_text(encoding="utf-8"))["items"]
    row = queue[0]
    assert row["human_review_status"] == "pending"
    assert row["affects_training_manifest"] is True
    assert "student_selected_for_training" in row["risk_reasons"]
    assert "low_confidence_selected" in row["risk_reasons"]


def test_final_student_export_requires_quality_gated_prompt_student(tmp_path: Path) -> None:
    from scripts.export_final_student import export_final_student

    round_dir = tmp_path / "round1"
    mstep = round_dir / "mstep"
    model = mstep / "voxtell_finetuned_model" / "fold_0"
    model.mkdir(parents=True)
    (mstep / "voxtell_finetuned_model" / "plans.json").write_text("{}", encoding="utf-8")
    (model / "checkpoint_final.pth").write_bytes(b"fake")
    (mstep / "voxtell_prompt_student_manifest.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    (mstep / "voxtell_prompt_mstep_result.json").write_text(json.dumps({
        "canonical_training_backend": "project_voxtell_prompt_distillation_student",
        "eligible_for_next_round_prompt_student": True,
        "is_official_voxtell_encoder_transfer": False,
        "trainer": "project_voxtell_prompt_distillation_student",
        "uses_official_voxtell_model": True,
    }), encoding="utf-8")
    result = export_final_student(round_dir, tmp_path / "export")
    assert result["ready_for_standalone_inference"] is True
    assert (tmp_path / "export" / "model" / "fold_0" / "checkpoint_final.pth").exists()


def test_student_auto_segmentation_cli_dry_run_has_no_teacher_dependencies(tmp_path: Path) -> None:
    import subprocess
    import sys

    target_config = tmp_path / "targets.json"
    target_config.write_text(json.dumps({
        "target_organs": ["liver"],
        "organ_to_prompt": {"liver": "segment the liver"},
    }), encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/student_auto_segmentation_cli.py",
        "--model-dir", str(tmp_path / "model"),
        "--image", str(tmp_path / "ct.nii.gz"),
        "--output-dir", str(tmp_path / "pred"),
        "--target-config", str(target_config),
        "--organs", "liver",
        "--device", "cpu",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    assert proc.returncode == 0
    audit = json.loads((tmp_path / "pred" / "student_auto_segmentation_audit.json").read_text(encoding="utf-8"))
    assert audit["uses_teacher"] is False
    assert audit["uses_autolabelcore"] is False
    assert audit["uses_estep"] is False
    assert audit["uses_labelcritic"] is False

def test_official_voxtell_pretrained_disabled_does_not_run_predictor(tmp_path: Path) -> None:
    from cli_anything.medai.core.voxtell_official_predictor import OfficialVoxTellPretrainedAdapter

    target_config = tmp_path / "targets.json"
    target_config.write_text(json.dumps({"target_organs": ["liver"]}), encoding="utf-8")
    adapter = OfficialVoxTellPretrainedAdapter(
        model_dir=tmp_path / "model",
        target_config=target_config,
        text_encoding_model=tmp_path / "qwen",
        mode="disabled",
        device="cpu",
    )
    result = adapter.segment(tmp_path / "ct.nii.gz", tmp_path / "out", prompts=["liver"], dry_run=True)
    assert result["status"] == "skipped"
    assert result["uses_official_voxtell_predictor"] is False
    assert result["allow_selection_by_autolabelcore"] is False
    assert result["used_for_training_manifest"] is False
    assert result["eligible_for_next_round_prompt_student"] is False


def test_run_registered_model_special_cases_official_voxtell_pretrained(tmp_path: Path, monkeypatch) -> None:
    from cli_anything.medai.core import registered_infer as ri

    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text("models: {}\n", encoding="utf-8")
    image = tmp_path / "ct.nii.gz"
    image.write_bytes(b"fake")

    captured: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, model_dir, target_config, mode="baseline_only", device="cuda", backend="official_python_api", text_encoding_model=None):
            captured["init"] = {
                "model_dir": Path(model_dir),
                "target_config": Path(target_config),
                "mode": mode,
                "device": device,
                "backend": backend,
                "text_encoding_model": text_encoding_model,
            }

        def segment(self, ct_image, output_dir, prompts=None, dry_run=False, timeout_sec=1800, prompt_batch_size=16, prompt_overrides=None):
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / "liver.nii.gz").write_bytes(b"mask")
            captured["segment"] = {
                "ct_image": Path(ct_image),
                "output_dir": output,
                "prompts": prompts,
                "dry_run": dry_run,
                "timeout_sec": timeout_sec,
                "prompt_batch_size": prompt_batch_size,
                "prompt_overrides": prompt_overrides,
            }
            return {
                "status": "success",
                "provider": "official_voxtell_pretrained",
                "official_voxtell_mode": "candidate",
                "allow_selection_by_autolabelcore": True,
                "active_as_teacher_candidate": True,
                "used_for_selected_pseudo_label": None,
                "used_for_training_manifest": None,
            }

    monkeypatch.setattr(ri, "OfficialVoxTellPretrainedAdapter", FakeAdapter)
    monkeypatch.setenv("MEDAI_EXPERIMENT_PROFILE", "enhanced_candidate_pool")
    monkeypatch.setenv("MEDAI_VOXTELL_MODEL_DIR", str(tmp_path / "voxtell_model"))
    target_config = tmp_path / "targets.json"
    target_config.write_text(json.dumps({"target_organs": ["liver"]}), encoding="utf-8")
    monkeypatch.setenv("MEDAI_PROMPT_TARGET_CONFIG", str(target_config))
    monkeypatch.setenv("MEDAI_TEXT_ENCODING_MODEL", str(tmp_path / "qwen"))

    result = ri.run_registered_model(
        image_path=image,
        output_folder=tmp_path / "outputs",
        model_key="official_voxtell_pretrained",
        registry_path=registry_path,
        case_id="case001",
        dry_run=False,
        device="cpu",
        extra_context={
            "requested_organs": ["arms", "liver"],
            "prompt_overrides": {"liver": "segment the liver carefully"},
        },
    )

    assert result["status"] == "success"
    assert result["supported_organs"] == ["liver"]
    assert result["skipped_unsupported_organs"] == ["arms"]
    assert captured["segment"]["prompts"] == ["liver"]
    assert result["backend"] == "official_voxtell_pretrained"
    assert result["requested_organs"] == ["arms", "liver"]
    assert captured["init"]["mode"] == "candidate"
    assert captured["init"]["device"] == "cpu"
    assert captured["segment"]["prompts"] == ["liver"]
    assert captured["segment"]["prompt_batch_size"] == 1
    assert captured["segment"]["prompt_overrides"] == {"liver": "segment the liver carefully"}
    assert (tmp_path / "outputs" / "case001" / "inference_summary.json").exists()
    assert (tmp_path / "outputs" / "case001" / "per_model" / "official_voxtell_pretrained" / "run_meta.json").exists()


def test_project_student_preflight_requires_reload_prompt_smoke_and_quality_gate(tmp_path: Path) -> None:
    from cli_anything.medai.core.voxtell_project_student_predictor import project_student_predictor_preflight

    model = tmp_path / "student"
    (model / "fold_0").mkdir(parents=True)
    (model / "plans.json").write_text("{}", encoding="utf-8")
    (model / "fold_0" / "checkpoint_final.pth").write_bytes(b"fake")
    result = project_student_predictor_preflight(model)
    assert result["prompt_conditioned"] is True
    assert result["eligible_for_next_round_prompt_student"] is False
    assert result["status"] == "requires_smoke_and_quality_gate"

    (model / "project_student_metadata.json").write_text(json.dumps({
        "prompt_conditioned": True,
        "checkpoint_reload_success": True,
        "prompt_inference_smoke_success": True,
        "quality_gate_success": True,
    }), encoding="utf-8")
    passed = project_student_predictor_preflight(model)
    assert passed["eligible_for_next_round_prompt_student"] is True
    assert passed["status"] == "passed"


def test_project_student_preflight_rejects_non_prompt_metadata(tmp_path: Path) -> None:
    from cli_anything.medai.core.voxtell_project_student_predictor import project_student_predictor_preflight

    model = tmp_path / "student"
    (model / "fold_0").mkdir(parents=True)
    (model / "plans.json").write_text("{}", encoding="utf-8")
    (model / "fold_0" / "checkpoint_final.pth").write_bytes(b"fake")
    (model / "project_student_metadata.json").write_text(json.dumps({"prompt_conditioned": False}), encoding="utf-8")
    result = project_student_predictor_preflight(model)
    assert result["status"] == "failed"
    assert result["eligible_for_next_round_prompt_student"] is False
    assert result["failure_reason"] == "metadata_prompt_conditioned_false"
