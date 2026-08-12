from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_HARNESS = REPO_ROOT / "agent-harness"
if str(AGENT_HARNESS) not in sys.path:
    sys.path.insert(0, str(AGENT_HARNESS))


def _save(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_airrc_valid_raw_masks_recover_to_candidate_selection_and_delivery(tmp_path: Path):
    from tools.dataset_delivery.task2_formal_validator import validate_case_group_run
    from tools.dataset_delivery.task2_recovery import recover_case_group

    root = tmp_path / "formal"
    case_id = "BDMAP_00000002"
    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), tmp_path / case_id / "ct.nii.gz")
    raw = np.zeros((3, 3, 3), dtype=np.uint8)
    raw[1, 1, 1] = 1
    raw_path = root / "cases" / case_id / "airrc" / "run_loop" / "raw_predictions" / "airrc" / case_id / "segmentations" / "lung_pulmonary_arteries.nii.gz"
    _save(raw, raw_path)

    report = recover_case_group(
        output_root=root,
        case_id=case_id,
        group="airrc",
        targets=["lung_pulmonary_arteries"],
        ct_path=ct,
        models=["airrc"],
        apply=True,
    )

    assert report["recovered_target_count"] == 1
    target = report["target_rows"][0]
    assert target["candidate_count"] >= 1
    assert target["teacher_names"] == ["airrc"]
    assert target["selected_candidate"]
    assert "missing_candidate" not in json.dumps(report)
    row = validate_case_group_run(output_root=root, case_id=case_id, group="airrc", target="lung_pulmonary_arteries", ct_path=ct)
    assert row["final_status"] == "generated_valid_mask"
    assert row["terminal_state"] == "SUCCESS"
    assert Path(row["mask_path"]).exists()
    candidate_doc = json.loads((root / "cases" / case_id / "airrc" / "run_loop" / "candidate_recovery.json").read_text(encoding="utf-8"))
    candidate = candidate_doc["candidate_rows"][0]
    assert candidate["teacher"] == "airrc"
    assert candidate["valid"] is True
    assert Path(candidate["candidate_path"]).exists()


def test_cads_recovery_idempotent_and_empty_mask_is_not_fake_training_label(tmp_path: Path):
    from tools.dataset_delivery.task2_recovery import recover_case_group

    root = tmp_path / "formal"
    case_id = "BDMAP_00010000"
    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), tmp_path / case_id / "ct.nii.gz")
    raw = np.zeros((3, 3, 3), dtype=np.uint8)
    raw[1, 1, 1] = 1
    raw_path = root / "cases" / case_id / "cads" / "run_loop" / "raw_predictions" / "cads557" / case_id / "segmentations" / "blood.nii.gz"
    _save(raw, raw_path)
    zero_path = root / "cases" / case_id / "cads" / "run_loop" / "raw_predictions" / "cads557" / case_id / "segmentations" / "face.nii.gz"
    _save(np.zeros((3, 3, 3), dtype=np.uint8), zero_path)

    first = recover_case_group(
        output_root=root,
        case_id=case_id,
        group="cads",
        targets=["blood", "face"],
        ct_path=ct,
        models=["cads557"],
        apply=True,
    )
    second = recover_case_group(
        output_root=root,
        case_id=case_id,
        group="cads",
        targets=["blood", "face"],
        ct_path=ct,
        models=["cads557"],
        apply=True,
    )

    assert first["target_rows"][0]["status"] == "SUCCESS"
    assert first["target_rows"][1]["status"] == "COMPLETED_NO_NONZERO"
    assert second["target_rows"][0]["status"] == "SUCCESS"
    assert (root / "cases" / case_id / "cads" / "run_loop" / "annotation_versions" / case_id / "updated" / "blood.nii.gz").exists()
    assert not (root / "cases" / case_id / "cads" / "run_loop" / "annotation_versions" / case_id / "updated" / "face.nii.gz").exists()
    recovery_csv = _rows(root / "cases" / case_id / "cads" / "run_loop" / "candidate_recovery.csv")
    assert [row for row in recovery_csv if row["target"] == "face"][0]["valid"] == "False"


def test_task1_formal_mapping_decisions_match_round1_contract():
    mapping = _rows(REPO_ROOT / "configs" / "dataset_delivery" / "task1" / "organ_rename_mapping_373.csv")
    by_pair = {(row["source_name"], row["target_name"]): row for row in mapping}

    assert ("celiac_aa", "celiac_aa") not in by_pair
    assert by_pair[("celiac_artery", "celiac_aa")]["status"] == "confirmed"
    assert by_pair[("coronary_arteries", "coronary_artery")]["status"] == "pending_review"
    assert by_pair[("parotid_gland", "parotid_glands")]["status"] == "pending_review"
    assert by_pair[("submandibular_gland", "submandibular_glands")]["status"] == "pending_review"
    for source, target in {
        "left_lung_lower_lobe": "lung_lower_left_lobe",
        "lung_lobe_lower_left": "lung_lower_left_lobe",
        "lung_lower_lobe_left": "lung_lower_left_lobe",
        "right_lung_lower_lobe": "lung_lower_right_lobe",
        "lung_lobe_lower_right": "lung_lower_right_lobe",
        "lung_lower_lobe_right": "lung_lower_right_lobe",
        "right_lung_middle_lobe": "lung_middle_right_lobe",
        "lung_lobe_middle_right": "lung_middle_right_lobe",
        "lung_middle_lobe_right": "lung_middle_right_lobe",
        "left_lung_upper_lobe": "lung_upper_left_lobe",
        "lung_lobe_upper_left": "lung_upper_left_lobe",
        "lung_upper_lobe_left": "lung_upper_left_lobe",
        "right_lung_upper_lobe": "lung_upper_right_lobe",
        "lung_lobe_upper_right": "lung_upper_right_lobe",
        "lung_upper_lobe_right": "lung_upper_right_lobe",
    }.items():
        assert by_pair[(source, target)]["status"] == "confirmed"


def test_task1_alias_conflict_report_keeps_provenance(tmp_path: Path):
    from tools.dataset_delivery.delivery_lib import apply_rename

    names = ["target_a", *(f"organ_{idx:03d}" for idx in range(372))]
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(json.dumps({"target_organs": list(names)}), encoding="utf-8")
    seg = tmp_path / "data" / "case" / "segmentations"
    seg.mkdir(parents=True)
    (seg / "source_a.nii.gz").write_text("a", encoding="utf-8")
    (seg / "source_b.nii.gz").write_text("b", encoding="utf-8")
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "source_name,target_name,status,reason,notes\n"
        "source_a,target_a,confirmed,test,\n"
        "source_b,target_a,confirmed,test,\n",
        encoding="utf-8",
    )
    aliases = tmp_path / "aliases.csv"
    aliases.write_text(
        "alias_group,target_name,source_name,status,evidence\n"
        "alias_target_a,target_a,source_a,confirmed,test\n"
        "alias_target_a,target_a,source_b,confirmed,test\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.csv"

    apply_rename(tmp_path / "data", mapping, taxonomy, report, apply=True, alias_groups=aliases, output_data_root=tmp_path / "out")

    rows = _rows(report)
    assert {row["status"] for row in rows} == {"conflict"}
    assert all(row["source_sha256"] for row in rows)
    assert all(row["alias_conflict_sources"] == "source_a;source_b" for row in rows)
    assert all(row["provenance_policy"] == "preserve_original_masks_no_silent_overwrite" for row in rows)


def test_personal_workspace_staging_writes_manifest_to_writable_copy(tmp_path: Path):
    from tools.dataset_delivery.task2_workspace_staging import stage_workspace

    image_root = tmp_path / "public_images"
    mask_root = tmp_path / "public_masks"
    manifest = tmp_path / "cases.csv"
    rows = []
    for idx in range(2):
        case_id = f"BDMAP_{idx:08d}"
        ct = _save(np.zeros((2, 2, 2), dtype=np.int16), image_root / case_id / "ct.nii.gz")
        seg = mask_root / case_id / "segmentations"
        _save(np.ones((2, 2, 2), dtype=np.uint8), seg / "liver.nii.gz")
        rows.append({"index": idx, "case_id": case_id, "ct_path": str(ct), "annotation_folder": str(seg)})
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = stage_workspace(
        cases_manifest=manifest,
        workspace_root=tmp_path / "workspace",
        image_root=image_root,
        mask_root=mask_root,
        expected_case_count=2,
        resume=True,
    )

    assert report["status"] == "READY"
    staged_manifest = Path(report["output_manifest"])
    assert staged_manifest.exists()
    staged = _rows(staged_manifest)
    assert all("/workspace/inputs/" in row["ct_path"] for row in staged)
    assert all("/workspace/inputs/" in row["annotation_folder"] for row in staged)
    assert (image_root / "BDMAP_00000000" / "ct.nii.gz").exists()


def test_formal_manifest_can_switch_to_task1_renamed_masks(tmp_path: Path):
    from tools.dataset_delivery.task2_manifest_masks import rewrite_manifest_mask_root

    manifest = tmp_path / "cases.csv"
    renamed_root = tmp_path / "workspace" / "task1" / "masks_renamed"
    rows = []
    for idx in range(2):
        case_id = f"BDMAP_{idx:08d}"
        _save(np.ones((2, 2, 2), dtype=np.uint8), renamed_root / case_id / "segmentations" / "liver.nii.gz")
        rows.append({
            "index": idx,
            "case_id": case_id,
            "ct_path": str(tmp_path / case_id / "ct.nii.gz"),
            "annotation_folder": str(tmp_path / "original" / case_id / "segmentations"),
        })
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = rewrite_manifest_mask_root(
        input_manifest=manifest,
        output_manifest=tmp_path / "workspace" / "manifests" / "cases_103_manifest_renamed.csv",
        mask_root=renamed_root,
        expected_case_count=2,
    )

    assert report["status"] == "READY"
    updated = _rows(Path(report["output_manifest"]))
    assert all("/task1/masks_renamed/" in row["annotation_folder"] for row in updated)
    assert all(row["mask_root_policy"] == "task1_renamed_personal_workspace" for row in updated)


def test_multi_teacher_plan_preserves_all_eligible_candidates():
    from cli_anything.medai.core.model_registry import load_registry
    from cli_anything.medai.core.multimodel_loop import _build_case_execution_plan

    registry = load_registry(REPO_ROOT / "configs" / "model_registry.yaml")
    plan = _build_case_execution_plan(
        registry=registry,
        project_root=REPO_ROOT,
        organs=["lung_lower_left_lobe"],
        requested_models=["vista3d", "cads551"],
        preseeded_model_dirs=None,
        candidate_mode="route_pruned_with_competition",
    )
    eligible = plan["per_organ"]["lung_lower_left_lobe"]["eligible_teachers"]
    assert set(eligible) >= {"vista3d", "cads551"}
    assert len(eligible) >= 2


def test_formal_launcher_command_keeps_shapekit_and_labelcritic_enabled(tmp_path: Path):
    from tools.dataset_delivery.task2_formal_launcher import _command_for_group

    command = _command_for_group(
        group="airrc",
        python=Path("/usr/bin/python"),
        case_csv=tmp_path / "case.csv",
        registry=REPO_ROOT / "configs" / "model_registry.yaml",
        run_out=tmp_path / "run_out",
        timeout_sec=60,
        checkpoint_root=tmp_path / "checkpoints",
        nnunet_predict_executable=tmp_path / "nnUNetv2_predict",
        unest_python_executable=Path("/usr/bin/python"),
        enable_shapekit=True,
        enable_critic=True,
        critic_vlm_model="Qwen/Qwen2-VL-72B-Instruct-AWQ",
    )

    assert "--no-enable-shapekit" not in command
    assert "--debug-allow-no-shapekit" not in command
    assert "--no-enable-critic" not in command
    assert command[command.index("--critic-vlm-model") + 1] == "Qwen/Qwen2-VL-72B-Instruct-AWQ"
    assert "--no-use-annotation-folder-reference" in command
    assert "--strict-delivery-targets" in command


def test_formal_72b_labelcritic_selects_without_legacy_benchmark_artifact(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as mm

    ct = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "ct.nii.gz")
    mask_a = _save(np.eye(4, dtype=np.uint8).reshape(4, 4, 1).repeat(4, axis=2), tmp_path / "a.nii.gz")
    mask_b_data = np.zeros((4, 4, 4), dtype=np.uint8)
    mask_b_data[2:, 2:, 2:] = 1
    mask_b = _save(mask_b_data, tmp_path / "b.nii.gz")

    def fake_compare(*args, **kwargs):
        output_json = Path(args[4])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        result = {"status": "success", "decision": {"winner": "b", "reason": "better boundary"}}
        output_json.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(mm, "run_labelcritic_compare", fake_compare)
    selected, selection = mm._select_candidate(
        ct=ct,
        organ="liver",
        candidates=[
            {"case_id": "case01", "organ": "liver", "model": "teacher_a", "prediction": str(mask_a), "candidate_id": "a", "eligible_for_labelcritic": True, "candidate_qc_score": 1.0},
            {"case_id": "case01", "organ": "liver", "model": "teacher_b", "prediction": str(mask_b), "candidate_id": "b", "eligible_for_labelcritic": True, "candidate_qc_score": 1.0},
        ],
        out=tmp_path / "out",
        case_id="case01",
        enable_critic=True,
        critic_backend="labelcritic",
        critic_base_url="http://localhost",
        critic_port=8000,
        timeout_sec=60,
        dry_run=False,
        use_official_pairwise=True,
        formal_72b_selection_ready=True,
    )

    assert selected["model"] == "teacher_b"
    assert selection["selection_method"] == "label_critic"
    assert selection["selection_status"] == "selected"
    assert selection["formal_winner"] == "teacher_b"
    assert selection["audit_winner"] is None


def test_labelcritic_endpoint_resolves_from_service_root_at_selection_time(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as mm

    service_root = tmp_path / "labelcritic_72b_service"
    service_root.mkdir()
    (service_root / "base_url.txt").write_text("http://h100-node\n", encoding="utf-8")
    (service_root / "port.txt").write_text("8123\n", encoding="utf-8")
    (service_root / "endpoint.host").write_text("h100-node\n", encoding="utf-8")
    monkeypatch.setenv("LABELCRITIC_SERVICE_ROOT", str(service_root))
    monkeypatch.setenv("MEDAI_LABELCRITIC_ENDPOINT_WAIT_SEC", "0")

    base_url, port, meta = mm._resolve_runtime_labelcritic_endpoint("http://localhost", 8000)

    assert base_url == "http://h100-node"
    assert port == 8123
    assert meta["status"] == "RESOLVED_FROM_SERVICE_ROOT"
    assert "h100-node" in os.environ["NO_PROXY"]


def test_em_gate_accepts_labelcritic_selected_teacher_against_student_candidate():
    from cli_anything.medai.core import multimodel_loop as mm

    selected = {
        "model": "teacher_a",
        "prediction": "/tmp/teacher_a.nii.gz",
        "candidate_qc_status": "pass",
        "eligible_for_labelcritic": True,
    }
    gated_selected, gated = mm._apply_em_student_vs_previous_gate(
        selected=selected,
        selection={
            "selection_method": "label_critic",
            "selection_status": "selected",
            "selected_model": "teacher_a",
            "labelcritic_decisive": True,
            "quality_flags": [],
            "review_flags": [],
        },
        candidates=[
            {"model": "round_prev_selected", "candidate_exists": True, "prediction": "/tmp/prev.nii.gz"},
            {"model": "student_prev", "candidate_exists": True, "candidate_qc_status": "pass"},
            selected,
        ],
    )

    assert gated_selected is selected
    assert gated["source_model"] == "teacher_a"
    assert gated["em_student_vs_previous_gate"]["status"] == "teacher_or_model_replacement_accepted"
    assert gated["should_enter_student_training"] is True


def test_hierarchy_validation_cycle_and_multiple_roots(tmp_path: Path):
    from tools.dataset_delivery.hierarchy_config import validate_hierarchy_config

    cfg = tmp_path / "hierarchy.json"
    cfg.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "parent_a", "root": True, "expression": {"op": "OR", "args": ["child_a", "child_b"]}},
                    {"id": "parent_b", "root": True, "expression": {"op": "AND", "args": ["child_b"]}},
                    {"id": "child_a", "expression": {"op": "NOT", "args": "parent_a"}},
                    {"id": "child_b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    report = validate_hierarchy_config(cfg, known_nodes={"parent_a", "parent_b", "child_a", "child_b"})
    assert report["root_count"] == 2
    assert report["status"] == "BLOCKED"
    assert any(error["type"] == "cycle_detected" for error in report["errors"])
