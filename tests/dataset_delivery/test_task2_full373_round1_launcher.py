from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.dataset_delivery import task2_full373_round1_launcher as full373
from tools.dataset_delivery import task2_round1_orchestrator as orch


def _case_manifest(path: Path, case_ids: list[str], *, with_source_mask: bool = False) -> Path:
    lines = ["case_id,ct_path,annotation_folder"]
    for case_id in case_ids:
        ann = path.parent / "ann" / case_id
        ann.mkdir(parents=True, exist_ok=True)
        if with_source_mask:
            (ann / "organ_a.nii.gz").write_bytes(b"source-mask")
        lines.append(f"{case_id},{path.parent / case_id / 'ct.nii.gz'},{ann}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _target_config(path: Path, targets: list[str]) -> Path:
    path.write_text(json.dumps({"target_organs": targets}), encoding="utf-8")
    return path


def _patch_routes(monkeypatch, routes: dict[str, list[str]]) -> None:
    registry = {
        "models": {
            "teacher1": {"enabled": True, "status": "ready", "evidence_family": "fam1", "checkpoint_path": "/ckpt/t1"},
            "teacher2": {"enabled": True, "status": "ready", "evidence_family": "fam2", "checkpoint_path": "/ckpt/t2"},
            "teacher3": {"enabled": True, "status": "ready", "evidence_family": "fam3", "checkpoint_path": "/ckpt/t3"},
        }
    }
    monkeypatch.setattr(full373, "load_registry", lambda path: registry)
    monkeypatch.setattr(full373, "candidate_models_for_organs", lambda registry, organs: routes)


def test_authoritative_373_routing_from_existing_configs_has_complete_target_space(tmp_path: Path):
    scope = full373.build_full_round1_scope(
        case_manifest=_case_manifest(tmp_path / "cases.csv", ["CASE001"]),
        registry_path=Path("configs/model_registry.yaml"),
        target_config=Path("configs/student_3d_prompt_target_organs.json"),
        output_root=tmp_path / "out",
        expected_case_count=1,
    )

    assert scope["status"] == "READY"
    assert scope["canonical_target_count"] == 373
    assert scope["unroutable_target_count"] == 0
    assert "candidate_models_for_organs" in scope["authoritative_routing_source"]
    assert scope["target_teacher_pairs"] > 373


def test_multi_model_target_expands_all_eligible_teachers_and_single_model_target(monkeypatch, tmp_path: Path):
    _patch_routes(monkeypatch, {"organ_a": ["teacher1", "teacher2", "teacher3"], "organ_b": ["teacher1"]})
    scope = full373.build_full_round1_scope(
        case_manifest=_case_manifest(tmp_path / "cases.csv", ["CASE001"]),
        registry_path=tmp_path / "registry.yaml",
        target_config=_target_config(tmp_path / "targets.json", ["organ_a", "organ_b"]),
        output_root=tmp_path / "out",
        expected_case_count=1,
    )

    assert scope["multi_teacher_target_count"] == 1
    assert scope["single_teacher_target_count"] == 1
    assert scope["target_teacher_pairs"] == 4
    assert {(row["target"], row["teacher"]) for row in scope["task_rows"]} == {
        ("organ_a", "teacher1"),
        ("organ_a", "teacher2"),
        ("organ_a", "teacher3"),
        ("organ_b", "teacher1"),
    }


def test_existing_source_mask_does_not_skip_teacher_candidate_planning(monkeypatch, tmp_path: Path):
    _patch_routes(monkeypatch, {"organ_a": ["teacher1", "teacher2"]})
    scope = full373.build_full_round1_scope(
        case_manifest=_case_manifest(tmp_path / "cases.csv", ["CASE001"], with_source_mask=True),
        registry_path=tmp_path / "registry.yaml",
        target_config=_target_config(tmp_path / "targets.json", ["organ_a"]),
        output_root=tmp_path / "out",
        expected_case_count=1,
    )

    assert scope["total_logical_candidate_tasks"] == 2
    assert [row["teacher"] for row in scope["task_rows"]] == ["teacher1", "teacher2"]


def test_candidate_artifacts_have_distinct_ids_per_teacher(monkeypatch, tmp_path: Path):
    _patch_routes(monkeypatch, {"organ_a": ["teacher1", "teacher2"]})
    scope = full373.build_full_round1_scope(
        case_manifest=_case_manifest(tmp_path / "cases.csv", ["CASE001"]),
        registry_path=tmp_path / "registry.yaml",
        target_config=_target_config(tmp_path / "targets.json", ["organ_a"]),
        output_root=tmp_path / "out",
        expected_case_count=1,
    )

    assert len({row["candidate_id"] for row in scope["task_rows"]}) == 2


def test_task2_outputs_can_be_reused_as_exact_candidate_cache(monkeypatch, tmp_path: Path):
    _patch_routes(monkeypatch, {"organ_a": ["teacher1", "teacher2"]})
    cache = tmp_path / "task2_cache"
    mask = cache / "masks" / "CASE001" / "organ_a.nii.gz"
    mask.parent.mkdir(parents=True, exist_ok=True)
    mask.write_bytes(b"mask")
    (cache / "task2_formal_case_target_status.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "case_id": "CASE001",
                        "target_name": "organ_a",
                        "selected_model": "teacher1",
                        "mask_path": str(mask),
                        "final_status": "generated_valid_mask",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scope = full373.build_full_round1_scope(
        case_manifest=_case_manifest(tmp_path / "cases.csv", ["CASE001"]),
        registry_path=tmp_path / "registry.yaml",
        target_config=_target_config(tmp_path / "targets.json", ["organ_a"]),
        output_root=tmp_path / "out",
        cache_roots=[cache],
        expected_case_count=1,
    )

    assert scope["cached_reusable_candidate_count"] == 1
    assert scope["new_inference_candidate_count"] == 1
    assert scope["cache_audit"]["reusable_candidates"][0]["cache_status"] == "REUSED_VALID_CANDIDATE"


def test_full373_execute_uses_run_loop_all_routed_teachers_and_labelcritic(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "tasks.csv"
    manifest.write_text(
        "task_index,case_id,ct_path,annotation_folder,registry_path,target_config,checkpoint_root,nnunet_predict_executable,unest_python_executable,python\n"
        f"0,CASE001,{tmp_path / 'ct.nii.gz'},{tmp_path / 'ann'},configs/model_registry.yaml,configs/student_3d_prompt_target_organs.json,/ckpt,nnUNetv2_predict,/venv/bin/python,{Path('/usr/bin/python')}\n",
        encoding="utf-8",
    )
    captured: list[str] = []

    def fake_run(command, **kwargs):
        captured.extend([str(part) for part in command])
        return subprocess.CompletedProcess(command, 0, '{"status":"success"}', "")

    monkeypatch.setattr(full373.subprocess, "run", fake_run)
    result = full373.execute_task_index(0, manifest, tmp_path / "out")

    assert result["status"] == "COMPLETED"
    assert "--models" in captured
    assert captured[captured.index("--models") + 1] == ""
    assert "--organs" in captured and captured[captured.index("--organs") + 1] == "student_373"
    assert "--enable-critic" in captured
    assert "--no-use-annotation-folder-reference" in captured


def test_full373_gate_requires_complete_case_target_scope(tmp_path: Path):
    out = tmp_path / "out"
    full373.write_json(
        out / "full_round1_scope.json",
        {
            "case_count": 1,
            "canonical_target_count": 2,
            "task_rows": [
                {"case_id": "CASE001", "target": "organ_a", "teacher": "teacher1"},
                {"case_id": "CASE001", "target": "organ_b", "teacher": "teacher1"},
            ],
        },
    )
    case = out / "run_loop_cases" / "CASE001"
    full373.write_json(case / "case_373_target_summary.json", {"complete_case_373": True})
    full373.write_json(case / "full_case_373_manifest.json", {"status": "success", "items": [{"case_id": "CASE001", "organ": "organ_a"}]})

    assert full373.aggregate_full373_estep(out)["status"] == "RUNNING"
    full373.write_json(
        case / "full_case_373_manifest.json",
        {"status": "success", "items": [{"case_id": "CASE001", "organ": "organ_a"}, {"case_id": "CASE001", "organ": "organ_b"}]},
    )
    full373.write_json(case / "training_manifest.json", {"items": [{"case_id": "CASE001", "organ": "organ_a", "mask_path": "/m", "training_weight": 1.0}]})
    assert full373.aggregate_full373_estep(out)["status"] == "PASSED"


def test_mstep_cannot_be_released_by_old_22_target_task2_status(tmp_path: Path):
    args = orch._args(tmp_path) if hasattr(orch, "_args") else None
    from types import SimpleNamespace

    if args is None:
        args = SimpleNamespace(
            state_root=tmp_path / "state",
            case_manifest=tmp_path / "cases.csv",
            base_manifest=tmp_path / "base.csv",
            target_config=Path("configs/student_3d_prompt_target_organs.json"),
        )
    formal_root = tmp_path / full373.FULL373_ROOT_NAME
    full373.write_json(formal_root / "task2_formal_case_target_status.json", {"rows": [{"case_id": "CASE001"}]})
    orch._save_state(args.state_root, formal_root=str(formal_root))

    result = orch.build_mstep_manifest(args)

    assert result["status"] == "FAILED"
    assert result["failure_reason"] == "full_373_training_manifest_missing"
