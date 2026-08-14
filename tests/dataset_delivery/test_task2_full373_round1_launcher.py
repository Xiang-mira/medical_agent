from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import nibabel as nib
import numpy as np

from tools.dataset_delivery import task2_full373_round1_launcher as full373
from tools.dataset_delivery import task2_round1_orchestrator as orch


def _save_mask(path: Path, *, value: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    if value:
        data[0, 0, 0] = 1
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _case_manifest(path: Path, case_ids: list[str], *, with_source_mask: bool = False) -> Path:
    lines = ["case_id,ct_path,annotation_folder"]
    for case_id in case_ids:
        ann = path.parent / "ann" / case_id
        ann.mkdir(parents=True, exist_ok=True)
        if with_source_mask:
            _save_mask(ann / "organ_a.nii.gz")
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
    _save_mask(mask)
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


def test_full373_execute_runs_one_teacher_target_candidate_without_labelcritic(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "tasks.csv"
    manifest.write_text(
        "task_index,case_id,target,teacher,candidate_id,ct_path,annotation_folder,registry_path,target_config,checkpoint_root,nnunet_predict_executable,unest_python_executable,python\n"
        f"0,CASE001,organ_a,teacher1,cand1,{tmp_path / 'ct.nii.gz'},{tmp_path / 'ann'},configs/model_registry.yaml,configs/student_3d_prompt_target_organs.json,/ckpt,nnUNetv2_predict,/venv/bin/python,{Path('/usr/bin/python')}\n",
        encoding="utf-8",
    )
    captured: list[str] = []

    def fake_run(command, **kwargs):
        captured.extend([str(part) for part in command])
        case_output = tmp_path / "out" / "candidate_runs" / "CASE001" / "organ_a" / "teacher1"
        mask = case_output / "annotation_versions" / "CASE001" / "updated" / "organ_a.nii.gz"
        _save_mask(mask)
        full373.write_json(
            case_output / "annotation_versions" / "CASE001" / "selection_metadata.json",
            {
                "ct_path": str(tmp_path / "ct.nii.gz"),
                "selection_rows": [{"organ": "organ_a", "candidate_predictions": [{"model": "teacher1", "prediction": str(mask), "candidate_exists": True, "candidate_qc_status": "pass"}]}],
                "selected_organs": [{"organ": "organ_a", "final_mask": str(mask), "target_type": "positive_hard"}],
            },
        )
        return subprocess.CompletedProcess(command, 0, '{"status":"success"}', "")

    monkeypatch.setattr(full373.subprocess, "run", fake_run)
    result = full373.execute_task_index(0, manifest, tmp_path / "out")

    assert result["status"] == "COMPLETED"
    assert "--models" in captured
    assert captured[captured.index("--models") + 1] == "teacher1"
    assert "--organs" in captured and captured[captured.index("--organs") + 1] == "organ_a"
    assert "--no-enable-critic" in captured
    assert "--enable-critic" not in captured
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


def _write_scope(root: Path, *, cases: list[str], routes: dict[str, list[str]]) -> None:
    rows = []
    for case_id in cases:
        for target, teachers in routes.items():
            for teacher in teachers:
                rows.append({"case_id": case_id, "target": target, "teacher": teacher, "candidate_id": f"{case_id}_{target}_{teacher}"})
    full373.write_json(
        root / "full_round1_scope.json",
        {
            "case_count": len(cases),
            "canonical_target_count": len(routes),
            "routes": routes,
            "task_rows": rows,
            "total_logical_candidate_tasks": len(rows),
        },
    )


def _write_candidate_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_index", "case_id", "target", "teacher", "candidate_id", "ct_path", "annotation_folder"]
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(field, "")) for field in fieldnames) + "\n")
    return path


def _candidate(root: Path, case_id: str, target: str, teacher: str, status: str = "SUCCESS") -> dict:
    mask = root / "masks" / case_id / target / f"{teacher}.nii.gz"
    if status == "SUCCESS":
        _save_mask(mask)
    state = {
        "status": status,
        "case_id": case_id,
        "target": target,
        "teacher": teacher,
        "model": teacher,
        "candidate_id": f"{case_id}_{target}_{teacher}",
        "candidate_exists": status == "SUCCESS",
        "prediction": str(mask) if status == "SUCCESS" else "",
        "eligible_for_labelcritic": status == "SUCCESS",
        "candidate_qc_status": "pass",
    }
    full373.publish_candidate_state(root, state)
    return state


def test_case_target_waits_until_all_eligible_teacher_candidates_terminal(tmp_path: Path):
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1", "teacher2", "teacher3"]})
    _candidate(tmp_path, "CASE001", "organ_a", "teacher1")
    _candidate(tmp_path, "CASE001", "organ_a", "teacher2")
    state = full373.recompute_case_target_readiness(tmp_path, case_id="CASE001", target="organ_a")
    assert state["status"] == "WAITING_FOR_CANDIDATES"

    _candidate(tmp_path, "CASE001", "organ_a", "teacher3")
    claim = full373.claim_next_case_target(tmp_path, worker_id="critic1", labelcritic_ready=True)
    assert claim["status"] == "CLAIMED"
    assert claim["case_target"]["case_id"] == "CASE001"
    assert claim["case_target"]["target"] == "organ_a"


def test_case_target_ready_does_not_wait_for_global_teacher_completion(tmp_path: Path):
    _write_scope(tmp_path, cases=["CASE001", "CASE002"], routes={"organ_a": ["teacher1", "teacher2"]})
    _candidate(tmp_path, "CASE001", "organ_a", "teacher1")
    _candidate(tmp_path, "CASE001", "organ_a", "teacher2")
    _candidate(tmp_path, "CASE002", "organ_a", "teacher1")

    claim = full373.claim_next_case_target(tmp_path, worker_id="critic1", labelcritic_ready=True)

    assert claim["status"] == "CLAIMED"
    assert claim["case_target"]["case_id"] == "CASE001"


def test_labelcritic_pending_accumulates_ready_backlog_without_failure(tmp_path: Path):
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1", "teacher2"]})
    _candidate(tmp_path, "CASE001", "organ_a", "teacher1")
    _candidate(tmp_path, "CASE001", "organ_a", "teacher2")

    claim = full373.claim_next_case_target(tmp_path, worker_id="critic1", labelcritic_ready=False)

    assert claim["status"] == "WAITING_FOR_LABELCRITIC"
    assert claim["queue_depth"] == 1
    telemetry = full373.build_estep_telemetry(tmp_path)
    assert telemetry["labelcritic"]["queue_depth"] == 1


def test_labelcritic_ready_consumes_backlog_and_passes_all_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1", "teacher2", "teacher3"]})
    for teacher in ["teacher1", "teacher2", "teacher3"]:
        _candidate(tmp_path, "CASE001", "organ_a", teacher)
    captured: dict[str, int] = {}

    def fake_select_candidate(**kwargs):
        captured["candidate_count"] = len(kwargs["candidates"])
        return kwargs["candidates"][1], {
            "selection_method": "label_critic",
            "selection_status": "selected",
            "selected_model": kwargs["candidates"][1]["model"],
            "labelcritic_records": [{"status": "success"}],
        }

    monkeypatch.setattr(full373, "_select_candidate", fake_select_candidate)
    claim = full373.claim_next_case_target(tmp_path, worker_id="critic1", labelcritic_ready=True)
    result = full373.select_case_target(
        tmp_path,
        case_id=claim["case_target"]["case_id"],
        target=claim["case_target"]["target"],
        critic_base_url="http://node",
        critic_port=8000,
    )

    assert captured["candidate_count"] == 3
    assert result["status"] == "SELECTED"
    assert result["selected_model"] == "teacher2"


def test_single_teacher_target_uses_existing_single_candidate_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_b": ["teacher1"]})
    _candidate(tmp_path, "CASE001", "organ_b", "teacher1")

    def fake_select_candidate(**kwargs):
        return kwargs["candidates"][0], {
            "selection_method": "single_teacher_provisional",
            "selection_status": "provisional",
            "selected_model": "teacher1",
        }

    monkeypatch.setattr(full373, "_select_candidate", fake_select_candidate)
    result = full373.select_case_target(tmp_path, case_id="CASE001", target="organ_b", critic_base_url="http://node", critic_port=8000)
    assert result["status"] == "VALID_SINGLE_TEACHER_ACCEPTED"


def test_same_logical_candidate_cannot_be_claimed_twice(tmp_path: Path):
    first = full373.claim_work(tmp_path, claim_kind="candidate", claim_key="CASE001|organ|teacher", worker_id="gpu_t4")
    second = full373.claim_work(tmp_path, claim_kind="candidate", claim_key="CASE001|organ|teacher", worker_id="gpu_a100")
    assert first["status"] == "CLAIMED"
    assert second["status"] == "BUSY"


def test_shared_ready_queue_claim_is_not_profile_bound(tmp_path: Path):
    manifest = tmp_path / "shared.csv"
    manifest.write_text(
        "task_index,case_id,target,teacher,candidate_id,ct_path,annotation_folder\n"
        "0,CASE001,organ_a,teacher1,cand_a,/ct,/ann\n"
        "1,CASE001,organ_a,teacher2,cand_b,/ct,/ann\n",
        encoding="utf-8",
    )
    full373.write_json(
        tmp_path / "full_round1_scope.json",
        {
            "case_count": 1,
            "canonical_target_count": 1,
            "routes": {"organ_a": ["teacher1", "teacher2"]},
            "task_rows": [
                {"case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_a"},
                {"case_id": "CASE001", "target": "organ_a", "teacher": "teacher2", "candidate_id": "cand_b"},
            ],
            "total_logical_candidate_tasks": 2,
        },
    )
    full373.seed_candidate_states(tmp_path, task_manifest=manifest)
    claim = full373.claim_next_ready_candidate(
        tmp_path,
        task_manifest=manifest,
        worker_id="gpu_a100_worker",
        profile="gpu_a100",
        resource_class="GPU_HIGH_MEMORY_A100",
    )

    assert claim["status"] == "CLAIMED"
    assert claim["candidate"]["profile"] == "gpu_a100"
    assert claim["candidate"]["status"] == "CLAIMED"


def test_queue_worker_does_not_call_global_seed_and_loads_manifest_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = _write_candidate_manifest(
        tmp_path / "shared.csv",
        [
            {"task_index": "0", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_a", "ct_path": "/ct", "annotation_folder": "/ann"},
            {"task_index": "1", "case_id": "CASE002", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_b", "ct_path": "/ct", "annotation_folder": "/ann"},
        ],
    )
    _write_scope(tmp_path, cases=["CASE001", "CASE002"], routes={"organ_a": ["teacher1"]})
    full373.seed_candidate_states(tmp_path, task_manifest=manifest)
    seed_calls = {"count": 0}
    manifest_loads = {"count": 0}
    original_rows = full373._candidate_queue_rows

    def fail_seed(*args, **kwargs):
        seed_calls["count"] += 1
        raise AssertionError("queue worker must not perform global candidate seed")

    def counted_rows(*args, **kwargs):
        manifest_loads["count"] += 1
        return original_rows(*args, **kwargs)

    def fake_execute(row, output_root, **kwargs):
        full373.publish_candidate_state(
            output_root,
            {
                "status": "SUCCESS",
                "case_id": row["case_id"],
                "target": row["target"],
                "teacher": row["teacher"],
                "candidate_id": row["candidate_id"],
                "candidate_exists": False,
                "prediction": "",
            },
        )
        return {"status": "COMPLETED", "candidate_status": "SUCCESS"}

    monkeypatch.setattr(full373, "seed_candidate_states", fail_seed)
    monkeypatch.setattr(full373, "_candidate_queue_rows", counted_rows)
    monkeypatch.setattr(full373, "_execute_candidate_row", fake_execute)

    result = full373.run_candidate_queue_worker(
        tmp_path,
        task_manifest=manifest,
        worker_id="worker_a",
        profile="gpu_t4",
        resource_class="GPU_LIGHT_T4",
        max_tasks=2,
        max_idle_sec=0,
    )

    assert result["status"] == "MAX_TASKS_REACHED"
    assert result["completed"] == 2
    assert seed_calls["count"] == 0
    assert manifest_loads["count"] == 1


def test_queue_worker_refreshes_seed_marker_without_reloading_manifest_or_seeding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = _write_candidate_manifest(
        tmp_path / "shared.csv",
        [
            {"task_index": "0", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_a", "ct_path": "/ct", "annotation_folder": "/ann"},
        ],
    )
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1"]})
    seed_calls = {"count": 0}
    manifest_loads = {"count": 0}
    sleep_calls = {"count": 0}
    original_rows = full373._candidate_queue_rows

    def fail_seed(*args, **kwargs):
        seed_calls["count"] += 1
        raise AssertionError("queue worker must wait for controller seed, not seed itself")

    def counted_rows(*args, **kwargs):
        manifest_loads["count"] += 1
        return original_rows(*args, **kwargs)

    def fake_sleep(_seconds):
        sleep_calls["count"] += 1
        full373.publish_candidate_state(
            tmp_path,
            {
                "status": "READY",
                "case_id": "CASE001",
                "target": "organ_a",
                "teacher": "teacher1",
                "candidate_id": "cand_a",
            },
            recompute_target=False,
        )
        identity = full373._manifest_identity(manifest, original_rows(tmp_path, task_manifest=manifest))
        full373.atomic_write_json(
            tmp_path / "queues" / "candidate_seed_complete.json",
            {
                "schema_version": "candidate_seed_complete_v1",
                "logical_task_count": 1,
                **identity,
                "seeded": 1,
                "retained": 0,
                "completed_at": full373.utc_now(),
            },
        )

    def fake_execute(row, output_root, **kwargs):
        full373.publish_candidate_state(
            output_root,
            {
                "status": "SUCCESS",
                "case_id": row["case_id"],
                "target": row["target"],
                "teacher": row["teacher"],
                "candidate_id": row["candidate_id"],
                "candidate_exists": False,
                "prediction": "",
            },
        )
        return {"status": "COMPLETED", "candidate_status": "SUCCESS"}

    monkeypatch.setattr(full373, "seed_candidate_states", fail_seed)
    monkeypatch.setattr(full373, "_candidate_queue_rows", counted_rows)
    monkeypatch.setattr(full373.time, "sleep", fake_sleep)
    monkeypatch.setattr(full373, "_execute_candidate_row", fake_execute)

    result = full373.run_candidate_queue_worker(
        tmp_path,
        task_manifest=manifest,
        worker_id="worker_a",
        profile="gpu_t4",
        resource_class="GPU_LIGHT_T4",
        poll_sec=1,
        max_tasks=1,
        max_idle_sec=60,
    )

    assert result["status"] == "MAX_TASKS_REACHED"
    assert result["completed"] == 1
    assert seed_calls["count"] == 0
    assert manifest_loads["count"] == 1
    assert sleep_calls["count"] == 1


def test_seed_candidate_states_does_not_recompute_case_target_for_each_ready_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = _write_candidate_manifest(
        tmp_path / "shared.csv",
        [
            {"task_index": "0", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_a", "ct_path": "/ct", "annotation_folder": "/ann"},
            {"task_index": "1", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher2", "candidate_id": "cand_b", "ct_path": "/ct", "annotation_folder": "/ann"},
        ],
    )
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1", "teacher2"]})
    calls = {"count": 0}

    def fail_recompute(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("global READY seed must not recompute case-target readiness per candidate")

    monkeypatch.setattr(full373, "recompute_case_target_readiness", fail_recompute)

    seed = full373.seed_candidate_states(tmp_path, task_manifest=manifest)

    assert seed["seeded"] == 2
    assert seed["logical_task_count"] == 2
    assert calls["count"] == 0


def test_restart_recovers_seed_marker_from_complete_existing_queue_without_overwriting_terminal(tmp_path: Path):
    manifest = _write_candidate_manifest(
        tmp_path / "shared.csv",
        [
            {"task_index": "0", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_success", "ct_path": "/ct", "annotation_folder": "/ann"},
            {"task_index": "1", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher2", "candidate_id": "cand_empty", "ct_path": "/ct", "annotation_folder": "/ann"},
            {"task_index": "2", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher3", "candidate_id": "cand_ready", "ct_path": "/ct", "annotation_folder": "/ann"},
        ],
    )
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1", "teacher2", "teacher3"]})
    _candidate(tmp_path, "CASE001", "organ_a", "teacher1", status="SUCCESS")
    _candidate(tmp_path, "CASE001", "organ_a", "teacher2", status="COMPLETED_NO_NONZERO")
    full373.publish_candidate_state(
        tmp_path,
        {
            "status": "READY",
            "case_id": "CASE001",
            "target": "organ_a",
            "teacher": "teacher3",
            "candidate_id": "cand_ready",
        },
        recompute_target=False,
    )

    marker = recover = full373.recover_candidate_seed_marker(tmp_path, task_manifest=manifest)

    assert recover["status"] == "READY"
    assert marker["execution_schema_version"] == full373.CANDIDATE_TASK_V1
    assert marker["logical_task_count"] == 3
    assert marker["validated_existing_count"] == 3
    assert marker["seeded_count"] == 0
    assert marker["retained_terminal_count"] == 2
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="organ_a", teacher="teacher1")["status"] == "SUCCESS"
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="organ_a", teacher="teacher2")["status"] == "COMPLETED_NO_NONZERO"


def test_stale_running_candidate_with_missing_claim_requeues_and_reclaims(tmp_path: Path):
    manifest = _write_candidate_manifest(
        tmp_path / "shared.csv",
        [
            {"task_index": "0", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_stale", "ct_path": "/ct", "annotation_folder": "/ann"},
        ],
    )
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1"]})
    full373.publish_candidate_state(
        tmp_path,
        {
            "status": "RUNNING",
            "case_id": "CASE001",
            "target": "organ_a",
            "teacher": "teacher1",
            "candidate_id": "cand_stale",
        },
        recompute_target=False,
    )
    full373.recover_candidate_seed_marker(tmp_path, task_manifest=manifest)

    recovery = full373.recover_stale_candidate_claims(tmp_path, task_manifest=manifest)
    claim = full373.claim_next_ready_candidate(
        tmp_path,
        task_manifest=manifest,
        worker_id="replacement",
        profile="gpu_t4",
        resource_class="GPU_LIGHT_T4",
    )

    assert recovery["recovered_count"] == 1
    assert claim["status"] == "CLAIMED"
    assert claim["candidate"]["status"] == "CLAIMED"


def test_targeted_failed_final_repair_requeues_known_infra_failures_only(tmp_path: Path):
    manifest = _write_candidate_manifest(
        tmp_path / "shared.csv",
        [
            {"task_index": "0", "case_id": "CASE001", "target": "vertebrae_t2", "teacher": "vista3d", "candidate_id": "cand_target", "ct_path": "/ct", "annotation_folder": "/ann"},
            {"task_index": "1", "case_id": "CASE001", "target": "liver", "teacher": "totalsegmentator", "candidate_id": "cand_ts", "ct_path": "/ct", "annotation_folder": "/ann"},
            {"task_index": "2", "case_id": "CASE001", "target": "spleen", "teacher": "teacher1", "candidate_id": "cand_other", "ct_path": "/ct", "annotation_folder": "/ann"},
            {"task_index": "3", "case_id": "CASE001", "target": "pancreas", "teacher": "teacher1", "candidate_id": "cand_success", "ct_path": "/ct", "annotation_folder": "/ann"},
            {"task_index": "4", "case_id": "CASE001", "target": "kidney_left", "teacher": "teacher1", "candidate_id": "cand_empty", "ct_path": "/ct", "annotation_folder": "/ann"},
        ],
    )
    _write_scope(
        tmp_path,
        cases=["CASE001"],
        routes={
            "vertebrae_t2": ["vista3d"],
            "liver": ["totalsegmentator"],
            "spleen": ["teacher1"],
            "pancreas": ["teacher1"],
            "kidney_left": ["teacher1"],
        },
    )
    full373.publish_candidate_state(tmp_path, {"status": "FAILED_FINAL", "case_id": "CASE001", "target": "vertebrae_t2", "teacher": "vista3d", "candidate_id": "cand_target", "failure_reason": full373.FORMAL_TARGET_CONTRACT_FAILURE}, recompute_target=False)
    full373.publish_candidate_state(tmp_path, {"status": "FAILED_FINAL", "case_id": "CASE001", "target": "liver", "teacher": "totalsegmentator", "candidate_id": "cand_ts", "failure_reason": "FileNotFoundError: [Errno 2] No such file or directory: 'TotalSegmentator'"}, recompute_target=False)
    full373.publish_candidate_state(tmp_path, {"status": "FAILED_FINAL", "case_id": "CASE001", "target": "spleen", "teacher": "teacher1", "candidate_id": "cand_other", "failure_reason": "model crashed"}, recompute_target=False)
    _candidate(tmp_path, "CASE001", "pancreas", "teacher1", status="SUCCESS")
    _candidate(tmp_path, "CASE001", "kidney_left", "teacher1", status="COMPLETED_NO_NONZERO")
    full373.recompute_case_target_readiness(tmp_path, case_id="CASE001", target="vertebrae_t2")

    repair = full373.repair_recoverable_failed_candidates(
        tmp_path,
        task_manifest=manifest,
        target_config=Path("configs/student_3d_prompt_target_organs.json"),
        totalseg_executable_ready=True,
        execution_attempt_id="attempt_b",
    )

    assert repair["repaired_count"] == 2
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="vertebrae_t2", teacher="vista3d")["status"] == "RETRY_PENDING"
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="liver", teacher="totalsegmentator")["recovery_reason"] == "totalsegmentator_executable_fixed"
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="spleen", teacher="teacher1")["status"] == "FAILED_FINAL"
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="pancreas", teacher="teacher1")["status"] == "SUCCESS"
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="kidney_left", teacher="teacher1")["status"] == "COMPLETED_NO_NONZERO"
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="vertebrae_t2", teacher="vista3d")["execution_attempt_id"] == "attempt_b"
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="vertebrae_t2", teacher="vista3d")["previous_status"] == "FAILED_FINAL"
    assert full373.load_candidate_state(tmp_path, case_id="CASE001", target="vertebrae_t2", teacher="vista3d")["previous_failure_reason"]
    assert full373._read_json(tmp_path / "queues" / "case_target_states" / "CASE001" / "vertebrae_t2.json")["status"] == "WAITING_FOR_CANDIDATES"

    full373.recover_candidate_seed_marker(tmp_path, task_manifest=manifest)
    claim = full373.claim_next_ready_candidate(tmp_path, task_manifest=manifest, worker_id="worker", profile="gpu_t4")
    assert claim["status"] == "CLAIMED"
    assert claim["candidate"]["candidate_id"] in {"cand_target", "cand_ts"}


def test_raw_candidate_survives_shapekit_failure_without_failed_final(tmp_path: Path):
    case_output = tmp_path / "case_runs" / "BDMAP_00003038" / "portal_vein_and_splenic_vein" / "vsmtrans"
    raw_mask = _save_mask(case_output / "annotation_versions" / "BDMAP_00003038" / "raw" / "portal_vein_and_splenic_vein.nii.gz")
    metadata_path = case_output / "annotation_versions" / "BDMAP_00003038" / "selection_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "selection_rows": [
                    {
                        "organ": "portal_vein_and_splenic_vein",
                        "candidate_predictions": [
                            {
                                "model": "vsmtrans",
                                "candidate_id": "cand_portal",
                                "prediction": str(raw_mask),
                                "candidate_raw_prediction": str(raw_mask),
                                "candidate_cleaned_prediction": "",
                                "candidate_qc_status": "fail",
                                "candidate_qc_flags": ["high_risk_postprocess_required", "postprocess_failed"],
                                "candidate_shapekit_status": "postprocess_failed",
                                "candidate_shapekit_reason": "ShapeKit returned non-zero code 1",
                                "eligible_for_labelcritic": False,
                            }
                        ],
                    }
                ],
                "selected_organs": [
                    {
                        "organ": "portal_vein_and_splenic_vein",
                        "pre_shapekit_mask": str(raw_mask),
                        "shapekit_status": "postprocess_failed",
                        "shapekit_reason": "ShapeKit returned non-zero code 1",
                        "selected_candidate_qc_status": "fail",
                        "selected_candidate_qc_flags": ["high_risk_postprocess_required", "postprocess_failed"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidate = full373._candidate_from_single_teacher_run(
        case_output,
        case_id="BDMAP_00003038",
        target="portal_vein_and_splenic_vein",
        teacher="vsmtrans",
    )

    assert candidate["status"] == "SUCCESS"
    assert candidate["candidate_exists"] is True
    assert candidate["prediction"] == str(raw_mask)
    assert candidate["candidate_raw_prediction"] == str(raw_mask)
    assert candidate["candidate_cleaned_prediction"] == ""
    assert candidate["candidate_qc_status"] == "fail"
    assert "postprocess_failed" in candidate["candidate_qc_flags"]
    assert candidate["candidate_shapekit_status"] == "postprocess_failed"
    assert candidate["candidate_shapekit_reason"] == "ShapeKit returned non-zero code 1"
    assert candidate["raw_candidate_survived_shapekit_failure"] is True
    assert candidate["eligible_for_labelcritic"] is False


def test_worker_exports_configured_totalsegmentator_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = _write_candidate_manifest(
        tmp_path / "shared.csv",
        [
            {"task_index": "0", "case_id": "CASE001", "target": "liver", "teacher": "totalsegmentator", "candidate_id": "cand_ts", "ct_path": "/ct", "annotation_folder": "/ann", "totalsegmentator_executable": "/opt/totalseg/bin/TotalSegmentator"},
        ],
    )
    full373.seed_candidate_states(tmp_path, task_manifest=manifest)
    captured_env = {}

    def fake_run(command, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        case_output = tmp_path / "candidate_runs" / "CASE001" / "liver" / "totalsegmentator"
        mask = case_output / "annotation_versions" / "CASE001" / "updated" / "liver.nii.gz"
        _save_mask(mask)
        full373.write_json(
            case_output / "annotation_versions" / "CASE001" / "selection_metadata.json",
            {"selected_organs": [{"organ": "liver", "final_mask": str(mask)}], "selection_rows": []},
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(full373.subprocess, "run", fake_run)

    result = full373.execute_task_index(0, manifest, tmp_path, worker_id="worker")

    assert result["status"] == "COMPLETED"
    assert captured_env["TOTAL_SEGMENTATOR_EXECUTABLE"] == "/opt/totalseg/bin/TotalSegmentator"


def test_build_submission_manifest_does_not_seed_before_dynamic_worker_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    targets = [f"organ_{index:03d}" for index in range(373)]
    _patch_routes(monkeypatch, {target: ["teacher1"] for target in targets})
    case_manifest = _case_manifest(tmp_path / "cases.csv", ["CASE001"])
    original_ann = tmp_path / "workspace" / "inputs" / "masks_original" / "CASE001" / "segmentations"
    canonical_ann = tmp_path / "workspace" / "inputs" / "masks_373_canonical" / "CASE001" / "segmentations"
    case_manifest.write_text(
        "case_id,ct_path,annotation_folder,original_annotation_folder,canonical_annotation_folder\n"
        f"CASE001,{tmp_path / 'ct.nii.gz'},{original_ann},{original_ann},{canonical_ann}\n",
        encoding="utf-8",
    )
    target_config = _target_config(tmp_path / "targets.json", targets)

    def fail_seed(*args, **kwargs):
        raise AssertionError("full373 planning must not seed; dynamic submitter seeds the actual worker manifest once")

    monkeypatch.setattr(full373, "seed_candidate_states", fail_seed)

    summary = full373.build_submission_manifest(
        case_manifest=case_manifest,
        output_root=tmp_path / "out",
        registry_path=tmp_path / "registry.yaml",
        target_config=target_config,
        python=Path("/usr/bin/python"),
        checkpoint_root=tmp_path / "checkpoints",
        nnunet_predict_executable=tmp_path / "nnUNetv2_predict",
        unest_python_executable=tmp_path / "unest_python",
        totalsegmentator_executable="/opt/totalseg/bin/TotalSegmentator",
        expected_case_count=1,
    )

    assert summary["status"] == "READY"
    assert summary["task_count"] == 373
    assert "candidate_seed" not in summary
    rows = full373.read_csv_rows(Path(summary["groups"][full373.FULL373_GROUP]["task_manifest"]))
    assert rows[0]["totalsegmentator_executable"] == "/opt/totalseg/bin/TotalSegmentator"
    assert rows[0]["annotation_folder"] == str(original_ann)
    assert rows[0]["original_annotation_folder"] == str(original_ann)
    assert rows[0]["canonical_annotation_folder"] == str(canonical_ann)


def test_full373_merge_prefers_task1_source_mask_with_provenance_without_changing_teacher_scope(tmp_path: Path):
    ct = tmp_path / "ct.nii.gz"
    _save_mask(ct)
    canonical_dir = tmp_path / "workspace" / "inputs" / "masks_373_canonical" / "CASE001" / "segmentations"
    task1_mask = _save_mask(canonical_dir / "organ_a.nii.gz")
    teacher_mask = _save_mask(tmp_path / "teacher" / "organ_a.nii.gz")
    manifest = tmp_path / "out" / "slurm" / "full373_task_manifest.csv"
    _write_candidate_manifest(
        manifest,
        [
            {
                "task_index": "0",
                "case_id": "CASE001",
                "target": "organ_a",
                "teacher": "teacher1",
                "candidate_id": "cand_a",
                "ct_path": str(ct),
                "annotation_folder": str(tmp_path / "workspace" / "inputs" / "masks_original" / "CASE001" / "segmentations"),
                "original_annotation_folder": str(tmp_path / "workspace" / "inputs" / "masks_original" / "CASE001" / "segmentations"),
                "canonical_annotation_folder": str(canonical_dir),
            }
        ],
    )
    full373.write_json(
        tmp_path / "out" / "full_round1_scope.json",
        {
            "case_count": 1,
            "canonical_target_count": 1,
            "routes": {"organ_a": ["teacher1"]},
            "task_rows": [{"case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_a"}],
            "total_logical_candidate_tasks": 1,
        },
    )
    full373.publish_candidate_state(
        tmp_path / "out",
        {
            "status": "SUCCESS",
            "case_id": "CASE001",
            "target": "organ_a",
            "teacher": "teacher1",
            "candidate_id": "cand_a",
            "candidate_exists": True,
            "prediction": str(teacher_mask),
        },
        recompute_target=True,
    )
    state_path = tmp_path / "out" / "queues" / "case_target_states" / "CASE001" / "organ_a.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "VALID_SINGLE_TEACHER_ACCEPTED",
            "ct_path": str(ct),
            "target": "organ_a",
            "organ": "organ_a",
            "selected_model": "teacher1",
            "final_mask": str(teacher_mask),
            "teacher_names": ["teacher1"],
        }
    )
    full373.atomic_write_json(state_path, state)

    report = full373.aggregate_full373_estep(tmp_path / "out", expected_cases=1, expected_targets=1)
    manifest_doc = json.loads((tmp_path / "out" / "training_manifest.json").read_text(encoding="utf-8"))
    item = manifest_doc["items"][0]

    assert report["status"] == "PASSED"
    assert item["mask_path"] == str(task1_mask)
    assert item["source"] == "task1_source"
    assert item["source_role"] == "task1_source"
    assert item["source_stage"] == "task1_373_canonical_source_mask"
    assert item["training_eligible"] is True
    assert item["contract_failures"] == []


def test_two_workers_share_one_seed_and_keep_o_excl_claim_authority(tmp_path: Path):
    manifest = _write_candidate_manifest(
        tmp_path / "shared.csv",
        [
            {"task_index": "0", "case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_a", "ct_path": "/ct", "annotation_folder": "/ann"},
        ],
    )
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1"]})
    seed = full373.seed_candidate_states(tmp_path, task_manifest=manifest)
    rows = full373._candidate_queue_rows(tmp_path, task_manifest=manifest)

    first = full373.claim_next_ready_candidate_from_rows(
        tmp_path,
        manifest_rows=rows,
        task_manifest=manifest,
        worker_id="worker_a",
        profile="gpu_t4",
        resource_class="GPU_LIGHT_T4",
    )
    second = full373.claim_next_ready_candidate_from_rows(
        tmp_path,
        manifest_rows=rows,
        task_manifest=manifest,
        worker_id="worker_b",
        profile="gpu_a100",
        resource_class="GPU_HIGH_MEMORY_A100",
    )

    assert seed["logical_task_count"] == 1
    assert first["status"] == "CLAIMED"
    assert second["status"] != "CLAIMED"
    marker = json.loads((tmp_path / "queues" / "candidate_seed_complete.json").read_text(encoding="utf-8"))
    assert marker["seeded"] == 1


def test_expired_candidate_claim_returns_to_retry_pending_and_reclaims(tmp_path: Path):
    manifest = tmp_path / "shared.csv"
    manifest.write_text(
        "task_index,case_id,target,teacher,candidate_id,ct_path,annotation_folder\n"
        "0,CASE001,organ_a,teacher1,cand_a,/ct,/ann\n",
        encoding="utf-8",
    )
    full373.write_json(
        tmp_path / "full_round1_scope.json",
        {
            "case_count": 1,
            "canonical_target_count": 1,
            "routes": {"organ_a": ["teacher1"]},
            "task_rows": [{"case_id": "CASE001", "target": "organ_a", "teacher": "teacher1", "candidate_id": "cand_a"}],
            "total_logical_candidate_tasks": 1,
        },
    )
    full373.seed_candidate_states(tmp_path, task_manifest=manifest)
    full373.publish_candidate_state(
        tmp_path,
        {
            "status": "RUNNING",
            "case_id": "CASE001",
            "target": "organ_a",
            "teacher": "teacher1",
            "candidate_id": "cand_a",
        },
    )
    full373.claim_work(
        tmp_path,
        claim_kind="candidate",
        claim_key="cand_a",
        worker_id="dead_worker",
        lease_sec=1,
    )
    claim_path = tmp_path / "queues" / "candidate_claims" / "cand_a.json"
    claim_doc = json.loads(claim_path.read_text(encoding="utf-8"))
    claim_doc["heartbeat_time"] = -100
    claim_doc["created_time"] = -100
    full373.atomic_write_json(claim_path, claim_doc)

    claim = full373.claim_next_ready_candidate(
        tmp_path,
        task_manifest=manifest,
        worker_id="replacement_worker",
        profile="gpu_t4",
        resource_class="GPU_LIGHT_T4",
        lease_sec=1,
    )

    assert claim["status"] == "CLAIMED"
    state = full373.load_candidate_state(tmp_path, case_id="CASE001", target="organ_a", teacher="teacher1")
    assert state["status"] == "CLAIMED"


def test_same_case_target_cannot_be_selected_by_two_workers(tmp_path: Path):
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1"]})
    _candidate(tmp_path, "CASE001", "organ_a", "teacher1")
    first = full373.claim_next_case_target(tmp_path, worker_id="critic1", labelcritic_ready=True)
    second = full373.claim_next_case_target(tmp_path, worker_id="critic2", labelcritic_ready=True)
    assert first["status"] == "CLAIMED"
    assert second["status"] != "CLAIMED"


def test_estep_gate_requires_all_case_targets_terminal_not_just_candidates_ready(tmp_path: Path):
    _write_scope(tmp_path, cases=["CASE001"], routes={"organ_a": ["teacher1"], "organ_b": ["teacher1"]})
    _candidate(tmp_path, "CASE001", "organ_a", "teacher1")
    report = full373.aggregate_full373_estep(tmp_path, expected_cases=1, expected_targets=2)
    assert report["status"] == "RUNNING"
    assert report["manifest_targets"] == 0
