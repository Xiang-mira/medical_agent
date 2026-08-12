from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_group_plan(root: Path, group: str, count: int) -> dict[str, str | int]:
    manifest = root / group / f"{group}_task_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_index", "case_id", "model_group", "status"])
        writer.writeheader()
        for index in range(count):
            writer.writerow({"task_index": index, "case_id": f"BDMAP_{index:08d}", "model_group": group, "status": "planned"})
    sbatch = root / "slurm" / f"{group}_task2_array.sbatch"
    sbatch.parent.mkdir(parents=True, exist_ok=True)
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH --job-name=task2_{group}\n"
        "#SBATCH --partition=gpu\n"
        "#SBATCH --gres=gpu:T4:1\n"
        "#SBATCH --cpus-per-task=8\n"
        "#SBATCH --mem=64G\n"
        "#SBATCH --time=06:00:00\n"
        f"#SBATCH --output={root / 'slurm' / (group + '_%A_%a.out')}\n"
        f"#SBATCH --error={root / 'slurm' / (group + '_%A_%a.err')}\n"
        "cd /repo\n"
        "python tools/dataset_delivery/task2_formal_launcher.py \\\n"
        '  --execute-task-index "$SLURM_ARRAY_TASK_ID" \\\n'
        f"  --task-manifest {manifest} \\\n"
        f"  --output-root {root}\n",
        encoding="utf-8",
    )
    return {"task_count": count, "task_manifest": str(manifest), "sbatch_file": str(sbatch)}


def _summary(root: Path, counts: dict[str, int]) -> Path:
    groups = {group: _write_group_plan(root, group, count) for group, count in counts.items()}
    path = root / "formal_task2_submission_manifest.json"
    path.write_text(json.dumps({"status": "READY", "groups": groups}), encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_dynamic_submitter_overrequests_target_30_to_40_and_uses_generic_gpu(tmp_path: Path):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 103, "atm": 103, "airrc": 103, "unest": 103})
    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=30,
        overrequest_workers=None,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00",
        groups=["cads", "atm", "airrc", "unest"],
        dry_run=True,
    )

    assert plan["status"] == "DRY_RUN"
    assert plan["planned_target_workers"] == 30
    assert plan["planned_overrequest_workers"] == 40
    assert plan["total_array_concurrency"] == 40
    assert plan["group_concurrency"] == {"cads": 18, "atm": 6, "airrc": 8, "unest": 8}
    rows = _rows(tmp_path / "slurm" / "submitted_jobs.csv")
    assert {row["gres"] for row in rows} == {"gpu:1"}
    assert "gpu:T4:1" not in (tmp_path / "slurm" / "dynamic" / "cads_generic_gpu_task2_array.sbatch").read_text(encoding="utf-8")


def test_dynamic_submitter_profile_shards_do_not_duplicate_source_task_indices(tmp_path: Path):
    from tools.dataset_delivery.task2_dynamic_gpu_submitter import build_dynamic_submission_plan

    summary = _summary(tmp_path, {"cads": 8})
    plan = build_dynamic_submission_plan(
        summary_path=summary,
        output_root=tmp_path,
        state_root=tmp_path / "state",
        target_workers=4,
        overrequest_workers=6,
        profile_specs="generic_gpu|gpu|gpu:1|8|64G|06:00:00,a100|gpu|gpu:A100:1|8|80G|06:00:00",
        groups=["cads"],
        dry_run=True,
    )

    assert plan["total_array_concurrency"] == 6
    shard_paths = [Path(row["task_manifest"]) for row in plan["jobs"]]
    source_indices: list[str] = []
    for path in shard_paths:
        rows = _rows(path)
        assert [row["task_index"] for row in rows] == [str(index) for index in range(len(rows))]
        source_indices.extend(row["source_task_index"] for row in rows)
    assert sorted(source_indices, key=int) == [str(index) for index in range(8)]
