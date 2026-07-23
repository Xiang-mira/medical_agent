from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path
from .planner import build_plan, submit_plan
from .preflight import run_preflight
from .state import read_statuses, write_status
from .utils import SchedulerError, read_json, write_json_atomic
from .executor import execute_task


def emit(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _load_plan(run_id_or_dir: str) -> dict[str, Any]:
    path = Path(run_id_or_dir)
    if not path.exists():
        path = Path("runs") / run_id_or_dir
    plan_path = path / "plan.json"
    if not plan_path.exists():
        raise SchedulerError(f"Could not find plan.json for {run_id_or_dir}")
    return read_json(plan_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scheduler.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate-config")
    p.add_argument("--config", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--config", required=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--require-hpc-paths", action="store_true")

    p = sub.add_parser("plan")
    p.add_argument("--config", required=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--backend", choices=["local", "slurm"], required=True)
    p.add_argument("--run-id", default=None)

    p = sub.add_parser("submit")
    p.add_argument("--config", required=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-id", default=None)

    p = sub.add_parser("status")
    p.add_argument("--run-id", required=True)

    p = sub.add_parser("retry-failed")
    p.add_argument("--run-id", required=True)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("cancel")
    p.add_argument("--run-id", required=True)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("smoke-plan")
    p.add_argument("--config", required=True)
    p.add_argument("--pipeline", required=True)

    p = sub.add_parser("select-resources")
    p.add_argument("--config", required=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--from-smoke", required=True)

    p = sub.add_parser("run-task")
    p.add_argument("--task", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--array-index", type=int, default=None)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "validate-config":
            cfg = load_config(args.config)
            emit({"status": "success", "config": str(cfg.path)})
        elif args.cmd == "preflight":
            cfg = load_config(args.config)
            emit(run_preflight(cfg, require_hpc_paths=args.require_hpc_paths))
        elif args.cmd == "plan":
            cfg = load_config(args.config)
            emit(build_plan(cfg, args.pipeline, backend=args.backend, run_id=args.run_id))
        elif args.cmd == "submit":
            cfg = load_config(args.config)
            if args.run_id and not args.dry_run:
                run_root = resolve_path(cfg.paths.get("run_root") or "runs")
                assert run_root is not None
                jobs_path = run_root / args.run_id / "jobs.json"
                if jobs_path.exists():
                    existing = read_json(jobs_path)
                    job_ids = [v.get("job_id") for v in existing.values() if isinstance(v, dict) and v.get("job_id")]
                    if job_ids:
                        raise SchedulerError(f"Run {args.run_id} already has submitted jobs: {job_ids}")
            plan = build_plan(cfg, args.pipeline, backend="slurm", run_id=args.run_id)
            emit(submit_plan(plan, dry_run=args.dry_run))
        elif args.cmd == "status":
            plan = _load_plan(args.run_id)
            emit({"run_id": plan["run_id"], "run_dir": plan["run_dir"], "statuses": read_statuses(Path(plan["run_dir"]))})
        elif args.cmd == "retry-failed":
            plan = _load_plan(args.run_id)
            run_dir = Path(plan["run_dir"])
            failed = [s for s in read_statuses(Path(plan["run_dir"])) if s.get("status") == "failed"]
            jobs = []
            by_task: dict[str, list[int | None]] = {}
            for row in failed:
                by_task.setdefault(str(row.get("task_id")), []).append(row.get("array_index"))
            for task_id, indices in by_task.items():
                script = run_dir / "generated_slurm" / f"{task_id}.sbatch"
                cmd = ["sbatch"]
                numeric = sorted(int(x) for x in indices if x is not None)
                if numeric:
                    cmd.append("--array=" + ",".join(str(x) for x in numeric))
                cmd.append(str(script))
                if args.dry_run:
                    jobs.append({"task_id": task_id, "status": "dry_run", "command": cmd})
                else:
                    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    if proc.returncode != 0:
                        raise SchedulerError(f"retry sbatch failed for {task_id}: {proc.stderr}")
                    jobs.append({"task_id": task_id, "status": "submitted", "command": cmd, "stdout": proc.stdout.strip()})
            emit({"status": "success", "dry_run": args.dry_run, "failed_count": len(failed), "jobs": jobs})
        elif args.cmd == "cancel":
            plan = _load_plan(args.run_id)
            jobs = read_json(Path(plan["run_dir"]) / "jobs.json")
            job_ids = [v.get("job_id") for v in jobs.values() if isinstance(v, dict) and v.get("job_id")]
            commands = []
            if not args.dry_run:
                for job_id in job_ids:
                    proc = subprocess.run(["scancel", str(job_id)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    if proc.returncode != 0:
                        raise SchedulerError(f"scancel failed for {job_id}: {proc.stderr}")
                    commands.append(["scancel", str(job_id)])
            emit({"status": "success", "dry_run": args.dry_run, "job_ids": job_ids, "commands": commands})
        elif args.cmd == "smoke-plan":
            cfg = load_config(args.config)
            plan = build_plan(cfg, args.pipeline, backend="slurm", run_id=None)
            smoke_tasks = [t for t in plan["tasks"] if "resource_selection" in t["task_name"] or "smoke" in t["task_name"]]
            emit({"status": "success", "run_id": plan["run_id"], "smoke_tasks": smoke_tasks})
        elif args.cmd == "select-resources":
            emit({"status": "planned", "from_smoke": args.from_smoke, "policy": "select first successful lower-scarcity tier, else fallback upward; preserve tier labels in final reports"})
        elif args.cmd == "run-task":
            emit(execute_task(Path(args.run_dir), args.task, array_index=args.array_index))
    except SchedulerError as exc:
        emit({"status": "failed", "error": str(exc)})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
