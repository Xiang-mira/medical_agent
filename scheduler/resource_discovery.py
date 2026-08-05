from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .utils import utc_now, write_json_atomic


GPU_PARTITIONS = {"gpu": "T4", "gpua100": "A100", "gpuh100": "H100"}
BAD_NODE_STATE_MARKERS = ("DOWN", "DRAIN", "DRAINING", "FAIL", "MAINT", "NOT_RESPONDING", "NO_RESPOND")


def _run(args: list[str]) -> tuple[int, str, str]:
    if shutil.which(args[0]) is None:
        return 127, "", f"{args[0]} unavailable"
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, "", "(null)", "N/A"}:
            return default
        return int(float(str(value).rstrip("MGT")))
    except Exception:
        return default


def _parse_tres(text: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    if not text or text == "(null)":
        return out
    for item in str(text).split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key.lower()] = _safe_int(value)
    return out


def _is_bad_state(state: str) -> bool:
    upper = state.upper()
    return any(marker in upper for marker in BAD_NODE_STATE_MARKERS)


def _kv_pairs(text: str) -> dict[str, str]:
    return dict(re.findall(r"(\S+?)=(\S+)", text.replace("\n   ", " ")))


def parse_scontrol_nodes(text: str) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for block in re.split(r"\n(?=NodeName=)", text.strip()):
        if not block.strip():
            continue
        pairs = _kv_pairs(block)
        gres = pairs.get("Gres", "")
        gpu_type = None
        configured_gpu = 0
        match = re.search(r"gpu:([A-Za-z0-9_]+):(\d+)", gres)
        if match:
            gpu_type = match.group(1).upper()
            configured_gpu = int(match.group(2))
        cfg_tres = _parse_tres(pairs.get("CfgTRES"))
        alloc_tres = _parse_tres(pairs.get("AllocTRES"))
        configured_gpu = configured_gpu or cfg_tres.get("gres/gpu", 0)
        allocated_gpu = alloc_tres.get("gres/gpu", 0)
        state = pairs.get("State", "")
        nodes.append(
            {
                "name": pairs.get("NodeName", ""),
                "state": state,
                "schedulable": not _is_bad_state(state),
                "partitions": [p.rstrip("*") for p in pairs.get("Partitions", "").split(",") if p],
                "cpus": _safe_int(pairs.get("CPUTot")),
                "alloc_cpus": _safe_int(pairs.get("CPUAlloc")),
                "real_memory_mb": _safe_int(pairs.get("RealMemory")),
                "alloc_memory_mb": _safe_int(pairs.get("AllocMem")),
                "gpu_type": gpu_type,
                "gpus_configured": configured_gpu,
                "gpus_allocated": min(allocated_gpu, configured_gpu) if configured_gpu else allocated_gpu,
                "gpus_free": max(0, configured_gpu - allocated_gpu),
            }
        )
    return nodes


def parse_sinfo_json(text: str) -> dict[str, Any]:
    doc = json.loads(text)
    partitions = {}
    for item in doc.get("sinfo") or doc.get("partitions") or []:
        part = item.get("partition") or item.get("name")
        if part:
            partitions[str(part).rstrip("*")] = item
    return partitions


def parse_sinfo_fallback(text: str) -> dict[str, Any]:
    partitions: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.strip() or line.lower().startswith("partition"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name, avail, timelimit, nodes = parts[:4]
        partitions[name.rstrip("*")] = {"partition": name.rstrip("*"), "availability": avail, "time": timelimit, "nodes": nodes}
    return partitions


def parse_squeue_fallback(text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("JOBID|"):
            continue
        parts = line.split("|")
        if len(parts) >= 8:
            rows.append({"job_id": parts[0], "partition": parts[1].rstrip("*"), "state": parts[2], "user": parts[3], "reason": parts[4], "tres": parts[5], "nodes": parts[6], "array_task": parts[7]})
        elif len(parts) >= 6:
            rows.append({"job_id": parts[0], "partition": parts[1].rstrip("*"), "state": parts[2], "user": parts[3], "reason": parts[4], "tres": parts[5], "nodes": "1", "array_task": ""})
    return rows


def parse_scontrol_job(text: str) -> dict[str, Any]:
    pairs = _kv_pairs(text)
    req_tres = _parse_tres(pairs.get("ReqTRES"))
    tres_per_node = pairs.get("TresPerNode", "")
    gres = pairs.get("Gres", "")
    gpu = req_tres.get("gres/gpu", 0)
    typed = {k: v for k, v in req_tres.items() if k.startswith("gres/gpu:")}
    if typed and not gpu:
        gpu = max(typed.values())
    for source in (tres_per_node, gres):
        m = re.search(r"gres/gpu(?::[A-Za-z0-9_]+)?[:=](\d+)", source, flags=re.I)
        if m:
            gpu = max(gpu, int(m.group(1)))
    nodes = _safe_int(pairs.get("NumNodes"), 1)
    partition = pairs.get("Partition", "")
    reason = pairs.get("Reason", "")
    state = pairs.get("JobState", "")
    return {
        "job_id": pairs.get("JobId", ""),
        "partition": partition.rstrip("*"),
        "state": state,
        "reason": reason,
        "gpu_per_job": gpu,
        "node_demand": nodes,
        "shape": gpu_shape(gpu),
        "req_tres": pairs.get("ReqTRES"),
        "tres_per_node": tres_per_node,
    }


def gpu_shape(gpu_count: int) -> str:
    if gpu_count == 1:
        return "1 GPU/job"
    if gpu_count == 2:
        return "2 GPUs on one node"
    if gpu_count == 4:
        return "4 GPUs on one node"
    return "other"


def _job_details(job_ids: list[str], limit: int = 200) -> tuple[list[dict[str, Any]], str]:
    details = []
    raw = []
    for job_id in job_ids[:limit]:
        code, out, err = _run(["scontrol", "show", "job", "-o", str(job_id)])
        raw.append(out or err)
        if code == 0 and out.strip():
            details.append(parse_scontrol_job(out))
    return details, "\n".join(raw)


def _queue_rows() -> tuple[list[dict[str, str]], str, str]:
    code, out, err = _run(["squeue", "--json"])
    if code == 0:
        try:
            doc = json.loads(out)
            rows = []
            for job in doc.get("jobs", []):
                tres = str(job.get("tres_req_str") or job.get("tres_per_node") or job.get("gres_detail") or "")
                rows.append(
                    {
                        "job_id": str(job.get("job_id", "")),
                        "partition": str(job.get("partition") or "").rstrip("*"),
                        "state": str(job.get("job_state") or job.get("state") or ""),
                        "user": str(job.get("user_name") or job.get("user") or ""),
                        "reason": str(job.get("state_reason") or job.get("reason") or ""),
                        "tres": tres,
                        "nodes": str(job.get("nodes") or job.get("node_count") or "1"),
                        "array_task": str(job.get("array_task_id") or ""),
                    }
                )
            return rows, "squeue_json", out
        except Exception:
            pass
    code, out, err = _run(["squeue", "-h", "-o", "%i|%P|%T|%u|%R|%b|%D|%K"])
    if code == 0:
        return parse_squeue_fallback(out), "squeue_text", out
    return [], "unavailable", err


def _cluster_name() -> tuple[str, str]:
    code, out, err = _run(["scontrol", "show", "config"])
    if code == 0:
        match = re.search(r"ClusterName\s*=\s*(\S+)", out)
        if match:
            return match.group(1), "scontrol_show_config"
    env = os.environ.get("SLURM_CLUSTER_NAME")
    if env:
        return env, "SLURM_CLUSTER_NAME"
    return "unknown", f"unavailable:{err.strip() if err else 'not found'}"


def _account_qos(config_account: str | None, config_qos: str | None) -> tuple[str | None, str, str | None, str]:
    if config_account:
        account, account_source = config_account, "config"
    elif os.environ.get("SLURM_ACCOUNT"):
        account, account_source = os.environ["SLURM_ACCOUNT"], "SLURM_ACCOUNT"
    else:
        account, account_source = None, "unresolved"
    if config_qos:
        qos, qos_source = config_qos, "config"
    elif os.environ.get("SLURM_QOS"):
        qos, qos_source = os.environ["SLURM_QOS"], "SLURM_QOS"
    else:
        qos, qos_source = None, "unresolved"
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if shutil.which("sacctmgr") and (account is None or qos is None) and user:
        code, out, _ = _run(["sacctmgr", "-n", "-P", "show", "assoc", f"user={user}", "format=Cluster,Account,User,DefaultQOS,QOS"])
        if code == 0:
            for line in out.splitlines():
                parts = line.split("|")
                if len(parts) >= 5 and parts[2] == user:
                    if account is None and parts[1]:
                        account, account_source = parts[1], "sacctmgr_association"
                    if qos is None and parts[3]:
                        qos, qos_source = parts[3], "sacctmgr_default_qos"
                    break
    return account, account_source, qos, qos_source


def _queue_summary(pending: list[dict[str, str]], details: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(d.get("job_id")): d for d in details if d.get("job_id")}
    total_gpu = 0
    node_demand = 0
    by_reason: Counter[str] = Counter()
    by_shape: Counter[str] = Counter()
    for job in pending:
        detail = by_id.get(str(job.get("job_id"))) or {}
        tres = _parse_tres(job.get("tres"))
        gpu = int(detail.get("gpu_per_job") or tres.get("gres/gpu") or 0)
        nodes = int(detail.get("node_demand") or _safe_int(job.get("nodes"), 1) or 1)
        total_gpu += gpu
        node_demand += nodes
        reason = str(detail.get("reason") or job.get("reason") or "unknown")
        by_reason[reason] += 1
        by_shape[gpu_shape(gpu)] += 1
    return {
        "queued_jobs": len(pending),
        "queued_gpu_demand_total": total_gpu,
        "queued_gpu_demand": total_gpu,
        "queued_node_demand": node_demand,
        "queued_jobs_by_reason": dict(sorted(by_reason.items())),
        "queued_jobs_by_gpu_shape": dict(sorted(by_shape.items())),
    }


def validate_snapshot_invariants(snapshot: dict[str, Any]) -> dict[str, Any]:
    errors = []
    for name, row in snapshot.get("partitions", {}).items():
        if "gpu_type" not in row:
            continue
        physical = row.get("physical_configured_total")
        allocatable = row.get("allocatable_configured_total")
        allocated = row.get("allocated_estimate")
        idle = row.get("idle_estimate")
        for key, value in (("physical_configured_total", physical), ("allocatable_configured_total", allocatable), ("allocated_estimate", allocated), ("idle_estimate", idle)):
            if value is not None and value < 0:
                errors.append(f"{name}.{key} is negative")
        if None not in {physical, allocated} and allocated > physical:
            errors.append(f"{name}.allocated_estimate exceeds physical_configured_total")
        if None not in {allocatable, idle} and idle > allocatable:
            errors.append(f"{name}.idle_estimate exceeds allocatable_configured_total")
    return {"status": "success" if not errors else "failed", "errors": errors}


def discover_resource_snapshot(
    config_source: str | Path | None = None,
    *,
    account: str | None = None,
    qos: str | None = None,
    output_dir: str | Path | None = None,
    include_raw: bool = False,
) -> dict[str, Any]:
    raw: dict[str, str] = {}
    code, out, err = _run(["scontrol", "show", "nodes"])
    raw["scontrol_nodes_raw.txt"] = out or err
    nodes = parse_scontrol_nodes(out) if code == 0 else []
    sinfo_source = "unavailable"
    code, out, err = _run(["sinfo", "--json"])
    raw["sinfo_raw.txt"] = out or err
    sinfo = {}
    if code == 0:
        try:
            sinfo = parse_sinfo_json(out)
            sinfo_source = "sinfo_json"
        except Exception:
            sinfo = {}
    if not sinfo:
        code, out, err = _run(["sinfo", "-h", "-o", "%P|%a|%l|%D"])
        raw["sinfo_raw.txt"] = out or err
        if code == 0:
            sinfo = parse_sinfo_fallback(out)
            sinfo_source = "sinfo_text"
    code, out, err = _run(["sinfo", "-N", "-h", "-o", "%N|%P|%t|%G|%C|%m"])
    raw["sinfo_nodes_raw.txt"] = out or err
    queues, queue_source, squeue_raw = _queue_rows()
    raw["squeue_raw.txt"] = squeue_raw
    pending_ids = [q["job_id"] for q in queues if "PEND" in q.get("state", "").upper() or q.get("state") == "PD"]
    job_details, sample_raw = _job_details(pending_ids)
    raw["sample_pending_jobs_raw.txt"] = sample_raw
    cluster, cluster_source = _cluster_name()
    resolved_account, account_source, resolved_qos, qos_source = _account_qos(account, qos)
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    by_partition_queue: dict[str, list[dict[str, str]]] = defaultdict(list)
    for job in queues:
        by_partition_queue[job.get("partition", "")].append(job)
    by_partition_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in job_details:
        by_partition_details[str(detail.get("partition", ""))].append(detail)

    partition_rows: dict[str, Any] = {}
    for part in ("cpu", "gpu", "gpua100", "gpuh100"):
        part_nodes = [n for n in nodes if part in n.get("partitions", [])]
        schedulable = [n for n in part_nodes if n.get("schedulable")]
        qrows = by_partition_queue.get(part, [])
        pending = [j for j in qrows if "PEND" in j.get("state", "").upper() or j.get("state") == "PD"]
        running = [j for j in qrows if "RUN" in j.get("state", "").upper() or j.get("state") == "R"]
        if part == "cpu":
            cpus_total = sum(int(n.get("cpus") or 0) for n in schedulable)
            cpus_alloc = sum(int(n.get("alloc_cpus") or 0) for n in schedulable)
            partition_rows[part] = {
                "partition": part,
                "nodes_total": len(part_nodes),
                "nodes_idle": sum(1 for n in schedulable if str(n.get("state", "")).upper().startswith("IDLE")),
                "nodes_mixed": sum(1 for n in schedulable if "MIX" in str(n.get("state", "")).upper()),
                "nodes_allocated": sum(1 for n in schedulable if "ALLOC" in str(n.get("state", "")).upper()),
                "nodes_down_or_drain": len(part_nodes) - len(schedulable),
                "cpus_total": cpus_total,
                "cpus_idle_estimate": max(0, cpus_total - cpus_alloc),
                "memory_summary": {"total_mb": sum(int(n.get("real_memory_mb") or 0) for n in schedulable)},
                "time_limit": (sinfo.get(part) or {}).get("time") or (sinfo.get(part) or {}).get("time_limit"),
                "queued_jobs": len(pending),
                "running_jobs": len(running),
                "user_queued_jobs": sum(1 for j in pending if j.get("user") == user),
                "user_running_jobs": sum(1 for j in running if j.get("user") == user),
            }
            continue
        gpu_type = GPU_PARTITIONS[part]
        physical_nodes = [n for n in part_nodes if str(n.get("gpu_type") or "").upper() == gpu_type or n.get("gpus_configured", 0)]
        sched_gpu_nodes = [n for n in physical_nodes if n.get("schedulable")]
        physical_total = sum(int(n.get("gpus_configured") or 0) for n in physical_nodes)
        allocatable_total = sum(int(n.get("gpus_configured") or 0) for n in sched_gpu_nodes)
        allocated = min(physical_total, sum(int(n.get("gpus_allocated") or 0) for n in sched_gpu_nodes))
        idle = max(0, allocatable_total - allocated)
        free_by_node = [max(0, int(n.get("gpus_configured") or 0) - int(n.get("gpus_allocated") or 0)) for n in sched_gpu_nodes]
        qsummary = _queue_summary(pending, by_partition_details.get(part, []))
        partition_rows[part] = {
            "gpu_type": gpu_type,
            "partition": part,
            "nodes_total": len(part_nodes),
            "physical_configured_total": physical_total,
            "allocatable_configured_total": allocatable_total,
            "allocated_estimate": allocated,
            "idle_estimate": idle,
            "down_or_drain_gpu_count": sum(int(n.get("gpus_configured") or 0) for n in physical_nodes if not n.get("schedulable")),
            "unknown_state_gpu_count": sum(int(n.get("gpus_configured") or 0) for n in physical_nodes if not n.get("state")),
            "gpus_configured_total": physical_total,
            "gpus_allocated_estimate": allocated,
            "gpus_idle_estimate": idle,
            "nodes_with_at_least_1_free_gpu": sum(1 for v in free_by_node if v >= 1),
            "nodes_with_at_least_2_free_gpus": sum(1 for v in free_by_node if v >= 2),
            "nodes_with_at_least_4_free_gpus": sum(1 for v in free_by_node if v >= 4),
            "max_gpus_per_node": max([int(n.get("gpus_configured") or 0) for n in physical_nodes] or [0]),
            **qsummary,
            "running_jobs": len(running),
            "time_limit": (sinfo.get(part) or {}).get("time") or (sinfo.get(part) or {}).get("time_limit"),
        }
    snapshot = {
        "snapshot_time": utc_now(),
        "hostname": socket.gethostname(),
        "slurm_cluster_name": cluster,
        "cluster_source": cluster_source,
        "user": user,
        "account": resolved_account,
        "account_source": account_source,
        "qos": resolved_qos,
        "qos_source": qos_source,
        "queue_estimate_confidence": "lower" if account_source == "unresolved" or qos_source == "unresolved" else "normal",
        "config_source": str(config_source or ""),
        "sinfo_source": sinfo_source,
        "squeue_source": queue_source,
        "partitions": partition_rows,
        "limitations": ["This is an estimate and does not reserve resources."],
    }
    snapshot["invariants"] = validate_snapshot_invariants(snapshot)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out_dir / "resource_snapshot.json", snapshot)
        (out_dir / "resource_snapshot.md").write_text(snapshot_markdown(snapshot), encoding="utf-8")
        if include_raw:
            for name, text in raw.items():
                (out_dir / name).write_text(text, encoding="utf-8")
    return snapshot


def snapshot_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        f"Resource snapshot: {snapshot.get('snapshot_time')}",
        "",
        "This is an estimate and does not reserve resources.",
        "",
        "| Resource | Partition | Physical | Allocatable | Idle estimate | Valid nodes | Queue | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    cpu = snapshot.get("partitions", {}).get("cpu", {})
    lines.append(f"| CPU | cpu | {cpu.get('cpus_total', 0)} CPUs | {cpu.get('cpus_total', 0)} CPUs | {cpu.get('cpus_idle_estimate', 0)} CPUs | {cpu.get('nodes_idle', 0)} idle | {cpu.get('queued_jobs', 0)} | batch |")
    for part in ("gpu", "gpua100", "gpuh100"):
        row = snapshot.get("partitions", {}).get(part, {})
        lines.append(
            f"| {row.get('gpu_type', part)} | {part} | {row.get('physical_configured_total', 'unknown')} | "
            f"{row.get('allocatable_configured_total', 'unknown')} | {row.get('idle_estimate', 'unknown')} | "
            f">=1:{row.get('nodes_with_at_least_1_free_gpu', 0)} >=2:{row.get('nodes_with_at_least_2_free_gpus', 0)} >=4:{row.get('nodes_with_at_least_4_free_gpus', 0)} | "
            f"{row.get('queued_jobs', 0)} / gpu demand {row.get('queued_gpu_demand_total', 'unknown')} | shape estimate |"
        )
    return "\n".join(lines) + "\n"
