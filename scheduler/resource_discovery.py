from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import SchedulerError, utc_now


GPU_PARTITIONS = {
    "gpu": "T4",
    "gpua100": "A100",
    "gpuh100": "H100",
}
BAD_NODE_STATES = ("DOWN", "DRAIN", "DRAINING", "FAIL", "MAINT")


def _run(args: list[str]) -> tuple[int, str, str]:
    if shutil.which(args[0]) is None:
        return 127, "", f"{args[0]} unavailable"
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _parse_tres(text: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    if not text:
        return out
    for item in str(text).split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            out[key] = int(float(value))
        except ValueError:
            continue
    return out


def parse_scontrol_nodes(text: str) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for block in re.split(r"\n(?=NodeName=)", text.strip()):
        if not block.strip():
            continue
        pairs = dict(re.findall(r"(\S+?)=(\S+)", block.replace("\n   ", " ")))
        gres = pairs.get("Gres", "")
        gpu_type = None
        configured_gpu = 0
        m = re.search(r"gpu:([A-Za-z0-9_]+):(\d+)", gres)
        if m:
            gpu_type = m.group(1).upper()
            configured_gpu = int(m.group(2))
        cfg_tres = _parse_tres(pairs.get("CfgTRES"))
        alloc_tres = _parse_tres(pairs.get("AllocTRES"))
        configured_gpu = configured_gpu or cfg_tres.get("gres/gpu", 0)
        allocated_gpu = alloc_tres.get("gres/gpu", 0)
        nodes.append(
            {
                "name": pairs.get("NodeName", ""),
                "state": pairs.get("State", ""),
                "partitions": pairs.get("Partitions", "").split(",") if pairs.get("Partitions") else [],
                "cpus": int(pairs.get("CPUTot", "0") or 0),
                "alloc_cpus": int(pairs.get("CPUAlloc", "0") or 0),
                "real_memory_mb": int(pairs.get("RealMemory", "0") or 0),
                "alloc_memory_mb": int(pairs.get("AllocMem", "0") or 0),
                "gpu_type": gpu_type,
                "gpus_configured": configured_gpu,
                "gpus_allocated": allocated_gpu,
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
            partitions[str(part)] = item
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
        if len(parts) >= 6:
            rows.append({"job_id": parts[0], "partition": parts[1], "state": parts[2], "user": parts[3], "reason": parts[4], "tres": parts[5]})
    return rows


def _queue_rows() -> tuple[list[dict[str, str]], str]:
    code, out, _ = _run(["squeue", "--json"])
    if code == 0:
        try:
            doc = json.loads(out)
            rows = []
            for job in doc.get("jobs", []):
                rows.append(
                    {
                        "job_id": str(job.get("job_id", "")),
                        "partition": str(job.get("partition") or ""),
                        "state": str(job.get("job_state") or job.get("state") or ""),
                        "user": str(job.get("user_name") or job.get("user") or ""),
                        "reason": str(job.get("state_reason") or job.get("reason") or ""),
                        "tres": str(job.get("tres_req_str") or job.get("tres_alloc_str") or ""),
                    }
                )
            return rows, "squeue_json"
        except Exception:
            pass
    code, out, _ = _run(["squeue", "-h", "-o", "%i|%P|%T|%u|%R|%b"])
    if code == 0:
        return parse_squeue_fallback(out), "squeue_text"
    return [], "unavailable"


def discover_resource_snapshot(config_source: str | Path | None = None, *, account: str | None = None, qos: str | None = None) -> dict[str, Any]:
    code, out, _ = _run(["scontrol", "show", "nodes"])
    nodes = parse_scontrol_nodes(out) if code == 0 else []
    sinfo_source = "unavailable"
    code, out, _ = _run(["sinfo", "--json"])
    sinfo = {}
    if code == 0:
        try:
            sinfo = parse_sinfo_json(out)
            sinfo_source = "sinfo_json"
        except Exception:
            sinfo = {}
    if not sinfo:
        code, out, _ = _run(["sinfo", "-h", "-o", "%P|%a|%l|%D"])
        if code == 0:
            sinfo = parse_sinfo_fallback(out)
            sinfo_source = "sinfo_text"
    queues, queue_source = _queue_rows()
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    by_partition_queue: dict[str, list[dict[str, str]]] = defaultdict(list)
    for job in queues:
        by_partition_queue[job.get("partition", "")].append(job)

    partition_rows: dict[str, Any] = {}
    for part in ("cpu", "gpu", "gpua100", "gpuh100"):
        part_nodes = [n for n in nodes if part in n.get("partitions", [])]
        usable = [n for n in part_nodes if not any(s in str(n.get("state", "")).upper() for s in BAD_NODE_STATES)]
        qrows = by_partition_queue.get(part, [])
        pending = [j for j in qrows if "PEND" in j.get("state", "").upper() or j.get("state") == "PD"]
        running = [j for j in qrows if "RUN" in j.get("state", "").upper() or j.get("state") == "R"]
        if part == "cpu":
            cpus_total = sum(int(n.get("cpus") or 0) for n in usable)
            cpus_alloc = sum(int(n.get("alloc_cpus") or 0) for n in usable)
            partition_rows[part] = {
                "partition": part,
                "nodes_total": len(part_nodes),
                "nodes_idle": sum(1 for n in usable if str(n.get("state", "")).upper().startswith("IDLE")),
                "nodes_mixed": sum(1 for n in usable if "MIX" in str(n.get("state", "")).upper()),
                "nodes_allocated": sum(1 for n in usable if "ALLOC" in str(n.get("state", "")).upper()),
                "nodes_down_or_drain": len(part_nodes) - len(usable),
                "cpus_total": cpus_total,
                "cpus_idle_estimate": max(0, cpus_total - cpus_alloc),
                "memory_summary": {"total_mb": sum(int(n.get("real_memory_mb") or 0) for n in usable)},
                "time_limit": (sinfo.get(part) or {}).get("time") or (sinfo.get(part) or {}).get("time_limit"),
                "queued_jobs": len(pending),
                "running_jobs": len(running),
                "user_queued_jobs": sum(1 for j in pending if j.get("user") == user),
                "user_running_jobs": sum(1 for j in running if j.get("user") == user),
            }
        else:
            gpu_type = GPU_PARTITIONS[part]
            gpu_nodes = [n for n in usable if str(n.get("gpu_type") or "").upper() == gpu_type or n.get("gpus_configured", 0)]
            free_by_node = [int(n.get("gpus_free") or 0) for n in gpu_nodes]
            partition_rows[part] = {
                "gpu_type": gpu_type,
                "partition": part,
                "nodes_total": len(part_nodes),
                "gpus_configured_total": sum(int(n.get("gpus_configured") or 0) for n in gpu_nodes),
                "gpus_allocated_estimate": sum(int(n.get("gpus_allocated") or 0) for n in gpu_nodes),
                "gpus_idle_estimate": sum(free_by_node),
                "nodes_with_at_least_1_free_gpu": sum(1 for v in free_by_node if v >= 1),
                "nodes_with_at_least_2_free_gpus": sum(1 for v in free_by_node if v >= 2),
                "nodes_with_at_least_4_free_gpus": sum(1 for v in free_by_node if v >= 4),
                "max_gpus_per_node": max([int(n.get("gpus_configured") or 0) for n in gpu_nodes] or [0]),
                "queued_jobs": len(pending),
                "queued_gpu_demand": sum(_parse_tres(j.get("tres")).get("gres/gpu", 0) for j in pending),
                "running_jobs": len(running),
                "time_limit": (sinfo.get(part) or {}).get("time") or (sinfo.get(part) or {}).get("time_limit"),
            }
    return {
        "snapshot_time": utc_now(),
        "hostname": socket.gethostname(),
        "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME", "unknown"),
        "user": user,
        "account": account,
        "qos": qos,
        "config_source": str(config_source or ""),
        "sinfo_source": sinfo_source,
        "squeue_source": queue_source,
        "partitions": partition_rows,
        "limitations": ["This is an estimate and does not reserve resources."],
    }


def snapshot_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        f"Resource snapshot: {snapshot.get('snapshot_time')}",
        "",
        "This is an estimate and does not reserve resources.",
        "",
        "| Resource | Partition | Configured | Idle estimate | Valid nodes | Queue | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    cpu = snapshot.get("partitions", {}).get("cpu", {})
    lines.append(f"| CPU | cpu | {cpu.get('cpus_total', 0)} CPUs | {cpu.get('cpus_idle_estimate', 0)} CPUs | {cpu.get('nodes_idle', 0)} idle nodes | {cpu.get('queued_jobs', 0)} | batch |")
    for part in ("gpu", "gpua100", "gpuh100"):
        row = snapshot.get("partitions", {}).get(part, {})
        lines.append(
            f"| {row.get('gpu_type', part)} | {part} | {row.get('gpus_configured_total', 0)} GPUs | "
            f"{row.get('gpus_idle_estimate', 0)} GPUs | >=1:{row.get('nodes_with_at_least_1_free_gpu', 0)} "
            f">=2:{row.get('nodes_with_at_least_2_free_gpus', 0)} >=4:{row.get('nodes_with_at_least_4_free_gpus', 0)} | "
            f"{row.get('queued_jobs', 0)} | allocatable shape estimate |"
        )
    return "\n".join(lines) + "\n"
