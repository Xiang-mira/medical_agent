from __future__ import annotations

import csv
import io
import subprocess
import time
from typing import Any


def inspect_gpu_resources(min_free_mib: int = 40000, max_utilization: int = 10, samples: int = 1, interval_sec: float = 0.0) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for sample in range(max(1, samples)):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if gpu.returncode != 0:
            return {"status": "skipped_resource_unavailable", "reason": gpu.stderr.strip() or "nvidia-smi failed"}
        rows = list(csv.reader(io.StringIO(gpu.stdout)))
        parsed = [{
            "index": int(row[0].strip()), "name": row[1].strip(), "total_mib": int(row[2]),
            "used_mib": int(row[3]), "free_mib": int(row[4]), "utilization_pct": int(row[5]),
        } for row in rows if len(row) >= 6]
        process_rows = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
        observations.append({"gpus": parsed, "compute_processes": process_rows})
        if sample + 1 < samples and interval_sec > 0:
            time.sleep(interval_sec)
    busy_reasons: list[str] = []
    for observation in observations:
        if observation["compute_processes"]:
            busy_reasons.append("foreign_compute_process_present")
        for gpu in observation["gpus"]:
            if gpu["free_mib"] < min_free_mib:
                busy_reasons.append(f"gpu_{gpu['index']}_free_memory_below_{min_free_mib}_mib")
            if gpu["utilization_pct"] > max_utilization:
                busy_reasons.append(f"gpu_{gpu['index']}_utilization_above_{max_utilization}_pct")
    return {
        "status": "ready" if not busy_reasons else "skipped_resource_busy",
        "busy_reasons": sorted(set(busy_reasons)),
        "policy": {"min_free_mib": min_free_mib, "max_utilization_pct": max_utilization, "foreign_process_policy": "block"},
        "observations": observations,
    }
