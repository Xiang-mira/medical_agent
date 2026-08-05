from __future__ import annotations

import re
import shutil
import subprocess
from statistics import median
from typing import Any


def sbatch_supports_test_only() -> bool:
    if shutil.which("sbatch") is None:
        return False
    proc = subprocess.run(["sbatch", "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return "--test-only" in (proc.stdout + proc.stderr)


def estimate_from_sbatch_test_only(script_path: str) -> dict[str, Any]:
    if not sbatch_supports_test_only():
        return {"wait_estimate": "unknown", "estimate_method": "sbatch_test_only_unavailable", "confidence": "low", "sample_count": 0}
    proc = subprocess.run(["sbatch", "--test-only", script_path], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    text = (proc.stdout + "\n" + proc.stderr).strip()
    match = re.search(r"start(?:ing)? at ([^\n]+)", text, flags=re.I)
    return {
        "wait_estimate": "slurm_start_estimate" if match else "unknown",
        "slurm_estimated_start": match.group(1).strip() if match else None,
        "estimate_method": "sbatch_test_only",
        "confidence": "medium" if match else "low",
        "sample_count": 1 if match else 0,
        "limitations": ["Slurm test-only is an estimate and does not submit or reserve resources."],
    }


def estimate_from_history(delays_seconds: list[float]) -> dict[str, Any]:
    if len(delays_seconds) < 3:
        return {"wait_estimate": "unknown", "estimate_method": "history", "confidence": "low", "sample_count": len(delays_seconds)}
    values = sorted(float(x) for x in delays_seconds)
    def pct(q: float) -> float:
        return values[min(len(values) - 1, int(round((len(values) - 1) * q)))]
    return {
        "estimated_wait_p50": median(values),
        "estimated_wait_p80": pct(0.8),
        "estimated_wait_p90": pct(0.9),
        "estimate_method": "history",
        "confidence": "medium" if len(values) >= 10 else "low",
        "sample_count": len(values),
        "limitations": ["Historical queue delay is not a guarantee for current Slurm priority/backfill behavior."],
    }


def heuristic_wait(partition_row: dict[str, Any], *, gpu_count: int = 0) -> dict[str, Any]:
    if not partition_row:
        return {"wait_estimate": "unknown", "estimate_method": "current_queue_heuristic", "confidence": "low", "sample_count": 0}
    if gpu_count:
        shape_key = f"nodes_with_at_least_{gpu_count}_free_gpus" if gpu_count in {1, 2, 4} else "nodes_with_at_least_1_free_gpu"
        allocatable = int(partition_row.get(shape_key) or 0)
        queue_expected = allocatable <= 0
    else:
        queue_expected = int(partition_row.get("cpus_idle_estimate") or 0) <= 0
    return {
        "estimated_wait_p50": "unknown" if queue_expected else 0,
        "estimated_wait_p80": "unknown" if queue_expected else 0,
        "estimated_wait_p90": "unknown" if queue_expected else 0,
        "estimate_method": "current_queue_heuristic",
        "confidence": "low",
        "sample_count": int(partition_row.get("queued_jobs") or 0),
        "queue_expected": queue_expected,
        "limitations": ["Current idle resources may be taken before submission; dependency, QOS and priority can dominate waiting time."],
    }
