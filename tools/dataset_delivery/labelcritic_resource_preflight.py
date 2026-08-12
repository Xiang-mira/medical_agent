#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.labelcritic_wrapper import run_labelcritic_compare  # noqa: E402
from scheduler.resource_discovery import discover_resource_snapshot  # noqa: E402
from scheduler.resource_recommender import LABELCRITIC_72B_MODEL_ID, select_labelcritic_72b_profile  # noqa: E402
from tools.dataset_delivery.delivery_lib import write_json  # noqa: E402


def _http_health(base_url: str, *, timeout_sec: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/health"
    started = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
        return {"ok": True, "url": url, "latency_sec": round(time.time() - started, 3), "body": body[:500]}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "url": url, "latency_sec": round(time.time() - started, 3), "error": str(exc)}


def build_labelcritic_preflight(
    *,
    output_json: Path | None = None,
    base_url: str = "http://127.0.0.1:8000",
    ct_path: Path | None = None,
    mask_a: Path | None = None,
    mask_b: Path | None = None,
    organ: str = "lung_pulmonary_arteries",
    run_request: bool = False,
    timeout_sec: int = 180,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = snapshot or discover_resource_snapshot()
    selection = select_labelcritic_72b_profile(snapshot)
    checks: list[dict[str, Any]] = [
        {"name": "official_model_locked", "ok": selection["model_id"] == LABELCRITIC_72B_MODEL_ID, "value": selection["model_id"]},
        {"name": "no_silent_downgrade", "ok": selection.get("silent_downgrade") is False},
        {"name": "resource_profile_available", "ok": selection["status"] == "READY", "reason": selection.get("reason", "")},
    ]
    request_result: dict[str, Any] | None = None
    if run_request:
        for name, path in (("ct_path", ct_path), ("mask_a", mask_a), ("mask_b", mask_b)):
            checks.append({"name": name, "ok": bool(path and path.exists()), "path": str(path or "")})
        health = _http_health(base_url, timeout_sec=min(timeout_sec, 30))
        checks.append({"name": "api_health", "ok": bool(health.get("ok")), "details": health})
        if all(check["ok"] for check in checks) and ct_path and mask_a and mask_b:
            smoke_json = (output_json.parent if output_json else Path.cwd()) / "labelcritic_real_request_smoke.json"
            started = time.time()
            request_result = run_labelcritic_compare(
                ct_path,
                mask_a,
                mask_b,
                organ,
                smoke_json,
                backend="labelcritic",
                base_url=base_url.rsplit(":", 1)[0],
                port=int(base_url.rsplit(":", 1)[1]) if ":" in base_url.rsplit("/", 1)[-1] else 8000,
                dry_run=False,
                timeout_sec=timeout_sec,
                candidate_context=[
                    {"candidate_id": "smoke_a", "qc_status": "pass"},
                    {"candidate_id": "smoke_b", "qc_status": "pass"},
                ],
            )
            request_result["latency_sec"] = round(time.time() - started, 3)
            checks.append({"name": "real_labelcritic_request", "ok": request_result.get("status") == "success", "details": request_result})
    status = "READY" if all(check["ok"] for check in checks) else ("insufficient_resource" if selection["status"] == "insufficient_resource" else "BLOCKED")
    report = {
        "status": status,
        "model_id": LABELCRITIC_72B_MODEL_ID,
        "resource_selection": selection,
        "checks": checks,
        "run_request": run_request,
        "request_result": request_result,
        "formal_backend": True,
        "silent_model_downgrade_allowed": False,
    }
    if output_json:
        write_json(output_json, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal LabelCritic 72B resource and real-request preflight.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ct-path", type=Path)
    parser.add_argument("--mask-a", type=Path)
    parser.add_argument("--mask-b", type=Path)
    parser.add_argument("--organ", default="lung_pulmonary_arteries")
    parser.add_argument("--run-request", action="store_true")
    parser.add_argument("--timeout-sec", default=180, type=int)
    args = parser.parse_args()
    report = build_labelcritic_preflight(
        output_json=args.output_json.resolve() if args.output_json else None,
        base_url=args.base_url,
        ct_path=args.ct_path.resolve() if args.ct_path else None,
        mask_a=args.mask_a.resolve() if args.mask_a else None,
        mask_b=args.mask_b.resolve() if args.mask_b else None,
        organ=args.organ,
        run_request=bool(args.run_request),
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps({"status": report["status"], "model_id": report["model_id"], "resource_selection": report["resource_selection"]}, indent=2))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
