#!/usr/bin/env python3
"""Run a small no-GT LabelCritic model/prompt/projection benchmark matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/em_round1_25case_pseudo_label_20260709/round1/mstep/voxtell_prompt_student_manifest.json"
DEFAULT_MODELS = [
    ROOT / "checkpoints/Qwen/Qwen2-VL-7B-Instruct",
    ROOT / "checkpoints/Qwen/Qwen2.5-VL-7B-Instruct",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def served_model(base_url: str, port: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception as exc:
        return {"online": False, "url": url, "reason": repr(exc)}
    ids = [str(row.get("id") or "") for row in body.get("data", []) if isinstance(row, dict)]
    return {"online": True, "url": url, "model_ids": ids, "raw": body}


def model_matches(served_ids: list[str], model: Path) -> bool:
    model_s = str(model.resolve())
    name = model.name
    return any(model_s in sid or name in sid for sid in served_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--models", type=Path, nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--prompt-modes", nargs="*", default=["official_dual", "strict_single"])
    ap.add_argument("--projection-modes", nargs="*", default=["ap", "multiview_audit"])
    ap.add_argument("--organs", nargs="*", default=["liver", "kidney_left", "aorta", "spleen", "stomach"])
    ap.add_argument("--corruptions", nargs="*", default=["random_blob"])
    ap.add_argument("--max-per-organ", type=int, default=1)
    ap.add_argument("--base-url", default="http://localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--allow-model-mismatch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = args.output_dir.resolve()
    service = served_model(args.base_url, args.port)
    rows: list[dict[str, Any]] = []
    if not service.get("online") and not args.dry_run:
        payload = {
            "stage": "labelcritic_benchmark_matrix",
            "status": "blocked",
            "blocker": "vllm_offline",
            "service": service,
            "rows": rows,
        }
        write_json(out / "labelcritic_repair_benchmark_summary.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2

    served_ids = list(service.get("model_ids") or [])
    for model in args.models:
        model = model.resolve()
        if not model.exists():
            rows.append({"model": str(model), "status": "skipped", "reason": "model_path_missing"})
            continue
        if not args.allow_model_mismatch and not args.dry_run and not model_matches(served_ids, model):
            rows.append({
                "model": str(model),
                "status": "skipped",
                "reason": "served_vllm_model_mismatch",
                "served_model_ids": served_ids,
            })
            continue
        for prompt_mode in args.prompt_modes:
            for projection_mode in args.projection_modes:
                combo_dir = out / model.name / prompt_mode / projection_mode
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts/benchmark_labelcritic_synthetic.py"),
                    "--manifest", str(args.manifest),
                    "--output-dir", str(combo_dir),
                    "--max-per-organ", str(args.max_per_organ),
                    "--base-url", args.base_url,
                    "--port", str(args.port),
                    "--timeout-sec", str(args.timeout_sec),
                    "--prompt-mode", prompt_mode,
                    "--projection-mode", projection_mode,
                    "--model-id", str(model),
                    "--organs", *args.organs,
                    "--corruptions", *args.corruptions,
                ]
                if args.dry_run:
                    cmd.append("--dry-run")
                proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
                summary_path = combo_dir / "labelcritic_synthetic_benchmark_summary.json"
                summary = {}
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except Exception:
                        summary = {}
                rows.append({
                    "model": str(model),
                    "prompt_mode": prompt_mode,
                    "projection_mode": projection_mode,
                    "status": summary.get("status") or "failed",
                    "pass": bool(summary.get("pass")),
                    "blocker": summary.get("blocker"),
                    "summary_path": str(summary_path),
                    "return_code": proc.returncode,
                    "stdout_tail": proc.stdout[-1000:],
                    "stderr_tail": proc.stderr[-1000:],
                    "projection_success_rate": summary.get("projection_success_rate"),
                    "parser_success_rate": summary.get("parser_success_rate"),
                    "ab_ba_consistency_rate": summary.get("ab_ba_consistency_rate"),
                    "known_better_pick_rate": summary.get("known_better_pick_rate"),
                })

    any_pass = any(bool(row.get("pass")) for row in rows)
    all_dry_run = bool(rows) and all(row.get("status") == "dry_run" for row in rows)
    payload = {
        "stage": "labelcritic_benchmark_matrix",
        "status": "passed" if any_pass else "dry_run" if all_dry_run else "failed",
        "blocker": None if any_pass or all_dry_run else "model_or_prompt_capacity_blocker",
        "service": service,
        "manifest": str(args.manifest),
        "rows": rows,
        "policy": "No GT is used. Formal reselection is allowed only if at least one matrix cell passes known-better checks.",
    }
    write_json(out / "labelcritic_repair_benchmark_summary.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2, ensure_ascii=False))
    return 0 if any_pass or all_dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
