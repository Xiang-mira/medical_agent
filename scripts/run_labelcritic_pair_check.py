#!/usr/bin/env python3
"""Check one LabelCritic A/B comparison pair.

Use `--backend stub` to validate projection/artifact generation without a VLM
server. Use `--backend labelcritic` when an OpenAI-compatible VLM endpoint is
online, for example `http://localhost:8000/v1/models`.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.labelcritic_wrapper import run_labelcritic_compare


def parse_args() -> argparse.Namespace:
    default_base = ROOT / "outputs/audit_21_models/real_teacher_subset_e2e/estep_epai_mock/cases/PanTS_00000026/raw_predictions"
    ap = argparse.ArgumentParser(description="Run one LabelCritic pair check.")
    ap.add_argument("--ct", default=str(ROOT / "data/PanTS/ImageTr/PanTS_00000026/ct.nii.gz"))
    ap.add_argument("--mask-a", default=str(default_base / "epai_20250421/PanTS_00000026/segmentations/pancreas.nii.gz"))
    ap.add_argument("--mask-b", default=str(default_base / "mock_seg/PanTS_00000026/segmentations/pancreas.nii.gz"))
    ap.add_argument("--organ", default="pancreas")
    ap.add_argument("--output-json", default=str(ROOT / "outputs/audit_21_models/labelcritic_pair_check/pancreas_epai_vs_mock.json"))
    ap.add_argument("--backend", default="stub", choices=["stub", "labelcritic"])
    ap.add_argument("--base-url", default="http://localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout-sec", type=int, default=300)
    ap.add_argument("--no-dice-check", action="store_true", default=False, help="Ablation: force VLM comparison even when 2D projections are similar.")
    ap.add_argument("--no-dual-confirmation", action="store_true", default=False, help="Ablation: use non-dual prompt path.")
    ap.add_argument("--simple-prompt-ablation", action="store_true", default=False, help="Ablation: remove organ-specific prompt details.")
    ap.add_argument("--conservative-dual", action="store_true", default=False, help="Use stricter dual-confirmation parsing.")
    ap.add_argument("--skip-organ-presence-gate", action="store_true", default=False, help="Diagnostic: bypass LabelCritic's initial organ-presence gate.")
    ap.add_argument("--strict-choice-prompt", action="store_true", default=False, help="Diagnostic: force overlay 1/overlay 2/tie answer format.")
    return ap.parse_args()


def endpoint_status(base_url: str, port: int) -> dict:
    url = f"{base_url.rstrip('/')}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            body = resp.read(1000).decode("utf-8", errors="ignore")
        return {"online": True, "url": url, "body_sample": body[:500]}
    except Exception as exc:
        return {"online": False, "url": url, "reason": repr(exc)}


def main() -> int:
    args = parse_args()
    service = endpoint_status(args.base_url, args.port)
    result = run_labelcritic_compare(
        args.ct,
        args.mask_a,
        args.mask_b,
        args.organ,
        args.output_json,
        backend=args.backend,
        base_url=args.base_url,
        port=args.port,
        dry_run=False,
        timeout_sec=args.timeout_sec,
        no_dice_check=args.no_dice_check,
        no_dual_confirmation=args.no_dual_confirmation,
        simple_prompt_ablation=args.simple_prompt_ablation,
        conservative_dual=args.conservative_dual,
        skip_organ_presence_gate=args.skip_organ_presence_gate,
        strict_choice_prompt=args.strict_choice_prompt,
    )
    summary = {
        "stage": "labelcritic_pair_check",
        "status": result.get("status"),
        "backend": args.backend,
        "service": service,
        "decision": result.get("decision"),
        "labelcritic_options": result.get("labelcritic_options"),
        "output_json": str(Path(args.output_json).resolve()),
        "projection_status": (result.get("projection") or {}).get("status"),
        "accuracy_warning": "This checks A/B selection plumbing; it is not a segmentation accuracy metric.",
    }
    summary_path = Path(args.output_json).resolve().with_name(Path(args.output_json).stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.backend == "labelcritic" and not service.get("online"):
        return 2
    return 0 if result.get("status") in {"success", "stub", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
