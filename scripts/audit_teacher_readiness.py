#!/usr/bin/env python3
"""Audit enabled teacher registry readiness without running heavy inference."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.model_registry import load_registry
from cli_anything.medai.core.registered_infer import run_registered_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit teacher registry/checkpoint readiness.")
    ap.add_argument("--registry", default=str(ROOT / "configs/model_registry.yaml"))
    ap.add_argument("--ct-image", default=str(ROOT / "data/PanTS/ImageTr/PanTS_00000026/ct.nii.gz"))
    ap.add_argument("--case-id", default="PanTS_00000026")
    ap.add_argument("--output", default=str(ROOT / "outputs/audit_21_models/teacher_readiness_audit.json"))
    ap.add_argument("--models", default="", help="Optional comma-separated model subset.")
    ap.add_argument("--include-disabled", action="store_true")
    return ap.parse_args()


def path_status(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    p = Path(str(value))
    if not p.is_absolute():
        p = ROOT / p
    return {"path": str(p.resolve()), "exists": p.exists(), "is_dir": p.is_dir(), "is_file": p.is_file()}


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry).resolve()
    registry = load_registry(registry_path)
    models_doc = registry.get("models", {})
    requested = [x.strip() for x in args.models.split(",") if x.strip()]
    keys = requested or sorted(models_doc.keys())
    rows: list[dict[str, Any]] = []

    for key in keys:
        entry = models_doc.get(key)
        if not entry:
            rows.append({"model_key": key, "status": "missing_registry_entry"})
            continue
        if entry.get("enabled") is False and not args.include_disabled:
            continue
        command_probe: dict[str, Any] = {}
        try:
            command_probe = run_registered_model(
                args.ct_image,
                ROOT / "outputs/audit_21_models/teacher_readiness_dryrun" / key,
                key,
                registry_path=registry_path,
                case_id=args.case_id,
                dry_run=True,
                timeout_sec=30,
                device="cuda",
            )
        except Exception as exc:
            command_probe = {"status": "failed", "reason": str(exc)}
        command = command_probe.get("command") or command_probe.get("commands")
        row = {
            "model_key": key,
            "enabled": entry.get("enabled", True),
            "registry_status": entry.get("status"),
            "runner": entry.get("runner"),
            "covered_organs_count": len(entry.get("covered_organs") or []),
            "covered_organs_sample": (entry.get("covered_organs") or [])[:20],
            "checkpoint_path": path_status(entry.get("checkpoint_path")),
            "source_code_path": path_status(entry.get("source_code_path")),
            "label_map_path": path_status(entry.get("label_map_path")),
            "dry_run_status": command_probe.get("status"),
            "dry_run_reason": command_probe.get("reason"),
            "dry_run_has_command": bool(command),
            "dry_run_command_sample": str(command)[:1000] if command else None,
            "dry_run_selected_subtasks": command_probe.get("selected_subtasks"),
            "dry_run_skipped_subtasks": command_probe.get("skipped_subtasks"),
        }
        rows.append(row)

    dependency_status = {
        "TotalSegmentator": shutil.which("TotalSegmentator"),
        "nnUNetv2_predict": shutil.which("nnUNetv2_predict"),
        "nnUNetv2_predict_from_modelfolder": shutil.which("nnUNetv2_predict_from_modelfolder"),
    }
    summary = {
        "stage": "teacher_readiness_audit",
        "status": "success",
        "registry": str(registry_path),
        "num_models": len(rows),
        "dependency_status": dependency_status,
        "models_ready_for_command": sum(1 for r in rows if r.get("dry_run_has_command")),
        "models_missing_checkpoint_path": [
            r["model_key"] for r in rows
            if r.get("checkpoint_path") and not r["checkpoint_path"].get("exists")
        ],
        "models": rows,
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "num_models": summary["num_models"],
        "models_ready_for_command": summary["models_ready_for_command"],
        "models_missing_checkpoint_path": summary["models_missing_checkpoint_path"][:20],
        "output": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
