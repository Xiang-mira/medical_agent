#!/usr/bin/env python3
"""Offline recompute for VoxTell M-step sampling/gradient audit artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_voxtell_prompt_student import (  # noqa: E402
    build_sampling_gradient_audits,
    write_csv,
    write_json,
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--loss-history", type=Path, required=True)
    ap.add_argument("--sampling-audit", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = read_json(args.manifest)
    loss_doc = read_json(args.loss_history)
    sampling_doc = read_json(args.sampling_audit) if args.sampling_audit else {}
    legacy_organ_audit = sampling_doc.get("organ_gradient_audit") or {}

    bundle = build_sampling_gradient_audits(
        training_rows=manifest.get("items") or [],
        loss_history=loss_doc.get("history") or [],
        legacy_organ_sample_counts=legacy_organ_audit.get("sample_counts") or {},
        legacy_gradient_shares=legacy_organ_audit.get("gradient_shares") or {},
    )
    out = args.output_dir.resolve()
    payload = {
        "stage": "offline_voxtell_sampling_gradient_audit_recompute",
        "status": "completed",
        "manifest": str(args.manifest.resolve()),
        "loss_history": str(args.loss_history.resolve()),
        "sampling_audit": str(args.sampling_audit.resolve()) if args.sampling_audit else None,
        "legacy_audit_status": legacy_organ_audit.get("status"),
        "legacy_underrepresented_organs": legacy_organ_audit.get("underrepresented_organs", []),
        "organ_gradient_audit": bundle["organ_gradient_audit"],
        "organ_exposure_audit": bundle["organ_exposure_audit"],
        "nonfinite_gradient_audit": bundle["nonfinite_gradient_audit"],
        "student_sampling_gradient_diagnosis": bundle["student_sampling_gradient_diagnosis"],
        "interpretation": (
            "No GT is used. For legacy loss histories without positive_organs "
            "metadata, this tool diagnoses the prior audit failure but cannot "
            "recover exact finite per-organ gradient mass."
        ),
    }
    write_json(out / "student_sampling_gradient_diagnosis.json", payload["student_sampling_gradient_diagnosis"])
    write_json(out / "organ_exposure_audit.json", payload["organ_exposure_audit"])
    write_json(out / "nonfinite_gradient_audit.json", payload["nonfinite_gradient_audit"])
    write_json(out / "organ_gradient_audit.json", payload["organ_gradient_audit"])
    write_json(out / "offline_sampling_gradient_recompute_summary.json", payload)
    write_csv(out / "organ_exposure_audit.csv", bundle["organ_rows"])
    write_csv(out / "student_sampling_gradient_diagnosis.csv", bundle["organ_rows"])
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
