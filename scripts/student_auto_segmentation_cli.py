#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.voxtell_student import VoxTellStudent


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone project Student CT auto-segmentation. No teachers, AutoLabelCore, E-step, or LabelCritic are used.")
    ap.add_argument("--model-dir", required=True, help="Exported final Student model dir or export root containing model/")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-config", default=str(ROOT / "configs/student_3d_prompt_target_organs.json"))
    ap.add_argument("--organs", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    model = Path(args.model_dir)
    if (model / "model").exists():
        model = model / "model"
    student = VoxTellStudent(model_dir=model, target_config=args.target_config, device=args.device)
    result = student.segment(
        ct_image=args.image,
        output_dir=args.output_dir,
        prompts=args.organs,
        dry_run=args.dry_run,
    )
    result.update({
        "stage": "student_auto_segmentation_cli",
        "deployment_mode": "final_student_only",
        "uses_teacher": False,
        "uses_autolabelcore": False,
        "uses_estep": False,
        "uses_labelcritic": False,
    })
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "student_auto_segmentation_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if args.dry_run or result.get("status") in {"success", "partial_success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
