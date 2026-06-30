#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.voxtell_nnunet_encoder import convert_manifest_to_nnunet_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert MedAI VoxTell prompt manifest to nnU-Net raw dataset for official VoxTell encoder-transfer fine-tune.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--dataset-id", required=True, type=int)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--target-config", default="configs/student_3d_prompt_target_organs.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    result = convert_manifest_to_nnunet_dataset(
        manifest_path=args.manifest,
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        target_config=args.target_config,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in {"success", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
