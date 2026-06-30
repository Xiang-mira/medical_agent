#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha1_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def export_final_student(round_dir: Path, output_dir: Path, target_config: Path | None = None) -> dict[str, Any]:
    mstep = round_dir / "mstep"
    model_dir = mstep / "voxtell_finetuned_model"
    result_path = mstep / "voxtell_prompt_mstep_result.json"
    manifest_path = mstep / "voxtell_prompt_student_manifest.json"
    result = read_json(result_path, {})
    plans = model_dir / "plans.json"
    ckpt = model_dir / "fold_0" / "checkpoint_final.pth"
    ready = bool(
        plans.exists()
        and ckpt.exists()
        and result.get("eligible_for_next_round_prompt_student", result.get("checkpoint_eligible_for_next_round"))
        and result.get("canonical_training_backend") in {None, "project_voxtell_prompt_distillation_student"}
        and result.get("is_official_voxtell_encoder_transfer") is not True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if ready:
        dst_model = output_dir / "model"
        if dst_model.exists():
            shutil.rmtree(dst_model)
        shutil.copytree(model_dir, dst_model)
    export = {
        "stage": "final_student_export",
        "status": "success" if ready else "not_ready",
        "ready_for_standalone_inference": ready,
        "round_dir": str(round_dir.resolve()),
        "source_model_dir": str(model_dir),
        "export_model_dir": str((output_dir / "model").resolve()) if ready else None,
        "mstep_result": str(result_path),
        "manifest": str(manifest_path),
        "manifest_sha1": sha1_file(manifest_path),
        "checkpoint_sha1": sha1_file(ckpt),
        "target_config": str(target_config.resolve()) if target_config else None,
        "training_lineage": {
            "canonical_training_backend": result.get("canonical_training_backend"),
            "trainer": result.get("trainer"),
            "uses_official_voxtell_model": result.get("uses_official_voxtell_model"),
            "uses_autolabelcore_confidence": result.get("uses_autolabelcore_confidence"),
            "uses_abcd_training_weight": result.get("uses_abcd_training_weight"),
        },
        "reason": None if ready else "missing checkpoint/plans or prompt-student quality gate not passed",
    }
    (output_dir / "final_student_export.json").write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    return export


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a quality-gated project VoxTell prompt Student for standalone inference.")
    ap.add_argument("--round-dir", required=True, help="Round directory containing mstep/voxtell_finetuned_model")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-config", default=str(ROOT / "configs/student_3d_prompt_target_organs.json"))
    args = ap.parse_args()
    result = export_final_student(Path(args.round_dir), Path(args.output_dir), Path(args.target_config))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ready_for_standalone_inference"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
