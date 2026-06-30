#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "VoxTell"


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return {
        "cmd": cmd,
        "return_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def build_audit() -> dict:
    audit = {
        "stage": "voxtell_vendor_audit",
        "official_repo": "https://github.com/MIC-DKFZ/VoxTell",
        "vendor_path": str(VENDOR),
        "protected_vendor_paths": [
            "voxtell/model/*",
            "voxtell/inference/*",
            "voxtell/utils/*",
        ],
        "adapter_policy": "Keep project-specific CLI, scoring, manifests, and EM integration outside third_party/VoxTell.",
    }
    if not VENDOR.exists():
        audit.update({"status": "failed", "reason": "vendor_path_missing"})
        return audit
    commit = _run(["git", "-C", str(VENDOR), "rev-parse", "--short", "HEAD"])
    status = _run(["git", "-C", str(VENDOR), "status", "--short"])
    remote = _run(["git", "-C", str(VENDOR), "remote", "-v"])
    import_check = _run([
        sys.executable,
        "-c",
        f"import sys, pathlib; sys.path.insert(0, {str(VENDOR)!r}); import voxtell; print(pathlib.Path(voxtell.__file__).resolve())",
    ])
    training_import_check = _run([
        sys.executable,
        "-c",
        f"import sys; sys.path.insert(0, {str(VENDOR)!r}); import voxtell.training.run_finetuning as r; from voxtell.training import VoxTellTrainer, VoxTellTrainer_noMirroring; print(r, VoxTellTrainer, VoxTellTrainer_noMirroring)",
    ])
    pyproject = VENDOR / "pyproject.toml"
    pyproject_text = pyproject.read_text(encoding="utf-8") if pyproject.exists() else ""
    dirty_files = [line for line in status["stdout"].splitlines() if line.strip()]
    audit.update({
        "status": "ok" if status["return_code"] == 0 and commit["return_code"] == 0 else "failed",
        "commit": commit["stdout"] or None,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "remote": remote["stdout"],
        "import_path": import_check["stdout"] or None,
        "import_check_return_code": import_check["return_code"],
        "import_check_stderr": import_check["stderr"],
        "voxtell_predict_entrypoint": shutil.which("voxtell-predict"),
        "voxtell_finetune_entrypoint": shutil.which("voxtell-finetune"),
        "pyproject_has_voxtell_predict": "voxtell-predict" in pyproject_text,
        "pyproject_has_voxtell_finetune": "voxtell-finetune" in pyproject_text,
        "training_import_return_code": training_import_check["return_code"],
        "training_import_stdout": training_import_check["stdout"],
        "training_import_stderr": training_import_check["stderr"],
        "fine_tune_available": bool(shutil.which("voxtell-finetune")) and training_import_check["return_code"] == 0,
        "fine_tune_semantics": "official_voxtell_nnunet_encoder_baseline: transfers VoxTell image encoder into nnU-Net multi-class segmentation; baseline/ablation only, not prompt-conditioned VoxTell Student training.",
    })
    return audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    audit = build_audit()
    text = json.dumps(audit, indent=2, ensure_ascii=False)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if audit.get("status") == "ok" and not audit.get("dirty") else 1


if __name__ == "__main__":
    raise SystemExit(main())
