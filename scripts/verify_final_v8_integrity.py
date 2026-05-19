#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "run_medai_cli.py",
    "configs/model_registry.yaml",
    "configs/class_checkpoint_map.xlsx",
    "configs/class_checkpoint_map.parsed.csv",
    "configs/atlasnet_label_map.json",
    "scripts/patch_epai_enable_all_organs.py",
    "scripts/prepare_colab_checkpoint_links.py",
    "scripts/validate_case_list_50.py",
    "agent-harness/cli_anything/medai/core/model_registry.py",
    "agent-harness/cli_anything/medai/core/mstep_runner.py",
    "agent-harness/cli_anything/medai/core/labelcritic_projection_runner.py",
    "agent-harness/cli_anything/medai/core/labelcritic_wrapper.py",
    "agent-harness/cli_anything/medai/core/report_supervision.py",
    "third_party/LabelCritic-main/CompareOrgan.py",
    "third_party/LabelCritic-main/ProjectDatasetFlex_single.py",
    "third_party/ShapeKit-main/main.py",
    "third_party/ePAI-main/README.md",
    "third_party/TotalSegmentator-master/README.md",
    "third_party/VISTA3D-Inference-Pipeline-master/README.md",
    "third_party/PanTS-main/README.md",
    "docs/FINAL_AUDIT_AND_USAGE_V8.md",
    "docs/START_ENVIRONMENT_AND_TRAINING_V8.md",
    "docs/raw_materials/meeting2.docx",
    "docs/raw_materials/task_zhou.docx",
]


def run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout, proc.stderr


def parse_json_stdout(out: str) -> dict:
    try:
        return json.loads(out)
    except Exception:
        return {"parse_error": out[:1000]}


def main() -> int:
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).exists()]

    pycache_count = len(list(ROOT.rglob("__pycache__")))
    pyc_count = len(list(ROOT.rglob("*.pyc")))

    rc_inventory, out_inventory, err_inventory = run([sys.executable, "run_medai_cli.py", "--json", "model-inventory"])
    inv = parse_json_stdout(out_inventory)

    rc_route, out_route, err_route = run([sys.executable, "run_medai_cli.py", "--json", "route-models", "--organs", "pancreas,liver,aorta,pancreatic_duct,kidney_cortex"])
    route = parse_json_stdout(out_route)

    rc_mstep_epai, out_mstep_epai, err_mstep_epai = run([
        sys.executable,
        "run_medai_cli.py",
        "--json",
        "mstep-update",
        "--training-manifest",
        "data_manifest/case_list_50_tumor_template.csv",
        "--output-folder",
        "/tmp/medai_v8_verify_epai",
        "--target-model",
        "epai_20250421",
        "--dry-run",
    ])
    epai = parse_json_stdout(out_mstep_epai)

    rc_mstep_totseg, out_mstep_totseg, err_mstep_totseg = run([
        sys.executable,
        "run_medai_cli.py",
        "--json",
        "mstep-update",
        "--training-manifest",
        "data_manifest/case_list_50_tumor_template.csv",
        "--output-folder",
        "/tmp/medai_v8_verify_totalseg",
        "--target-model",
        "totalsegmentator",
        "--dry-run",
    ])
    totseg = parse_json_stdout(out_mstep_totseg)

    checks = {
        "stage": "verify_final_v8_integrity",
        "status": "success" if not missing and rc_inventory == 0 and rc_route == 0 and rc_mstep_epai == 0 and rc_mstep_totseg == 0 else "error",
        "missing": missing,
        "pycache_dirs": pycache_count,
        "pyc_files": pyc_count,
        "inventory_status": inv.get("status"),
        "num_models_excluding_mock": inv.get("num_models_excluding_mock"),
        "num_trainable_or_finetunable_if_checkpoint_present": inv.get("num_trainable_or_finetunable_if_checkpoint_present"),
        "route_status": route.get("status"),
        "pancreas_primary": route.get("routing", {}).get("pancreas", {}).get("primary_model"),
        "pancreas_mstep_target": route.get("routing", {}).get("pancreas", {}).get("mstep_target_model"),
        "aorta_primary": route.get("routing", {}).get("aorta", {}).get("primary_model"),
        "aorta_mstep_target": route.get("routing", {}).get("aorta", {}).get("mstep_target_model"),
        "mstep_epai_status": epai.get("status"),
        "mstep_epai_backend": epai.get("mstep_backend"),
        "mstep_totalseg_status": totseg.get("status"),
        "mstep_totalseg_backend": totseg.get("mstep_backend"),
        "errors": {
            "inventory": err_inventory[-1000:],
            "route": err_route[-1000:],
            "mstep_epai": err_mstep_epai[-1000:],
            "mstep_totalseg": err_mstep_totseg[-1000:],
        },
    }
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0 if checks["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
