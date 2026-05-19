#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "run_medai_cli.py",
    "configs/model_registry.yaml",
    "agent-harness/cli_anything/medai/core/mstep_runner.py",
    "agent-harness/cli_anything/medai/core/model_registry.py",
    "docs/SELECTED_MODEL_AWARE_MSTEP_V7.md",
    "docs/MODEL_INVENTORY_AND_TRAINABILITY_V7.md",
]

def run(args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

missing = [x for x in REQUIRED if not (ROOT / x).exists()]
checks = {"missing": missing}

rc, out, err = run([sys.executable, "run_medai_cli.py", "--json", "model-inventory"])
try:
    inv = json.loads(out)
except Exception:
    inv = {"parse_error": out[-1000:], "stderr": err[-1000:]}
checks["model_inventory_rc"] = rc
checks["num_models_excluding_mock"] = inv.get("num_models_excluding_mock")
checks["inventory_status"] = inv.get("status")

rc, out, err = run([sys.executable, "run_medai_cli.py", "--json", "route-models", "--organs", "pancreas,liver,aorta,kidney_cortex"])
try:
    route = json.loads(out)
except Exception:
    route = {"parse_error": out[-1000:], "stderr": err[-1000:]}
checks["route_models_rc"] = rc
checks["route_status"] = route.get("status")
checks["pancreas_mstep_target"] = route.get("routing", {}).get("pancreas", {}).get("mstep_target_model")
checks["aorta_mstep_target"] = route.get("routing", {}).get("aorta", {}).get("mstep_target_model")

rc, out, err = run([sys.executable, "run_medai_cli.py", "--json", "mstep-update", "--training-manifest", "data_manifest/case_list_50_tumor_template.csv", "--output-folder", "/tmp/medai_v7_verify_epai", "--target-model", "epai_20250421", "--dry-run"])
try:
    mstep_epai = json.loads(out)
except Exception:
    mstep_epai = {"parse_error": out[-1000:], "stderr": err[-1000:]}
checks["mstep_epai_rc"] = rc
checks["mstep_epai_status"] = mstep_epai.get("status")
checks["mstep_epai_backend"] = mstep_epai.get("mstep_backend")

rc, out, err = run([sys.executable, "run_medai_cli.py", "--json", "mstep-update", "--training-manifest", "data_manifest/case_list_50_tumor_template.csv", "--output-folder", "/tmp/medai_v7_verify_totalseg", "--target-model", "totalsegmentator", "--dry-run"])
try:
    mstep_ts = json.loads(out)
except Exception:
    mstep_ts = {"parse_error": out[-1000:], "stderr": err[-1000:]}
checks["mstep_totalseg_rc"] = rc
checks["mstep_totalseg_status"] = mstep_ts.get("status")
checks["mstep_totalseg_backend"] = mstep_ts.get("mstep_backend")

ok = (
    not missing and
    checks.get("inventory_status") == "success" and checks.get("num_models_excluding_mock") == 18 and
    checks.get("route_status") == "success" and checks.get("pancreas_mstep_target") == "epai_20250421" and
    checks.get("mstep_epai_status") == "dry_run" and
    checks.get("mstep_totalseg_status") == "not_trainable_in_current_project"
)
result = {"stage": "verify_final_v7_integrity", "status": "success" if ok else "failed", **checks}
print(json.dumps(result, indent=2, ensure_ascii=False))
raise SystemExit(0 if ok else 1)
