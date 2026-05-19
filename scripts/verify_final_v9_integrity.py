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
    "third_party/TotalSegmentator-master/resources/train_nnunet.md",
    "third_party/TotalSegmentator-master/resources/train_nnunet.sh",
    "third_party/TotalSegmentator-master/resources/convert_dataset_to_nnunet.py",
    "third_party/VISTA3D-Inference-Pipeline-master/README.md",
    "third_party/VISTA3D-Inference-Pipeline-master/configs/train.json",
    "third_party/VISTA3D-Inference-Pipeline-master/configs/train_continual.json",
    "third_party/VISTA3D-Inference-Pipeline-master/configs/multi_gpu_train.json",
    "third_party/VISTA3D-Inference-Pipeline-master/scripts/trainer.py",
    "third_party/PanTS-main/README.md",
    "docs/FINAL_AUDIT_AND_USAGE_V9.md",
    "docs/TOTALSEG_VISTA3D_TRAINING_RECIPE_V9.md",
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


def find_model(inv: dict, key: str) -> dict:
    for m in inv.get("models", []):
        if m.get("model_key") == key:
            return m
    return {}


def cleanup_python_cache() -> tuple[int, int]:
    pyc_files = list(ROOT.rglob("*.pyc"))
    pycache_dirs = sorted(ROOT.rglob("__pycache__"), key=lambda p: len(p.parts), reverse=True)
    removed_pyc = 0
    removed_dirs = 0
    for path in pyc_files:
        try:
            path.unlink()
            removed_pyc += 1
        except FileNotFoundError:
            pass
    for path in pycache_dirs:
        try:
            path.rmdir()
            removed_dirs += 1
        except (FileNotFoundError, OSError):
            pass
    return removed_dirs, removed_pyc


def main() -> int:
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).exists()]
    verify_root = ROOT / "outputs" / "_verify_v9"
    loop_dir = verify_root / "loop"
    totseg_dir = verify_root / "mstep_ts"
    vista_dir = verify_root / "mstep_vista"

    pre_pycache_count = len(list(ROOT.rglob("__pycache__")))
    pre_pyc_count = len(list(ROOT.rglob("*.pyc")))

    rc_inventory, out_inventory, err_inventory = run([sys.executable, "run_medai_cli.py", "--json", "model-inventory"])
    inv = parse_json_stdout(out_inventory)
    totseg_inv = find_model(inv, "totalsegmentator")
    vista_inv = find_model(inv, "vista3d")

    rc_route, out_route, err_route = run([sys.executable, "run_medai_cli.py", "--json", "route-models", "--organs", "pancreas,liver,aorta,pancreatic_duct,kidney_cortex"])
    route = parse_json_stdout(out_route)

    # Create a minimal empty manifest through the dry-run loop. Real manifests are generated after PanTS data is available.
    rc_loop, out_loop, err_loop = run([
        sys.executable, "run_medai_cli.py", "--json", "run-loop",
        "--case-list", "data_manifest/case_list_50_tumor_template.csv",
        "--models", "mock_seg",
        "--organs", "pancreas,liver,aorta",
        "--output", str(loop_dir),
        "--dry-run",
        "--critic-backend", "stub",
    ])

    manifest = str(loop_dir / "training_manifest.json")

    rc_mstep_totseg, out_mstep_totseg, err_mstep_totseg = run([
        sys.executable, "run_medai_cli.py", "--json", "mstep-update",
        "--training-manifest", manifest,
        "--output-folder", str(totseg_dir),
        "--target-model", "totalsegmentator",
        "--dry-run",
    ])
    totseg = parse_json_stdout(out_mstep_totseg)

    rc_mstep_vista, out_mstep_vista, err_mstep_vista = run([
        sys.executable, "run_medai_cli.py", "--json", "mstep-update",
        "--training-manifest", manifest,
        "--output-folder", str(vista_dir),
        "--target-model", "vista3d",
        "--dry-run",
    ])
    vista = parse_json_stdout(out_mstep_vista)

    rc_compile, out_compile, err_compile = run([sys.executable, "-m", "compileall", "-q", "agent-harness/cli_anything/medai", "scripts"])
    cleanup_dirs, cleanup_pyc = cleanup_python_cache()
    final_pycache_count = len(list(ROOT.rglob("__pycache__")))
    final_pyc_count = len(list(ROOT.rglob("*.pyc")))

    checks = {
        "stage": "verify_final_v9_integrity",
        "status": "success" if not missing and rc_inventory == 0 and rc_route == 0 and rc_loop == 0 and rc_mstep_totseg == 0 and rc_mstep_vista == 0 and rc_compile == 0 else "error",
        "missing": missing,
        "pycache_dirs": final_pycache_count,
        "pyc_files": final_pyc_count,
        "pre_cleanup_pycache_dirs": pre_pycache_count,
        "pre_cleanup_pyc_files": pre_pyc_count,
        "inventory_status": inv.get("status"),
        "num_models_excluding_mock": inv.get("num_models_excluding_mock"),
        "num_trainable_or_finetunable_if_checkpoint_present": inv.get("num_trainable_or_finetunable_if_checkpoint_present"),
        "totalsegmentator_trainable": totseg_inv.get("trainable"),
        "totalsegmentator_mstep_backend": totseg_inv.get("mstep_backend"),
        "vista3d_trainable": vista_inv.get("trainable"),
        "vista3d_mstep_backend": vista_inv.get("mstep_backend"),
        "route_status": route.get("status"),
        "pancreas_primary": route.get("routing", {}).get("pancreas", {}).get("primary_model"),
        "pancreas_mstep_target": route.get("routing", {}).get("pancreas", {}).get("mstep_target_model"),
        "aorta_primary": route.get("routing", {}).get("aorta", {}).get("primary_model"),
        "aorta_mstep_target": route.get("routing", {}).get("aorta", {}).get("mstep_target_model"),
        "mstep_totalseg_status": totseg.get("status"),
        "mstep_totalseg_backend": totseg.get("mstep_backend"),
        "mstep_totalseg_recipe_files": totseg.get("training_result", {}).get("recipe_files", {}).get("exists"),
        "mstep_vista3d_status": vista.get("status"),
        "mstep_vista3d_backend": vista.get("mstep_backend"),
        "mstep_vista3d_recipe_files": vista.get("training_result", {}).get("recipe_files", {}).get("exists"),
        "compile_returncode": rc_compile,
        "post_compile_cleanup_pycache_dirs": cleanup_dirs,
        "post_compile_cleanup_pyc_files": cleanup_pyc,
        "errors": {
            "inventory": err_inventory[-1000:],
            "route": err_route[-1000:],
            "loop": err_loop[-1000:],
            "mstep_totalseg": err_mstep_totseg[-1000:],
            "mstep_vista3d": err_mstep_vista[-1000:],
            "compile": err_compile[-1000:],
        },
    }
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0 if checks["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
