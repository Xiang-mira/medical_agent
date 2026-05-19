"""
Batch import + run agent-loop on all 50 PanTS cases.

Usage:
    python scripts/batch_run_pants50.py
    python scripts/batch_run_pants50.py --dry-run
    python scripts/batch_run_pants50.py --skip-import   # if already imported
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANTS_ROOT = PROJECT_ROOT / "third_party" / "PanTS-main"
DATA_ROOT = PROJECT_ROOT / "data" / "pants_real"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pants_batch_totalseg"
PYTHON = sys.executable
CLI = str(PROJECT_ROOT / "run_medai_cli.py")


def run_cmd(cmd: list[str], label: str) -> tuple[int, str, str]:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode, r.stdout, r.stderr


def import_case(case_id: str, dry_run: bool) -> dict:
    dest = DATA_ROOT / case_id
    if dest.exists():
        return {"case_id": case_id, "import_status": "already_exists"}
    if dry_run:
        return {"case_id": case_id, "import_status": "dry_run"}
    cmd = [PYTHON, CLI, "pants-import-case",
           "--pants-root", str(PANTS_ROOT),
           "--case-id", case_id,
           "--output-root", str(DATA_ROOT)]
    rc, out, err = run_cmd(cmd, case_id)
    return {"case_id": case_id, "import_status": "success" if rc == 0 else "failed",
            "return_code": rc, "stderr_tail": err[-500:]}


def run_agent(case_id: str, dry_run: bool, use_shapekit: bool = False) -> dict:
    patient_folder = DATA_ROOT / case_id
    case_output = OUTPUT_ROOT / case_id  # each case gets its own output folder
    shapekit_root = PROJECT_ROOT / "third_party" / "ShapeKit-main"
    postprocess = "shapekit" if use_shapekit else "none"
    cmd = [PYTHON, CLI, "--json", "agent-loop",
           "--patient-folder", str(patient_folder),
           "--output-folder", str(case_output),
           "--backend", "totalseg",
           "--postprocess", postprocess,
           "--shapekit-root", str(shapekit_root),
           "--fast",
           "--organ", "pancreas",
           "--expected-organs", "pancreas,liver,aorta,postcava"]
    if dry_run:
        cmd.append("--dry-run")
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = round(time.time() - t0, 1)
    try:
        result = json.loads(r.stdout)
    except Exception:
        result = {"status": "parse_error", "stdout_tail": r.stdout[-300:]}
    result["elapsed_sec"] = elapsed
    result["case_id"] = case_id
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-import", action="store_true", help="Skip import step if already done")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all)")
    parser.add_argument("--shapekit", action="store_true", help="Enable ShapeKit post-processing (slower, adds anatomy-aware refinement)")
    args = parser.parse_args()

    image_dir = PANTS_ROOT / "data" / "ImageTr"
    if not image_dir.exists():
        print(f"ERROR: ImageTr not found at {image_dir}", file=sys.stderr)
        sys.exit(1)

    cases = sorted([p.name for p in image_dir.iterdir() if p.is_dir()])
    if args.limit > 0:
        cases = cases[:args.limit]

    print(f"=== Batch run: {len(cases)} cases | dry_run={args.dry_run} | shapekit={args.shapekit} ===")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    import_results = []
    if not args.skip_import:
        print(f"\n[1/2] Importing {len(cases)} cases into {DATA_ROOT} ...")
        for i, case_id in enumerate(cases, 1):
            r = import_case(case_id, args.dry_run)
            import_results.append(r)
            print(f"  [{i:02d}/{len(cases)}] {case_id}: {r['import_status']}")
    else:
        print("\n[1/2] Skipping import (--skip-import)")

    print(f"\n[2/2] Running agent-loop on {len(cases)} cases ...")
    agent_results = []
    success = failed = warning = 0
    for i, case_id in enumerate(cases, 1):
        r = run_agent(case_id, args.dry_run, use_shapekit=args.shapekit)
        agent_results.append(r)
        status = r.get("status", "unknown")
        exec_status = r.get("execution_status", "?")
        num_masks = None
        scan_results = r.get("scan_results", [])
        if scan_results:
            num_masks = scan_results[0].get("infer_num_masks")
        masks_str = f" masks={num_masks}" if num_masks is not None else ""
        print(f"  [{i:02d}/{len(cases)}] {case_id}: status={status} exec={exec_status}{masks_str} ({r.get('elapsed_sec')}s)")
        if exec_status == "success":
            success += 1
        elif status == "failed":
            failed += 1
        else:
            warning += 1

    summary = {
        "total": len(cases),
        "execution_success": success,
        "execution_warning": warning,
        "execution_failed": failed,
        "import_results": import_results,
        "agent_results": agent_results,
    }
    summary_path = OUTPUT_ROOT / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Done ===")
    print(f"  Success: {success} / Warning: {warning} / Failed: {failed}")
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
