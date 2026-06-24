#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime

def compare_organ(ct_path, mask1_path, mask2_path, organ,base_url,
                  base_output="./comparison_results", base_csv="./results",
                  port="8000", log_file="./comparison_summary.log", run_id=None,
                  no_dice_check=False, no_dual_confirmation=False,
                  simple_prompt_ablation=False, conservative_dual=False,
                  skip_organ_presence_gate=False, strict_choice_prompt=False):
    """
    Compare two segmentation masks using vLLM pipeline and write a detailed log.
    """
    
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(base_output, f"{run_id}", f"{organ}")
    csv_dir = os.path.join(base_csv, f"{run_id}")
    csv_path = os.path.join(csv_dir, f"{organ}.csv")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Step 1. Projection
    result = subprocess.run([
        "python3", "ProjectDatasetFlex_single.py",
        "--ct_good", ct_path,
        "--mask_good", mask1_path,
        "--ct_bad", ct_path,
        "--mask_bad", mask2_path,
        "--organ", organ,
        "--output_dir", output_dir
    ], capture_output=True, text=True)
    print("Projection STDOUT:\n", result.stdout)
    print("Projection STDERR:\n", result.stderr)
    result.check_returncode()
    if True:
        # Step 2. Run API
        run_api_cmd = [
            "python3", "RunAPI_single.py",
            "--path", output_dir,
            "--organ", organ,
            "--csv_path", csv_path,
            "--port", str(port),
            "--base_url", base_url
        ]
        if no_dice_check:
            run_api_cmd.append("--no_dice_check")
        if no_dual_confirmation:
            run_api_cmd.append("--no_dual_confirmation")
        if simple_prompt_ablation:
            run_api_cmd.append("--simple_prompt_ablation")
        if conservative_dual:
            run_api_cmd.append("--conservative_dual")
        if skip_organ_presence_gate:
            run_api_cmd.append("--skip_organ_presence_gate")
        if strict_choice_prompt:
            run_api_cmd.append("--strict_choice_prompt")
        subprocess.run(run_api_cmd, check=True)

        # Step 3. Parse result
        better = None
        with open(csv_path, "r") as f:
            header = f.readline().strip().split(",")
            row = f.readline().strip().split(",")
            if len(row) >= 2:
                answer = row[1].strip()
                if answer == "1":
                    better = "mask1"
                elif answer == "2":
                    better = "mask2"
                elif answer == "0.5":
                    better = "uncertain"
                    print("Undecided case (answer=0.5):", ",".join(row))
            elif len(header) >= 2:
                print("No comparison rows were produced; LabelCritic likely skipped all projection cases.")

        # Determine which path is better; uncertain means VLM could not decide
        if better == "mask1":
            best_path = mask1_path
        elif better == "mask2":
            best_path = mask2_path
        else:
            best_path = None

        # Step 4. Write to log file
        with open(log_file, "a") as log:
            log.write(
                f"[{datetime.now().isoformat()}] Run ID: {run_id}\n"
                f"  Organ: {organ}\n"
                f"  CT: {ct_path}\n"
                f"  Mask1: {mask1_path}\n"
                f"  Mask2: {mask2_path}\n"
                f"  Better: {best_path if best_path is not None else 'uncertain'}\n\n"
            )
        print(mask1_path, mask2_path)
        print(best_path)
    else:
        best_path = mask1_path  # Default to mask1 if not running API
    return best_path


def compare_organ_batch(manifest_path, default_base_url, default_port="8000", default_log_file="./comparison_summary.log"):
    """Run multiple CompareOrgan jobs in one Python process.

    Manifest format:
    {"items": [{"ct": ..., "mask1": ..., "mask2": ..., "organ": ...,
                 "base_output": ..., "base_csv": ..., "run_id": ...,
                 "base_url": ..., "port": ..., "log_file": ...,
                 "no_dice_check": true, ...}]}
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    items = doc.get("items", doc if isinstance(doc, list) else [])
    results = []
    for idx, item in enumerate(items, start=1):
        try:
            best = compare_organ(
                item["ct"], item["mask1"], item["mask2"], item["organ"],
                item.get("base_url", default_base_url),
                base_output=item.get("base_output", "./comparison_results"),
                base_csv=item.get("base_csv", "./results"),
                port=str(item.get("port", default_port)),
                log_file=item.get("log_file", default_log_file),
                run_id=item.get("run_id"),
                no_dice_check=bool(item.get("no_dice_check", False)),
                no_dual_confirmation=bool(item.get("no_dual_confirmation", False)),
                simple_prompt_ablation=bool(item.get("simple_prompt_ablation", False)),
                conservative_dual=bool(item.get("conservative_dual", False)),
                skip_organ_presence_gate=bool(item.get("skip_organ_presence_gate", False)),
                strict_choice_prompt=bool(item.get("strict_choice_prompt", False)),
            )
            results.append({"index": idx, "status": "success", "best": best, "organ": item.get("organ"), "run_id": item.get("run_id")})
        except Exception as exc:
            results.append({"index": idx, "status": "failed", "reason": str(exc), "organ": item.get("organ"), "run_id": item.get("run_id")})
            if bool(doc.get("fail_fast", False)):
                raise
    output_json = doc.get("output_json")
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"status": "success", "num_items": len(items), "results": results}, f, ensure_ascii=False, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two segmentation masks using vLLM and log the result.")
    parser.add_argument("--batch_manifest", default=None, help="Optional JSON manifest for batched comparisons")
    parser.add_argument("--ct", required=False, help="Path to CT volume (.nii.gz)")
    parser.add_argument("--mask1", required=False, help="Path to segmentation directory for model 1")
    parser.add_argument("--mask2", required=False, help="Path to segmentation directory for model 2")
    parser.add_argument("--organ", required=False, help="Organ name (e.g. aorta, pancreas)")
    parser.add_argument("--base_output", default="./comparison_results", help="Base folder for projection results")
    parser.add_argument("--base_csv", default="./results", help="Base folder for CSV results")
    parser.add_argument("--port", default="8000", help="API server port (vLLM)")
    parser.add_argument("--log_file", default="./comparison_summary.log", help="File to append final comparison results")
    parser.add_argument("--base_url", default="http://udc-ba02-35")
    parser.add_argument("--run_id", default=None, help="Optional run id for reproducible log scoping")
    parser.add_argument("--no_dice_check", action="store_true", default=False, help="Do not skip high-2D-Dice projections")
    parser.add_argument("--no_dual_confirmation", action="store_true", default=False, help="Use non-dual LabelCritic prompt path")
    parser.add_argument("--simple_prompt_ablation", action="store_true", default=False, help="Use simplified comparison prompt")
    parser.add_argument("--conservative_dual", action="store_true", default=False, help="Require strict dual confirmation")
    parser.add_argument("--skip_organ_presence_gate", action="store_true", default=False, help="Diagnostic only: bypass organ-present gate")
    parser.add_argument("--strict_choice_prompt", action="store_true", default=False, help="Diagnostic only: use forced-choice prompt")

    args = parser.parse_args()

    if args.batch_manifest:
        compare_organ_batch(args.batch_manifest, args.base_url, default_port=args.port, default_log_file=args.log_file)
        raise SystemExit(0)
    missing = [name for name in ("ct", "mask1", "mask2", "organ") if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments for single compare: " + ", ".join(missing))

    best = compare_organ(
        args.ct, args.mask1, args.mask2, args.organ, args.base_url,
        base_output=args.base_output, base_csv=args.base_csv,
        port=args.port, log_file=args.log_file, run_id=args.run_id,
        no_dice_check=args.no_dice_check,
        no_dual_confirmation=args.no_dual_confirmation,
        simple_prompt_ablation=args.simple_prompt_ablation,
        conservative_dual=args.conservative_dual,
        skip_organ_presence_gate=args.skip_organ_presence_gate,
        strict_choice_prompt=args.strict_choice_prompt,
    )
    print(best)
