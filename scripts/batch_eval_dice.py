"""
Batch Dice evaluation: TotalSegmentator predictions vs PanTS reference labels.
Runs pants-eval-case on all 50 cases and prints a summary table.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_BASE = PROJECT_ROOT / "outputs" / "pants_batch_totalseg" / "scan_outputs"
REF_BASE = PROJECT_ROOT / "data" / "pants_real"
PYTHON = sys.executable
CLI = str(PROJECT_ROOT / "run_medai_cli.py")

EVAL_ORGANS = "pancreas,liver,spleen,kidney_left,kidney_right,aorta,stomach"


def eval_case(case_id: str) -> dict | None:
    # Prediction folder: scan_outputs/<case_id>/raw_predictions/<case_id>_<case_id>/segmentations
    pred_folder = PRED_BASE / case_id / "raw_predictions" / f"{case_id}_{case_id}" / "segmentations"
    ref_folder = REF_BASE / case_id / "reference_labels" / case_id / "segmentations"
    if not pred_folder.exists():
        return {"case_id": case_id, "status": "no_predictions"}
    if not ref_folder.exists():
        return {"case_id": case_id, "status": "no_reference"}
    cmd = [PYTHON, CLI, "--json", "pants-eval-case",
           "--pred-folder", str(pred_folder),
           "--reference-label-folder", str(ref_folder),
           "--organs", EVAL_ORGANS]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        result = json.loads(r.stdout)
        result["case_id"] = case_id
        return result
    except Exception:
        return {"case_id": case_id, "status": "parse_error", "stdout": r.stdout[:200]}


def main():
    cases = sorted([p.name for p in PRED_BASE.iterdir() if p.is_dir()])
    print(f"Evaluating {len(cases)} cases on organs: {EVAL_ORGANS}\n")

    all_results = []
    organ_scores: dict[str, list[float]] = {}

    for i, case_id in enumerate(cases, 1):
        r = eval_case(case_id)
        all_results.append(r)
        if not r or r.get("status") in ("no_predictions", "no_reference", "parse_error"):
            print(f"[{i:02d}/50] {case_id}: SKIP ({r.get('status')})")
            continue
        mean_dice = r.get("mean_dice", 0)
        per_organ = {x["mask"].replace(".nii.gz", ""): x["dice"]
                     for x in r.get("results", []) if x.get("status") == "success"}
        for organ, dice in per_organ.items():
            organ_scores.setdefault(organ, []).append(dice)
        organs_str = " ".join(f"{o}={d:.3f}" for o, d in per_organ.items())
        print(f"[{i:02d}/50] {case_id}: mean={mean_dice:.3f}  {organs_str}")

    print("\n" + "=" * 60)
    print("SUMMARY (mean Dice across all cases):")
    print("=" * 60)
    grand_dices = []
    for organ in EVAL_ORGANS.split(","):
        scores = organ_scores.get(organ, [])
        if scores:
            avg = sum(scores) / len(scores)
            grand_dices.append(avg)
            print(f"  {organ:<20} {avg:.4f}  (n={len(scores)})")
    if grand_dices:
        overall = sum(grand_dices) / len(grand_dices)
        print(f"\n  {'OVERALL MEAN':<20} {overall:.4f}")

    out_path = PROJECT_ROOT / "outputs" / "pants_batch_totalseg" / "dice_eval_summary.json"
    out_path.write_text(json.dumps({"organ_eval_organs": EVAL_ORGANS, "cases": all_results,
                                    "organ_means": {o: round(sum(s)/len(s), 6) for o, s in organ_scores.items()}},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull results saved: {out_path}")


if __name__ == "__main__":
    main()
