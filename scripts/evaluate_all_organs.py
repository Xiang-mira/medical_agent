#!/usr/bin/env python3
"""
全器官评估脚本
对有 ground truth 的器官：计算真实 DSC
对没有 ground truth 的器官：
  - 方法1：与最优 teacher 预测的重叠度（teacher_overlap）
  - 方法2：预测体积合理性（volume_ratio = student体积 / teacher体积）
用法：
  python evaluate_all_organs.py --rounds 1 2 3
"""
import argparse, csv, json, yaml
import numpy as np
import nibabel as nib
from pathlib import Path

PROJECT_ROOT = Path("/home/teacher1/JHU-project1/medical_agent")
CASE_LIST    = PROJECT_ROOT / "data_manifest/case_list_50_tumor.csv"
TEACHER_MAP  = PROJECT_ROOT / "configs/teacher_branch_map.yaml"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"
TEACHERS = ["totalsegmentator","epai_20250421","vsmtrans","cads","moose","moose3_0",
            "nnunet_private","saros_nnunet","airrc","atm","vsnet","unest","vista3d"]


def load_nii(path):
    return np.asanyarray(nib.load(str(path)).dataobj) > 0


def dice(a, b):
    inter = (a & b).sum()
    total = a.sum() + b.sum()
    return float(2 * inter / total) if total > 0 else None


def best_teacher_pred(estep_dir, case_id, organ):
    """找该 case 该 organ 体积最大的 teacher 预测（作为参考）"""
    best_path, best_vol = None, 0
    for t in TEACHERS:
        p = estep_dir / case_id / "raw_predictions" / t / case_id / "segmentations" / f"{organ}.nii.gz"
        if not p.exists():
            continue
        arr = np.asanyarray(nib.load(str(p)).dataobj) > 0
        if arr.sum() > best_vol:
            best_vol = arr.sum()
            best_path = p
    return best_path, best_vol


def evaluate_round(round_idx, cases, all_organs, estep_dir):
    pred_dir = OUTPUT_ROOT / f"round{round_idx}" / "student_predictions"
    if not pred_dir.exists():
        return None

    organ_stats = {}  # organ -> {gt_dsc, teacher_overlap, volume_ratio, n_cases}

    for case in cases:
        case_id = case["case_id"]
        ann_folder = Path(case.get("annotation_folder", "")) if case.get("annotation_folder") else None

        for organ in all_organs:
            student_path = pred_dir / case_id / f"{organ}.nii.gz"
            if not student_path.exists():
                continue

            student_arr = load_nii(student_path)
            if student_arr.sum() == 0:
                continue

            if organ not in organ_stats:
                organ_stats[organ] = {"gt_dsc": [], "teacher_overlap": [], "volume_ratio": [], "n_cases": 0}
            organ_stats[organ]["n_cases"] += 1

            # 方法1：有 ground truth → 计算真实 DSC
            if ann_folder and ann_folder.exists():
                ref = ann_folder / f"{organ}.nii.gz"
                if ref.exists():
                    ref_arr = load_nii(ref)
                    if ref_arr.sum() > 0:
                        d = dice(student_arr, ref_arr)
                        if d is not None:
                            organ_stats[organ]["gt_dsc"].append(d)

            # 方法2：与最优 teacher 的重叠度
            teacher_path, teacher_vol = best_teacher_pred(estep_dir, case_id, organ)
            if teacher_path is not None:
                teacher_arr = load_nii(teacher_path)
                overlap = dice(student_arr, teacher_arr)
                if overlap is not None:
                    organ_stats[organ]["teacher_overlap"].append(overlap)
                # 体积比
                if teacher_vol > 0:
                    ratio = float(student_arr.sum()) / teacher_vol
                    organ_stats[organ]["volume_ratio"].append(ratio)

    return organ_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--output", default=str(OUTPUT_ROOT / "organ_evaluation_report.json"))
    args = ap.parse_args()

    cases = []
    with open(CASE_LIST) as f:
        for row in csv.DictReader(f):
            cases.append(row)

    with open(TEACHER_MAP) as f:
        branch_map = yaml.safe_load(f)
    all_organs = [o for o, info in branch_map.items() if info.get("vista3d_label_id", 0) > 0]

    report = {}

    for rnd in args.rounds:
        estep_dir = OUTPUT_ROOT / f"round{rnd}" / "estep" / "cases"
        if not estep_dir.exists():
            print(f"Round {rnd} E-step 目录不存在，跳过")
            continue

        print(f"\n评估 Round {rnd}...")
        stats = evaluate_round(rnd, cases, all_organs, estep_dir)
        if stats is None:
            print(f"  Round {rnd} student_predictions 不存在，跳过")
            continue

        report[f"round{rnd}"] = {}
        gt_organs, no_gt_organs = [], []

        for organ in sorted(stats.keys()):
            s = stats[organ]
            entry = {
                "n_cases": s["n_cases"],
                "has_gt": len(s["gt_dsc"]) > 0,
            }
            if s["gt_dsc"]:
                entry["gt_dsc_mean"] = round(float(np.mean(s["gt_dsc"])), 4)
                entry["gt_dsc_std"]  = round(float(np.std(s["gt_dsc"])), 4)
                gt_organs.append((organ, entry["gt_dsc_mean"]))
            if s["teacher_overlap"]:
                entry["teacher_overlap_mean"] = round(float(np.mean(s["teacher_overlap"])), 4)
            if s["volume_ratio"]:
                entry["volume_ratio_mean"] = round(float(np.mean(s["volume_ratio"])), 4)
            report[f"round{rnd}"][organ] = entry
            if not s["gt_dsc"]:
                no_gt_organs.append((organ, entry.get("teacher_overlap_mean", 0)))

        # 打印摘要
        gt_dscs = [v for _, v in gt_organs]
        no_gt_overlaps = [v for _, v in no_gt_organs if v > 0]
        print(f"  有 GT 的器官: {len(gt_organs)}个, 平均 DSC = {np.mean(gt_dscs):.4f}" if gt_dscs else "  无 GT 器官")
        print(f"  无 GT 的器官: {len(no_gt_organs)}个, 平均 teacher_overlap = {np.mean(no_gt_overlaps):.4f}" if no_gt_overlaps else "  无 teacher_overlap 数据")

    # 跨轮对比
    if len(args.rounds) > 1:
        print("\n" + "="*70)
        print("跨轮对比（有 GT 的器官）")
        print(f"{'器官':30s}", end="")
        for rnd in args.rounds:
            if f"round{rnd}" in report:
                print(f"  R{rnd}-DSC  ", end="")
        print()
        print("-"*70)

        all_organs_with_gt = set()
        for rnd in args.rounds:
            key = f"round{rnd}"
            if key in report:
                for o, v in report[key].items():
                    if v.get("has_gt"):
                        all_organs_with_gt.add(o)

        for organ in sorted(all_organs_with_gt):
            print(f"{organ:30s}", end="")
            prev = None
            for rnd in args.rounds:
                key = f"round{rnd}"
                if key in report and organ in report[key]:
                    v = report[key][organ].get("gt_dsc_mean", float("nan"))
                    diff = f"({v-prev:+.3f})" if prev is not None else ""
                    print(f"  {v:.4f}{diff:8s}", end="")
                    prev = v
                else:
                    print(f"  {'N/A':14s}", end="")
            print()

        print("\n无 GT 器官 teacher_overlap 对比（前20个）")
        print(f"{'器官':30s}", end="")
        for rnd in args.rounds:
            if f"round{rnd}" in report:
                print(f"  R{rnd}-overlap", end="")
        print()
        print("-"*70)

        no_gt_set = set()
        for rnd in args.rounds:
            key = f"round{rnd}"
            if key in report:
                for o, v in report[key].items():
                    if not v.get("has_gt") and v.get("teacher_overlap_mean"):
                        no_gt_set.add(o)

        rows = []
        for organ in sorted(no_gt_set):
            vals = []
            for rnd in args.rounds:
                key = f"round{rnd}"
                v = report.get(key, {}).get(organ, {}).get("teacher_overlap_mean", None)
                vals.append(v)
            if any(v is not None for v in vals):
                rows.append((organ, vals))

        rows.sort(key=lambda x: -(x[1][-1] or 0))
        for organ, vals in rows[:20]:
            print(f"{organ:30s}", end="")
            prev = None
            for v in vals:
                if v is not None:
                    diff = f"({v-prev:+.3f})" if prev is not None else ""
                    print(f"  {v:.4f}{diff:8s}", end="")
                    prev = v
                else:
                    print(f"  {'N/A':14s}", end="")
            print()

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n完整报告已保存: {args.output}")


if __name__ == "__main__":
    main()
