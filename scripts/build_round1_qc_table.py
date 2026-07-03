#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


ABDOMEN_TERMS = [
    "liver", "spleen", "pancreas", "kidney", "adrenal", "aorta", "stomach",
    "duodenum", "colon", "bowel", "intestine", "bladder", "gall", "bile",
    "celiac", "iliac", "renal_vein", "vena_cava", "portal", "hepatic",
    "abdominal", "rectum", "prostate", "uterus", "ovary", "gonad",
    "esophagus", "mesenteric", "peritone",
]
THORAX_TERMS = [
    "lung", "heart", "pulmonary", "bronch", "trachea", "airway", "mediast",
    "rib", "sternum", "aortic_arch", "atrial", "coronary", "mammary",
    "breast", "brachiocephalic_vein",
]
HEAD_TERMS = [
    "brain", "cerebellum", "brainstem", "ventricle", "caudate",
    "central_sulcus", "insular", "internal_capsule", "gray_matter", "optic",
    "eyeball", "eye_", "lens", "cochlear", "auditory", "nasal", "cheek",
    "mandible", "maxilla", "teeth", "tooth", "tongue", "palate", "parotid",
    "submandibular", "masseter", "pterygoid", "temporalis", "zygomatic",
    "lips", "buccal", "lacrimal",
]
NECK_TERMS = [
    "larynx", "pharynx", "thyroid", "hyoid", "cricoid", "arytenoid",
    "glottis", "scalene", "sternocleidomastoid", "trapezius",
    "prevertebral", "digastric", "platysma", "cricopharyngeus", "cervical",
    "styloid",
]
EXTREMITY_TERMS = [
    "humerus", "radius", "ulna", "carpal", "metacarpal", "fingers", "femur",
    "fibula", "tibia", "patella", "tarsal", "metatarsal", "toes", "hip",
    "scapula", "clavicula", "phalanges",
]

KEY_CORE_ORGANS = {
    "liver", "spleen", "pancreas", "kidney_left", "kidney_right", "kidney",
    "aorta", "adrenal_gland_left", "adrenal_gland_right", "stomach",
    "duodenum", "colon", "small_bowel", "intestine", "abdominal_cavity",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Round1 QC tables from formal E/M-step outputs.")
    ap.add_argument(
        "--round-dir",
        type=Path,
        default=ROOT / "outputs/formal_round1_final_20260627/round1",
        help="Round directory containing estep/dashboards and metrics.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <round-dir>/qc",
    )
    return ap.parse_args()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def region_for_organ(organ: str) -> str:
    s = organ.lower()
    if any(term in s for term in ABDOMEN_TERMS):
        return "abdomen_pelvis"
    if any(term in s for term in THORAX_TERMS):
        return "thorax"
    if any(term in s for term in HEAD_TERMS):
        return "head_brain"
    if any(term in s for term in NECK_TERMS):
        return "neck"
    if any(term in s for term in EXTREMITY_TERMS):
        return "extremity_bone"
    return "other_unclear"


def organ_complexity(organ: str) -> str:
    s = organ.lower()
    if any(x in s for x in ["vein", "artery", "vessel", "duct", "airway", "bronch", "trachea"]):
        return "thin_tubular_structure"
    if any(x in s for x in ["segment", "lobe", "atrium", "ventricle", "cortex", "medulla"]):
        return "substructure"
    if any(x in s for x in ["rib", "clavicula", "scapula", "sternum", "bone", "cartilage"]):
        return "bone_or_fragmented_structure"
    if organ in KEY_CORE_ORGANS:
        return "core_large_or_common_organ"
    return "general_or_longtail"


def yesno(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def classify_student(mean_dsc: float | None, coverage: int, d_count: int) -> str:
    if coverage == 0:
        return "no_final_label_or_likely_out_of_fov"
    if mean_dsc is None:
        if coverage >= 8:
            return "teacher_pool_exists_but_student_not_evaluated"
        return "sparse_teacher_pool"
    if mean_dsc >= 0.8:
        return "good"
    if mean_dsc >= 0.5:
        return "usable_needs_improvement"
    if mean_dsc >= 0.2:
        return "needs_repair_or_more_training"
    return "very_poor"


def likely_failure_mode(row: pd.Series) -> str:
    organ = str(row["organ"])
    region = row["anatomy_region"]
    coverage = int(row["coverage"])
    mean_dsc = row.get("mean_dsc")
    ab_count = int(row["A_count"] + row["B_count"])
    d_count = int(row["D_count"])
    single_family = int(row["single_family_count"])
    fallback_count = int(row["fallback_count"])
    missing_case_count = int(row["missing_case_count"])
    complexity = row["organ_complexity"]

    if coverage == 0:
        if region in {"head_brain", "neck", "extremity_bone"}:
            return "likely_out_of_fov_for_abdominal_ct"
        if organ in {"bladder", "rectum", "uterus", "prostate", "gonads"}:
            return "likely_pelvic_fov_absent_or_incomplete"
        return "no_teacher_output_or_fov_unconfirmed"
    if mean_dsc is None:
        return "teacher_labels_exist_but_no_student_metric"
    if mean_dsc < 0.2 and ab_count >= 6:
        return "student_or_mapping_failure_despite_high_teacher_grade"
    if complexity == "thin_tubular_structure" and mean_dsc < 0.5:
        return "thin_structure_boundary_or_connectivity_failure"
    if single_family >= max(3, coverage // 2) and mean_dsc < 0.5:
        return "single_teacher_dependency_or_weak_consensus"
    if fallback_count >= max(3, coverage // 2) and mean_dsc < 0.5:
        return "fallback_selection_noise"
    if d_count >= max(5, coverage // 2):
        return "low_quality_teacher_pool"
    if missing_case_count > 0:
        return "partial_case_coverage"
    if mean_dsc < 0.5:
        return "student_underfit_or_insufficient_training_examples"
    return "acceptable_but_monitor"


def recommended_action(row: pd.Series) -> str:
    category = row["qc_category"]
    mode = row["likely_failure_mode"]
    organ = str(row["organ"])
    if category == "good":
        return "keep_for_training_and_report_as_round1_success"
    if category == "usable_needs_improvement":
        return "keep_with_weighting; review low-case outliers before round2"
    if mode.startswith("likely_out_of_fov") or mode.startswith("likely_pelvic_fov"):
        return "exclude_from_accuracy_metrics; add FOV/presence gate"
    if mode == "student_or_mapping_failure_despite_high_teacher_grade":
        return "high_priority_debug_mapping_prompt_and_prediction_for_this_organ"
    if mode == "thin_structure_boundary_or_connectivity_failure":
        return "evaluate_with_surface_or_centerline_metrics; add topology/connected-component QC"
    if mode in {"single_teacher_dependency_or_weak_consensus", "fallback_selection_noise", "low_quality_teacher_pool"}:
        return "do_not_use_as_hard_label; require second teacher_or_manual_review"
    if organ in KEY_CORE_ORGANS:
        return "core_organ_priority_repair_before_round2"
    if category in {"very_poor", "needs_repair_or_more_training"}:
        return "exclude_or_downweight_until_repaired"
    return "review"


def build_organ_qc(round_dir: Path) -> pd.DataFrame:
    organ_dashboard = pd.read_csv(round_dir / "estep/dashboards/organ_capability_dashboard.csv")
    student_summary = pd.read_csv(round_dir / "metrics/student_organ_summary.csv")
    case_organ = pd.read_csv(round_dir / "estep/dashboards/case_organ_label_scores.csv")

    selected = case_organ[case_organ["selected_model"].notna()].copy()
    fallback = selected.groupby("organ")["selection_status"].apply(lambda s: int((s == "fallback").sum())).rename("fallback_count")
    selected_count = selected.groupby("organ")["organ"].size().rename("selected_count")
    distill = selected.groupby("organ")["distillation_eligible"].apply(lambda s: int(sum(yesno(v) for v in s))).rename("distillation_eligible_count")

    qc = organ_dashboard.merge(student_summary[["organ", "n", "mean_dsc", "std_dsc"]], on="organ", how="left")
    qc = qc.merge(fallback, on="organ", how="left")
    qc = qc.merge(selected_count, on="organ", how="left")
    qc = qc.merge(distill, on="organ", how="left")

    for col in [
        "A_count", "B_count", "C_count", "D_count", "coverage", "missing_case_count",
        "single_family_count", "shapekit_fallback_count", "training_excluded_count",
        "fallback_count", "selected_count", "distillation_eligible_count",
    ]:
        qc[col] = pd.to_numeric(qc.get(col, 0), errors="coerce").fillna(0).astype(int)
    qc["mean_dsc"] = pd.to_numeric(qc["mean_dsc"], errors="coerce")
    qc["std_dsc"] = pd.to_numeric(qc["std_dsc"], errors="coerce")
    qc["mean_reliability_score"] = pd.to_numeric(qc["mean_reliability_score"], errors="coerce")

    qc["anatomy_region"] = qc["organ"].map(region_for_organ)
    qc["organ_complexity"] = qc["organ"].map(organ_complexity)
    qc["AB_count"] = qc["A_count"] + qc["B_count"]
    qc["AB_rate"] = (qc["AB_count"] / 10).round(3)
    qc["D_rate"] = (qc["D_count"] / 10).round(3)
    qc["fallback_rate_among_selected"] = (
        qc["fallback_count"] / qc["selected_count"].replace(0, pd.NA)
    ).astype("Float64").round(3)
    qc["single_family_rate"] = (
        qc["single_family_count"] / qc["coverage"].replace(0, pd.NA)
    ).astype("Float64").round(3)
    qc["distillation_eligible_rate"] = (
        qc["distillation_eligible_count"] / qc["selected_count"].replace(0, pd.NA)
    ).astype("Float64").round(3)
    qc["qc_category"] = qc.apply(
        lambda r: classify_student(None if pd.isna(r["mean_dsc"]) else float(r["mean_dsc"]), int(r["coverage"]), int(r["D_count"])),
        axis=1,
    )
    qc["likely_failure_mode"] = qc.apply(likely_failure_mode, axis=1)
    qc["recommended_action"] = qc.apply(recommended_action, axis=1)
    qc["metric_warning"] = "student-vs-selected-pseudo-label consistency, not expert-GT accuracy"
    qc["metric_target"] = "pseudo-label"
    qc["metric_subject"] = "student"
    qc["metric_comparison"] = "student_vs_selected_pseudo_label"
    qc["metric_interpretation"] = "pseudo_label_consistency"
    qc["student_vs_selected_pseudo_mean_dsc"] = qc["mean_dsc"]

    preferred = [
        "organ", "qc_category", "recommended_action", "likely_failure_mode",
        "anatomy_region", "organ_complexity", "coverage", "selected_count",
        "distillation_eligible_count", "distillation_eligible_rate",
        "A_count", "B_count", "C_count", "D_count", "AB_count", "AB_rate", "D_rate",
        "mean_reliability_score", "n", "mean_dsc", "student_vs_selected_pseudo_mean_dsc", "std_dsc",
        "fallback_count", "fallback_rate_among_selected", "single_family_count",
        "single_family_rate", "shapekit_fallback_count", "missing_case_count",
        "training_excluded_count", "status", "student_checkpoint_status",
        "metric_target", "metric_subject", "metric_comparison", "metric_interpretation",
        "metric_warning",
    ]
    return qc[[c for c in preferred if c in qc.columns]].sort_values(
        ["qc_category", "mean_dsc", "coverage", "organ"],
        ascending=[True, False, False, True],
    )


def build_case_qc(round_dir: Path) -> pd.DataFrame:
    student = pd.read_csv(round_dir / "metrics/student_dice_per_organ.csv")
    case_organ = pd.read_csv(round_dir / "estep/dashboards/case_organ_label_scores.csv")
    selected = case_organ[case_organ["selected_model"].notna()].copy()

    student_case = student.groupby("case_id").agg(
        evaluated_organs=("dsc", "size"),
        mean_dsc=("dsc", "mean"),
        median_dsc=("dsc", "median"),
        dsc_ge_08=("dsc", lambda s: int((s >= 0.8).sum())),
        dsc_ge_05=("dsc", lambda s: int((s >= 0.5).sum())),
        dsc_lt_02=("dsc", lambda s: int((s < 0.2).sum())),
        zero_dsc=("dsc", lambda s: int((s == 0).sum())),
    ).reset_index()

    selected_case = selected.groupby("case_id").agg(
        selected_organs=("organ", "size"),
        A=("grade", lambda s: int((s == "A").sum())),
        B=("grade", lambda s: int((s == "B").sum())),
        C=("grade", lambda s: int((s == "C").sum())),
        D=("grade", lambda s: int((s == "D").sum())),
        fallback_count=("selection_status", lambda s: int((s == "fallback").sum())),
        single_family_count=("independent_family_count", lambda s: int((pd.to_numeric(s, errors="coerce") < 2).sum())),
        distillation_eligible_count=("distillation_eligible", lambda s: int(sum(yesno(v) for v in s))),
        mean_reliability=("estimated_reliability", "mean"),
        mean_selected_pseudo_consistency=("selected_pseudo_consistency_dice", "mean"),
    ).reset_index()

    qc = selected_case.merge(student_case, on="case_id", how="outer")
    for num in ["mean_dsc", "median_dsc", "mean_reliability", "mean_selected_pseudo_consistency"]:
        qc[num] = pd.to_numeric(qc[num], errors="coerce").round(4)
    qc["pct_dsc_ge_05"] = (qc["dsc_ge_05"] / qc["evaluated_organs"] * 100).round(1)
    qc["pct_dsc_lt_02"] = (qc["dsc_lt_02"] / qc["evaluated_organs"] * 100).round(1)
    qc["fallback_rate"] = (qc["fallback_count"] / qc["selected_organs"] * 100).round(1)
    qc["distillation_eligible_rate"] = (qc["distillation_eligible_count"] / qc["selected_organs"] * 100).round(1)
    qc["case_risk"] = qc.apply(
        lambda r: "high" if r["mean_dsc"] < 0.3 or r["pct_dsc_lt_02"] >= 50
        else ("medium" if r["mean_dsc"] < 0.36 or r["pct_dsc_lt_02"] >= 40 else "lower"),
        axis=1,
    )
    qc["recommended_action"] = qc["case_risk"].map({
        "high": "manual_review_core_organs_and_exclude_bad_labels_before_round2",
        "medium": "review_low_dice_organs_and_downweight_C_D",
        "lower": "usable_for_filtered_training_with_monitoring",
    })
    qc["metric_warning"] = "student-vs-selected-pseudo-label consistency, not expert-GT accuracy"
    qc["metric_target"] = "pseudo-label"
    qc["metric_subject"] = "student"
    qc["metric_comparison"] = "student_vs_selected_pseudo_label"
    qc["metric_interpretation"] = "pseudo_label_consistency"
    qc["student_vs_selected_pseudo_mean_dsc"] = qc["mean_dsc"]
    return qc.sort_values(["case_risk", "mean_dsc"], ascending=[True, True])


def build_summary(organ_qc: pd.DataFrame, case_qc: pd.DataFrame, round_dir: Path) -> dict[str, Any]:
    gate = read_json(round_dir / "estep/formal_gate.json", {})
    metrics = read_json(round_dir / "metrics/round_metrics.json", {})
    category_counts = organ_qc["qc_category"].value_counts().to_dict()
    region_by_category = (
        organ_qc.groupby(["anatomy_region", "qc_category"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    high_priority = organ_qc[
        organ_qc["recommended_action"].isin([
            "high_priority_debug_mapping_prompt_and_prediction_for_this_organ",
            "core_organ_priority_repair_before_round2",
        ])
    ]["organ"].tolist()
    return {
        "scope": str(round_dir),
        "metric_warning": "QC uses pseudo-label consistency and auto-label evidence. It is not expert-ground-truth accuracy.",
        "metric_target": "pseudo-label",
        "metric_subject": "student",
        "metric_comparison": "student_vs_selected_pseudo_label",
        "metric_interpretation": "pseudo_label_consistency",
        "round_status": {
            "estep_status": gate.get("status"),
            "num_cases": gate.get("num_cases_with_selection_metadata"),
            "num_selected_organs": gate.get("num_selected_organs"),
            "grade_counts": gate.get("grade_counts"),
            "overall_student_pseudo_consistency_mean_dsc": metrics.get("overall_mean_dsc"),
            "metric_target": metrics.get("metric_target", "pseudo-label"),
            "metric_comparison": metrics.get("metric_comparison", "student_vs_selected_pseudo_label"),
            "n_student_evaluations": metrics.get("n_evaluations"),
        },
        "organ_qc_category_counts": category_counts,
        "region_by_category": region_by_category,
        "case_risk_counts": case_qc["case_risk"].value_counts().to_dict(),
        "high_priority_organs": high_priority,
        "key_findings": [
            "Only a small core of large abdominal organs currently reaches high student-pseudo consistency.",
            "Most formal target organs have no final label in these 10 abdominal CT cases, so they should be FOV/presence-gated rather than counted as segmentation failures.",
            "Several in-FOV abdominal organs have weak student consistency despite teacher coverage; these are the most important round2 repair targets.",
            "Dice should be supplemented with surface distance, volume error, empty-mask rate, connected-component checks, and anatomy plausibility QC.",
        ],
    }


def write_markdown(path: Path, summary: dict[str, Any], organ_qc: pd.DataFrame, case_qc: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Round1 segmentation QC report\n")
    lines.append("This report evaluates pseudo-label and student consistency, not expert ground-truth accuracy.\n")
    lines.append("## Overall\n")
    rs = summary["round_status"]
    lines.append(f"- E-step status: `{rs.get('estep_status')}`")
    lines.append(f"- Cases: {rs.get('num_cases')}")
    lines.append(f"- Selected pseudo-labels: {rs.get('num_selected_organs')}")
    lines.append(f"- Grade counts: {rs.get('grade_counts')}")
    lines.append(f"- Student-vs-selected-pseudo mean DSC: {rs.get('overall_student_pseudo_consistency_mean_dsc')}\n")
    lines.append("## Organ QC category counts\n")
    for key, value in summary["organ_qc_category_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("\n## Case risk table\n")
    lines.append("| case_id | risk | Student vs selected pseudo-label Dice | median student-vs-pseudo Dice | student-vs-pseudo Dice<0.2 % | selected | fallback % | recommendation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for _, row in case_qc.iterrows():
        lines.append(
            f"| {row['case_id']} | {row['case_risk']} | {row['mean_dsc']:.3f} | "
            f"{row['median_dsc']:.3f} | {row['pct_dsc_lt_02']:.1f} | "
            f"{int(row['selected_organs'])} | {row['fallback_rate']:.1f} | {row['recommended_action']} |"
        )
    lines.append("\n## High-priority organ repair/debug list\n")
    for organ in summary["high_priority_organs"]:
        lines.append(f"- {organ}")
    lines.append("\n## Best current organs\n")
    best = organ_qc[organ_qc["qc_category"].isin(["good", "usable_needs_improvement"])].sort_values("mean_dsc", ascending=False).head(30)
    for _, row in best.iterrows():
        lines.append(f"- {row['organ']}: {row['qc_category']}, student_vs_selected_pseudo_mean_dsc={row['mean_dsc']:.3f}, action={row['recommended_action']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    round_dir = args.round_dir.resolve()
    output_dir = (args.output_dir or (round_dir / "qc")).resolve()

    organ_qc = build_organ_qc(round_dir)
    case_qc = build_case_qc(round_dir)
    summary = build_summary(organ_qc, case_qc, round_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    organ_qc.to_csv(output_dir / "round1_organ_qc.csv", index=False)
    case_qc.to_csv(output_dir / "round1_case_qc.csv", index=False)
    write_json(output_dir / "round1_qc_summary.json", summary)
    write_markdown(output_dir / "ROUND1_QC_REPORT.md", summary, organ_qc, case_qc)

    print(json.dumps({
        "output_dir": str(output_dir),
        "organ_qc": str(output_dir / "round1_organ_qc.csv"),
        "case_qc": str(output_dir / "round1_case_qc.csv"),
        "summary": str(output_dir / "round1_qc_summary.json"),
        "report": str(output_dir / "ROUND1_QC_REPORT.md"),
        "organ_qc_category_counts": summary["organ_qc_category_counts"],
        "case_risk_counts": summary["case_risk_counts"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
