#!/usr/bin/env python3
"""Three-phase blind-10 protocol: student -> full agent -> sealed evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

TEACHERS = [
    "cads551", "cads552", "cads553", "cads554", "cads555", "cads556",
    "cads557", "cads558", "cads559", "moose666", "moose888",
    "nnunet_private", "saros_nnunet", "atm", "airrc", "lvp", "daps",
    "epai_20250421", "vsmtrans", "vista3d", "unest", "totalsegmentator",
]


def args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["student", "agent", "evaluate"])
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/blind10_batch1.csv"))
    ap.add_argument("--student-model-dir")
    ap.add_argument("--output-root", default=str(ROOT / "outputs/blind10_batch1"))
    ap.add_argument("--reference-root", default=str(ROOT / "data/PanTS/LabelTr"))
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) != {"case_id", "ct_path"}:
            raise SystemExit("Blind case list must contain exactly case_id,ct_path")
        rows = list(reader)
    for row in rows:
        if "Label" in row["ct_path"] or "annotation" in row["ct_path"].lower():
            raise SystemExit(f"Reference-like path forbidden in blind inference: {row}")
    return rows


def run_student(cases: list[dict[str, str]], model_dir: Path, out: Path, device: str) -> None:
    root = out / "student_only"
    for case in cases:
        case_out = root / case["case_id"]
        result_path = case_out / "student_auto_segmentation_result.json"
        if result_path.exists():
            continue
        subprocess.run([
            sys.executable, str(ROOT / "scripts/student_auto_segmentation_cli.py"),
            "--model-dir", str(model_dir), "--image", case["ct_path"],
            "--output-dir", str(case_out), "--device", device,
        ], check=True)
    (out / "student_phase_complete.json").write_text(json.dumps({
        "status": "complete", "model_dir": str(model_dir.resolve()), "cases": [x["case_id"] for x in cases],
        "reference_access": False,
    }, indent=2))


def find_student_segmentations(base: Path, case_id: str) -> Path:
    candidates = [
        base / case_id,
        base / case_id / "segmentations",
        base / case_id / case_id / "segmentations",
    ]
    candidates.extend((base / case_id).glob("**/segmentations"))
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.nii.gz")):
            return candidate
    raise FileNotFoundError(f"No student segmentations found for {case_id}")


def run_agent(case_list: Path, cases: list[dict[str, str]], out: Path, device: str) -> None:
    from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop
    from cli_anything.medai.core.target_space import canonical_target_name, load_student_target_space

    student_base = out / "student_only"
    preseed_root = out / "preseeded_student"
    for case in cases:
        source = find_student_segmentations(student_base, case["case_id"])
        destination = preseed_root / case["case_id"] / "segmentations"
        destination.mkdir(parents=True, exist_ok=True)
        for mask in source.glob("*.nii.gz"):
            # The Student writes canonical names plus official prompt aliases.
            # Seed only canonical files so ShapeKit and candidate scoring do not
            # process the same prediction twice.
            if mask.name.startswith("ct_segment_the_"):
                continue
            target = destination / mask.name
            if not target.exists():
                target.symlink_to(mask.resolve())
    doc = load_student_target_space()
    organs = list(dict.fromkeys(canonical_target_name(x) for x in doc.get("target_organs", [])))
    result = run_multimodel_annotation_loop(
        case_list=case_list,
        output_folder=out / "full_agent",
        models=TEACHERS,
        organs=organs,
        registry_path=ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=True,
        shapekit_root=ROOT / "third_party/ShapeKit-main",
        enable_shapekit=True,
        enable_critic=True,
        critic_backend="labelcritic",
        critic_base_url="http://localhost",
        critic_port=8000,
        dry_run=False,
        timeout_sec=3600,
        device=device,
        resume=True,
        teacher_inference_mode="hierarchical_roi",
        preseeded_model_dirs={"student_blind": preseed_root},
    )
    (out / "agent_phase_complete.json").write_text(json.dumps({
        "status": result.get("status"), "cases": [x["case_id"] for x in cases],
        "reference_access": False, "result": result,
    }, indent=2, default=str))


def evaluate(case_list: Path, out: Path, reference_root: Path) -> None:
    if not (out / "student_phase_complete.json").exists() or not (out / "agent_phase_complete.json").exists():
        raise SystemExit("Both inference phases must complete before references can be opened")
    commands = [
        ("student", out / "student_only"),
        ("agent", out / "full_agent"),
    ]
    for name, prediction_root in commands:
        subprocess.run([
            sys.executable, str(ROOT / "scripts/evaluate_blind_segmentation.py"),
            "--case-list", str(case_list), "--prediction-root", str(prediction_root),
            "--reference-root", str(reference_root), "--reference-kind", "historical_pseudo",
            "--output-dir", str(out / "evaluation" / name),
        ], check=True)


def main() -> int:
    a = args()
    case_list = Path(a.case_list).resolve()
    output = Path(a.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = read_cases(case_list)
    if a.phase == "student":
        if not a.student_model_dir:
            raise SystemExit("--student-model-dir is required for student phase")
        run_student(cases, Path(a.student_model_dir).resolve(), output, a.device)
    elif a.phase == "agent":
        run_agent(case_list, cases, output, a.device)
    else:
        evaluate(case_list, output, Path(a.reference_root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
