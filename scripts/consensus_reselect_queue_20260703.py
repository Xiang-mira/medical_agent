#!/usr/bin/env python3
"""Queue the advisor-aligned consensus reselect diagnostics.

This script is intentionally orchestration-only.  It does not change the core
E-step selector or trainer.  It reuses completed E-step metadata/teacher cache,
splits the full case×373 manifest into the three approved views, then runs only
the gated short student diagnostics that are safe for the available evidence.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/home/teacher1/JHU-project1/medical_agent")
QUEUE_ROOT = Path(os.getenv("MEDAI_CONSENSUS_QUEUE_ROOT", ROOT / "outputs/consensus_reselect_queue_20260703")).resolve()
PYTHON = sys.executable

FRESH_3CASE_SOURCE_ESTEP = ROOT / "outputs/fresh_3case_estep_20260703/round1/estep"
FRESH_3CASE_LIST = ROOT / "outputs/labelcritic_373_repair_20260703/fresh_3case_case_list.csv"
FORMAL_10CASE_SOURCE_ESTEP = ROOT / "outputs/formal_round1_final_20260627/round1/estep"
FORMAL_10CASE_LIST = ROOT / "outputs/formal_round1_final_20260627/case_list_10.csv"
REGISTRY = ROOT / "configs/model_registry.yaml"

MODEL_DIR = ROOT / "checkpoints/VoxTell/voxtell_v1.1"
TEXT_MODEL = ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B"
EMBEDDING_BANK = ROOT / "checkpoints/VoxTell/embeddings/voxtell_v1.1/text_embeddings.npz"
TARGET_CONFIG = ROOT / "configs/student_3d_prompt_target_organs.json"
PROMPT_MAP = ROOT / "configs/voxtell_official_prompt_map.json"
ALL_TEACHERS = [
    "cads551", "cads552", "cads553", "cads554", "cads555", "cads556",
    "cads557", "cads558", "cads559", "moose666", "moose888",
    "nnunet_private", "saros_nnunet", "atm", "airrc", "lvp", "daps",
    "epai_20250421", "vsmtrans", "vista3d", "unest", "totalsegmentator",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def run(cmd: list[str], *, env: dict[str, str] | None = None, allow_fail: bool = False) -> subprocess.CompletedProcess:
    log("RUN " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed with return code {proc.returncode}: {' '.join(cmd)}")
    return proc


def case_map(case_list: Path) -> dict[str, dict[str, str]]:
    with case_list.open(encoding="utf-8-sig", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle) if row.get("case_id")}


def prompt_map() -> dict[str, str]:
    doc = read_json(PROMPT_MAP, {}) or {}
    out: dict[str, str] = {}
    for row in doc.get("mappings", []) or []:
        if isinstance(row, dict) and row.get("project_class"):
            out[str(row["project_class"])] = str(row.get("canonical_prompt") or row["project_class"]).strip()
    return out


def is_positive(row: dict[str, Any]) -> bool:
    return str(row.get("target_type") or "").lower() in {"positive_hard", "positive_soft", "hard", "soft"}


def is_negative_absent(row: dict[str, Any]) -> bool:
    return str(row.get("target_type") or "").lower() == "negative_absent"


def build_train_ready_manifest(split_manifest: Path, case_list: Path, estep_root: Path, output: Path) -> dict[str, Any]:
    """Create a trainer-ready prompt/mask manifest from an approved split view."""
    split = read_json(split_manifest, {}) or {}
    cases = case_map(case_list)
    prompts = prompt_map()
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in split.get("items", []) or []:
        try:
            weight = float(row.get("training_weight") or 0.0)
        except Exception:
            weight = 0.0
        if weight <= 0:
            continue
        case_id = str(row.get("case_id") or "")
        organ = str(row.get("organ") or "")
        ct = Path(str(row.get("image") or row.get("ct_path") or cases.get(case_id, {}).get("ct_path") or ""))
        if not ct.is_absolute():
            ct = ROOT / ct
        mask_candidates = [
            row.get("mask"),
            row.get("mask_path"),
            row.get("final_mask"),
            row.get("selected_prediction"),
        ]
        if is_positive(row) and case_id and organ:
            mask_candidates.insert(0, estep_root / "annotation_versions" / case_id / "updated" / f"{organ}.nii.gz")
        if is_negative_absent(row) and case_id:
            mask_candidates.insert(0, estep_root / "annotation_versions" / case_id / "updated" / "negative_targets" / "zero_mask.nii.gz")
        mask: Path | None = None
        for candidate in mask_candidates:
            if not candidate:
                continue
            p = Path(str(candidate))
            if not p.is_absolute():
                p = ROOT / p
            if p.exists():
                mask = p
                break
        prompt = str(row.get("prompt") or row.get("prompt_text") or prompts.get(organ) or organ.replace("_", " ")).strip()
        if not (case_id and organ and ct.exists() and mask and mask.exists() and prompt):
            skipped.append({"case_id": case_id, "organ": organ, "reason": "missing_case_ct_mask_or_prompt"})
            continue
        item = dict(row)
        item.update({
            "case_id": case_id,
            "organ": organ,
            "image": str(ct),
            "ct_path": str(ct),
            "mask": str(mask),
            "mask_path": str(mask),
            "prompt": prompt,
            "prompt_text": prompt,
            "training_weight": weight,
            "distillation_eligible": True,
            "should_enter_student_training": True,
        })
        if is_negative_absent(row):
            item.update({
                "supervision_type": "negative",
                "target_type": "negative_absent",
                "grade": str(row.get("grade") or "A"),
                "scoring_schema_version": "autolabel_core_v3_absent_negative",
                "negative_source": str(row.get("negative_source") or "case_373_expected_absent"),
                "zero_mask_role": "negative_absent_target_mask",
            })
        else:
            item.update({
                "supervision_type": "positive",
                "target_type": "positive_hard" if str(row.get("target_type") or "").lower() in {"positive_hard", "hard"} else "positive_soft",
                "grade": str(row.get("grade") or "A"),
                "scoring_schema_version": str(row.get("scoring_schema_version") or "autolabel_core_v2"),
            })
        items.append(item)
    counts = Counter(str(item.get("supervision_type") or "positive") for item in items)
    target_counts = Counter(str(item.get("target_type") or "") for item in items)
    doc = {
        "stage": f"train_ready_from_{split.get('stage') or split_manifest.stem}",
        "status": "success",
        "source_split_manifest": str(split_manifest.resolve()),
        "case_list": str(case_list.resolve()),
        "estep_root": str(estep_root.resolve()),
        "target_config": str(TARGET_CONFIG),
        "num_items": len(items),
        "num_cases": len({item["case_id"] for item in items}),
        "num_positive_items": counts.get("positive", 0),
        "num_negative_items": counts.get("negative", 0),
        "num_distillation_eligible_items": len(items),
        "target_type_counts": dict(target_counts),
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:50],
        "items": items,
    }
    write_json(output, doc)
    return doc


def split_and_audit(name: str, source_estep: Path, case_list: Path) -> dict[str, Any]:
    out_root = QUEUE_ROOT / name
    estep = out_root / "estep"
    log(f"=== {name}: cached metadata replay/reselect ===")
    run([
        PYTHON, "scripts/rebuild_mstep_from_existing_estep.py",
        "--source-estep", str(source_estep),
        "--output-root", str(out_root),
        "--case-list", str(case_list),
        "--absent-negative-training-weight", "0.1",
    ])
    full_manifest = estep / "full_case_373_manifest.json"
    audit_path = estep / "full_case_373_manifest_audit.json"
    audit_proc = run([
        PYTHON, "scripts/audit_full_case373_manifest.py",
        "--manifest", str(full_manifest),
        "--output", str(audit_path),
    ], allow_fail=True)
    split_dir = estep / "labelcritic_repair_split_manifests"
    run([
        PYTHON, "scripts/split_labelcritic_repair_manifests.py",
        "--input", str(full_manifest),
        "--output-dir", str(split_dir),
    ])
    # Code-level regression for "family changes do not affect winner".
    pytest_proc = run([
        PYTHON, "-m", "pytest",
        "agent-harness/tests/test_labelcritic_373_repair_contract.py",
        "-k", "geometric_consensus_ignores_outlier_and_family_metadata",
        "-q",
    ], allow_fail=True)
    summary = read_json(split_dir / "manifest_split_summary.json", {}) or {}
    audit = read_json(audit_path, {}) or {}
    core = read_json(split_dir / "consensus_core_manifest.json", {}) or {}
    audit_manifest = read_json(split_dir / "labelcritic_audit_manifest.json", {}) or {}
    core_positives = [r for r in core.get("items", []) if str(r.get("selection_method")) == "geometric_teacher_consensus"]
    audit_leaks = [
        r for r in audit_manifest.get("items", [])
        if float(r.get("training_weight") or 0.0) != 0.0
        or r.get("should_enter_student_training") is True
        or r.get("distillation_eligible") is True
    ]
    gate = {
        "stage": f"{name}_cached_reselect_gate",
        "status": "passed" if (
            audit_proc.returncode == 0
            and pytest_proc.returncode == 0
            and not audit_leaks
            and not [r for r in core_positives if r.get("winner_is_original_teacher") is not True]
        ) else "failed",
        "source_estep": str(source_estep),
        "case_list": str(case_list),
        "output_root": str(out_root),
        "full_manifest": str(full_manifest),
        "full_audit": audit,
        "split_summary": summary,
        "audit_only_training_leaks": len(audit_leaks),
        "geometric_positive_items": len(core_positives),
        "geometric_positive_organs": len({str(r.get("organ")) for r in core_positives}),
        "geometric_positive_cases": len({str(r.get("case_id")) for r in core_positives}),
        "geometric_winner_original_failures": [
            {"case_id": r.get("case_id"), "organ": r.get("organ"), "selected_model": r.get("selected_model")}
            for r in core_positives if r.get("winner_is_original_teacher") is not True
        ],
        "family_change_regression_pytest_returncode": pytest_proc.returncode,
    }
    train_ready_dir = out_root / "train_ready_manifests"
    gate["train_ready_core"] = str((train_ready_dir / "consensus_core_train_ready_manifest.json").resolve())
    gate["train_ready_ablation"] = str((train_ready_dir / "single_teacher_ablation_train_ready_manifest.json").resolve())
    core_train = build_train_ready_manifest(
        split_dir / "consensus_core_manifest.json",
        case_list,
        estep,
        train_ready_dir / "consensus_core_train_ready_manifest.json",
    )
    ablation_train = build_train_ready_manifest(
        split_dir / "single_teacher_ablation_manifest.json",
        case_list,
        estep,
        train_ready_dir / "single_teacher_ablation_train_ready_manifest.json",
    )
    gate["train_ready_core_counts"] = {k: core_train.get(k) for k in ("num_items", "num_positive_items", "num_negative_items", "num_cases", "skipped_count")}
    gate["train_ready_ablation_counts"] = {k: ablation_train.get(k) for k in ("num_items", "num_positive_items", "num_negative_items", "num_cases", "skipped_count")}
    write_json(out_root / "cached_reselect_gate.json", gate)
    return gate


def cached_models_present_for_all_cases(source_estep: Path, case_list: Path) -> tuple[list[str], dict[str, Any]]:
    cases = sorted(case_map(case_list))
    counts: dict[str, int] = {}
    examples_missing: dict[str, list[str]] = {}
    for teacher in ALL_TEACHERS:
        ok = 0
        missing: list[str] = []
        for case_id in cases:
            seg = source_estep / "cases" / case_id / "hierarchical_predictions" / teacher / "segmentations"
            if seg.exists() and any(seg.glob("*.nii.gz")):
                ok += 1
            else:
                missing.append(case_id)
        counts[teacher] = ok
        if missing:
            examples_missing[teacher] = missing[:5]
    all_present = [teacher for teacher in ALL_TEACHERS if counts.get(teacher) == len(cases)]
    return all_present, {
        "num_cases": len(cases),
        "per_teacher_present_cases": counts,
        "all_cases_present_teachers": all_present,
        "missing_case_examples": examples_missing,
    }


def current_selector_cached_reselect(name: str, source_estep: Path, case_list: Path) -> dict[str, Any]:
    """Rerun current selector only if explicitly allowed.

    The shared hierarchical selector can launch missing child/ROI inference even
    when major cached masks are preseeded. For this queue the user's hard
    constraint is no teacher re-inference, so the safe default is to block this
    fallback unless a future pure-cache selector is added.
    """
    out_root = QUEUE_ROOT / name
    out_root.mkdir(parents=True, exist_ok=True)
    log(f"=== {name}: current selector pure-cache replay safety check ===")
    all_present, cache_audit = cached_models_present_for_all_cases(source_estep, case_list)
    write_json(out_root / "teacher_cache_subset_audit.json", cache_audit)
    if os.getenv("MEDAI_ALLOW_CURRENT_SELECTOR_MAY_RUN_TEACHERS", "0").strip().lower() not in {"1", "true", "yes"}:
        gate = {
            "stage": f"{name}_gate",
            "status": "blocked",
            "reason": "current_selector_path_can_trigger_teacher_inference; blocked_to_preserve_cached_only_contract",
            "teacher_inference_rerun": False,
            "teacher_cache_subset_audit": cache_audit,
            "models_with_cache_for_all_cases": all_present,
        }
        write_json(out_root / "cached_reselect_gate.json", gate)
        return gate

    estep = out_root / "estep"
    if len(all_present) < 2:
        gate = {
            "stage": f"{name}_gate",
            "status": "failed",
            "reason": "fewer_than_two_teachers_have_cache_for_all_cases",
            "teacher_cache_subset_audit": cache_audit,
        }
        write_json(out_root / "cached_reselect_gate.json", gate)
        return gate

    organs = read_json(TARGET_CONFIG, {}).get("target_organs", [])
    preseeded = {
        teacher: source_estep / "cases" / "{case_id}" / "hierarchical_predictions" / teacher / "segmentations"
        for teacher in all_present
    }
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path('agent-harness').resolve()))\n"
        "from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop\n"
        f"case_list=Path({str(case_list.resolve())!r})\n"
        f"out=Path({str(estep.resolve())!r})\n"
        f"models={all_present!r}\n"
        f"organs={organs!r}\n"
        f"preseeded={{k: Path(v) for k, v in { {k: str(v) for k, v in preseeded.items()}!r}.items()}}\n"
        "result=run_multimodel_annotation_loop(\n"
        "    case_list=case_list, output_folder=out, models=models, organs=organs,\n"
        f"    registry_path=Path({str(REGISTRY.resolve())!r}), checkpoint_map_models=False,\n"
        "    enable_shapekit=True, shapekit_root=Path('third_party/ShapeKit-main').resolve(),\n"
        "    enable_critic=False, critic_backend='labelcritic', critic_base_url='http://localhost', critic_port=8000,\n"
        "    dry_run=False, timeout_sec=3600, device='cuda', perf_tracker_path=None, resume=False,\n"
        "    preseeded_model_dirs=preseeded, labelcritic_options={}, candidate_mode='route_pruned_with_competition',\n"
        "    teacher_inference_mode='hierarchical_roi', roi_margin_mm=20.0,\n"
        ")\n"
        "print(json.dumps(result, indent=2, ensure_ascii=False))\n"
        "raise SystemExit(0 if result.get('status') == 'success' else 2)\n"
    )
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"agent-harness:{env.get('PYTHONPATH', '')}",
        "MEDAI_DEBUG_ALLOW_NO_LABELCRITIC": "1",
        "MEDAI_ENABLE_CRITIC": "0",
        "MEDAI_EXPERIMENT_PROFILE": "advisor_aligned_default",
    })
    proc = subprocess.run([PYTHON, "-c", script], cwd=str(ROOT), env=env, check=False)
    if proc.returncode != 0:
        gate = {
            "stage": f"{name}_gate",
            "status": "failed",
            "reason": "current_selector_cached_replay_failed",
            "return_code": proc.returncode,
            "teacher_cache_subset_audit": cache_audit,
        }
        write_json(out_root / "cached_reselect_gate.json", gate)
        return gate

    full_manifest = estep / "full_case_373_manifest.json"
    audit_path = estep / "full_case_373_manifest_audit.json"
    audit_proc = run([
        PYTHON, "scripts/audit_full_case373_manifest.py",
        "--manifest", str(full_manifest),
        "--output", str(audit_path),
    ], allow_fail=True)
    split_dir = estep / "labelcritic_repair_split_manifests"
    run([
        PYTHON, "scripts/split_labelcritic_repair_manifests.py",
        "--input", str(full_manifest),
        "--output-dir", str(split_dir),
    ])
    pytest_proc = run([
        PYTHON, "-m", "pytest",
        "agent-harness/tests/test_labelcritic_373_repair_contract.py",
        "-k", "geometric_consensus_ignores_outlier_and_family_metadata",
        "-q",
    ], allow_fail=True)
    summary = read_json(split_dir / "manifest_split_summary.json", {}) or {}
    audit = read_json(audit_path, {}) or {}
    core = read_json(split_dir / "consensus_core_manifest.json", {}) or {}
    audit_manifest = read_json(split_dir / "labelcritic_audit_manifest.json", {}) or {}
    core_positives = [r for r in core.get("items", []) if str(r.get("selection_method")) == "geometric_teacher_consensus"]
    audit_leaks = [
        r for r in audit_manifest.get("items", [])
        if float(r.get("training_weight") or 0.0) != 0.0
        or r.get("should_enter_student_training") is True
        or r.get("distillation_eligible") is True
    ]
    gate = {
        "stage": f"{name}_current_selector_cached_reselect_gate",
        "status": "passed" if (
            audit_proc.returncode == 0
            and pytest_proc.returncode == 0
            and not audit_leaks
            and not [r for r in core_positives if r.get("winner_is_original_teacher") is not True]
        ) else "failed",
        "source_estep": str(source_estep),
        "case_list": str(case_list),
        "output_root": str(out_root),
        "teacher_inference_rerun": False,
        "teacher_cache_subset_audit": cache_audit,
        "models_used": all_present,
        "full_manifest": str(full_manifest),
        "full_audit": audit,
        "split_summary": summary,
        "audit_only_training_leaks": len(audit_leaks),
        "geometric_positive_items": len(core_positives),
        "geometric_positive_organs": len({str(r.get("organ")) for r in core_positives}),
        "geometric_positive_cases": len({str(r.get("case_id")) for r in core_positives}),
        "geometric_winner_original_failures": [
            {"case_id": r.get("case_id"), "organ": r.get("organ"), "selected_model": r.get("selected_model")}
            for r in core_positives if r.get("winner_is_original_teacher") is not True
        ],
        "family_change_regression_pytest_returncode": pytest_proc.returncode,
    }
    train_ready_dir = out_root / "train_ready_manifests"
    core_train = build_train_ready_manifest(
        split_dir / "consensus_core_manifest.json",
        case_list,
        estep,
        train_ready_dir / "consensus_core_train_ready_manifest.json",
    )
    ablation_train = build_train_ready_manifest(
        split_dir / "single_teacher_ablation_manifest.json",
        case_list,
        estep,
        train_ready_dir / "single_teacher_ablation_train_ready_manifest.json",
    )
    gate["train_ready_core"] = str((train_ready_dir / "consensus_core_train_ready_manifest.json").resolve())
    gate["train_ready_ablation"] = str((train_ready_dir / "single_teacher_ablation_train_ready_manifest.json").resolve())
    gate["train_ready_core_counts"] = {k: core_train.get(k) for k in ("num_items", "num_positive_items", "num_negative_items", "num_cases", "skipped_count")}
    gate["train_ready_ablation_counts"] = {k: ablation_train.get(k) for k in ("num_items", "num_positive_items", "num_negative_items", "num_cases", "skipped_count")}
    write_json(out_root / "cached_reselect_gate.json", gate)
    return gate


def diagnostic_threshold(gate: dict[str, Any]) -> dict[str, Any]:
    ok = (
        gate.get("status") == "passed"
        and int(gate.get("geometric_positive_items") or 0) >= 10
        and int(gate.get("geometric_positive_organs") or 0) >= 4
        and int(gate.get("geometric_positive_cases") or 0) >= 2
    )
    return {
        "status": "passed" if ok else "failed",
        "minimum_positive_case_organs": 10,
        "minimum_organs": 4,
        "minimum_cases": 2,
        "actual_positive_case_organs": int(gate.get("geometric_positive_items") or 0),
        "actual_organs": int(gate.get("geometric_positive_organs") or 0),
        "actual_cases": int(gate.get("geometric_positive_cases") or 0),
    }


def run_mstep_diagnostic(name: str, manifest: Path, max_steps: int) -> dict[str, Any]:
    out_root = QUEUE_ROOT / name
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"agent-harness:{env.get('PYTHONPATH', '')}",
        "MEDAI_OUTPUT_ROOT": str(out_root),
        "MEDAI_CASE_LIST": str(FORMAL_10CASE_LIST if "10case" in name else FRESH_3CASE_LIST),
        "MEDAI_STUDENT_BACKEND": "voxtell_style_3d_prompt",
        "MEDAI_EXPERIMENT_PROFILE": "advisor_aligned_default",
        "MEDAI_VOXTELL_MSTEP_MODE": "project_voxtell_prompt_distillation_student",
        "MEDAI_VOXTELL_TRAINING_PROFILE": "quality_weighted_ablation",
        "MEDAI_MSTEP_BATCH_SIZE": "1",
        "MEDAI_MAX_STEPS": str(max_steps),
        "MEDAI_FORMAL_MIN_STEPS": "50000",
        "MEDAI_VOXTELL_MODEL_DIR": str(MODEL_DIR),
        "MEDAI_TEXT_ENCODING_MODEL": str(TEXT_MODEL),
        "MEDAI_VOXTELL_EMBEDDING_BANK": str(EMBEDDING_BANK),
        "MEDAI_DEBUG_ALLOW_NO_LABELCRITIC": "1",
        "MEDAI_ALLOW_PROMPT_SEMANTIC_WARNINGS": "1",
    })
    script = (
        "import json\n"
        "from pathlib import Path\n"
        "import scripts.run_em_training as em\n"
        f"manifest=Path({str(manifest.resolve())!r})\n"
        "em.ensure_current_student_backend_allowed()\n"
        "result=em.run_prompt_student_mstep(1, manifest)\n"
        "print(json.dumps(result, indent=2, ensure_ascii=False))\n"
        "raise SystemExit(0 if result.get('status') == 'success' else 2)\n"
    )
    log(f"=== {name}: {max_steps}-step prompt-student diagnostic ===")
    proc = subprocess.run([PYTHON, "-c", script], cwd=str(ROOT), env=env, check=False)
    result = read_json(out_root / "round1/mstep/voxtell_prompt_mstep_result.json", {}) or {}
    summary = {
        "stage": name,
        "status": "passed" if proc.returncode == 0 and result.get("status") == "success" else "failed",
        "return_code": proc.returncode,
        "manifest": str(manifest.resolve()),
        "output_root": str(out_root),
        "max_steps": max_steps,
        "mstep_result": result,
        "safe_only_no_benefit_claim": True,
    }
    write_json(out_root / "diagnostic_summary.json", summary)
    return summary


def quality_score(summary: dict[str, Any]) -> tuple[float, float]:
    gate = ((summary.get("mstep_result") or {}).get("quality_gate") or {})
    return (float(gate.get("mean_dsc") or 0.0), float(gate.get("nonempty_recall") or 0.0))


def main() -> int:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"Queue root: {QUEUE_ROOT}")
    overall: dict[str, Any] = {
        "stage": "consensus_reselect_experiment_queue_20260703",
        "status": "running",
        "queue_root": str(QUEUE_ROOT),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": {},
    }
    write_json(QUEUE_ROOT / "queue_summary.json", overall)

    three_gate = split_and_audit("3case_cached_reselect", FRESH_3CASE_SOURCE_ESTEP, FRESH_3CASE_LIST)
    overall["steps"]["3case_cached_reselect"] = three_gate
    three_threshold = diagnostic_threshold(three_gate)
    overall["steps"]["3case_consensus_threshold"] = three_threshold
    write_json(QUEUE_ROOT / "queue_summary.json", overall)

    consensus_25: dict[str, Any] | None = None
    active_gate = three_gate
    active_case = "3case"
    if three_threshold["status"] == "passed":
        consensus_25 = run_mstep_diagnostic(
            "3case_consensus_only_25step",
            Path(three_gate["train_ready_core"]),
            25,
        )
    else:
        overall["steps"]["3case_consensus_only_25step"] = {
            "status": "skipped",
            "reason": "3case_consensus_core_below_minimum_positive_coverage",
            "threshold": three_threshold,
        }
        write_json(QUEUE_ROOT / "queue_summary.json", overall)
        ten_gate = split_and_audit("10case_cached_reselect", FORMAL_10CASE_SOURCE_ESTEP, FORMAL_10CASE_LIST)
        ten_threshold = diagnostic_threshold(ten_gate)
        overall["steps"]["10case_cached_reselect"] = ten_gate
        overall["steps"]["10case_consensus_threshold"] = ten_threshold
        write_json(QUEUE_ROOT / "queue_summary.json", overall)
        active_gate = ten_gate
        active_case = "10case"
        if ten_threshold["status"] == "passed":
            consensus_25 = run_mstep_diagnostic(
                "10case_consensus_only_25step",
                Path(ten_gate["train_ready_core"]),
                25,
            )
        else:
            current_gate = current_selector_cached_reselect(
                "10case_current_cached_reselect",
                FORMAL_10CASE_SOURCE_ESTEP,
                FORMAL_10CASE_LIST,
            )
            current_threshold = diagnostic_threshold(current_gate)
            overall["steps"]["10case_current_cached_reselect"] = current_gate
            overall["steps"]["10case_current_consensus_threshold"] = current_threshold
            write_json(QUEUE_ROOT / "queue_summary.json", overall)
            active_gate = current_gate
            active_case = "10case_current"
            if current_threshold["status"] == "passed":
                consensus_25 = run_mstep_diagnostic(
                    "10case_current_consensus_only_25step",
                    Path(current_gate["train_ready_core"]),
                    25,
                )
            else:
                overall["steps"]["10case_consensus_only_25step"] = {
                    "status": "skipped",
                    "reason": "10case_current_cached_consensus_core_below_minimum_positive_coverage",
                    "threshold": current_threshold,
                }

    if consensus_25 is not None:
        overall["steps"][f"{active_case}_consensus_only_25step"] = consensus_25
    write_json(QUEUE_ROOT / "queue_summary.json", overall)

    ablation_25: dict[str, Any] | None = None
    if consensus_25 and consensus_25.get("status") == "passed":
        ablation_25 = run_mstep_diagnostic(
            f"{active_case}_single_teacher_ablation_25step",
            Path(active_gate["train_ready_ablation"]),
            25,
        )
        overall["steps"][f"{active_case}_single_teacher_ablation_25step"] = ablation_25
    else:
        overall["steps"]["single_teacher_ablation_25step"] = {
            "status": "skipped",
            "reason": "consensus_only_25step_not_safely_passed",
        }
    write_json(QUEUE_ROOT / "queue_summary.json", overall)

    # Conservative "benefit" gate: no expert claim, only pseudo-consistency.
    if consensus_25 and consensus_25.get("status") == "passed" and ablation_25 and ablation_25.get("status") == "passed":
        c_dsc, c_recall = quality_score(consensus_25)
        a_dsc, a_recall = quality_score(ablation_25)
        directional = a_dsc >= c_dsc and a_recall >= c_recall
        benefit_gate = {
            "status": "passed" if directional else "failed",
            "metric_scope": "automatic_pseudo_label_consistency_only",
            "consensus_25_mean_dsc": c_dsc,
            "consensus_25_nonempty_recall": c_recall,
            "ablation_25_mean_dsc": a_dsc,
            "ablation_25_nonempty_recall": a_recall,
            "no_expert_benefit_claim": True,
        }
    else:
        benefit_gate = {
            "status": "failed",
            "reason": "required_25step_safety_runs_not_passed",
            "no_expert_benefit_claim": True,
        }
    overall["steps"]["round1_safety_benefit_gate"] = benefit_gate
    write_json(QUEUE_ROOT / "queue_summary.json", overall)

    if benefit_gate.get("status") == "passed":
        fifty = run_mstep_diagnostic(
            f"{active_case}_consensus_only_50step_round1_gate",
            Path(active_gate["train_ready_core"]),
            50,
        )
        overall["steps"][f"{active_case}_consensus_only_50step_round1_gate"] = fifty
    else:
        overall["steps"]["50step_round1_gate"] = {
            "status": "skipped",
            "reason": "no_safe_reproducible_directional_pseudo_consistency_gain",
            "benefit_gate": benefit_gate,
        }

    overall["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    overall["status"] = "complete"
    write_json(QUEUE_ROOT / "queue_summary.json", overall)
    log(f"Queue complete. Summary: {QUEUE_ROOT / 'queue_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
