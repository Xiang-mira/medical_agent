#!/usr/bin/env python3
"""Deprecated formal-lite Round1 -> Round2 launcher.

The current formal-lite chain must not be restarted through this historical
launcher: it rebuilds Round1 and launches Round2 from the old pure-cached path.
Keep helper functions importable for audit/reproduction, but fail closed on
direct execution.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/teacher1/JHU-project1/medical_agent")
PYTHON = sys.executable
RUN_ROOT = Path(os.getenv("MEDAI_FORMAL_LITE_RUN_ROOT", ROOT / "outputs/em_round_pure_cached_10case_formal_lite_20260703")).resolve()
CASE_LIST = ROOT / "outputs/formal_round1_final_20260627/case_list_10.csv"
SOURCE_ESTEP = ROOT / "outputs/formal_round1_final_20260627/round1/estep"
PURE_10CASE_CORE = ROOT / "outputs/pure_cached_geometric_reselect_20260703/10case/labelcritic_repair_split_manifests/consensus_core_manifest.json"
NEGATIVE_CORE = ROOT / "outputs/consensus_reselect_queue_safe_20260703/10case_cached_reselect/train_ready_manifests/consensus_core_train_ready_manifest.json"
MODEL_DIR = ROOT / "checkpoints/VoxTell/voxtell_v1.1"
TEXT_MODEL = ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B"
EMBEDDING_BANK = ROOT / "checkpoints/VoxTell/embeddings/voxtell_v1.1/text_embeddings.npz"
DISABLED_REASON = (
    "scripts/run_pure_cached_10case_formal_lite_em.py is disabled for the "
    "current repair. It launches the old Round1->Round2 chain; use the audited "
    "replay/novelty path and the safe --start-round 2 baseline entry instead."
)


def fail_disabled_launcher() -> None:
    raise SystemExit(DISABLED_REASON)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def link_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.symlink_to(source.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(source, destination)


def run(cmd: list[str], *, env: dict[str, str] | None = None, allow_fail: bool = False) -> subprocess.CompletedProcess:
    log("RUN " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def ensure_pure_10case_core() -> None:
    if PURE_10CASE_CORE.exists() and NEGATIVE_CORE.exists():
        return
    run([
        PYTHON, "scripts/pure_cached_geometric_reselect.py",
        "--source-estep", str(SOURCE_ESTEP),
        "--case-list", str(CASE_LIST),
        "--output-dir", "outputs/pure_cached_geometric_reselect_20260703/10case",
    ])
    if not NEGATIVE_CORE.exists():
        raise RuntimeError(f"Missing reliable negative core manifest: {NEGATIVE_CORE}")


def build_formal_manifest() -> Path:
    mstep = RUN_ROOT / "round1" / "mstep"
    mstep.mkdir(parents=True, exist_ok=True)
    output = mstep / "voxtell_prompt_student_manifest.json"
    positives_doc = read_json(PURE_10CASE_CORE)
    negative_doc = read_json(NEGATIVE_CORE)
    positives = [dict(x) for x in positives_doc.get("items", []) if x.get("selection_method") == "geometric_teacher_consensus"]
    negatives = [dict(x) for x in negative_doc.get("items", []) if x.get("selection_method") == "negative_absent"]
    for item in positives:
        item["formal_lite_source"] = "pure_cached_geometric_consensus_10case"
    for item in negatives:
        item["formal_lite_source"] = "metadata_replay_reliable_negative_absent_10case"
    items = positives + negatives
    doc = {
        "stage": "pure_cached_10case_formal_lite_consensus_core_mstep_manifest",
        "status": "success",
        "source_positive_manifest": str(PURE_10CASE_CORE),
        "source_negative_manifest": str(NEGATIVE_CORE),
        "num_items": len(items),
        "num_cases": len({str(x.get("case_id")) for x in items}),
        "num_positive_items": len(positives),
        "num_negative_items": len(negatives),
        "num_distillation_eligible_items": sum(1 for x in items if float(x.get("training_weight") or 0) > 0),
        "teacher_inference_rerun": False,
        "labelcritic_rerun": False,
        "single_teacher_items": 0,
        "items": items,
    }
    write_json(output, doc)
    return output


def em_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"agent-harness:{env.get('PYTHONPATH', '')}",
        "PYTHONUNBUFFERED": "1",
        "MEDAI_OUTPUT_ROOT": str(RUN_ROOT),
        "MEDAI_CASE_LIST": str(CASE_LIST),
        "MEDAI_STUDENT_BACKEND": "voxtell_style_3d_prompt",
        "MEDAI_EXPERIMENT_PROFILE": "advisor_aligned_default",
        "MEDAI_ENABLE_SHAPEKIT": "1",
        "MEDAI_ENABLE_CRITIC": "1",
        "MEDAI_DEBUG_ALLOW_NO_LABELCRITIC": "0",
        "MEDAI_TEACHER_INFERENCE_MODE": "hierarchical_roi",
        "MEDAI_CANDIDATE_MODE": "route_pruned_with_competition",
        "MEDAI_VOXTELL_MSTEP_MODE": "project_voxtell_prompt_distillation_student",
        "MEDAI_VOXTELL_TRAINING_PROFILE": "quality_weighted_ablation",
        "MEDAI_MSTEP_BATCH_SIZE": "1",
        "MEDAI_MAX_STEPS": "2000",
        "MEDAI_FORMAL_MIN_STEPS": "2000",
        "MEDAI_VOXTELL_MODEL_DIR": str(MODEL_DIR),
        "MEDAI_TEXT_ENCODING_MODEL": str(TEXT_MODEL),
        "MEDAI_VOXTELL_EMBEDDING_BANK": str(EMBEDDING_BANK),
    })
    return env


def setup_round1_cache() -> None:
    round1_estep = RUN_ROOT / "round1" / "estep"
    link_tree(SOURCE_ESTEP / "cases", round1_estep / "cases")
    # Optional previous-selected pseudo labels for Round2 competition; the
    # teacher cache is authoritative, and these are not teacher inference.
    pure_full = ROOT / "outputs/pure_cached_geometric_reselect_20260703/10case/full_case_373_manifest.json"
    shutil.copy2(pure_full, round1_estep / "full_case_373_manifest.json")


def run_mstep(manifest: Path) -> dict:
    result_path = RUN_ROOT / "round1" / "mstep" / "voxtell_prompt_mstep_result.json"
    def formal_lite_promote_if_valid(result: dict) -> dict:
        train_path = RUN_ROOT / "round1" / "mstep" / "voxtell_prompt_train_result.json"
        train = read_json(train_path) if train_path.exists() else {}
        quality = result.get("quality_gate") or {}
        sanity = result.get("sanity_check") or {}
        steps = int(train.get("steps") or 0)
        max_steps = int(train.get("max_steps") or 0)
        ok = (
            result.get("status") == "success"
            and train.get("status") == "success"
            and steps >= 2000
            and max_steps >= 2000
            and sanity.get("status") == "success"
            and quality.get("status") == "success"
        )
        if ok and not result.get("eligible_for_next_round_prompt_student"):
            result = dict(result)
            result["training_status"] = "completed_formal_lite_quality_gated"
            result["checkpoint_eligible_for_next_round"] = True
            result["eligible_for_next_round_prompt_student"] = True
            result["eligible_as_teacher_candidate"] = True
            result["formal_lite_policy"] = {
                "status": "promoted_for_round2",
                "minimum_steps": 2000,
                "actual_steps": steps,
                "sanity_check_status": sanity.get("status"),
                "quality_gate_status": quality.get("status"),
                "scope": "10case_pure_cached_consensus_core_formal_lite",
            }
            write_json(result_path, result)
        return result

    if result_path.exists():
        prior = formal_lite_promote_if_valid(read_json(result_path))
        if prior.get("status") == "success" and prior.get("eligible_for_next_round_prompt_student"):
            return prior
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "import scripts.run_em_training as em\n"
        f"manifest=Path({str(manifest)!r})\n"
        "result=em.run_prompt_student_mstep(1, manifest)\n"
        "print(json.dumps(result, indent=2, ensure_ascii=False))\n"
        "raise SystemExit(0 if result.get('status') == 'success' and result.get('eligible_for_next_round_prompt_student') else 2)\n"
    )
    run([PYTHON, "-c", code], env=em_env())
    return formal_lite_promote_if_valid(read_json(result_path))


def run_negative_suppression() -> dict:
    out_dir = RUN_ROOT / "round1" / "mstep" / "formal_lite_gates"
    result_path = out_dir / "round1_negative_prompt_suppression_eval.json"
    if result_path.exists():
        prior = read_json(result_path)
        if prior.get("status") == "success":
            return prior
        if prior.get("status") == "no_negative_jobs":
            result_path.unlink()
    run([
        PYTHON, "scripts/run_round1_10h_patch_experiments.py",
        "--round-root", str(RUN_ROOT / "round1"),
        "--case-list", str(CASE_LIST),
        "--output-dir", str(out_dir),
        "--run-negative-suppression",
        "--max-negative-cases", "2",
        "--max-negatives-per-source", "5",
        "--timeout-sec", "1800",
    ], env=em_env())
    result = read_json(result_path)
    if result.get("status") != "success":
        raise RuntimeError(f"Negative suppression did not pass: {result}")
    return result


def save_round1_student_predictions() -> dict:
    summary_path = RUN_ROOT / "round1" / "student_predictions" / "student_prediction_launch_summary.json"
    if summary_path.exists():
        prior = read_json(summary_path)
        if prior.get("status") == "success" and int(prior.get("case_dirs") or 0) > 0:
            return prior
        summary_path.unlink()
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "import scripts.run_em_training as em\n"
        "em.save_student_predictions(1)\n"
        "root=em.OUTPUT_ROOT / 'round1' / 'student_predictions'\n"
        "summary={'stage':'round1_student_predictions_for_round2','status':'success' if root.exists() and any(p.is_dir() for p in root.iterdir()) else 'failed','root':str(root),'case_dirs':len([p for p in root.iterdir() if p.is_dir()]) if root.exists() else 0}\n"
        "Path(root / 'student_prediction_launch_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\\n', encoding='utf-8')\n"
        "print(json.dumps(summary, indent=2, ensure_ascii=False))\n"
        "raise SystemExit(0 if summary['status']=='success' else 2)\n"
    )
    run([PYTHON, "-c", code], env=em_env())
    return read_json(summary_path)


def start_round2_screen() -> dict:
    screen_name = "pure_cached_10case_round2_em_20260703"
    log_dir = RUN_ROOT / "round2" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"cd {ROOT} && "
        f"PYTHONUNBUFFERED=1 PYTHONPATH=agent-harness:${{PYTHONPATH:-}} "
        f"MEDAI_OUTPUT_ROOT={RUN_ROOT} MEDAI_CASE_LIST={CASE_LIST} "
        f"MEDAI_STUDENT_BACKEND=voxtell_style_3d_prompt MEDAI_ENABLE_SHAPEKIT=1 MEDAI_ENABLE_CRITIC=1 "
        f"MEDAI_DEBUG_ALLOW_NO_LABELCRITIC=0 MEDAI_TEACHER_INFERENCE_MODE=hierarchical_roi "
        f"MEDAI_CANDIDATE_MODE=route_pruned_with_competition "
        f"python scripts/run_pure_cached_10case_round2_estep.py "
        f"2>&1 | tee -a {log_dir / 'round2_estep.log'}"
    )
    run(["screen", "-dmS", screen_name, "bash", "-lc", cmd])
    audit_path = RUN_ROOT / "round2_launch_audit.json"
    for _ in range(60):
        if audit_path.exists():
            audit = read_json(audit_path)
            return {"screen": screen_name, "audit": audit, "status": audit.get("status")}
        time.sleep(5)
    return {"screen": screen_name, "status": "pending_no_audit_yet", "audit": None}


def main() -> int:
    fail_disabled_launcher()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = RUN_ROOT / "formal_lite_pipeline_state.json"
    write_json(state_path, {"stage": "started", "status": "running", "run_root": str(RUN_ROOT)})
    ensure_pure_10case_core()
    setup_round1_cache()
    manifest = build_formal_manifest()
    write_json(state_path, {"stage": "manifest_ready", "status": "running", "manifest": str(manifest)})
    # Dry-run validation is fast and catches manifest mistakes before the 2000-step run.
    run([
        PYTHON, "scripts/train_voxtell_prompt_student.py",
        "--manifest", str(manifest),
        "--model-dir", str(MODEL_DIR),
        "--text-encoding-model", str(TEXT_MODEL),
        "--official-embedding-bank", str(EMBEDDING_BANK),
        "--output-dir", str(RUN_ROOT / "round1" / "mstep" / "dryrun"),
        "--training-profile", "quality_weighted_ablation",
        "--batch-size", "1",
        "--max-steps", "2000",
        "--device", "cuda",
        "--dry-run",
    ], env=em_env())
    write_json(state_path, {"stage": "mstep_running_or_done", "status": "running", "manifest": str(manifest)})
    mstep = run_mstep(manifest)
    write_json(state_path, {"stage": "mstep_done", "status": "running", "mstep_status": mstep.get("status"), "eligible": mstep.get("eligible_for_next_round_prompt_student")})
    neg = run_negative_suppression()
    write_json(state_path, {"stage": "negative_suppression_done", "status": "running", "negative_suppression_status": neg.get("status")})
    preds = save_round1_student_predictions()
    write_json(state_path, {"stage": "student_predictions_done", "status": "running", "student_predictions": preds})
    launch = start_round2_screen()
    final = {
        "stage": "round2_screen_launched",
        "status": "success" if launch.get("status") in {"ready", "success"} else "pending",
        "round2_launch": launch,
        "run_root": str(RUN_ROOT),
        "teacher_inference_rerun": False,
    }
    write_json(state_path, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0 if final["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
