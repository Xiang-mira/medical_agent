#!/usr/bin/env python3
"""Run VoxTell-style 3D prompt student inference for a completed round.

This replaces the old VISTA3D/127-class helper. The output layout remains:

    outputs/round<round>/student_predictions/<case_id>/<organ>.nii.gz

Round 2+ E-step can inject that folder as `student_prev` through
`preseeded_model_dirs`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-harness"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CASE_LIST = PROJECT_ROOT / "data_manifest/case_list_50_tumor.csv"
TARGET_CONFIG = PROJECT_ROOT / "configs/student_3d_prompt_target_organs.json"
VOXTELL_MODEL_DIR = Path(os.getenv("MEDAI_VOXTELL_MODEL_DIR", PROJECT_ROOT / "checkpoints/VoxTell/voxtell_v1.1"))
TEXT_ENCODING_MODEL = Path(os.getenv("MEDAI_TEXT_ENCODING_MODEL", PROJECT_ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B"))
QWEN_VLM_MODEL = Path(os.getenv("MEDAI_QWEN_VLM_MODEL", PROJECT_ROOT / "checkpoints/Qwen/Qwen2-VL-72B-Instruct-AWQ"))
if not QWEN_VLM_MODEL.is_absolute():
    QWEN_VLM_MODEL = PROJECT_ROOT / QWEN_VLM_MODEL
VLLM_BASE_URL = os.getenv("MEDAI_VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
LABELCRITIC_PORT = int(os.getenv("LABELCRITIC_PORT", "8000"))
LABELCRITIC_MODEL_ID = os.getenv("LABELCRITIC_MODEL_ID", "Qwen/Qwen2-VL-72B-Instruct-AWQ")
MANAGE_OWN_VLLM = os.getenv("MEDAI_MANAGE_OWN_VLLM", "0").strip().lower() in {"1", "true", "yes", "on"}
LOG_FILE = OUTPUT_ROOT / "training.log"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run 3D prompt student inference for Round N.")
    ap.add_argument("--round", type=int, default=1, dest="round_idx", help="Round index whose M-step checkpoint should be used.")
    ap.add_argument("--case-list", default=str(CASE_LIST), help="CSV with case_id,ct_path.")
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Project outputs root.")
    ap.add_argument("--target-config", default=str(TARGET_CONFIG), help="3D prompt target-organ config.")
    ap.add_argument("--model-dir", default=None, help="Explicit VoxTell model dir. Defaults to round finetuned dir, then official checkpoint.")
    ap.add_argument("--text-encoding-model", default=str(TEXT_ENCODING_MODEL), help="Local Qwen3 embedding model path.")
    ap.add_argument("--device", default=os.getenv("MEDAI_DEVICE", "cuda"))
    ap.add_argument("--gpu", type=int, default=int(os.getenv("MEDAI_GPU", "0")))
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--prompt-batch-size", type=int, default=16, help="Run VoxTell prompts in batches; do not push all 373 prompts at once.")
    ap.add_argument("--max-cases", type=int, default=0, help="Optional smoke-test case limit.")
    ap.add_argument("--prompts", default="", help="Optional comma-separated organ subset. Default: all 373 exact targets.")
    ap.add_argument("--dry-run", action="store_true", help="Write per-case VoxTell commands without running inference.")
    ap.add_argument("--postprocess", action="store_true", help="After student inference, write anatomy-containment postprocessed masks to a separate root.")
    ap.add_argument("--cascade", action="store_true", help="Run true parent-ROI cropped inference for configured child organs.")
    ap.add_argument("--postprocess-policy", default=str(PROJECT_ROOT / "configs/organ_postprocess_policy.yaml"))
    ap.add_argument("--taxonomy", default=str(PROJECT_ROOT / "configs/organ_taxonomy.json"))
    ap.add_argument("--teacher-root", default="", help="Optional teacher/selected-pseudo root to provide reliable parent organ masks.")
    ap.add_argument("--parent-root", action="append", default=[], help="Additional parent-mask root. Can be repeated.")
    ap.add_argument("--student-parent-allowlist", default="", help="Allowlist proving which student masks may provide a parent ROI.")
    ap.add_argument("--cascaded-output-root", default="", help="Default: outputs/roundN/student_predictions_cascaded")
    ap.add_argument("--postprocessed-output-root", default="", help="Default: outputs/roundN/student_predictions_postprocessed")
    ap.add_argument("--postprocess-overwrite", action="store_true")
    ap.add_argument("--restart-vllm", action="store_true", help="Legacy debug only; formal 72B AWQ requires external multi-GPU SLURM vLLM.")
    return ap.parse_args()


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_target_organs(target_config: Path, prompt_subset: str = "") -> list[str]:
    doc = json.loads(target_config.read_text(encoding="utf-8"))
    targets = list(doc.get("target_organs", []))
    if prompt_subset.strip():
        requested = [x.strip() for x in prompt_subset.split(",") if x.strip()]
        missing = [x for x in requested if x not in set(targets)]
        if missing:
            raise SystemExit(f"Unknown/non-target organs requested: {missing[:20]}")
        return requested
    return targets


def load_cases(case_list: Path, max_cases: int = 0) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with case_list.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("case_id") and row.get("ct_path"):
                rows.append(row)
    return rows[:max_cases] if max_cases > 0 else rows


def default_round_model_dir(output_root: Path, round_idx: int) -> Path:
    finetuned = output_root / f"round{round_idx}" / "mstep" / "voxtell_finetuned_model"
    if (finetuned / "plans.json").exists() and (finetuned / "fold_0" / "checkpoint_final.pth").exists():
        return finetuned
    return VOXTELL_MODEL_DIR


def restart_vllm() -> bool:
    if not MANAGE_OWN_VLLM:
        log("拒绝重启 vLLM：请使用外部 SLURM vLLM 服务")
        return False
    if "72b" in QWEN_VLM_MODEL.name.lower():
        log("拒绝重启 vLLM：72B AWQ 必须由外部多 GPU SLURM 作业启动")
        return False
    cmd = (
        "screen -dmS vllm_server bash -c '"
        f"cd {PROJECT_ROOT} && "
        "python -m vllm.entrypoints.openai.api_server "
        f"--model {QWEN_VLM_MODEL} "
        f"--served-model-name {LABELCRITIC_MODEL_ID} "
        f"--port {LABELCRITIC_PORT} "
        "--max-model-len 4096 "
        "--gpu-memory-utilization 0.4"
        "'"
    )
    subprocess.run(cmd, shell=True, check=False)
    log("vLLM 已在后台重启，等待上线...")
    import urllib.request
    for i in range(30):
        time.sleep(5)
        try:
            urllib.request.urlopen(f"{VLLM_BASE_URL}/v1/models", timeout=3)
            log(f"vLLM 在线 ({(i + 1) * 5}s)")
            return True
        except Exception:
            pass
    log("vLLM 启动超时，后续 LabelCritic 可能不可用")
    return False


def inference_case_acceptable(result: dict[str, Any], expected_organs: int) -> bool:
    """True when the case produced the requested standardized mask files.

    VoxTell may legitimately output an empty mask for an absent/out-of-FOV
    prompt (for example ``brain`` on an abdomen CT).  That makes the per-case
    result ``partial_success`` for review purposes, but it should still count as
    an inference-layout success when all requested masks were materialized and
    no organ failed.
    """
    status = str(result.get("status") or "")
    if status in {"success", "skipped_existing"}:
        return True
    if status != "partial_success":
        return False
    if int(result.get("num_failed_organs") or 0) > 0:
        return False
    return int(result.get("num_masks") or 0) >= int(expected_organs)


def existing_case_complete(case_pred_dir: Path, organs: list[str]) -> bool:
    """Only resume-skip a case with a successful result and every exact mask."""
    result_path = case_pred_dir / "voxtell_student_result.json"
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not inference_case_acceptable(result, len(organs)):
        return False
    return all((case_pred_dir / f"{organ}.nii.gz").is_file() for organ in organs)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    case_list = Path(args.case_list).resolve()
    target_config = Path(args.target_config).resolve()
    model_dir = Path(args.model_dir).resolve() if args.model_dir else default_round_model_dir(output_root, args.round_idx)
    pred_dir = output_root / f"round{args.round_idx}" / "student_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    from cli_anything.medai.core.voxtell_student import VoxTellStudent
    from cli_anything.medai.core.student_postprocess import process_student_root
    from cli_anything.medai.core.student_cascade import cascade_case

    organs = load_target_organs(target_config, args.prompts)
    cases = load_cases(case_list, args.max_cases)
    student = VoxTellStudent(
        model_dir=model_dir,
        target_config=target_config,
        text_encoding_model=Path(args.text_encoding_model).resolve(),
        device=args.device,
        gpu=args.gpu,
    )

    log("=" * 60)
    log(f"Round {args.round_idx} VoxTell-style 3D prompt student inference")
    log(f"  Cases: {len(cases)}, prompts: {len(organs)}")
    log(f"  Model dir: {model_dir}")
    log(f"  Target config: {target_config}")
    log(f"  Output: {pred_dir}")
    log(f"  Dry run: {args.dry_run}")
    log("=" * 60)

    saved = 0
    results: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        ct_path = Path(case["ct_path"]).resolve()
        case_id = case["case_id"]
        if not ct_path.exists() and not args.dry_run:
            log(f"  [{i}/{len(cases)}] {case_id}: CT 不存在，跳过")
            results.append({"case_id": case_id, "status": "skipped", "reason": f"CT not found: {ct_path}"})
            continue
        case_pred_dir = pred_dir / case_id
        if not args.dry_run and existing_case_complete(case_pred_dir, organs):
            saved += 1
            log(f"  [{i}/{len(cases)}] {case_id}: 已存在完整的 {len(organs)} masks，跳过")
            results.append({
                "case_id": case_id,
                "status": "skipped_existing",
                "num_masks": len(organs),
                "output_dir": str(case_pred_dir),
            })
            continue
        result = student.segment(
            ct_image=ct_path,
            prompts=organs,
            output_dir=case_pred_dir,
            dry_run=args.dry_run,
            timeout_sec=args.timeout_sec,
            prompt_batch_size=args.prompt_batch_size,
        )
        result["case_id"] = case_id
        results.append(result)
        if inference_case_acceptable(result, len(organs)):
            saved += 1
            status_note = "empty masks present" if result.get("status") == "partial_success" else "ok"
            log(f"  [{i}/{len(cases)}] {case_id}: ✓ {result.get('num_masks', 0)}/{len(organs)} masks ({status_note})")
        else:
            log(f"  [{i}/{len(cases)}] {case_id}: {result.get('status')} ({result.get('reason', 'see result json')})")

    run_complete = saved == len(cases)
    summary = {
        "stage": "round_student_inference",
        "student_backend": "voxtell_style_3d_prompt",
        "round": args.round_idx,
        "status": "dry_run" if args.dry_run else ("success" if run_complete else "failed"),
        "num_cases": len(cases),
        "num_prompts": len(organs),
        "prompt_batch_size": args.prompt_batch_size,
        "num_success_or_existing": saved,
        "success_count_policy": "success, skipped_existing, or partial_success with all requested masks materialized and no failed organs",
        "model_dir": str(model_dir),
        "target_config": str(target_config),
        "prediction_dir": str(pred_dir),
        "results": results,
    }
    (pred_dir / "student_inference_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"3D prompt student 推理完成/计划完成: {saved}/{len(cases)} cases")

    cascade_summary: dict[str, Any] | None = None
    if args.cascade:
        if not args.teacher_root and not args.parent_root:
            raise SystemExit("--cascade requires at least one reliable --teacher-root/--parent-root; raw student fallback is forbidden")
        cascade_root = (
            Path(args.cascaded_output_root).resolve()
            if args.cascaded_output_root
            else output_root / f"round{args.round_idx}" / "student_predictions_cascaded"
        )
        policy = __import__("yaml").safe_load(Path(args.postprocess_policy).read_text(encoding="utf-8")) or {}
        taxonomy = __import__("cli_anything.medai.core.organ_taxonomy", fromlist=["load_taxonomy"]).load_taxonomy(
            Path(args.taxonomy).resolve()
        )
        from cli_anything.medai.core.student_postprocess import containment_rule_for_organ
        cascade_organs = [
            organ for organ in organs
            if containment_rule_for_organ(organ, taxonomy, policy).enabled
        ]
        teacher_roots = [Path(x).resolve() for x in args.parent_root]
        if args.teacher_root:
            teacher_roots.insert(0, Path(args.teacher_root).resolve())

        def infer_roi(cropped_ct: Path, organ: str, roi_output: Path) -> dict[str, Any]:
            return student.segment(
                ct_image=cropped_ct,
                prompts=[organ],
                output_dir=roi_output,
                dry_run=args.dry_run,
                timeout_sec=args.timeout_sec,
                prompt_batch_size=1,
            )

        cascade_results = []
        for case in cases:
            cascade_results.append(cascade_case(
                case_id=case["case_id"],
                ct_path=Path(case["ct_path"]).resolve(),
                output_root=cascade_root,
                inference=infer_roi,
                teacher_parent_roots=teacher_roots,
                student_parent_root=pred_dir,
                student_parent_allowlist=Path(args.student_parent_allowlist).resolve() if args.student_parent_allowlist else None,
                taxonomy_path=Path(args.taxonomy).resolve(),
                policy_path=Path(args.postprocess_policy).resolve(),
                organs=cascade_organs,
                dry_run=args.dry_run,
            ))
        cascade_summary = {
            "stage": "parent_roi_cascaded_student_inference",
            "status": "dry_run" if args.dry_run else "success",
            "cases": len(cascade_results),
            "target_organs": cascade_organs,
            "output_root": str(cascade_root),
            "results": cascade_results,
        }
        (cascade_root / "student_cascade_summary.json").write_text(
            json.dumps(cascade_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary["cascade"] = cascade_summary
        (pred_dir / "student_inference_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.postprocess:
        post_root = Path(args.postprocessed_output_root).resolve() if args.postprocessed_output_root else output_root / f"round{args.round_idx}" / "student_predictions_postprocessed"
        parent_roots = [Path(x).resolve() for x in args.parent_root]
        if args.teacher_root:
            parent_roots.append(Path(args.teacher_root).resolve())
        if args.student_parent_allowlist:
            parent_roots.append(pred_dir.resolve())
        log(f"开始 student containment post-processing: {post_root}")
        post_summary = process_student_root(
            input_root=pred_dir.resolve(),
            output_root=post_root.resolve(),
            parent_roots=parent_roots,
            taxonomy_path=Path(args.taxonomy).resolve(),
            policy_path=Path(args.postprocess_policy).resolve(),
            case_list=case_list,
            organs=organs,
            max_cases=args.max_cases,
            overwrite=args.postprocess_overwrite,
            write_roi_masks=True,
            dry_run=args.dry_run,
        )
        summary["postprocess"] = post_summary
        summary["postprocessed_prediction_dir"] = str(post_root)
        (pred_dir / "student_inference_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"student containment post-processing 完成: {post_summary.get('processed_masks', 0)} masks")

    if args.restart_vllm:
        restart_vllm()
    return 0 if args.dry_run or run_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
