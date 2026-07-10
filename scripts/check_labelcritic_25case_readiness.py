#!/usr/bin/env python3
"""Readiness gate for the 25-case pseudo-label EM run.

This script is intentionally bounded: vLLM polling is capped by
--max-wait-sec, defaulting to 240 seconds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.labelcritic_wrapper import run_labelcritic_compare


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def endpoint_status(base_url: str, port: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            body = resp.read(4096).decode("utf-8", errors="ignore")
        return {"online": True, "url": url, "body_sample": body[:1000]}
    except Exception as exc:
        return {"online": False, "url": url, "reason": repr(exc)}


def start_vllm_screen(screen_name: str, model_path: Path, port: int) -> dict[str, Any]:
    cmd = (
        "screen -dmS {screen} bash -lc 'cd {root} && "
        "python -m vllm.entrypoints.openai.api_server "
        "--model {model} --port {port} --max-model-len 4096 "
        "--gpu-memory-utilization 0.4'"
    ).format(
        screen=screen_name,
        root=ROOT,
        model=model_path,
        port=port,
    )
    proc = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True, text=True)
    return {
        "screen_name": screen_name,
        "command": cmd,
        "return_code": proc.returncode,
        "stdout": proc.stdout[-1000:],
        "stderr": proc.stderr[-1000:],
    }


def wait_for_vllm(base_url: str, port: int, max_wait_sec: int) -> dict[str, Any]:
    started = time.time()
    checks: list[dict[str, Any]] = []
    while time.time() - started <= max_wait_sec:
        status = endpoint_status(base_url, port)
        checks.append(status)
        if status.get("online"):
            return {
                "online": True,
                "waited_sec": round(time.time() - started, 1),
                "checks": checks[-5:],
                "status": status,
            }
        time.sleep(10)
    return {
        "online": False,
        "waited_sec": round(time.time() - started, 1),
        "checks": checks[-5:],
        "status": checks[-1] if checks else {},
    }


def existing_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.resolve())


def target_coverage_audit() -> dict[str, Any]:
    target_doc = read_json(ROOT / "configs/student_3d_prompt_target_organs.json", {})
    prompt_map_doc = read_json(ROOT / "configs/voxtell_official_prompt_map.json", {})
    appearance_doc = read_json(ROOT / "configs/organ_ct_appearance_373.json", {})
    taxonomy_doc = read_json(ROOT / "configs/organ_taxonomy.json", {})
    targets = list(target_doc.get("target_organs") or [])
    organ_to_id = target_doc.get("organ_to_student_id") or {}
    organ_to_prompt = target_doc.get("organ_to_prompt") or {}
    raw_official_map = prompt_map_doc.get("mappings") or {}
    if isinstance(raw_official_map, list):
        official_map = {
            str(row.get("project_class") or row.get("organ") or ""): row
            for row in raw_official_map
            if isinstance(row, dict)
        }
    else:
        official_map = raw_official_map if isinstance(raw_official_map, dict) else {}
    appearances = appearance_doc.get("organs") or appearance_doc
    taxonomy = taxonomy_doc.get("organs") or taxonomy_doc
    missing: list[dict[str, str]] = []
    generic_fallback: list[str] = []
    for organ in targets:
        prompt = (
            (official_map.get(organ) or {}).get("canonical_prompt")
            if isinstance(official_map.get(organ), dict)
            else None
        ) or organ_to_prompt.get(organ) or organ.replace("_", " ")
        if organ not in organ_to_id:
            missing.append({"organ": organ, "missing": "student_target_id"})
        if not str(prompt or "").strip():
            missing.append({"organ": organ, "missing": "prompt"})
        if organ not in appearances and organ not in taxonomy:
            generic_fallback.append(organ)
    return {
        "stage": "labelcritic_373_coverage_audit",
        "status": "passed" if not missing and len(targets) == 373 else "failed",
        "target_count": len(targets),
        "missing_examples": missing[:50],
        "missing_count": len(missing),
        "generic_prompt_fallback_examples": generic_fallback[:50],
        "generic_prompt_fallback_count": len(generic_fallback),
        "policy": "Every target must have a student id and a usable prompt; LabelCritic uses organ-specific CT appearance when present and a generic 373-target anatomical prompt fallback otherwise.",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-url", default="http://localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-wait-sec", type=int, default=240)
    ap.add_argument("--start-vllm-if-offline", action="store_true")
    ap.add_argument("--vllm-screen", default="vllm_labelcritic_25case")
    ap.add_argument("--vllm-model", type=Path, default=ROOT / "checkpoints/Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--skip-real-compare", action="store_true", help="Only audit files/endpoint/target coverage.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    lc_root = ROOT / "third_party/LabelCritic-main"
    files = {
        "compare_organ": str(lc_root / "CompareOrgan.py"),
        "error_detector": str(lc_root / "ErrorDetector.py"),
        "qwen_vlm_model": str(args.vllm_model),
    }
    for key, raw in files.items():
        if not Path(raw).exists():
            failures.append(f"{key}_missing")

    service = endpoint_status(args.base_url, args.port)
    vllm_start = None
    wait = {"online": service.get("online"), "status": service, "waited_sec": 0}
    if not service.get("online") and args.start_vllm_if_offline:
        vllm_start = start_vllm_screen(args.vllm_screen, args.vllm_model, args.port)
        wait = wait_for_vllm(args.base_url, args.port, max(0, min(args.max_wait_sec, 240)))
        service = wait.get("status") or service
    if not service.get("online"):
        failures.append("vllm_offline")

    coverage = target_coverage_audit()
    if coverage.get("status") != "passed":
        failures.append("labelcritic_373_coverage_failed")

    compare_results: dict[str, Any] = {"status": "skipped"}
    student_compare_results: dict[str, Any] = {"status": "skipped"}
    if service.get("online") and not args.skip_real_compare:
        ct = Path(existing_file(ROOT / "data/PanTS/ImageTr/PanTS_00000100/ct.nii.gz"))
        mask_a = Path(existing_file(ROOT / "outputs/formal_round1_final_20260627/round1/estep/cases/PanTS_00000100/hierarchical_predictions/vsmtrans/segmentations/kidney_left.nii.gz"))
        mask_b = Path(existing_file(ROOT / "outputs/formal_round1_final_20260627/round1/estep/cases/PanTS_00000100/hierarchical_predictions/cads551/segmentations/kidney_left.nii.gz"))
        forward_json = out / "kidney_left_vsmtrans_vs_cads551.json"
        reverse_json = out / "kidney_left_cads551_vs_vsmtrans.json"
        forward = run_labelcritic_compare(
            ct, mask_a, mask_b, "kidney_left", forward_json,
            backend="labelcritic", base_url=args.base_url, port=args.port,
            timeout_sec=args.timeout_sec, no_dice_check=True,
        )
        reverse = run_labelcritic_compare(
            ct, mask_b, mask_a, "kidney_left", reverse_json,
            backend="labelcritic", base_url=args.base_url, port=args.port,
            timeout_sec=args.timeout_sec, no_dice_check=True,
        )
        fw = (forward.get("decision") or {}).get("winner")
        rv = (reverse.get("decision") or {}).get("winner")
        consistent = (
            fw == "uncertain"
            or rv == "uncertain"
            or (fw == "a" and rv == "b")
            or (fw == "b" and rv == "a")
        )
        compare_results = {
            "status": "passed" if forward.get("status") == "success" and reverse.get("status") == "success" and consistent else "failed",
            "forward": forward,
            "reverse": reverse,
            "ab_ba_consistent": consistent,
        }
        if compare_results["status"] != "passed":
            failures.append("labelcritic_ab_compare_failed")

        student_mask = Path(existing_file(ROOT / "outputs/em_round_pure_cached_10case_formal_lite_20260703/round1/student_predictions_postprocessed_v2/PanTS_00000100/liver.nii.gz"))
        pseudo_mask = Path(existing_file(ROOT / "outputs/em_round1_25case_pseudo_label_20260709/round1/estep/annotation_versions/PanTS_00000100/updated/liver.nii.gz"))
        student_compare = run_labelcritic_compare(
            ct, student_mask, pseudo_mask, "liver", out / "student_vs_selected_pseudo_liver.json",
            backend="labelcritic", base_url=args.base_url, port=args.port,
            timeout_sec=args.timeout_sec, no_dice_check=True,
        )
        student_compare_results = {
            "status": "passed" if student_compare.get("status") == "success" else "failed",
            "result": student_compare,
        }
        if student_compare_results["status"] != "passed":
            failures.append("student_vs_pseudo_compare_failed")

    payload = {
        "stage": "labelcritic_25case_readiness",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "output_dir": str(out),
        "bounded_polling_policy": "vLLM polling is capped at 240 seconds.",
        "files": files,
        "service": service,
        "vllm_start": vllm_start,
        "wait": wait,
        "labelcritic_373_coverage_audit": coverage,
        "teacher_pair_compare": compare_results,
        "student_vs_selected_pseudo_compare": student_compare_results,
        "interpretation": "Readiness checks pseudo-label comparison plumbing, not external-label accuracy.",
    }
    write_json(out / "labelcritic_readiness.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
