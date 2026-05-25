#!/usr/bin/env python3
"""
补跑 Round 1 M-step 和 student 推理。
pseudo_labels 和 datalist 已存在，直接从 M-step 开始。
完成后修改 round_summary.json，让主循环不会重跑 round 1。
"""
import json, sys, time, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-harness"))

PROJECT_ROOT  = Path("/home/teacher1/JHU-project1/medical_agent")
OUTPUT_ROOT   = PROJECT_ROOT / "outputs"
VISTA3D_ROOT  = PROJECT_ROOT / "checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master"
VISTA3D_MODEL = VISTA3D_ROOT / "models/model.pt"
TEACHER_MAP   = PROJECT_ROOT / "configs/teacher_branch_map.yaml"
CASE_LIST     = PROJECT_ROOT / "data_manifest/case_list_50_tumor.csv"
QWEN_MODEL    = PROJECT_ROOT / "checkpoints/Qwen/Qwen2-VL-7B-Instruct"
LOG_FILE      = OUTPUT_ROOT / "training.log"

LEARNING_RATE   = 5e-5
FINETUNE_EPOCHS = 50


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def stop_vllm():
    r = subprocess.run(["pgrep", "-f", "vllm"], capture_output=True, text=True)
    pids = r.stdout.strip().split()
    if not pids:
        log("  vLLM 未运行")
        return
    for pid in pids:
        subprocess.run(["kill", "-TERM", pid], check=False)
    log(f"  已停止 vLLM (pids={pids})，等待退出...")
    for _ in range(30):
        time.sleep(2)
        r2 = subprocess.run(["pgrep", "-f", "vllm"], capture_output=True, text=True)
        if not r2.stdout.strip():
            break
    else:
        for pid in pids:
            subprocess.run(["kill", "-KILL", pid], check=False)
    log("  vLLM 已停止，GPU 显存已释放")


def restart_vllm():
    cmd = (
        f"screen -dmS vllm_server bash -c '"
        f"cd {PROJECT_ROOT} && "
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {QWEN_MODEL} "
        f"--port 8000 "
        f"--max-model-len 4096 "
        f"--gpu-memory-utilization 0.4"
        f"'"
    )
    subprocess.run(cmd, shell=True, check=False)
    log("  vLLM 已在后台重启")
    for i in range(30):
        time.sleep(5)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
            log(f"  vLLM 在线 (等待了 {(i+1)*5}s)")
            return
        except Exception:
            pass
    log("  vLLM 启动超时")

def main():
    log("=" * 60)
    log("补跑 Round 1 M-step + student 推理")
    log("=" * 60)

    round_idx = 1
    out_dir = OUTPUT_ROOT / f"round{round_idx}" / "mstep"
    datalist_path = out_dir / "datalist.json"

    if not datalist_path.exists():
        log(f"ERROR: datalist 不存在: {datalist_path}")
        sys.exit(1)

    # ── M-step ──────────────────────────────────────────────────────
    log(f"=== Round {round_idx} M-step 开始 (局部更新) ===")

    # 停止 vLLM 释放显存
    stop_vllm()

    from cli_anything.medai.core.vista3d_student import VISTA3DStudent
    import yaml

    student = VISTA3DStudent(vista3d_root=VISTA3D_ROOT, model_path=VISTA3D_MODEL, device="cuda")

    with open(TEACHER_MAP) as f:
        branch_map = yaml.safe_load(f)

    mstep_result = student.continual_finetune(
        pseudo_label_dir=out_dir / "pseudo_labels",
        ct_dir=PROJECT_ROOT / "data/PanTS/ImageTr",
        target_organs=list(branch_map.keys()),
        output_dir=out_dir,
        learning_rate=LEARNING_RATE,
        max_epochs=FINETUNE_EPOCHS,
        freeze_backbone=True,
        global_consolidation=False,
        dry_run=False,
        timeout_sec=86400,
        prebuilt_datalist_path=datalist_path,
    )

    log(f"M-step 完成: status={mstep_result.get('status')}, "
        f"checkpoint={mstep_result.get('finetuned_checkpoint')}")

    if mstep_result.get("status") != "success":
        log("ERROR: M-step 失败，请检查 stderr")
        log(mstep_result.get("stderr_tail", "")[-2000:])
        sys.exit(1)

    # M-step 完成后重启 vLLM
    restart_vllm()

    # ── Student 推理 ─────────────────────────────────────────────────
    log(f"保存 Round {round_idx} student 推理结果...")
    pred_dir = OUTPUT_ROOT / f"round{round_idx}" / "student_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ckpt = out_dir / "model_finetune.pt"
    student2 = VISTA3DStudent(vista3d_root=VISTA3D_ROOT, model_path=ckpt, device="cuda")

    import csv as csv_mod
    cases = []
    with open(CASE_LIST) as f:
        for row in csv_mod.DictReader(f):
            cases.append(row)

    key_organs = "pancreas,liver,spleen,kidney_left,kidney_right,aorta,postcava,stomach,duodenum,colon,lung,heart,bladder,spinal_cord"
    saved = 0
    for case in cases:
        ct_path = Path(case["ct_path"])
        if not ct_path.exists():
            continue
        case_pred_dir = pred_dir / case["case_id"]
        if case_pred_dir.exists() and any(case_pred_dir.glob("*.nii.gz")):
            saved += 1
            continue
        result = student2.segment(
            ct_image=ct_path,
            prompts=key_organs.split(","),
            output_dir=case_pred_dir,
            dry_run=False,
        )
        if result.get("status") == "success":
            saved += 1
            log(f"  {case['case_id']}: {result.get('num_segmented', 0)} organs")
        else:
            log(f"  {case['case_id']}: 推理失败 - {result.get('stderr_tail','')[-200:]}")

    log(f"Student 推理完成: {saved}/{len(cases)} 个 case")

    # ── 更新 round_summary.json ──────────────────────────────────────
    summary_path = OUTPUT_ROOT / f"round{round_idx}" / "round_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    summary["mstep_status"] = mstep_result.get("status")
    summary["finetuned_checkpoint"] = mstep_result.get("finetuned_checkpoint")
    summary["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"round_summary.json 已更新: mstep_status={summary['mstep_status']}")

    log("=" * 60)
    log("Round 1 补跑完成，可以启动主训练循环继续 Round 2")
    log("=" * 60)

if __name__ == "__main__":
    main()
