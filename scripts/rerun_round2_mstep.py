#!/usr/bin/env python3
"""
Round 2 M-step 재실행 스크립트.
수정 사항: LR 이중 감소 버그 수정 (5e-7 → 5e-6), n_train_samples 10 → 50
E-step은 이미 완료되었으므로 M-step + student 추론만 재실행.
"""
import sys, time, json, csv, subprocess
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

LEARNING_RATE        = 5e-5
CONSOLIDATION_LR     = 5e-6
CONSOLIDATION_EPOCHS = 20
KEY_ORGANS = None  # None = 使用 teacher_branch_map 里全部127个器官


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def stop_vllm():
    import subprocess, time
    # 只杀 vLLM 进程，不能用 nvidia-smi 杀所有 GPU 进程（会误杀训练进程）
    r = subprocess.run(["pgrep", "-f", "vllm"], capture_output=True, text=True)
    for pid in r.stdout.strip().split():
        subprocess.run(["kill", "-9", pid], check=False)
    subprocess.run("ps aux | grep -i 'VLLM\\|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9",
                   shell=True, capture_output=True)
    time.sleep(5)
    used = subprocess.run(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader",
        shell=True, capture_output=True, text=True
    ).stdout.strip()
    log(f"  vLLM 已停止，GPU 显存使用: {used}")


def restart_vllm():
    import urllib.request
    cmd = (f"screen -dmS vllm_server bash -c 'cd {PROJECT_ROOT} && "
           f"python -m vllm.entrypoints.openai.api_server "
           f"--model {QWEN_MODEL} --port 8000 --max-model-len 4096 "
           f"--gpu-memory-utilization 0.4'")
    subprocess.run(cmd, shell=True, check=False)
    log("  vLLM 已在后台重启，等待上线...")
    for i in range(30):
        time.sleep(5)
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
            log(f"  vLLM 在线 ({(i+1)*5}s)")
            return
        except Exception:
            pass
    log("  vLLM 启动超时")


def main():
    log("=" * 60)
    log("Round 2 M-step 重跑 (修复 LR 5e-7→5e-6, n_train 10→50)")
    log("=" * 60)

    round_idx = 2
    out_dir = OUTPUT_ROOT / f"round{round_idx}" / "mstep"
    datalist_path = out_dir / "datalist.json"

    if not datalist_path.exists():
        log(f"ERROR: datalist 不存在: {datalist_path}")
        sys.exit(1)

    prev_ckpt = OUTPUT_ROOT / "round1/mstep/model_finetune.pt"
    if not prev_ckpt.exists():
        log(f"ERROR: Round 1 checkpoint 不存在: {prev_ckpt}")
        sys.exit(1)

    log(f"  从 Round 1 checkpoint 开始: {prev_ckpt}")
    log(f"  LR={CONSOLIDATION_LR}, epochs={CONSOLIDATION_EPOCHS}, n_train=50")

    # 清理上次失败的输出
    for f in ["continual_finetune_result.json", "continual_config_override.json",
              "continual_datalist.json"]:
        (out_dir / f).unlink(missing_ok=True)
    import shutil
    for d in ["checkpoints", "eval"]:
        p = out_dir / d
        if p.exists():
            shutil.rmtree(p)

    # 停止 vLLM 释放显存
    stop_vllm()

    from cli_anything.medai.core.vista3d_student import VISTA3DStudent
    import yaml

    student = VISTA3DStudent(vista3d_root=VISTA3D_ROOT, model_path=prev_ckpt, device="cuda")

    with open(TEACHER_MAP) as f:
        branch_map = yaml.safe_load(f)

    result = student.continual_finetune(
        pseudo_label_dir=out_dir / "pseudo_labels",
        ct_dir=PROJECT_ROOT / "data/PanTS/ImageTr",
        target_organs=list(branch_map.keys()),
        output_dir=out_dir,
        learning_rate=CONSOLIDATION_LR,
        max_epochs=CONSOLIDATION_EPOCHS,
        freeze_backbone=False,
        global_consolidation=True,
        dry_run=False,
        timeout_sec=86400,
        prebuilt_datalist_path=datalist_path,
    )

    log(f"M-step 完成: status={result.get('status')}, checkpoint={result.get('finetuned_checkpoint')}")

    if result.get("status") != "success":
        log("ERROR: M-step 失败")
        log(result.get("stderr_tail", "")[-2000:])
        sys.exit(1)

    # 重启 vLLM
    restart_vllm()

    # Round 2 student 推理
    log("保存 Round 2 student 推理结果...")
    ckpt = out_dir / "model_finetune.pt"
    student2 = VISTA3DStudent(vista3d_root=VISTA3D_ROOT, model_path=ckpt, device="cuda")

    cases = []
    with open(CASE_LIST) as f:
        for row in csv.DictReader(f):
            cases.append(row)

    pred_dir = OUTPUT_ROOT / "round2/student_predictions"
    # 清理旧的空目录
    if pred_dir.exists():
        import shutil
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, case in enumerate(cases, 1):
        ct_path = Path(case["ct_path"])
        case_id = case["case_id"]
        if not ct_path.exists():
            continue
        case_pred_dir = pred_dir / case_id
        case_pred_dir.mkdir(parents=True, exist_ok=True)
        # 使用全部127个器官
        import yaml as _yaml
        with open(PROJECT_ROOT / "configs/teacher_branch_map.yaml") as _f:
            _bmap = _yaml.safe_load(_f)
        all_organs = [o for o, info in _bmap.items() if info.get("vista3d_label_id", 0) > 0]
        r = student2.segment(ct_image=ct_path, prompts=all_organs,
                             output_dir=case_pred_dir, dry_run=False, timeout_sec=900)
        n = r.get("num_segmented", 0)
        if r.get("status") == "success":
            saved += 1
            log(f"  [{i}/{len(cases)}] {case_id}: ✓ {n}/{len(all_organs)} organs")
        else:
            log(f"  [{i}/{len(cases)}] {case_id}: ✗")

    log(f"Student 推理完成: {saved}/{len(cases)} 个 case")

    # 更新 round_summary
    summary_path = OUTPUT_ROOT / "round2/round_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    summary["mstep_status"] = result.get("status")
    summary["finetuned_checkpoint"] = result.get("finetuned_checkpoint")
    summary["mstep_lr_fix"] = "5e-6 (was 5e-7)"
    summary["mstep_n_train_fix"] = "50 (was 10)"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log("=" * 60)
    log("Round 2 M-step 重跑完成")
    log("=" * 60)


if __name__ == "__main__":
    main()
