#!/usr/bin/env python3
"""
Step 1: Round 1 student 推理（用 fine-tuned checkpoint）
Step 2: 启动主训练循环（round 1 已标记完成，直接跑 round 2）
"""
import sys, time, json, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-harness"))

PROJECT_ROOT  = Path("/home/teacher1/JHU-project1/medical_agent")
OUTPUT_ROOT   = PROJECT_ROOT / "outputs"
VISTA3D_ROOT  = PROJECT_ROOT / "checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master"
CASE_LIST     = PROJECT_ROOT / "data_manifest/case_list_50_tumor.csv"
LOG_FILE      = OUTPUT_ROOT / "training.log"
CKPT          = OUTPUT_ROOT / "round1/mstep/model_finetune.pt"
QWEN_MODEL    = PROJECT_ROOT / "checkpoints/Qwen/Qwen2-VL-7B-Instruct"

KEY_ORGANS = None  # None = 使用 teacher_branch_map 里全部127个器官


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def restart_vllm():
    import subprocess
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
    log("vLLM 已在后台重启，等待上线...")
    import urllib.request
    for i in range(30):
        time.sleep(5)
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
            log(f"vLLM 在线 ({(i+1)*5}s)")
            return True
        except Exception:
            pass
    log("vLLM 启动超时，LabelCritic 本轮不可用")
    return False


def main():
    log("=" * 60)
    log("Round 1 student 推理 → Round 2 训练")
    log("=" * 60)

    if not CKPT.exists():
        log(f"ERROR: checkpoint 不存在: {CKPT}")
        sys.exit(1)

    from cli_anything.medai.core.vista3d_student import VISTA3DStudent

    student = VISTA3DStudent(vista3d_root=VISTA3D_ROOT, model_path=CKPT, device="cuda")

    cases = []
    with open(CASE_LIST) as f:
        for row in csv.DictReader(f):
            cases.append(row)

    pred_dir = OUTPUT_ROOT / "round1" / "student_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, case in enumerate(cases, 1):
        ct_path = Path(case["ct_path"])
        case_id = case["case_id"]
        if not ct_path.exists():
            log(f"  [{i}/{len(cases)}] {case_id}: CT 不存在，跳过")
            continue
        case_pred_dir = pred_dir / case_id
        if case_pred_dir.exists() and any(case_pred_dir.glob("*.nii.gz")):
            log(f"  [{i}/{len(cases)}] {case_id}: 已完成，跳过")
            saved += 1
            continue
        case_pred_dir.mkdir(parents=True, exist_ok=True)
        import yaml as _yaml
        with open(PROJECT_ROOT / "configs/teacher_branch_map.yaml") as _f:
            _bmap = _yaml.safe_load(_f)
        all_organs = [o for o, info in _bmap.items() if info.get("vista3d_label_id", 0) > 0]
        result = student.segment(
            ct_image=ct_path,
            prompts=all_organs,
            output_dir=case_pred_dir,
            dry_run=False,
            timeout_sec=900,
        )
        n = result.get("num_segmented", 0)
        status = result.get("status")
        if status == "success":
            saved += 1
            log(f"  [{i}/{len(cases)}] {case_id}: ✓ {n} organs")
        else:
            log(f"  [{i}/{len(cases)}] {case_id}: ✗ {status} - {result.get('stderr_tail','')[-200:]}")

    log(f"Student 推理完成: {saved}/{len(cases)} 个 case")

    # 重启 vLLM 供 round 2 LabelCritic 使用
    restart_vllm()

    log("=" * 60)
    log("启动 Round 2 训练循环")
    log("=" * 60)


if __name__ == "__main__":
    main()
