#!/usr/bin/env python3
"""
完整 EM Loop 训练脚本 v2
架构：13个 teacher 模型 → pseudo-label → VISTA3D student continual fine-tuning

修复：
- E-step/M-step 直接调用 Python 函数，无 subprocess timeout 问题
- 断点续跑：检测已完成 case，跳过重跑
- vLLM LabelCritic 接入（Qwen2-VL-7B @ localhost:8000）
- ShapeKit 开启提升 mask 质量
- save_round_predictions 直接调用，无 timeout
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-harness"))

# ── 配置 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/teacher1/JHU-project1/medical_agent")
CASE_LIST    = PROJECT_ROOT / "data_manifest/case_list_50_tumor.csv"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"
LOG_FILE     = OUTPUT_ROOT / "training.log"

VISTA3D_ROOT  = PROJECT_ROOT / "checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master"
VISTA3D_MODEL = VISTA3D_ROOT / "models/model.pt"
TEACHER_MAP   = PROJECT_ROOT / "configs/teacher_branch_map.yaml"
ALL_ORGANS    = PROJECT_ROOT / "configs/all_organs.json"   # 所有teacher覆盖的358个器官
PERF_TRACKER  = OUTPUT_ROOT / "organ_model_performance.json"

QWEN_MODEL    = PROJECT_ROOT / "checkpoints/Qwen/Qwen2-VL-7B-Instruct"
VLLM_BASE_URL = "http://localhost:8000"

ALL_TEACHERS = [
    "totalsegmentator",
    "epai_20250421",
    "vsmtrans",
    "cads",
    "moose",
    "moose3_0",
    "nnunet_private",
    "saros_nnunet",
    "airrc",
    "atm",
    "vsnet",
    "unest",
    "vista3d",
]

NUM_ROUNDS             = 3
CONSOLIDATION_INTERVAL = 2
FINETUNE_EPOCHS        = 100   # 增加 epoch 数，让模型充分学习
CONSOLIDATION_EPOCHS   = 50    # 全局整合也需要足够 epoch
LEARNING_RATE          = 5e-5
CONSOLIDATION_LR       = 5e-6

# 单个模型推理 timeout（秒）。CADS/VISTA3D 约 3-5 分钟/case，10分钟足够
INFER_TIMEOUT_SEC = 600

# ShapeKit：CPU密集型，每case约2-3分钟，13模型×50case=650次≈30小时，关闭加速
ENABLE_SHAPEKIT = False

# LabelCritic：Qwen2-VL-7B @ vLLM server
ENABLE_CRITIC   = True
CRITIC_BACKEND  = "labelcritic"

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def check_vllm_server() -> bool:
    """检查 vLLM server 是否在线"""
    try:
        import urllib.request
        urllib.request.urlopen(f"{VLLM_BASE_URL}/v1/models", timeout=5)
        return True
    except Exception:
        return False


def stop_vllm_for_mstep():
    """M-step 前停止 vLLM server 释放显存（只杀 vLLM 进程，不影响其他 GPU 进程）"""
    import subprocess as _sp
    # 只杀 vLLM 相关进程，不能用 nvidia-smi 杀所有 GPU 进程（会误杀训练进程）
    r = _sp.run(["pgrep", "-f", "vllm"], capture_output=True, text=True)
    pids = [p for p in r.stdout.strip().split() if p]
    if pids:
        for pid in pids:
            _sp.run(["kill", "-9", pid], check=False)
        import time as _t; _t.sleep(5)
    # 也通过进程名杀 VLLM::EngineCore
    _sp.run("ps aux | grep -i 'VLLM\\|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9",
            shell=True, capture_output=True)
    import time as _t; _t.sleep(3)
    used = _sp.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader",
                   shell=True, capture_output=True, text=True).stdout.strip()
    log(f"  vLLM 已停止，GPU 显存使用: {used}")


def restart_vllm_after_mstep():
    """M-step 后在后台重启 vLLM server"""
    try:
        import subprocess
        vllm_cmd = (
            f"screen -dmS vllm_server bash -c '"
            f"cd {PROJECT_ROOT} && "
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {QWEN_MODEL} "
            f"--port 8000 "
            f"--max-model-len 4096 "
            f"--gpu-memory-utilization 0.4"
            f"'"
        )
        subprocess.run(vllm_cmd, shell=True, check=False)
        log("  vLLM 已在后台重启 (screen: vllm_server)")
        # 等待启动
        import time as _t
        for i in range(30):
            _t.sleep(5)
            if check_vllm_server():
                log(f"  vLLM 在线 (等待了 {(i+1)*5}s)")
                return True
        log("  vLLM 启动超时，LabelCritic 本轮将不可用")
        return False
    except Exception as e:
        log(f"  重启 vLLM 时出错（忽略）: {e}")
        return False


def completed_cases(round_idx: int) -> set:
    """返回本轮已完成推理的 case_id 集合（用于断点续跑）"""
    cases_dir = OUTPUT_ROOT / f"round{round_idx}" / "estep" / "cases"
    if not cases_dir.exists():
        return set()
    done = set()
    for case_dir in cases_dir.iterdir():
        if not case_dir.is_dir():
            continue
        # 至少有一个模型成功输出了 mask 才算完成
        pred_root = case_dir / "raw_predictions"
        if pred_root.exists():
            for model_dir in pred_root.iterdir():
                seg = model_dir / case_dir.name / "segmentations"
                if seg.exists() and any(seg.glob("*.nii.gz")):
                    done.add(case_dir.name)
                    break
    return done


# ── E-step ────────────────────────────────────────────────────────────────────

def run_estep(round_idx: int) -> dict:
    """
    E-step：所有 teacher 模型推理，生成全身器官 pseudo-label。
    - 直接调用 Python 函数，无 subprocess timeout
    - 断点续跑：跳过已完成的 case
    - Round 2+ 由 OrganModelPerformance 追踪器自动选 top-2 teacher
    - Round 2+ 把上一轮 student 预测注入 teacher 候选池，参与 DICE 竞争
    """
    log(f"=== Round {round_idx} E-step 开始 ===")
    out_dir = OUTPUT_ROOT / f"round{round_idx}" / "estep"

    # 检查断点续跑
    done = completed_cases(round_idx)
    if done:
        log(f"  断点续跑：已完成 {len(done)}/50 个 case，跳过这些 case")

    # 检查 vLLM server
    vllm_ok = check_vllm_server()
    enable_critic = ENABLE_CRITIC and vllm_ok
    if ENABLE_CRITIC and not vllm_ok:
        log("  ⚠️  vLLM server 不可用，本轮关闭 LabelCritic")

    from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop
    import yaml, json

    # E-step 用所有 teacher 覆盖的 358 个器官（全身完整识别）
    with open(ALL_ORGANS) as f:
        organ_list = json.load(f)

    # M-step 蒸馏只用 VISTA3D 支持的 127 个器官（branch_map 的键）
    with open(TEACHER_MAP) as f:
        branch_map = yaml.safe_load(f)

    # Round 2+：把上一轮 student 预测注入 teacher 候选池
    # student_predictions/<case_id>/<organ>.nii.gz 直接参与 DICE 竞争
    preseeded: dict = {}
    if round_idx > 1:
        prev_pred_dir = OUTPUT_ROOT / f"round{round_idx - 1}" / "student_predictions"
        if prev_pred_dir.exists() and any(prev_pred_dir.iterdir()):
            preseeded["student_prev"] = prev_pred_dir
            log(f"  注入上一轮 student 预测参与竞争: {prev_pred_dir}")
        else:
            log(f"  上一轮 student 预测不存在，跳过注入: {prev_pred_dir}")

    log(f"  模型: {len(ALL_TEACHERS)}个, 器官: {len(organ_list)}个, "
        f"待处理 case: {50 - len(done)}/50, "
        f"ShapeKit: {'开' if ENABLE_SHAPEKIT else '关'}, "
        f"LabelCritic: {'开' if enable_critic else '关'}"
        + (f", student_prev: 注入" if preseeded else ""))

    # 如果所有 case 都完成了，直接返回成功
    if len(done) >= 50:
        log("  所有 case 已完成，跳过 E-step")
        return {"status": "success", "num_cases": 50, "total_updated": 0, "skipped": True}

    result = run_multimodel_annotation_loop(
        case_list=CASE_LIST,
        output_folder=out_dir,
        models=ALL_TEACHERS,
        organs=organ_list,
        registry_path=PROJECT_ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=False,
        enable_shapekit=ENABLE_SHAPEKIT,
        shapekit_root=PROJECT_ROOT / "third_party/ShapeKit-main",
        enable_critic=enable_critic,
        critic_backend=CRITIC_BACKEND,
        critic_base_url=VLLM_BASE_URL,
        critic_port=8000,
        dry_run=False,
        timeout_sec=INFER_TIMEOUT_SEC,
        device="cuda",
        perf_tracker_path=PERF_TRACKER,
        resume=True,
        preseeded_model_dirs=preseeded if preseeded else None,
    )

    status = result.get("status", "unknown")
    log(f"E-step 完成: status={status}, "
        f"cases={result.get('num_cases',0)}, "
        f"updated={result.get('total_updated',0)}")
    return result


# ── 数据集构建 ────────────────────────────────────────────────────────────────

def build_vista3d_dataset(round_idx: int) -> Path:
    """
    将 E-step 的 pseudo-label 转换为 VISTA3D continual fine-tuning 格式。
    优先级：真实标注 > 上一轮 student 预测 > OrganModelPerformance 最优 teacher > yaml 配置 teacher
    """
    import yaml, nibabel as nib, numpy as np, csv
    perf_tracker_path = PERF_TRACKER

    log(f"构建 VISTA3D 训练数据集...")
    out_dir = OUTPUT_ROOT / f"round{round_idx}" / "mstep"
    label_dir = out_dir / "pseudo_labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    with open(TEACHER_MAP) as f:
        branch_map = yaml.safe_load(f)

    cases = []
    with open(CASE_LIST) as f:
        for row in csv.DictReader(f):
            cases.append(row)

    estep_dir = OUTPUT_ROOT / f"round{round_idx}" / "estep" / "cases"
    # 上一轮 student 预测目录（round 2+ 才存在）
    prev_student_dir = OUTPUT_ROOT / f"round{round_idx - 1}" / "student_predictions" if round_idx > 1 else None

    datalist = {"training": [], "validation": []}
    built = 0
    student_used = 0

    for case in cases:
        case_id = case["case_id"]
        ct_path = Path(case["ct_path"])
        ann_folder = Path(case.get("annotation_folder", "")) if case.get("annotation_folder") else None

        if not ct_path.exists():
            continue

        # 检查是否已经构建过（断点续跑）
        dst = label_dir / case_id / "combined_label.nii.gz"
        if dst.exists():
            entry = {"image": str(ct_path), "label": str(dst)}
            datalist["training"].append(entry)
            built += 1
            continue

        try:
            ref_img = nib.load(str(ct_path))
            shape = np.asanyarray(ref_img.dataobj).shape
            combined = np.zeros(shape, dtype=np.uint8)
            organs_added = 0
            case_student_used = 0

            for organ, info in branch_map.items():
                label_id = info.get("vista3d_label_id", 0)
                if label_id == 0:
                    continue

                mask_path = None

                # 1. 真实标注优先
                if ann_folder and ann_folder.exists():
                    real = ann_folder / f"{organ}.nii.gz"
                    if real.exists():
                        mask_path = real

                # 2. 上一轮 student 预测（比 teacher 更可信，因为已经经过蒸馏）
                if mask_path is None and prev_student_dir is not None:
                    student_cand = prev_student_dir / case_id / f"{organ}.nii.gz"
                    if student_cand.exists():
                        mask_path = student_cand
                        case_student_used += 1

                # 3. E-step 最优 teacher 输出
                # 优先用 OrganModelPerformance tracker 里 DSC 最高的 teacher
                if mask_path is None:
                    teacher = info.get("teacher_model", "")
                    fallbacks = info.get("fallback_teachers", [])
                    # 如果 tracker 存在，动态选该 organ 实际 DSC 最高的 teacher
                    if perf_tracker_path and perf_tracker_path.exists():
                        try:
                            import json as _json
                            perf = _json.loads(perf_tracker_path.read_text())
                            if organ in perf:
                                ranked = sorted(perf[organ].items(),
                                                key=lambda x: x[1].get("mean_dice", 0), reverse=True)
                                dynamic_order = [t for t, _ in ranked if t != "student_prev"]
                                # 合并：tracker 排序 + yaml fallback 补充
                                all_teachers = dynamic_order + [t for t in [teacher] + fallbacks
                                                                if t not in dynamic_order]
                            else:
                                all_teachers = [teacher] + fallbacks
                        except Exception:
                            all_teachers = [teacher] + fallbacks
                    else:
                        all_teachers = [teacher] + fallbacks

                    for t in all_teachers:
                        cand = estep_dir / case_id / "raw_predictions" / t / case_id / "segmentations" / f"{organ}.nii.gz"
                        if cand.exists():
                            mask_path = cand
                            break

                if mask_path is None:
                    continue

                try:
                    m_img = nib.load(str(mask_path))
                    m_arr = np.asanyarray(m_img.dataobj) > 0
                    if m_arr.shape == shape:
                        combined[m_arr] = label_id
                        organs_added += 1
                except Exception:
                    pass

            if organs_added == 0:
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(combined, ref_img.affine), str(dst))
            entry = {"image": str(ct_path), "label": str(dst)}
            datalist["training"].append(entry)
            built += 1
            if case_student_used > 0:
                student_used += 1

        except Exception as e:
            log(f"  {case_id} 构建失败: {e}")

    n_val = max(1, len(datalist["training"]) // 10)
    datalist["validation"] = datalist["training"][:n_val]

    datalist_path = out_dir / "datalist.json"
    with open(datalist_path, "w") as f:
        json.dump(datalist, f, indent=2)

    log(f"数据集构建完成: {built} 个 case"
        + (f"，其中 {student_used} 个 case 使用了上一轮 student 预测" if student_used > 0 else ""))
    return datalist_path


# ── M-step ────────────────────────────────────────────────────────────────────

def run_mstep(round_idx: int, datalist_path: Path, global_consolidation: bool = False) -> dict:
    """VISTA3D continual fine-tuning，直接调用，无 timeout 限制"""
    log(f"=== Round {round_idx} M-step 开始 "
        f"({'全局整合' if global_consolidation else '局部更新'}) ===")

    out_dir = OUTPUT_ROOT / f"round{round_idx}" / "mstep"

    if round_idx == 1:
        model_path = VISTA3D_MODEL
    else:
        prev_ckpt = OUTPUT_ROOT / f"round{round_idx-1}" / "mstep" / "model_finetune.pt"
        model_path = prev_ckpt if prev_ckpt.exists() else VISTA3D_MODEL

    lr = CONSOLIDATION_LR if global_consolidation else LEARNING_RATE
    epochs = CONSOLIDATION_EPOCHS if global_consolidation else FINETUNE_EPOCHS

    from cli_anything.medai.core.vista3d_student import VISTA3DStudent
    import yaml

    # 停止 vLLM 释放显存，M-step 完成后重启
    stop_vllm_for_mstep()

    student = VISTA3DStudent(vista3d_root=VISTA3D_ROOT, model_path=model_path, device="cuda")

    with open(TEACHER_MAP) as f:
        branch_map = yaml.safe_load(f)

    result = student.continual_finetune(
        pseudo_label_dir=out_dir / "pseudo_labels",
        ct_dir=PROJECT_ROOT / "data/PanTS/ImageTr",
        target_organs=list(branch_map.keys()),
        output_dir=out_dir,
        learning_rate=lr,
        max_epochs=epochs,
        freeze_backbone=not global_consolidation,
        global_consolidation=global_consolidation,
        dry_run=False,
        timeout_sec=86400,
        prebuilt_datalist_path=datalist_path,
    )

    log(f"M-step 完成: status={result.get('status')}, checkpoint={result.get('finetuned_checkpoint')}")

    # M-step 完成后重启 vLLM，供下一轮 E-step 的 LabelCritic 使用
    restart_vllm_after_mstep()

    return result


# ── 评估指标 ──────────────────────────────────────────────────────────────────

def compute_round_metrics(round_idx: int) -> dict:
    """
    评估本轮 student 模型的分割效果（与 ground truth 对比）。
    这才是衡量蒸馏效果的正确指标，而不是 teacher 的 DSC。
    """
    import csv as csv_mod, nibabel as nib, numpy as np

    log(f"计算 Round {round_idx} student 评估指标...")
    metrics_dir = OUTPUT_ROOT / f"round{round_idx}" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    student_pred_dir = OUTPUT_ROOT / f"round{round_idx}" / "student_predictions"
    cases = []
    with open(CASE_LIST) as f:
        for row in csv_mod.DictReader(f):
            cases.append(row)

    dice_rows, organ_dices, case_dices = [], {}, {}

    for case in cases:
        case_id = case["case_id"]
        ann_folder = Path(case.get("annotation_folder", "")) if case.get("annotation_folder") else None
        if not ann_folder or not ann_folder.exists():
            continue
        pred_dir = student_pred_dir / case_id
        if not pred_dir.exists():
            continue
        for pred_mask in pred_dir.glob("*.nii.gz"):
            organ = pred_mask.stem.replace(".nii", "")
            ref_mask = ann_folder / f"{organ}.nii.gz"
            if not ref_mask.exists():
                continue
            try:
                pa = np.asanyarray(nib.load(str(pred_mask)).dataobj) > 0
                ra = np.asanyarray(nib.load(str(ref_mask)).dataobj) > 0
                if pa.shape != ra.shape:
                    continue
                inter = int((pa & ra).sum())
                total = int(pa.sum()) + int(ra.sum())
                dsc = round(2 * inter / total, 4) if total > 0 else 0.0
                dice_rows.append({"round": round_idx, "case_id": case_id, "organ": organ, "dsc": dsc})
                organ_dices.setdefault(organ, []).append(dsc)
                case_dices.setdefault(case_id, []).append(dsc)
            except Exception:
                continue

    if dice_rows:
        with open(metrics_dir / "student_dice_per_organ.csv", "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=["round", "case_id", "organ", "dsc"])
            w.writeheader(); w.writerows(dice_rows)

    organ_summary = []
    for organ, dscs in sorted(organ_dices.items()):
        arr = np.array(dscs)
        organ_summary.append({"round": round_idx, "organ": organ, "n": len(dscs),
                               "mean_dsc": round(float(arr.mean()), 4),
                               "std_dsc": round(float(arr.std()), 4)})
    if organ_summary:
        with open(metrics_dir / "student_organ_summary.csv", "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=["round", "organ", "n", "mean_dsc", "std_dsc"])
            w.writeheader(); w.writerows(organ_summary)

    all_dscs = [r["dsc"] for r in dice_rows]
    arr_all = np.array(all_dscs) if all_dscs else np.array([0.0])
    round_metrics = {
        "round": round_idx,
        "source": "student_predictions",
        "n_evaluations": len(dice_rows),
        "n_cases_with_labels": len(case_dices),
        "overall_mean_dsc": round(float(arr_all.mean()), 4),
        "overall_std_dsc": round(float(arr_all.std()), 4),
        "top5_organs": sorted(organ_summary, key=lambda x: -x["mean_dsc"])[:5],
        "bottom5_organs": sorted(organ_summary, key=lambda x: x["mean_dsc"])[:5],
    }
    with open(metrics_dir / "round_metrics.json", "w") as f:
        json.dump(round_metrics, f, indent=2)

    log(f"Student 指标完成: {len(dice_rows)} 条, mean DSC={round_metrics['overall_mean_dsc']:.4f}")
    return round_metrics


def save_round_predictions(round_idx: int):
    """保存本轮 VISTA3D student 推理结果（全部127个器官），直接调用，无 timeout"""
    log(f"保存 Round {round_idx} student 推理结果（127个器官）...")
    pred_dir = OUTPUT_ROOT / f"round{round_idx}" / "student_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ckpt = OUTPUT_ROOT / f"round{round_idx}" / "mstep" / "model_finetune.pt"
    model_path = ckpt if ckpt.exists() else VISTA3D_MODEL

    from cli_anything.medai.core.vista3d_student import VISTA3DStudent
    import csv as csv_mod, yaml

    student = VISTA3DStudent(vista3d_root=VISTA3D_ROOT, model_path=model_path, device="cuda")

    # 从 teacher_branch_map 获取全部127个器官（有 vista3d_label_id 的）
    with open(TEACHER_MAP) as f:
        branch_map = yaml.safe_load(f)
    all_organs = [o for o, info in branch_map.items() if info.get("vista3d_label_id", 0) > 0]

    cases = []
    with open(CASE_LIST) as f:
        for row in csv_mod.DictReader(f):
            cases.append(row)

    saved = 0
    for case in cases:
        ct_path = Path(case["ct_path"])
        case_id = case["case_id"]
        if not ct_path.exists():
            continue
        case_pred_dir = pred_dir / case_id
        # 跳过已完成的（有任意 mask 即视为完成）
        if case_pred_dir.exists() and any(case_pred_dir.glob("*.nii.gz")):
            saved += 1
            continue
        case_pred_dir.mkdir(parents=True, exist_ok=True)
        result = student.segment(
            ct_image=ct_path,
            prompts=all_organs,
            output_dir=case_pred_dir,
            dry_run=False,
            timeout_sec=900,
        )
        n = result.get("num_segmented", 0)
        if result.get("status") == "success":
            saved += 1
            log(f"  {case_id}: ✓ {n}/{len(all_organs)} organs")
        else:
            log(f"  {case_id}: ✗ {result.get('status')}")

    log(f"Student 推理完成: {saved}/{len(cases)} 个 case")


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("EM Loop 训练 v2（断点续跑 + vLLM LabelCritic）")
    log(f"  轮数: {NUM_ROUNDS}, Teacher: {len(ALL_TEACHERS)}个, Case: 50")
    log(f"  ShapeKit: {'开' if ENABLE_SHAPEKIT else '关'}, LabelCritic: {'开' if ENABLE_CRITIC else '关'}")
    log(f"  vLLM: {VLLM_BASE_URL} ({'在线' if check_vllm_server() else '离线'})")
    log(f"  VISTA3D: {VISTA3D_MODEL}")
    log("=" * 60)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    for round_idx in range(1, NUM_ROUNDS + 1):
        round_start = time.time()

        # 检查本轮是否已完全完成（断点续跑跳过整轮）
        summary_path = OUTPUT_ROOT / f"round{round_idx}" / "round_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            if s.get("estep_status") == "success" and s.get("mstep_status") == "success":
                log(f"\nRound {round_idx} 已完成，跳过")
                continue

        log(f"\n{'='*60}\nRound {round_idx}/{NUM_ROUNDS}\n{'='*60}")

        # E-step
        estep_result = run_estep(round_idx)
        if estep_result.get("status") == "failed":
            log(f"E-step 失败，跳过 Round {round_idx}")
            continue

        # 构建训练数据
        datalist_path = build_vista3d_dataset(round_idx)

        # M-step — 실패 시 1회 재시도
        is_consolidation = (round_idx % CONSOLIDATION_INTERVAL == 0)
        mstep_result = run_mstep(round_idx, datalist_path, global_consolidation=is_consolidation)

        if mstep_result.get("status") != "success":
            log(f"⚠️  M-step 失败 (return_code={mstep_result.get('return_code')}), 清理后重试一次...")
            # 清理失败的输出，重试
            import shutil as _shutil
            for _f in ["continual_finetune_result.json", "continual_config_override.json",
                       "continual_datalist.json"]:
                (OUTPUT_ROOT / f"round{round_idx}" / "mstep" / _f).unlink(missing_ok=True)
            for _d in ["checkpoints", "eval"]:
                _dp = OUTPUT_ROOT / f"round{round_idx}" / "mstep" / _d
                if _dp.exists():
                    _shutil.rmtree(_dp)
            mstep_result = run_mstep(round_idx, datalist_path, global_consolidation=is_consolidation)

        if mstep_result.get("status") != "success":
            log(f"❌ M-step 重试后仍失败，跳过 Round {round_idx} 的 student 推理和评估")
            log(f"   stderr: {mstep_result.get('stderr_tail','')[-500:]}")
            summary = {
                "round": round_idx,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "estep_status": estep_result.get("status"),
                "mstep_status": "failed",
                "mstep_type": "global_consolidation" if is_consolidation else "local_update",
                "finetuned_checkpoint": None,
                "metrics": {},
                "round_elapsed_hours": round((time.time() - round_start) / 3600, 2),
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            continue

        # 先保存 student 推理结果，再评估（评估依赖 student_predictions）
        save_round_predictions(round_idx)

        # 评估 student 模型效果（与 ground truth 对比，才是真正的蒸馏效果指标）
        round_metrics = compute_round_metrics(round_idx)

        round_elapsed = time.time() - round_start
        log(f"Round {round_idx} 完成，耗时 {round_elapsed/3600:.1f} 小时")

        summary = {
            "round": round_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "estep_status": estep_result.get("status"),
            "mstep_status": mstep_result.get("status"),
            "mstep_type": "global_consolidation" if is_consolidation else "local_update",
            "finetuned_checkpoint": mstep_result.get("finetuned_checkpoint"),
            "metrics": {
                "overall_mean_dsc": round_metrics.get("overall_mean_dsc"),
                "top5_organs": round_metrics.get("top5_organs", []),
                "bottom5_organs": round_metrics.get("bottom5_organs", []),
            },
            "round_elapsed_hours": round(round_elapsed / 3600, 2),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log(f"摘要已保存: {summary_path}")

    # 跨轮对比报告
    total_elapsed = time.time() - total_start
    comparison = {"rounds": []}
    for r in range(1, NUM_ROUNDS + 1):
        sp = OUTPUT_ROOT / f"round{r}" / "round_summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            comparison["rounds"].append({
                "round": r,
                "overall_mean_dsc": s.get("metrics", {}).get("overall_mean_dsc"),
                "elapsed_hours": s.get("round_elapsed_hours"),
                "checkpoint": s.get("finetuned_checkpoint"),
            })

    dscs = [r["overall_mean_dsc"] for r in comparison["rounds"] if r.get("overall_mean_dsc")]
    if len(dscs) >= 2:
        comparison["improvement"] = {
            "round1_dsc": dscs[0], "final_dsc": dscs[-1],
            "absolute_gain": round(dscs[-1] - dscs[0], 4),
            "relative_gain_pct": round((dscs[-1] - dscs[0]) / max(dscs[0], 1e-6) * 100, 2),
        }

    with open(OUTPUT_ROOT / "training_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    log(f"\n{'='*60}")
    log(f"训练完成！总耗时 {total_elapsed/3600:.1f} 小时")
    if comparison.get("improvement"):
        imp = comparison["improvement"]
        log(f"DSC 提升: {imp['round1_dsc']:.4f} → {imp['final_dsc']:.4f} (+{imp['relative_gain_pct']:.1f}%)")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
