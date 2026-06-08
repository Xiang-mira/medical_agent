#!/usr/bin/env python3
"""
完整 EM Loop 训练脚本 v2
当前主线架构：teacher 模型 → pseudo-label → 3D prompt-based student manifest/training

默认 student backend 是 VoxTell-style 3D prompt student。VISTA3D student 只保留为
legacy/reference backend，不能再作为默认目标类别空间。

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


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


# ── 配置 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/teacher1/JHU-project1/medical_agent")
CASE_LIST    = PROJECT_ROOT / "data_manifest/case_list_50_tumor.csv"
_OUTPUT_ROOT_ENV = os.getenv("MEDAI_OUTPUT_ROOT")
OUTPUT_ROOT  = Path(_OUTPUT_ROOT_ENV).expanduser() if _OUTPUT_ROOT_ENV else PROJECT_ROOT / "outputs"
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT
LOG_FILE     = OUTPUT_ROOT / "training.log"

VISTA3D_ROOT  = PROJECT_ROOT / "checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master"
VISTA3D_MODEL = VISTA3D_ROOT / "models/model.pt"
TEACHER_MAP   = PROJECT_ROOT / "configs/teacher_branch_map.yaml"
ALL_ORGANS    = PROJECT_ROOT / "configs/all_organs.json"   # 所有teacher覆盖的358个器官
PERF_TRACKER  = OUTPUT_ROOT / "organ_model_performance.json"

STUDENT_BACKEND = os.getenv("MEDAI_STUDENT_BACKEND", "voxtell_style_3d_prompt")
PROMPT_TARGET_CONFIG = PROJECT_ROOT / "configs/student_3d_prompt_target_organs.json"
VOXTELL_MODEL_DIR = Path(os.getenv("MEDAI_VOXTELL_MODEL_DIR", PROJECT_ROOT / "checkpoints/VoxTell/voxtell_v1.1"))
VOXTELL_TRAIN_CMD = os.getenv("MEDAI_VOXTELL_TRAIN_CMD", "").strip()
VOXTELL_TEXT_ENCODING_MODEL = Path(os.getenv("MEDAI_TEXT_ENCODING_MODEL", PROJECT_ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B"))

QWEN_MODEL    = PROJECT_ROOT / "checkpoints/Qwen/Qwen2-VL-7B-Instruct"
VLLM_BASE_URL = "http://localhost:8000"

# Current formal teacher pool: 21 Drive-aligned models plus the separate
# official TotalSegmentator route. Keep these as concrete registry keys, not
# legacy family aliases such as "cads", "moose", or "vsnet".
ALL_TEACHERS = [
    "cads551",
    "cads552",
    "cads553",
    "cads554",
    "cads555",
    "cads556",
    "cads557",
    "cads558",
    "cads559",
    "moose666",
    "moose888",
    "nnunet_private",
    "saros_nnunet",
    "atm",
    "airrc",
    "lvp",
    "daps",
    "epai_20250421",
    "vsmtrans",
    "vista3d",
    "unest",
    "totalsegmentator",
]

NUM_ROUNDS             = int(os.getenv("MEDAI_NUM_ROUNDS", "3"))
CONSOLIDATION_INTERVAL = 2

# Cross-round convergence auto-stop: end the EM loop early when the student's
# round-over-round pseudo-consistency stops improving, so we do not burn GPU on
# rounds that no longer change the labels (de-human stopping criterion — no human
# decides when to stop).
CONVERGENCE_AUTOSTOP   = env_bool("MEDAI_CONVERGENCE_AUTOSTOP", default=True)
CONVERGENCE_DSC_DELTA  = float(os.getenv("MEDAI_CONVERGENCE_DSC_DELTA", "0.01"))
CONVERGENCE_MIN_ROUNDS = int(os.getenv("MEDAI_CONVERGENCE_MIN_ROUNDS", "2"))
FINETUNE_EPOCHS        = 100   # 增加 epoch 数，让模型充分学习
CONSOLIDATION_EPOCHS   = 50    # 全局整合也需要足够 epoch
LEARNING_RATE          = 5e-5
CONSOLIDATION_LR       = 5e-6

# 单个模型推理 timeout（秒）。CADS/VISTA3D 约 3-5 分钟/case，10分钟足够
INFER_TIMEOUT_SEC = 600

# ShapeKit：老师会议要求正式 pseudo-label 生成中所有 selected outputs 都要过 ShapeKit。
# 只有显式 fast-smoke/debug 时才关闭。
ENABLE_SHAPEKIT = env_bool(
    "MEDAI_ENABLE_SHAPEKIT",
    default=not env_bool("MEDAI_FAST_SMOKE", default=False),
)
DEBUG_ALLOW_NO_SHAPEKIT = (
    env_bool("MEDAI_DEBUG_ALLOW_NO_SHAPEKIT", default=False)
    or env_bool("MEDAI_FAST_SMOKE", default=False)
)

# LabelCritic：Qwen2-VL-7B @ vLLM server
ENABLE_CRITIC = env_bool("MEDAI_ENABLE_CRITIC", default=True)
DEBUG_ALLOW_NO_LABELCRITIC = (
    env_bool("MEDAI_DEBUG_ALLOW_NO_LABELCRITIC", default=False)
    or env_bool("MEDAI_FAST_SMOKE", default=False)
)
CRITIC_BACKEND  = "labelcritic"
LABELCRITIC_OPTIONS = {
    "no_dice_check": env_bool("MEDAI_LABELCRITIC_NO_DICE_CHECK", default=False),
    "no_dual_confirmation": env_bool("MEDAI_LABELCRITIC_NO_DUAL_CONFIRMATION", default=False),
    "simple_prompt_ablation": env_bool("MEDAI_LABELCRITIC_SIMPLE_PROMPT_ABLATION", default=False),
    "conservative_dual": env_bool("MEDAI_LABELCRITIC_CONSERVATIVE_DUAL", default=False),
    "skip_organ_presence_gate": env_bool("MEDAI_LABELCRITIC_SKIP_ORGAN_PRESENCE_GATE", default=False),
    "strict_choice_prompt": env_bool("MEDAI_LABELCRITIC_STRICT_CHOICE_PROMPT", default=False),
}

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
    """Return case_ids with complete E-step artifacts, not just raw outputs."""
    cases_dir = OUTPUT_ROOT / f"round{round_idx}" / "estep" / "cases"
    annotations_dir = OUTPUT_ROOT / f"round{round_idx}" / "estep" / "annotation_versions"
    if not cases_dir.exists():
        return set()
    done = set()
    for case_dir in cases_dir.iterdir():
        if not case_dir.is_dir():
            continue
        selection_meta = case_dir / "selection_metadata.json"
        updated_dir = annotations_dir / case_dir.name / "updated"
        if selection_meta.exists() and updated_dir.exists() and any(updated_dir.glob("*.nii.gz")):
            done.add(case_dir.name)
    return done


def load_student_target_organs() -> list[str]:
    """Current accepted exact target space for the 3D prompt student: 373 organs."""
    with open(PROMPT_TARGET_CONFIG, encoding="utf-8") as f:
        doc = json.load(f)
    return list(doc.get("target_organs", []))


def ensure_current_student_backend_allowed() -> None:
    """Prevent accidental return to the legacy VISTA3D/127-label student path."""
    if STUDENT_BACKEND == "vista3d_legacy" and os.getenv("MEDAI_ALLOW_VISTA3D_LEGACY") != "1":
        raise RuntimeError(
            "MEDAI_STUDENT_BACKEND=vista3d_legacy is disabled by default. "
            "The current teacher-approved mainline is VoxTell-style 3D prompt "
            "student over the 373 exact targets. Set MEDAI_ALLOW_VISTA3D_LEGACY=1 "
            "only for historical reproduction/reference runs."
        )
    if STUDENT_BACKEND not in {"voxtell_style_3d_prompt", "vista3d_legacy"}:
        raise ValueError(f"Unsupported MEDAI_STUDENT_BACKEND={STUDENT_BACKEND}")


def ensure_formal_teacher_pool_registered() -> None:
    """Fail fast if the formal E-step teacher pool drifts from registry keys."""
    from cli_anything.medai.core.model_registry import load_registry

    registry_path = PROJECT_ROOT / "configs/model_registry.yaml"
    registry = load_registry(registry_path)
    models = registry.get("models", {})
    missing = [key for key in ALL_TEACHERS if key not in models]
    disabled = [
        key for key in ALL_TEACHERS
        if key in models and models[key].get("enabled") is False
    ]
    if missing or disabled:
        raise RuntimeError(
            "Formal teacher pool is not registry-ready. "
            f"missing={missing}, disabled={disabled}, registry={registry_path}"
        )


def ensure_formal_quality_gates() -> None:
    """Enforce teacher-meeting requirements for formal non-smoke runs."""
    if not ENABLE_SHAPEKIT and not DEBUG_ALLOW_NO_SHAPEKIT:
        raise RuntimeError(
            "Formal run_em_training.py requires ShapeKit for all selected outputs. "
            "Set MEDAI_DEBUG_ALLOW_NO_SHAPEKIT=1 only for smoke/debug runs."
        )
    if not ENABLE_CRITIC and not DEBUG_ALLOW_NO_LABELCRITIC:
        raise RuntimeError(
            "Formal run_em_training.py requires LabelCritic for multi-candidate "
            "selection. Set MEDAI_DEBUG_ALLOW_NO_LABELCRITIC=1 only for "
            "smoke/debug runs."
        )


def _round_selected_pseudo_label_root(round_idx: int) -> Path:
    """Directory containing the previous round's selected/ShapeKit-final masks."""
    return OUTPUT_ROOT / f"round{round_idx}" / "estep" / "annotation_versions"


def _round_teacher_cache_dirs(round_idx: int) -> dict[str, Path]:
    """Return teacher raw-prediction roots from an earlier E-step.

    Round 2+ must reuse Round 1 teacher candidates instead of rerunning the
    full teacher pool. Each returned path is the base consumed by
    _resolve_preseeded_case_dir: <base>/<case_id>/segmentations/*.nii.gz.
    """
    cache: dict[str, Path] = {}
    cases_root = OUTPUT_ROOT / f"round{round_idx}" / "estep" / "cases"
    if not cases_root.exists():
        return cache
    for teacher in ALL_TEACHERS:
        teacher_ready = False
        for case_dir in cases_root.iterdir():
            teacher_root = case_dir / "raw_predictions" / teacher / case_dir.name
            summary = teacher_root / "inference_summary.json"
            seg_dir = teacher_root / "segmentations"
            if summary.exists() or (seg_dir.exists() and any(seg_dir.glob("*.nii.gz"))):
                teacher_ready = True
                break
        if teacher_ready:
            cache[teacher] = cases_root / "{case_id}" / "raw_predictions" / teacher
    return cache


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
        if not DEBUG_ALLOW_NO_LABELCRITIC:
            raise RuntimeError(
                "Formal E-step requires LabelCritic/vLLM to be online before "
                f"teacher candidate selection. Server unavailable: {VLLM_BASE_URL}. "
                "Start vLLM or set MEDAI_DEBUG_ALLOW_NO_LABELCRITIC=1 only for "
                "smoke/debug fallback runs."
            )
        log("  ⚠️  vLLM server 不可用，本轮以 debug/smoke 模式关闭 LabelCritic")

    from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop
    import json

    if STUDENT_BACKEND == "voxtell_style_3d_prompt":
        # 当前老师确认：缺失/不可一一对应的类别跳过，精确 student 目标为 373。
        organ_list = load_student_target_organs()
    else:
        # Legacy path: old teacher-covered organ list, kept only for reproduction.
        with open(ALL_ORGANS) as f:
            organ_list = json.load(f)

    # Round 2+：复用 Round 1 teacher cache，并把上一轮 best pseudo labels
    # 和上一轮 student 预测都注入候选池。这样后续 EM 轮次不重跑 22 个 teacher。
    # 这对应老师要求的“student 输出 vs 第一轮最好输出”竞争；student 不能自动覆盖 Round1。
    preseeded: dict = {}
    models_to_run = list(ALL_TEACHERS)
    if round_idx > 1:
        teacher_cache = _round_teacher_cache_dirs(1)
        missing_teacher_cache = sorted(set(ALL_TEACHERS) - set(teacher_cache))
        if missing_teacher_cache:
            raise RuntimeError(
                "Round 2+ requires Round 1 teacher cache so teachers are not rerun. "
                f"Missing cached teacher outputs for: {missing_teacher_cache}"
            )
        preseeded.update(teacher_cache)
        models_to_run = []
        log(f"  复用 Round 1 teacher cache: {len(teacher_cache)}/{len(ALL_TEACHERS)} 个 teacher")

        prev_selected_dir = _round_selected_pseudo_label_root(round_idx - 1)
        if prev_selected_dir.exists() and any(prev_selected_dir.iterdir()):
            preseeded["round_prev_selected"] = prev_selected_dir
            log(f"  注入上一轮 selected pseudo labels 参与竞争: {prev_selected_dir}")
        else:
            log(f"  上一轮 selected pseudo labels 不存在，跳过注入: {prev_selected_dir}")

        prev_pred_dir = OUTPUT_ROOT / f"round{round_idx - 1}" / "student_predictions"
        if prev_pred_dir.exists() and any(prev_pred_dir.iterdir()):
            preseeded["student_prev"] = prev_pred_dir
            log(f"  注入上一轮 student 预测参与竞争: {prev_pred_dir}")
        else:
            log(f"  上一轮 student 预测不存在，跳过注入: {prev_pred_dir}")

    log(f"  待推理模型: {len(models_to_run)}个, 候选teacher总数: {len(ALL_TEACHERS)}个, 器官: {len(organ_list)}个, "
        f"Student backend: {STUDENT_BACKEND}, "
        f"待处理 case: {50 - len(done)}/50, "
        f"ShapeKit: {'开' if ENABLE_SHAPEKIT else '关'}, "
        f"LabelCritic: {'开' if enable_critic else '关'}"
        + (f", preseeded: {','.join(preseeded.keys())}" if preseeded else ""))
    if any(LABELCRITIC_OPTIONS.values()):
        log(f"  LabelCritic diagnostic options: {LABELCRITIC_OPTIONS}")

    # 如果所有 case 都完成了，直接返回成功
    if len(done) >= 50:
        log("  所有 case 已完成，跳过 E-step")
        return {"status": "success", "num_cases": 50, "total_updated": 0, "skipped": True}

    result = run_multimodel_annotation_loop(
        case_list=CASE_LIST,
        output_folder=out_dir,
        models=models_to_run,
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
        labelcritic_options=LABELCRITIC_OPTIONS,
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
    优先级：真实标注 > E-step selected pseudo label > OrganModelPerformance 最优 teacher > yaml 配置 teacher。

    This is a legacy VISTA3D/127-label reproduction helper. It is guarded by
    `MEDAI_ALLOW_VISTA3D_LEGACY=1` and is not the current 373-organ mainline.
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
    datalist = {"training": [], "validation": []}
    built = 0
    selected_used = 0

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
            case_selected_used = 0

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

                # 2. 当前 E-step selected pseudo label（可能来自 teacher、round_prev_selected 或 student_prev）。
                # Student 预测必须先进入 E-step candidate selection，不能在 M-step 自动覆盖 teacher/Round1。
                if mask_path is None:
                    selected_cand = (
                        OUTPUT_ROOT / f"round{round_idx}" / "estep" /
                        "annotation_versions" / case_id / "updated" / f"{organ}.nii.gz"
                    )
                    if selected_cand.exists():
                        mask_path = selected_cand
                        case_selected_used += 1

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
            if case_selected_used > 0:
                selected_used += 1

        except Exception as e:
            log(f"  {case_id} 构建失败: {e}")

    n_val = max(1, len(datalist["training"]) // 10)
    datalist["validation"] = datalist["training"][:n_val]

    datalist_path = out_dir / "datalist.json"
    with open(datalist_path, "w") as f:
        json.dump(datalist, f, indent=2)

    log(f"数据集构建完成: {built} 个 case"
        + (f"，其中 {selected_used} 个 case 使用了当前 E-step selected pseudo labels" if selected_used > 0 else ""))
    return datalist_path


def build_3d_prompt_student_dataset(round_idx: int) -> Path:
    """Build VoxTell-style prompt/mask manifest from current E-step pseudo-labels."""
    from cli_anything.medai.core.voxtell_student import VoxTellStudent

    log("构建 3D prompt student 训练 manifest（373 exact organs）...")
    out_dir = OUTPUT_ROOT / f"round{round_idx}" / "mstep"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_root = OUTPUT_ROOT / f"round{round_idx}" / "estep" / "annotation_versions"
    manifest_path = out_dir / "voxtell_prompt_student_manifest.json"
    student = VoxTellStudent(
        model_dir=VOXTELL_MODEL_DIR,
        target_config=PROMPT_TARGET_CONFIG,
        device="cuda",
    )
    manifest = student.build_training_manifest(
        cases_root=cases_root,
        output_manifest=manifest_path,
        case_list=CASE_LIST,
        require_images=True,
    )
    log(
        f"3D prompt manifest 完成: items={manifest.get('num_items', 0)}, "
        f"cases={manifest.get('num_cases', 0)}, "
        f"missing_image_items={manifest.get('num_items_missing_image', 0)}"
    )
    return manifest_path


def build_student_dataset(round_idx: int) -> Path:
    ensure_current_student_backend_allowed()
    if STUDENT_BACKEND == "voxtell_style_3d_prompt":
        return build_3d_prompt_student_dataset(round_idx)
    if STUDENT_BACKEND == "vista3d_legacy":
        return build_vista3d_dataset(round_idx)
    raise ValueError(f"Unsupported MEDAI_STUDENT_BACKEND={STUDENT_BACKEND}")


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


def run_prompt_student_mstep(round_idx: int, manifest_path: Path, global_consolidation: bool = False) -> dict:
    """Run or plan the VoxTell-style 3D prompt student M-step.

    The upstream VoxTell release provides inference code but no ready project
    fine-tuning script. If MEDAI_VOXTELL_TRAIN_CMD is unset, we stop after
    producing the prompt/mask manifest and do not fabricate a checkpoint.
    """
    log(f"=== Round {round_idx} 3D prompt student M-step 开始 ===")
    out_dir = OUTPUT_ROOT / f"round{round_idx}" / "mstep"
    result_path = out_dir / "voxtell_prompt_mstep_result.json"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    result = {
        "stage": "voxtell_style_3d_prompt_mstep",
        "status": "pending" if VOXTELL_TRAIN_CMD else "manifest_ready",
        "training_status": "manifest_ready_no_trainer" if not VOXTELL_TRAIN_CMD else "pending",
        "student_backend": STUDENT_BACKEND,
        "manifest_path": str(manifest_path),
        "num_items": manifest.get("num_items", 0),
        "num_cases": manifest.get("num_cases", 0),
        "target_config": str(PROMPT_TARGET_CONFIG),
        "target_organs": len(load_student_target_organs()),
        "model_dir": str(VOXTELL_MODEL_DIR),
        "finetuned_checkpoint": None,
        "note": (
            "3D prompt-based student uses prompt/mask pairs and does not use "
            "VISTA3D 127-class label IDs."
        ),
    }

    if not VOXTELL_TRAIN_CMD:
        result["reason"] = (
            "MEDAI_VOXTELL_TRAIN_CMD is not configured. Manifest is ready; "
            "real training requires a project fine-tuning command around VoxTellModel. "
            "The EM loop must stop here instead of pretending that a student "
            "checkpoint was trained."
        )
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log("3D prompt M-step 已生成 manifest；未配置真实训练命令，因此暂停 EM，不生成 checkpoint。")
        return result

    stop_vllm_for_mstep()
    env = os.environ.copy()
    env.update({
        "MEDAI_PROMPT_STUDENT_MANIFEST": str(manifest_path),
        "MEDAI_PROMPT_STUDENT_OUTPUT_DIR": str(out_dir),
        "MEDAI_PROMPT_TARGET_CONFIG": str(PROMPT_TARGET_CONFIG),
        "MEDAI_VOXTELL_MODEL_DIR": str(VOXTELL_MODEL_DIR),
        "MEDAI_TEXT_ENCODING_MODEL": str(VOXTELL_TEXT_ENCODING_MODEL),
        "MEDAI_MSTEP_EPOCHS": str(CONSOLIDATION_EPOCHS if global_consolidation else FINETUNE_EPOCHS),
        "MEDAI_MSTEP_LR": str(CONSOLIDATION_LR if global_consolidation else LEARNING_RATE),
    })
    start = time.time()
    proc = subprocess.run(
        VOXTELL_TRAIN_CMD,
        shell=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.time() - start
    train_result_path = out_dir / "voxtell_prompt_train_result.json"
    train_result = {}
    if train_result_path.exists():
        try:
            with open(train_result_path, encoding="utf-8") as f:
                train_result = json.load(f)
        except Exception:
            train_result = {}
    inference_model_dir = Path(train_result.get("inference_model_dir", out_dir / "voxtell_finetuned_model"))
    inference_checkpoint = inference_model_dir / "fold_0" / "checkpoint_final.pth"
    finetuned_checkpoint = Path(train_result.get("finetuned_checkpoint", out_dir / "model_finetune.pth"))
    has_inference_model = (inference_model_dir / "plans.json").exists() and inference_checkpoint.exists()
    has_finetuned_checkpoint = finetuned_checkpoint.exists()
    result.update({
        "training_status": "completed" if proc.returncode == 0 else "failed",
        "status": "success" if proc.returncode == 0 and has_inference_model else "failed",
        "return_code": proc.returncode,
        "runtime_sec": round(elapsed, 3),
        "finetuned_checkpoint": str(finetuned_checkpoint) if has_finetuned_checkpoint else None,
        "inference_model_dir": str(inference_model_dir) if has_inference_model else None,
        "inference_checkpoint": str(inference_checkpoint) if has_inference_model else None,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    })
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    restart_vllm_after_mstep()
    log(f"3D prompt M-step 完成: status={result.get('status')}, checkpoint={result.get('finetuned_checkpoint')}")
    return result


def run_student_mstep(round_idx: int, dataset_path: Path, global_consolidation: bool = False) -> dict:
    ensure_current_student_backend_allowed()
    if STUDENT_BACKEND == "voxtell_style_3d_prompt":
        return run_prompt_student_mstep(round_idx, dataset_path, global_consolidation=global_consolidation)
    if STUDENT_BACKEND == "vista3d_legacy":
        return run_mstep(round_idx, dataset_path, global_consolidation=global_consolidation)
    raise ValueError(f"Unsupported MEDAI_STUDENT_BACKEND={STUDENT_BACKEND}")


# ── 评估指标 ──────────────────────────────────────────────────────────────────

def compute_round_metrics(round_idx: int, reference_round: int = 1) -> dict:
    """
    Evaluate student consistency against selected pseudo labels.

    This project does not assume an expert fine-label set. Metrics produced here
    are student-vs-selected-pseudo-label consistency signals, not true accuracy.
    """
    import csv as csv_mod, nibabel as nib, numpy as np

    log(f"计算 Round {round_idx} student vs Round {reference_round} selected pseudo-label consistency 指标...")
    metrics_dir = OUTPUT_ROOT / f"round{round_idx}" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    student_pred_dir = OUTPUT_ROOT / f"round{round_idx}" / "student_predictions"
    pseudo_reference_root = OUTPUT_ROOT / f"round{reference_round}" / "estep" / "annotation_versions"
    cases = []
    with open(CASE_LIST) as f:
        for row in csv_mod.DictReader(f):
            cases.append(row)

    dice_rows, organ_dices, case_dices = [], {}, {}

    for case in cases:
        case_id = case["case_id"]
        ann_folder = pseudo_reference_root / case_id / "updated"
        if not ann_folder.exists():
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
                dice_rows.append({
                    "round": round_idx,
                    "reference_round": reference_round,
                    "case_id": case_id,
                    "organ": organ,
                    "dsc": dsc,
                    "pseudo_consistency_dsc": dsc,
                    "metric_family": "pseudo_consistency",
                    "metric_scope": "student_vs_selected_pseudo_label",
                    "ground_truth_status": "pseudo_label_candidate",
                })
                organ_dices.setdefault(organ, []).append(dsc)
                case_dices.setdefault(case_id, []).append(dsc)
            except Exception:
                continue

    if dice_rows:
        with open(metrics_dir / "student_dice_per_organ.csv", "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=[
                "round", "reference_round", "case_id", "organ", "dsc",
                "pseudo_consistency_dsc", "metric_family", "metric_scope",
                "ground_truth_status",
            ])
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
        "reference_round": reference_round,
        "source": "student_predictions",
        "metric_family": "pseudo_consistency",
        "metric_scope": "student_vs_selected_pseudo_label",
        "ground_truth_status": "pseudo_label_candidate",
        "pseudo_reference_root": str(pseudo_reference_root),
        "accuracy_warning": "No expert fine-label set is assumed. These are not true accuracy metrics.",
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
    """Legacy VISTA3D student prediction path.

    Only used when `MEDAI_STUDENT_BACKEND=vista3d_legacy`. The current default
    mainline uses `save_prompt_student_predictions()` for 373 exact prompt
    targets.
    """
    log(f"保存 Round {round_idx} legacy VISTA3D student 推理结果（127个器官）...")
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


def save_prompt_student_predictions(round_idx: int):
    """Save VoxTell-style 3D prompt student predictions for 373 exact organs."""
    log(f"保存 Round {round_idx} 3D prompt student 推理结果（373 organs）...")
    pred_dir = OUTPUT_ROOT / f"round{round_idx}" / "student_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    finetuned_model_dir = OUTPUT_ROOT / f"round{round_idx}" / "mstep" / "voxtell_finetuned_model"
    if (finetuned_model_dir / "plans.json").exists() and (finetuned_model_dir / "fold_0" / "checkpoint_final.pth").exists():
        model_dir = finetuned_model_dir
    else:
        model_dir = VOXTELL_MODEL_DIR

    from cli_anything.medai.core.voxtell_student import VoxTellStudent
    import csv as csv_mod

    student = VoxTellStudent(
        model_dir=model_dir,
        target_config=PROMPT_TARGET_CONFIG,
        device="cuda",
    )
    all_organs = load_student_target_organs()

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
        if case_pred_dir.exists() and any(case_pred_dir.glob("*.nii.gz")):
            saved += 1
            continue
        case_pred_dir.mkdir(parents=True, exist_ok=True)
        result = student.segment(
            ct_image=ct_path,
            prompts=all_organs,
            output_dir=case_pred_dir,
            dry_run=not model_dir.exists(),
            timeout_sec=1800,
        )
        if result.get("status") == "success":
            saved += 1
            log(f"  {case_id}: ✓ {result.get('num_masks', 0)}/{len(all_organs)} masks")
        else:
            log(f"  {case_id}: {result.get('status')} ({result.get('reason', 'see result json')})")

    log(f"3D prompt student 推理阶段完成/计划完成: {saved}/{len(cases)} 个 case")


def save_student_predictions(round_idx: int):
    ensure_current_student_backend_allowed()
    if STUDENT_BACKEND == "voxtell_style_3d_prompt":
        return save_prompt_student_predictions(round_idx)
    if STUDENT_BACKEND == "vista3d_legacy":
        return save_round_predictions(round_idx)
    raise ValueError(f"Unsupported MEDAI_STUDENT_BACKEND={STUDENT_BACKEND}")


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def convergence_reached(current_dsc, previous_dsc, threshold: float) -> bool:
    """Pure convergence test: True when both round scores are valid and the
    absolute round-over-round change is below ``threshold``. Kept side-effect-free
    so it is unit-testable without running the full EM loop."""
    if not isinstance(current_dsc, (int, float)) or not isinstance(previous_dsc, (int, float)):
        return False
    return abs(float(current_dsc) - float(previous_dsc)) < float(threshold)


def _round_mean_dsc(round_idx: int):
    """Read a completed round's student pseudo-consistency mean DSC, or None."""
    sp = OUTPUT_ROOT / f"round{round_idx}" / "round_summary.json"
    if not sp.exists():
        return None
    try:
        m = json.loads(sp.read_text(encoding="utf-8")).get("metrics", {})
    except Exception:
        return None
    v = m.get("overall_mean_dsc")
    return float(v) if isinstance(v, (int, float)) else None


def _student_manifest_weight_summary(round_idx: int) -> dict:
    """Reliability-weight distribution of this round's student manifest, so the
    self-cleaning trend (more strong / fewer zero-weight items over rounds) is
    auditable in round_summary."""
    mp = OUTPUT_ROOT / f"round{round_idx}" / "mstep" / "voxtell_prompt_student_manifest.json"
    if not mp.exists():
        return {}
    try:
        doc = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = doc.get("items") if isinstance(doc, dict) else (doc if isinstance(doc, list) else [])
    rows = rows or []
    weights = [float(r.get("training_weight") or 0.0) for r in rows if isinstance(r, dict)]
    if not weights:
        return {}
    return {
        "num_items": len(weights),
        "mean_training_weight": round(sum(weights) / len(weights), 4),
        "num_strong_items_weight_ge_0_5": sum(1 for w in weights if w >= 0.5),
        "num_zero_weight_items": sum(1 for w in weights if w == 0.0),
    }


def main():
    ensure_current_student_backend_allowed()
    ensure_formal_teacher_pool_registered()
    ensure_formal_quality_gates()
    log("=" * 60)
    log("EM Loop 训练 v2（断点续跑 + vLLM LabelCritic + 3D prompt student）")
    log(f"  轮数: {NUM_ROUNDS}, Teacher: {len(ALL_TEACHERS)}个, Case: 50")
    log(f"  Student backend: {STUDENT_BACKEND}")
    if STUDENT_BACKEND == "voxtell_style_3d_prompt":
        log(f"  Prompt target organs: {len(load_student_target_organs())}")
    log(f"  ShapeKit: {'开' if ENABLE_SHAPEKIT else '关'}, LabelCritic: {'开' if ENABLE_CRITIC else '关'}")
    if any(LABELCRITIC_OPTIONS.values()):
        log(f"  LabelCritic diagnostic options: {LABELCRITIC_OPTIONS}")
    log(f"  vLLM: {VLLM_BASE_URL} ({'在线' if check_vllm_server() else '离线'})")
    log(f"  VISTA3D legacy reference: {VISTA3D_MODEL}")
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
        dataset_path = build_student_dataset(round_idx)

        # M-step — 실패 시 1회 재시도
        is_consolidation = (round_idx % CONSOLIDATION_INTERVAL == 0)
        mstep_result = run_student_mstep(round_idx, dataset_path, global_consolidation=is_consolidation)

        if (
            STUDENT_BACKEND == "voxtell_style_3d_prompt"
            and mstep_result.get("training_status") == "manifest_ready_no_trainer"
        ):
            log("3D prompt student 训练器尚未配置：已生成训练 manifest，本次在 M-step 暂停，避免进入未训练 student 的后续轮次。")
            round_elapsed = time.time() - round_start
            summary = {
                "round": round_idx,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "estep_status": estep_result.get("status"),
                "mstep_status": "manifest_ready_no_trainer",
                "student_backend": STUDENT_BACKEND,
                "mstep_type": "global_consolidation" if is_consolidation else "local_update",
                "finetuned_checkpoint": None,
                "metrics": {
                    "status": "skipped",
                    "reason": "prompt student manifest is ready but no real trainer/checkpoint is configured",
                },
                "round_elapsed_hours": round(round_elapsed / 3600, 2),
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            log(f"摘要已保存: {summary_path}")
            break

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
            mstep_result = run_student_mstep(round_idx, dataset_path, global_consolidation=is_consolidation)

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

        if (
            STUDENT_BACKEND == "voxtell_style_3d_prompt"
            and mstep_result.get("training_status") == "manifest_ready_no_trainer"
        ):
            log("3D prompt student 训练器尚未配置，本轮跳过 student 推理和 pseudo-consistency 评估，避免伪指标。")
            round_metrics = {
                "status": "skipped",
                "reason": "prompt student manifest is ready but no real trainer/checkpoint is configured",
            }
        else:
            # 先保存 student 推理结果，再评估（评估依赖 student_predictions）
            save_student_predictions(round_idx)

            # 评估 student 与 Round1 selected pseudo label 的一致性；不能当真实 accuracy。
            round_metrics = compute_round_metrics(round_idx)

        round_elapsed = time.time() - round_start
        log(f"Round {round_idx} 完成，耗时 {round_elapsed/3600:.1f} 小时")

        summary = {
            "round": round_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "estep_status": estep_result.get("status"),
            "mstep_status": mstep_result.get("status"),
            "student_backend": STUDENT_BACKEND,
            "mstep_type": "global_consolidation" if is_consolidation else "local_update",
            "finetuned_checkpoint": mstep_result.get("finetuned_checkpoint"),
            "metrics": {
                "overall_mean_dsc": round_metrics.get("overall_mean_dsc"),
                "overall_mean_pseudo_consistency_dsc": round_metrics.get("overall_mean_dsc"),
                "metric_family": round_metrics.get("metric_family", "pseudo_consistency"),
                "metric_scope": round_metrics.get("metric_scope", "student_vs_selected_pseudo_label"),
                "accuracy_warning": round_metrics.get("accuracy_warning", "Not true accuracy."),
                "top5_organs": round_metrics.get("top5_organs", []),
                "bottom5_organs": round_metrics.get("bottom5_organs", []),
            },
            "round_elapsed_hours": round(round_elapsed / 3600, 2),
        }
        summary["reliability_weights"] = _student_manifest_weight_summary(round_idx)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log(f"摘要已保存: {summary_path}")

        # 跨轮收敛自动停机：student 伪一致性不再提升则提前结束，省 GPU，无需人工决定何时停。
        if CONVERGENCE_AUTOSTOP and round_idx >= CONVERGENCE_MIN_ROUNDS:
            cur_dsc = round_metrics.get("overall_mean_dsc") if isinstance(round_metrics, dict) else None
            prev_dsc = _round_mean_dsc(round_idx - 1)
            if isinstance(cur_dsc, (int, float)) and isinstance(prev_dsc, (int, float)):
                delta = abs(float(cur_dsc) - float(prev_dsc))
                log(f"收敛检查: Round{round_idx} mean_dsc={cur_dsc:.4f} vs Round{round_idx-1} {prev_dsc:.4f}, "
                    f"|Δ|={delta:.4f} (阈值 {CONVERGENCE_DSC_DELTA})")
                if convergence_reached(cur_dsc, prev_dsc, CONVERGENCE_DSC_DELTA):
                    log(f"✅ 跨轮收敛，提前停机（跳过剩余 {NUM_ROUNDS - round_idx} 轮）")
                    summary["converged"] = True
                    summary["convergence"] = {
                        "stopped_after_round": round_idx,
                        "overall_mean_dsc_current": cur_dsc,
                        "overall_mean_dsc_previous": prev_dsc,
                        "delta": round(delta, 6),
                        "threshold": CONVERGENCE_DSC_DELTA,
                        "metric_family": "pseudo_consistency",
                        "metric_scope": "student_vs_selected_pseudo_label",
                    }
                    with open(summary_path, "w") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    (OUTPUT_ROOT / "convergence_stop.json").write_text(
                        json.dumps(summary["convergence"], indent=2, ensure_ascii=False), encoding="utf-8")
                    break

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
                "overall_mean_pseudo_consistency_dsc": s.get("metrics", {}).get("overall_mean_dsc"),
                "elapsed_hours": s.get("round_elapsed_hours"),
                "checkpoint": s.get("finetuned_checkpoint"),
            })

    dscs = [r["overall_mean_pseudo_consistency_dsc"] for r in comparison["rounds"] if r.get("overall_mean_pseudo_consistency_dsc")]
    if len(dscs) >= 2:
        comparison["improvement"] = {
            "round1_pseudo_consistency_dsc": dscs[0],
            "final_pseudo_consistency_dsc": dscs[-1],
            "absolute_gain": round(dscs[-1] - dscs[0], 4),
            "relative_gain_pct": round((dscs[-1] - dscs[0]) / max(dscs[0], 1e-6) * 100, 2),
            "accuracy_warning": "This is pseudo-label consistency change, not true accuracy improvement.",
        }

    with open(OUTPUT_ROOT / "training_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    log(f"\n{'='*60}")
    log(f"训练完成！总耗时 {total_elapsed/3600:.1f} 小时")
    if comparison.get("improvement"):
        imp = comparison["improvement"]
        log(f"pseudo-consistency DSC 变化: {imp['round1_pseudo_consistency_dsc']:.4f} → {imp['final_pseudo_consistency_dsc']:.4f} (+{imp['relative_gain_pct']:.1f}%)")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
