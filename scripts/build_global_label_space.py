#!/usr/bin/env python3
"""build_global_label_space.py — 阶段0 配置基线生成器（幂等）

读取权威路由表 configs/organ_routing_from_xlsx.json，产出/校验：
  1) configs/global_label_space.json   —— 384 器官固定全局标签空间（id 1..384，0=背景）
  2) configs/routing_token_to_model.json —— 每个路由 token → {model, subtask, enabled, reason}

设计依据：docs/ARCHITECTURE_multimodel_routing.md §0/§3/§7.8/§9。

用法：
  python3 scripts/build_global_label_space.py            # 生成 + 校验
  python3 scripts/build_global_label_space.py --check    # 仅校验，不写文件

可重复运行（幂等）：相同输入产生相同输出。
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.normpath(os.path.join(HERE, "..", "configs"))
ROUTING = os.path.join(CONFIGS, "organ_routing_from_xlsx.json")
GLOBAL_LABEL_SPACE = os.path.join(CONFIGS, "global_label_space.json")
ROUTING_TOKEN_TO_MODEL = os.path.join(CONFIGS, "routing_token_to_model.json")


# ---------------------------------------------------------------------------
# 归一化辅助
# ---------------------------------------------------------------------------
def _norm_totalseg_subtask(token):
    """从 'Totalsegmentator (xxx)' / 'Totalsegmentator(xxx)' 提取规范化子任务名。
    空格不一致视为同一个。"""
    m = re.match(r"^Totalsegmentator\s*\(\s*(.*?)\s*\)$", token)
    if not m:
        return None
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# routing_token_to_model 规则（人工规则，按设计 §3/§7.8/§9）
# ---------------------------------------------------------------------------
def build_token_map(tokens):
    """根据出现的 token 集合，构造 token -> 条目映射。"""
    out = {}
    for tok in tokens:
        # --- CADS551..CADS559 ---
        m = re.match(r"^CADS(55\d)$", tok)
        if m:
            out[tok] = {
                "model": "cads" + m.group(1),
                "subtask": None,
                "enabled": True,
                "reason": None,
            }
            continue

        # --- 裸 CADS（按器官区分到 cads556/cads558，见 bare_cads_organ_overrides）---
        if tok == "CADS":
            out[tok] = {
                "model": None,
                "subtask": None,
                "enabled": True,
                "reason": None,
                "note": "需在 organ 级按器官指定 cads556/cads558，见 bare_cads_organ_overrides",
            }
            continue

        # --- Totalsegmentator (子任务) / Totalsegmentator(子任务) ---
        sub = _norm_totalseg_subtask(tok)
        if sub is not None:
            # body / total 等空格差异归并到同一 subtask
            out[tok] = {
                "model": "totalsegmentator",
                "subtask": sub,
                "enabled": True,
                "reason": None,
            }
            continue

        # --- MOOSE 子任务 ---
        mm = re.match(r"^MOOSE.*\(\s*(clin_ct_\w+)\s*\)$", tok)
        if mm:
            subtask = mm.group(1)
            if subtask == "clin_ct_peripheral_bones":
                out[tok] = {"model": "moose666", "subtask": subtask,
                            "enabled": True, "reason": None}
            elif subtask == "clin_ct_cardiac":
                out[tok] = {"model": "moose888", "subtask": subtask,
                            "enabled": True, "reason": None}
            else:
                out[tok] = {
                    "model": None,
                    "subtask": subtask,
                    "enabled": False,
                    "reason": "missing weight (only 666/888 available)",
                }
            continue

        # --- 其余单 token 模型 ---
        simple = {
            "VISTA3D": ("vista3d", True, None),
            "VSmTrans": ("vsmtrans", True, None),
            "ePAI": ("epai_20250421", True, None),
            "SAROS nnUNet": ("saros_nnunet", True, None),
            "AirRC": ("airrc", True, None),
            "LVP": ("lvp", True, None),
            "ATM": ("atm", True, None),
            "UNEST": ("unest", True, None),
            "Dataset224_AbdomenAtlas1.1": ("nnunet_private", True, None),
            "Duke": (None, False, "决策:忽略不封装"),
            "Dataset001_ATLASNet": (None, False, "决策:忽略不封装"),
            "DAPS": ("daps", False, "weight not downloaded yet"),
        }
        if tok in simple:
            model, enabled, reason = simple[tok]
            entry = {"model": model, "subtask": None,
                     "enabled": enabled, "reason": reason}
            if tok == "LVP":
                entry["note"] = "lvp 即 vsnet"
            out[tok] = entry
            continue

        raise ValueError("未处理的 token，需要补规则: %r" % tok)

    return out


BARE_CADS_OVERRIDES = {
    "pericardium": "cads556",
    "submandibular_gland_left": "cads558",
    "submandibular_gland_right": "cads558",
}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="仅校验，不写文件")
    args = ap.parse_args()

    with open(ROUTING, "r", encoding="utf-8") as f:
        routing = json.load(f)

    organ_to_models = routing["organ_to_models"]
    organs = sorted(organ_to_models.keys())

    # ----- (1) global_label_space.json -----
    if len(organs) != routing.get("total_organs", len(organs)):
        print("[WARN] organ_to_models 键数 %d != total_organs %d"
              % (len(organs), routing.get("total_organs")), file=sys.stderr)

    organ_to_id = {organ: i + 1 for i, organ in enumerate(organs)}
    id_to_organ = {str(i + 1): organ for i, organ in enumerate(organs)}
    gls = {
        "version": 1,
        "background": 0,
        "total_organs": len(organs),
        "organ_to_id": organ_to_id,
        "id_to_organ": id_to_organ,
    }

    # ----- (2) routing_token_to_model.json -----
    tokens = set()
    for ms in organ_to_models.values():
        tokens.update(ms)
    token_map = build_token_map(tokens)

    rttm = {
        "version": 1,
        "note": ("token 归一化映射；裸 CADS 按器官区分见 bare_cads_organ_overrides。"
                 "Totalsegmentator 空格差异归并到同一 subtask。"),
        "bare_cads_organ_overrides": BARE_CADS_OVERRIDES,
        "tokens": token_map,
    }

    # ----- 校验：token 全覆盖（零遗漏）-----
    missing = sorted(t for t in tokens if t not in token_map)
    if missing:
        print("[ERROR] 以下 token 在 routing_token_to_model 中无映射:", file=sys.stderr)
        for t in missing:
            print("   ", repr(t), file=sys.stderr)
        sys.exit(2)

    # ----- 校验：全局标签空间 id 连续无重复 -----
    ids = sorted(organ_to_id.values())
    assert ids == list(range(1, len(organs) + 1)), "id 不连续/有重复/有空缺"
    assert len(set(organ_to_id.values())) == len(organs), "id 有重复"

    if args.check:
        print("[OK] 校验通过：%d 器官，%d token 全覆盖" % (len(organs), len(tokens)))
        return

    with open(GLOBAL_LABEL_SPACE, "w", encoding="utf-8") as f:
        json.dump(gls, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(ROUTING_TOKEN_TO_MODEL, "w", encoding="utf-8") as f:
        json.dump(rttm, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("[OK] 写出 %s (%d 器官)" % (GLOBAL_LABEL_SPACE, len(organs)))
    print("[OK] 写出 %s (%d token)" % (ROUTING_TOKEN_TO_MODEL, len(token_map)))


if __name__ == "__main__":
    main()
