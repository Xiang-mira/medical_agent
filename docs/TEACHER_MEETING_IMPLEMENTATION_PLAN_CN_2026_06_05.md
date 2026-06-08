# 老师会议要求执行计划 - 3D Prompt Student 蒸馏主线

日期：2026-06-05

本文档把老师会议中明确提出的项目流程整理成一份可执行、可验收、可逐步修复的工程计划。后续所有修改都应围绕这份计划推进，避免重新回到 2D 路线、VISTA3D 127 类路线，或把 teacher 输出误写成真实 ground truth。

本计划同时记录当前现实约束：目前账号已经具备，但还没有拿到 JHU 组内精标集；合作方已确认缺失或不能一一对应的类别直接跳过；因此当前工程主线以 `373 organs` 为 exact target。老师会议中反复提到的 `377 classes` 在本文中保留为研究叙事目标：不要被 VISTA3D 的 127 类限制，应该面向多 teacher label space 扩展；但当前实际可执行、可验收的训练目标是 373。

## 0. 一句话目标

我们要做的不是单独证明某个模型好，也不是围绕 VISTA3D 做 127 类适配，而是建立一条可扩展的多 teacher 蒸馏 pipeline：

```text
多 3D CT
  -> 多个固定 teacher models 做 inference
  -> 收集每个 case-organ 的候选 masks
  -> 多候选用 Label Critic 选择最佳伪标签
  -> 所有 selected masks 过 ShapeKit
  -> 组装可审计的 pseudo-label dataset
  -> 训练成熟的 3D prompt-based student
  -> student 重新预测
  -> student 输出与 Round1 best pseudo label 竞争
  -> 挖出 student 学不会的 hard cases
  -> 人工复核/修正后进入下一轮
```

当前工程落地目标是 `373 organs`。老师会议里提到的 377 类应理解为“不被 VISTA3D 127 类限制、尽可能覆盖多 teacher label space”的研究目标；但当前我们已经按合作方反馈确认：缺失或不能一一对应的类直接跳过，因此本项目现在的 exact target 是 373。

## 0.1 老师原话到工程动作的映射

这部分用于防止后续讨论再次变成抽象概念。每条老师要求都必须落到具体模块、产物和验收标准。

| 老师要求 / 会议要点 | 我们的工程动作 | 产物 | 当前状态 |
| --- | --- | --- | --- |
| “第一步你有很多 CT” | 选择 CT case list，formal 阶段优先 50 个有 tumor annotation 的 CT | `data_manifest/case_list_50_tumor.csv` | 已有 case list，仍需 formal 全量运行 |
| “这么多模型 test 一下得到很多输出” | 21 个模型资料对齐；teacher 只做 fixed inference，不训练 | registry、每模型 wrapper、raw predictions | 静态审计已做，仍在逐个真实 runnable 验证 |
| “有些类只有一个模型能打，就当它是 ground truth” | 改写为 `single_teacher_default pseudo-label candidate`，不能写 true ground truth | `selection_metadata.json`、manifest | 已实现基础 metadata |
| “有些类好几个模型能打，需要 label critic 判断哪个最好” | Label Critic 前移到 teacher candidate selection 阶段 | `labelcritic_records`、`vlm_decisions.jsonl` | 小样本跑通，仍需多 teacher 大规模验证 |
| “所有输出都过一遍 ShapeKit” | selected mask 统一进入 ShapeKit；失败/fallback 写 metadata | `shapekit_status`、review queue | 默认已开启，仍需 373 全量统计 |
| “组装成 50 CT 和 377 类最理想输出的数据集” | 当前按 50 CT × 373 organs 组装 pseudo-label dataset，保留 377 作为研究目标叙事 | training manifest | tiny/small 验证已做，formal 未跑 |
| “去训 student 模型” | 使用 3D prompt-based / VoxTell-style student；不走 2D，不走 VISTA3D 127 类主线 | VoxTell manifest、train result、checkpoint | dry-run/ministep 已做，真实长训练未完成 |
| “训完继续打标签，这时不一定需要 label creators” | student 重新 inference，作为下一轮 candidate source | `student_predictions/<case>/<organ>.nii.gz` | dry-run 合同已验证，真实 inference 未完成 |
| “如果 student 学崩，还要用第一轮输出” | Round2 中 student vs Round1 best pseudo label 竞争；不自动覆盖 | Round2 selection manifest | 脚本骨架已有，大规模未跑 |
| “需要预测 student 输出和第一轮所谓 ground truth 之间的 Dice” | Dice 命名为 pseudo-label consistency，不叫真实 accuracy | failure mining CSV/JSON | 脚本已实现，需真实 student 输出 |
| “找到 student 怎么学都学不会的例子” | hard-case mining + manual review queue | `student_failure_cases.csv/json` | 初版 smoke 已通过 |
| “不要转 2D，2D 没有用” | CT 输入保持 3D，prompt 在 3D student 内部使用 | VoxTell-style student path | 默认路径已切换 |
| “VISTA3D 没什么用，不要作为核心” | VISTA3D 只保留 teacher/reference/legacy，不能限制 target | legacy guard、文档说明 | 默认主线已移除 VISTA3D student |

## 0.2 当前数值口径

- `384`：历史 xlsx / global label space 中的全量 organ 数。
- `381`：xlsx 中有 enabled best-model route 的 organ 数。
- `8`：SAROS 路由存在但 local label 粗粒度、无法精确一一对应的 organ；合作方已确认此类缺失或不能一一对应的类直接跳过。
- `3`：当前 no-enabled-route / 零候选 organ，也跳过。
- `373`：当前 accepted exact prompt target，也是后续训练和 merge 的工程验收目标。
- `377`：老师会议中用于强调“不被 VISTA3D 127 类限制”的研究目标口径；后续汇报可以说“原目标方向是多 teacher 覆盖的约 377 类，但当前按已核验可精确处理的 373 类落地”。

后续所有代码、报告和汇报必须遵守：当前项目验收写 `373 organs`；不能为了迎合 377 把不能一一对应的类别硬塞回训练标签；更不能把粗标签近似当精确 mask。

## 0.3 本次会议要求不遗漏清单

这部分把会议口述内容重新整理成必须实现的 checklist。后续如果某个实现和这里冲突，以这里为准。

| 会议要求 | 正确理解 | 必须实现的项目行为 |
| --- | --- | --- |
| “第一步你有很多 CT” | 输入是多个 3D CT case，不是 2D slice 数据集 | 保持 3D CT 输入；formal 阶段优先 50 个有 tumor annotation 的 CT |
| “这么多模型 test 一下” | teacher models 是固定推理模型，不在 E-step 训练 | 每个 teacher 只做 inference/test；记录 checkpoint、命令、输出 |
| “得到很多很多输出” | 每个 case-organ 可能有 0/1/多个候选 mask | 保存所有候选，不允许只保留最后一个或静默覆盖 |
| “只有一个模型能打，就当它是 ground truth” | 工程上不能写成 true ground truth，只能作为当前默认伪标签候选 | 标记为 `single_teacher_default` 和 `pseudo_label_candidate` |
| “好几个模型能打，需要 label critic 判断哪个最好” | Label Critic 必须前移到 teacher candidate selection 阶段 | 多候选 case-organ 先交给 Label Critic 或显式 fallback，再进入 ShapeKit |
| “所有输出都过一遍 ShapeKit” | 正式流程中 selected mask 必须统一后处理 | ShapeKit 默认开启；失败/fallback 写入 metadata 和 review queue |
| “组装成一个数据集” | 组装的是可审计 pseudo-label dataset，不是专家精标集 | manifest 必须保留 source、candidate、selection、ShapeKit、review、quality 字段 |
| “50 个 CT 和 377 类” | 老师强调目标不应被 VISTA3D 127 类限制；当前合作方策略下实际 exact target 是 373 | 汇报可解释为“面向多 teacher 大类别空间，当前可精确落地 373 organs” |
| “去训 student 模型” | student 学 selected + ShapeKit processed pseudo labels，而不是单个 teacher | 使用 3D prompt-based / VoxTell-style student；不走 VISTA3D 127 类主线 |
| “训完继续去打标签” | student 训练后重新 inference，成为下一轮候选来源 | student outputs 写成 `<case>/<organ>.nii.gz`，并注入 Round2 candidate pool |
| “这个时候不一定需要 label creators” | 如果 student 质量足够，后续可减少 teacher/label creator 依赖；但不能盲目信任 student | Round2 让 student 与 Round1 best pseudo label 竞争，保留更可靠输出 |
| “如果 student 学崩，还得用第一轮输出” | Round1 best pseudo label 是安全回退 | student 不自动覆盖 Round1；低质量/空 mask/冲突输出进入 review |
| “需要预测 student 输出和第一轮所谓 ground truth 之间的 DICE” | 这里的 Dice 是 pseudo-label consistency，不是真实 accuracy | 计算 student vs Round1 selected pseudo label Dice，并明确指标含义 |
| “找到 student 怎么学都学不会的例子” | 需要 hard-case mining 和人工复核入口 | 输出 low Dice、empty mask、shape mismatch、volume outlier、Label Critic uncertain、ShapeKit fallback 的 case-organ list |
| “student 用老师发的论文 student” | 不从零搭不稳定 student，而是参考/迁移成熟 3D prompt student | 当前工程主线对应 VoxTell-style 3D prompt student；继续熟悉论文/源码并替换不稳定临时代码 |
| “Label Critic 的位置先调一下” | 先修流程关键位置，再追求大规模训练 | 当前 sprint 优先验证真实 teacher candidates -> Label Critic -> ShapeKit 的小闭环 |

## 1. 当前必须遵守的前提

- 当前 exact prompt target 是 `373 organs`。
- xlsx / 全局 label space 历史统计为 384 个 organ；其中 381 个有 enabled 路由；8 个 SAROS 粗粒度/非一一对应 organ 按合作方策略跳过；3 个零候选 organ 当前也跳过，因此当前可精确 merge 的目标为 373。
- 当前已经有账号，但还没有拿到 JHU 组内精标集访问/数据；因此现在不能声称 true accuracy，只能评估 pseudo-label consistency。
- Teacher models 在 E-step 阶段只做 inference / test，不训练 teacher，不修改 teacher checkpoint。
- Teacher 输出只能称为 pseudo-label candidate，不能称为 true ground truth。
- Label Critic 必须前移到 teacher candidate selection 阶段，而不是等 M-step 后或 Dice 低了才补救。
- 所有 selected outputs 在正式流程中都要经过 ShapeKit；只有 smoke/debug 才能临时关闭，并必须在 metadata 中明确记录。
- Student 必须保持 3D prompt-based / language-prompt-related 架构，不再走 3D CT 转 2D slice 的路线。
- VISTA3D / VSTA3D 只能作为 teacher pool 中的一个模型或参考组件，不能作为整个系统的核心标签空间。
- 当前 student 主线应基于老师推荐论文中已经调试较成熟的 3D prompt-based student 思路；本项目当前工程对应 VoxTell-style 3D prompt student 路线。
- 项目资料层面的 21 个模型必须和 Google Drive 对齐；训练/伪标签构建阶段实际启用的 teacher pool 可以先从可运行的 13 个或其子集开始，并且设计上要支持未来扩展到更多 teacher。

## 2. 老师口述流程的工程化版本

### Step 1：准备 CT 和固定 teacher pool

输入是多个 3D CT case，例如第一阶段优先选择 50 个有 tumor annotation 的 PanTS/PAINTS CT。tumor annotation 在这条主线里作为 case 上下文、质量分析和后续 report/VQA 旁线信息，不把 tumor mask 混成 373 organ target。teacher models 是已经训练好的固定模型，只负责 test / inference。

工程要求：

- 不训练 teacher。
- 不改 teacher checkpoint。
- 每个 teacher 保持独立 CLI / wrapper。
- 每次 inference 记录模型版本、checkpoint、命令、输入 CT、输出目录。
- 优先保留 tumor mask / tumor annotation 元数据，便于后续解释 organ mask 质量、生成 patient trace 和做展示。

验收标准：

- 对任意一个 case，能查到是哪些 teacher 被调用。
- 对任意一个 teacher，能查到使用的是哪个 checkpoint 和哪条命令。

### Step 2：teacher inference 生成候选输出

每个 teacher 对 CT 做 inference，输出它支持的 organ masks。不同 teacher 支持的 organ 不同，所以要把输出整理成统一的 case-organ-candidate 结构。

工程要求：

- 对每个 case、每个 organ，收集所有可用候选。
- 单个 teacher 能输出的 organ，标记为 `single_teacher_default`。
- 多个 teacher 都能输出的 organ，进入 Label Critic。
- 没有候选或 label 不能精确一一对应的 organ，直接 skip 并记录原因。

验收标准：

- 能回答“这个 case 的这个 organ 有几个候选，分别来自哪些模型”。
- skip 不是静默发生，必须写入 report / metadata。

### Step 3：Label Critic 前移并选择最佳候选

老师强调 Label Critic 的位置要提前。正确位置是在 teacher 输出候选之后、ShapeKit 之前。

工程要求：

- 多候选 organ 立刻调用 Label Critic。
- Label Critic 比较同一 case-organ 下的候选 masks。
- 选择当前最可信的 mask 作为 selected pseudo label。
- 如果 Label Critic 返回 uncertain、失败或暂时只能用 stub，必须 fallback，并把 fallback reason 写入 durable metadata。

验收标准：

- 每个 selected mask 都能追溯到完整 candidate list。
- 每个多候选选择都有 `labelcritic_records` 或 `fallback_reason`。
- 能回答“为什么这个 organ 选了这个模型”。

### Step 4：所有 selected outputs 经过 ShapeKit

老师原话要求所有输出过 ShapeKit。工程上应把 ShapeKit 放在 Label Critic 之后，作为统一后处理。

工程要求：

- 输入是 selected pseudo-label case layout。
- 正式 run 默认开启 ShapeKit。
- ShapeKit success / fallback / failed / missing 都必须写入 manifest。
- ShapeKit 失败时可以保留 pre-ShapeKit mask，但必须标记 review risk。

验收标准：

- 每个 manifest item 都有 `shapekit_status`。
- ShapeKit fallback 会进入 review queue 或 hard-case mining。
- smoke/debug 关闭 ShapeKit 时要写成 `skipped_debug_only`，不能伪装成正式结果。

### Step 5：组装 pseudo-label dataset

ShapeKit 后得到的 selected masks 要组装成 student 训练数据集。这个数据集不是 expert ground truth，而是可审计的 pseudo-label dataset。

当前目标：

- 小规模验证：2-5 个 CT，10-20 个代表性 organs。
- 中规模验证：更多真实 teacher、更多器官、更多 CT。
- formal 目标：50 个优先带 tumor annotation 的 CT × 373 organs。

每条 manifest 至少包含：

- `case_id`
- `ct_path`
- `organ`
- `prompt`
- `final_mask`
- `selected_model`
- `candidate_models`
- `selection_method`
- `selection_status`
- `fallback_reason`
- `labelcritic_records`
- `shapekit_status`
- `dataset_role = pseudo_label`
- `ground_truth_status = pseudo_label_candidate`
- `review_flags`
- `quality_flags`
- `tumor_context_available` 或对应 tumor annotation 元数据状态，如果当前 case 有 tumor annotation。

验收标准：

- 不只是有 mask 文件，还能追溯 mask 来源、选择逻辑和质量风险。
- 文档和日志中不把 pseudo-label dataset 写成 expert ground truth dataset。

### Step 6：训练 3D prompt-based student

老师明确说不要走 2D，student 应该是 3D prompt-based / language-prompt-related。prompt 发生在 3D student 内部，而不是通过 2D 化实现。

工程要求：

- 使用老师推荐论文中成熟 student 的思想和代码结构作为主要参考。
- 当前工程主线使用 VoxTell-style 3D prompt student。
- 输入保持 3D CT。
- 训练目标为 373 exact organs。
- student 学的是 selected + ShapeKit processed pseudo labels，而不是单个 teacher。
- 训练日志必须记录 target organ count、manifest hash、checkpoint、训练配置。

验收标准：

- 训练入口不是 VISTA3D 127 类 student。
- 可以用 organ prompt 输出 `<case_id>/<organ>.nii.gz`。
- 训练报告明确写 pseudo-label supervision，不写 expert-label supervision。

### Step 7：student 重新 inference

student 训练完成后，对 CT 重新预测，产生新的 organ masks。此时 student 只是新的候选来源，不能默认比 Round1 teacher-based pseudo label 更好。

工程要求：

- student 输出目录保持 case-organ layout。
- student 输出 source 标记为 `student_prev` 或对应 round。
- 输出要能注入下一轮 candidate pool。

验收标准：

- Round2 candidate pool 里能同时看到 Round1 best pseudo label 和 student prediction。
- student prediction 不会自动覆盖 Round1 best pseudo label。

### Step 8：Round2 中比较 student 与 Round1 best output

老师强调如果 student 学崩了，还要保留第一轮输出。因此 Round2 要把 student 输出和 Round1 best pseudo label 竞争，而不是盲目相信 student。

工程要求：

- 计算 student output 与 Round1 best pseudo label 的 Dice、体积比、空 mask、shape mismatch。
- 如果 student 和 Round1 明显冲突，调用 Label Critic 再判断。
- 如果 student 输出不稳定、空 mask 或明显异常，保留 Round1 best pseudo label。
- 选择结果同样进入 manifest 和 review queue。

验收标准：

- Round2 manifest 能记录 selected source 是 teacher/Round1 还是 student。
- 低质量 student 输出会被 fallback / review 捕捉。
- 不会因为 EM 迭代把错误 pseudo label 继续强化。

### Step 9：找 student 学不会的例子

老师特别强调要找到 student 怎么学都学不会的例子。这一步不是为了证明真实精度，而是为了定位困难 case-organ。

工程要求：

- 用 student output 与 Round1 selected pseudo label 计算 consistency Dice。
- 输出低 Dice、空 mask、体积异常、shape mismatch 的 case-organ。
- 默认把 `student_vs_selected_pseudo_label Dice < 0.5` 作为强 review 信号；0.5-0.8 作为 warning 区间，具体阈值可在小闭环后按 organ size 调整。
- 按 organ 汇总长期低 Dice 类别。
- 每轮汇总低质量 case-organ 数量，观察是否随 student/Label Critic/ShapeKit 迭代减少；这只是 pseudo-label consistency 指标，不是真实 accuracy。
- 合并 Label Critic uncertain、ShapeKit fallback、student failure 到 manual review queue。

输出形式：

```text
case_id, organ, selected_model, candidate_models, student_dice,
volume_ratio, shapekit_status, labelcritic_status, problem_type, review_action
```

验收标准：

- 输出 `student_failure_cases.csv/json`。
- 能回答哪些 organ / case 是 student 长期学不会的。
- 人工复核集中在困难样本，而不是盲目全量精标 373 organs。

### Step 10：人工复核与下一轮迭代

人工处理的重点不是从零标注所有类别，而是优先处理 pipeline 发现的高风险样本。

优先复核：

- student 长期低 Dice 的 case-organ。
- Label Critic uncertain 的多候选。
- ShapeKit fallback / failed 的 mask。
- teacher 分歧特别大的类别。
- 小器官、细结构、边界模糊结构。
- student 与 Round1 best pseudo label 冲突大的样本。

验收标准：

- manual review queue 有明确优先级。
- 人工修正结果能回写到下一轮 pseudo-label dataset。
- 后续拿到 JHU 精标集后，可以把人工修正和专家标签区分开。

### Step 11：展示、reasoning trace 与 report/VQA 旁线

这一步不是替代 373-organ student 主线，而是把老师会议里提到的研究价值和展示需求保留下来，避免流程只剩工程跑分。

工程要求：

- 从 E-step / ShapeKit / failure mining 中抽取少量典型 case，在 ITK-SNAP 中做人工 sanity check，保存截图或检查记录。
- 对每个 case 保留 patient trace：CT、tumor context、候选 teacher、Label Critic 选择、ShapeKit 状态、failure/review 原因。
- RadThinking-style reasoning trace / VQA dataset 作为旁线输出：先生成结构化 trace 样例，不影响 373 organ mask 训练主线。
- report supervision / `r_super_pseudo_masks.py` 相关能力可以作为 tumor/report-supervised 扩展模块，但不能阻塞当前 3D prompt student 蒸馏闭环。

验收标准：

- 至少有若干典型成功/失败案例可在 ITK-SNAP 复查和展示。
- patient trace 能解释“这个 organ 为什么选这个 mask、是否过 ShapeKit、是否需要人工 review”。
- reasoning trace / VQA 旁线明确标注为辅助研究资产，不误写成 expert label。

### Step 12：源码排查与质量闸门

老师明确说他不看源码，但也提醒现在流程很 high-level，中间任何一环都可能出问题。因此我们需要把源码排查作为正式工程阶段，而不是临时 debug。

必须检查的源码问题：

- Teacher inference 是否真的只调用 fixed checkpoints，没有在 E-step 里训练 teacher。
- 21 个模型的 wrapper 是否各自独立，且没有把 TotalSegmentator academic-license 任务混入普通私有 nnUNet 路线。
- CT 与 mask 是否保持同一坐标系、spacing、orientation、shape；发现 mismatch 必须进入 quality flags。
- 模型 local label、dataset.json label、global organ name、xlsx organ name 是否一一对齐。
- `model_label_aliases` 是否只做真实 alias，不做粗标签近似。
- 缺失或不能一一对应的 organ 是否显式 skip，并写入 metadata/report。
- 多 teacher candidates 是否都被保存，而不是只保留最后一个输出。
- Label Critic 输入是否公平：同一个 CT、同一个 organ、同一空间下的候选 masks。
- Label Critic uncertain / failed 时 fallback 是否可追溯，不能静默选模型。
- ShapeKit 是否对正式 selected outputs 执行；debug/smoke skipped 是否明确记录。
- Manifest 是否保留 source、candidate、selection、ShapeKit、review、quality metadata。
- Student training manifest 是否使用 373 target，而不是 VISTA3D 127 labels。
- Dice 计算是否类别对齐；Dice against pseudo label 不能写成 true accuracy。
- Round2 是否把 student 当作 candidate，而不是自动覆盖 Round1 best pseudo label。
- hard-case mining 是否合并 low Dice、empty mask、shape mismatch、volume outlier、Label Critic uncertain、ShapeKit fallback。

排查产物：

- 每次源码排查更新 `docs/TEACHER_MEETING_IMPLEMENTATION_AUDIT_2026_06_05.md`。
- 发现的问题必须记录：模块、风险、修复方式、验证命令、残余风险。
- 每次较大改动后至少运行 routing audit、teacher readiness audit、py_compile，以及一个 tiny/small pipeline check。

验收标准：

- 任意一个 manifest item 都能追溯到原始 CT、teacher candidate、selected model、选择原因、ShapeKit 状态和 review flags。
- 任意一个 skipped organ 都能解释为什么跳过。
- 任意一个 Dice 数字都能说明它是 pseudo-label consistency 还是 expert-label accuracy。
- 当前没有 JHU 精标集时，不允许出现“真实准确率已验证”的结论。

## 3. 当前项目状态对齐

### 已经基本落地

- 默认 student 方向已经切到 VoxTell-style 3D prompt student。
- `configs/student_3d_prompt_target_organs.json` 已记录当前 373 exact prompt target。
- 8 个 SAROS 粗粒度 / 非一一对应 organ 已按策略跳过，不再强行近似。
- 3 个无 enabled route organ 也记录为 no-enabled-route，不参与当前 exact target。
- VISTA3D student 旧路线已从默认路径移除或加 legacy guard，避免默认回到 127 类。
- E-step 已开始支持 source-aware candidate selection、Label Critic metadata、ShapeKit status、review flags。
- VoxTell manifest / dry-run training / tiny real training / tiny inference 路线已有初步验证。
- failure mining 脚本已能比较 student prediction 与 selected pseudo label，并输出 hard-case report。
- 21-model / 373-routing 静态审计已有脚本和初步结果。

### 仍未完全完成

- 还没有跑完整 formal 50 CT × 373 organs 的 teacher inference + Label Critic + ShapeKit + manifest 生成。
- 21 个 Google Drive 模型虽然已做静态/部分 live audit，但仍需继续逐个真实 runnable 验证。
- 全部 21 个模型并不是都已经在真实 CT 上完成 end-to-end 验证。
- Real Label Critic 已通过小样本链路，但还没有在大规模 multi-teacher candidate 上稳定验证。
- ShapeKit 在正式全量 373 organs 上的覆盖、失败率、fallback 情况还没有统计。
- VoxTell / CVPR student 只做过 ministep 或 dry-run 级验证，还没有完成有意义的长训练。
- Round2 中 student output 与 Round1 best pseudo label 的大规模竞争选择还没有完成。
- 当前已经有账号，但还没有 JHU 组内精标集访问/数据，因此没有 true expert-label accuracy 评估。

## 4. 分阶段修复计划

### Phase 1：模型资料和 Drive 对齐

目标：确保 teacher pool 的基础事实可靠。

任务：

- 对照 Google Drive 重新核验 21 个模型目录、checkpoint、dataset.json、plans.json、run_*.sh。
- 检查本地模型是否和 Drive 一致；缺失、旧版本、路径不一致的模型要列出并修复。
- 每个模型保持单独 CLI / wrapper，不混用入口。
- nnUNet 系列优先沿用 Drive 提供的 run_CADS / run_MOOSE / run_type1 / run_type2 参数。
- TotalSegmentator 系列必须按官方 TotalSegmentator 方式使用，不混进普通 nnUNet license 路线。
- VISTA3D 只作为 teacher/reference wrapper，使用方式参考 VISTA3D-Inference-Pipeline，但不限制 student target space。
- ePAI registry dataset_id 保持 1339，不回退到 1017。
- DAPS 使用 Drive 中的 `checkpoint_best.pth`，不是 final checkpoint。
- 输出一份 `21-model alignment audit`，记录每个模型：Drive 状态、本地状态、是否一致、CLI 状态、真实 runnable 状态。

验收标准：

- 21 个模型每个都有明确 status：ready / missing / mismatch / disabled-by-policy / needs-manual-fix。
- 所有 ready 模型都能生成 deterministic output layout 或至少通过 command dry-run。
- 所有 disabled / skipped 都有明确原因，不允许静默缺失。

当前下一步：

- 继续扩大真实 teacher subset，优先把 CADS552-559、ATM、AirRC、LVP、UNEST、VISTA3D、TotalSegmentator 等逐个跑小样本验证。
- 对每个 teacher 记录：能否 real inference、输出 mask 数量、是否能被 alias/merge 找到、ShapeKit 状态。

### Phase 2：373 organ routing 与 alias 稳定

目标：系统明确知道每个 case 应识别哪些 organ，以及每个 organ 从哪些模型得到候选。

任务：

- 以 xlsx 为准生成 organ-to-model 路由。
- 以 `configs/student_3d_prompt_target_organs.json` 为当前 student target source of truth。
- 确认 373 organs 全部存在 prompt、routing、merge 目标。
- 确认 8 个 SAROS unresolvable coarse labels 和 3 个 no-enabled-route organs 被显式 skip。
- 完善 `model_label_aliases`，确保模型原始 label 与全局 organ name 一一映射。
- 对没有一一对应的 label 不强行 merge，直接 skip 并记录。

验收标准：

- 代码中默认 target count 为 373。
- 每个 target organ 都能查到候选 teacher 或明确来源。
- 每个 skipped organ 都在 report 中可见。
- merge 阶段不会把粗标签误当精细 organ mask。

### Phase 3：E-step 真实候选生成与早期 Label Critic

目标：把老师要求的“多个模型输出后用 Label Critic 选最好”真正跑起来。

任务：

- 对真实 CT subset 跑多个 enabled teachers。
- subset 优先来自有 tumor annotation 的 PanTS/PAINTS case list，便于后续把 organ mask、tumor context 和 patient trace 串起来。
- 对每个 case-organ 收集所有 candidates。
- 单候选 organ 标记为 `single_teacher_default`。
- 多候选 organ 调用 Label Critic。
- Label Critic 结果写入 durable JSONL 和 manifest metadata。
- Label Critic uncertain / failed 时启用显式 fallback，并写入 review queue。
- 对参考标签存在的 smoke subset，可以继续用 Dice 做 wiring sanity check；但必须写明这是 pseudo/reference consistency，不是真实 expert accuracy。

验收标准：

- 每个 selected mask 可追溯到 candidate list。
- 能回答“这个 organ 为什么选了这个模型”。
- 多候选选择不是隐藏规则，而是 Label Critic / fallback metadata 可审计。

### Phase 4：ShapeKit 全量后处理

目标：满足老师要求的“所有输出都过 ShapeKit”。

任务：

- 将 selected pseudo-label case layout 作为 ShapeKit 输入。
- 正式 run 默认启用 ShapeKit。
- ShapeKit 输出缺失、失败、fallback 都记录到 manifest 和 review queue。
- 统计每个 organ 的 ShapeKit success / fallback / missing。

验收标准：

- Formal E-step 中 ShapeKit 默认开启。
- 每个 manifest item 都有 `shapekit_status`。
- 失败不静默吞掉，必须能进入 hard-case / review 列表。

### Phase 5：组装 pseudo-label dataset

目标：生成 student 可消费、可审计的训练数据。

任务：

- 从 selected + ShapeKit processed masks 生成 training manifest。
- 保留 source / candidate / Label Critic / ShapeKit / quality metadata。
- 明确 `dataset_role = pseudo_label`。
- 明确 `ground_truth_status = pseudo_label_candidate`。
- 逐步从 tiny subset 扩展到 formal 50 个优先带 tumor annotation 的 CT × 373 organs。

验收标准：

- 不只是有 mask 文件，还能追溯 mask 的来源、选择逻辑和质量风险。
- 不把 pseudo-label dataset 写成 expert ground truth dataset。

### Phase 6：3D prompt-based student 训练

目标：用老师认可方向训练统一 student。

任务：

- 继续以 VoxTell-style / 老师论文中成熟的 3D prompt student 为主线。
- 输入保持 3D CT。
- prompt 在 3D student 内部使用，不通过 2D 化实现。
- 训练目标为 373 exact organs。
- 先做可复现实验：tiny subset、small subset、再扩展到 formal training。
- 记录训练配置、checkpoint、manifest hash、target organ count。

验收标准：

- 训练入口不是 VISTA3D 127 类 student。
- 可以用 organ prompt 生成 `<case_id>/<organ>.nii.gz`。
- 训练日志明确写明 pseudo-label supervision，而不是真实精标监督。

### Phase 7：Round2 student 与 Round1 best pseudo label 竞争

目标：不盲目信任 student，而是把 student 当作新的 candidate。

任务：

- 用训练后的 student 对 CT 重新 inference。
- 将 student output 注入下一轮 candidate pool，来源记为 `student_prev`。
- 对 student output 与 Round1 best pseudo label 冲突的 case-organ 调用 Label Critic。
- 如果 student 学崩或明显不稳定，保留 Round1 best pseudo label。

验收标准：

- Manifest 能记录 selected source 是 teacher 还是 `student_prev`。
- Round2 不自动覆盖 Round1。
- 低质量 student 输出能被 fallback / review 捕捉。

### Phase 8：student failure mining

目标：实现老师强调的 hard-case mining。

任务：

- 计算 student output 与 Round1 best pseudo label 的 Dice。
- 记录 empty mask、shape mismatch、volume ratio outlier、low Dice。
- 默认阈值：Dice < 0.5 进入强 review；0.5-0.8 进入 warning/recheck；空 mask、shape mismatch、极端体积比直接进入 review。
- 按 case-organ 输出 hard case list。
- 按 organ 汇总长期低 Dice 类别。
- 每轮汇报低质量 case-organ 总数和按 organ 分布，目标是随着候选选择、ShapeKit、student 训练迭代逐步减少。
- 将 Label Critic uncertain、ShapeKit fallback、student failure 合并为 manual review queue。

验收标准：

- 输出 `student_failure_cases.csv/json`。
- 能回答哪些 organ / case 是 student 怎么学都学不会的。
- 人工复核集中在困难样本，而不是盲目全量标注。

### Phase 9：JHU 精标集接入与真实评估

目标：在拿到精标集后补齐真实 accuracy 评估。

任务：

- 当前账号已具备，继续跟进 JHU 组内精标集访问权限、数据路径、使用协议和评估 split。
- 拿到精标集后，建立 expert-label evaluation set。
- 分别评估 teacher candidates、selected pseudo labels、student outputs。
- 区分 pseudo-label consistency 和 true expert-label accuracy。

验收标准：

- 在没有精标集前不声称真实精度。
- 拿到精标后，报告中明确区分 Dice against pseudo label 与 Dice against expert label。

## 5. 代码与配置落点

关键配置：

- `configs/student_3d_prompt_target_organs.json`：当前 373 target source of truth。
- `configs/organ_routing_from_xlsx.json`：xlsx organ-model 路由。
- `configs/model_label_aliases.json`：模型输出 label 到全局 organ 的 alias。
- `configs/model_registry.yaml`：21 个模型 / official TotalSegmentator / wrapper 参数。

关键核心代码：

- `agent-harness/cli_anything/medai/core/organ_router.py`：organ routing 默认目标。
- `agent-harness/cli_anything/medai/core/multimodel_loop.py`：teacher candidate collection、Label Critic、ShapeKit、metadata。
- `agent-harness/cli_anything/medai/core/mstep_runner.py`：generic pseudo-label manifest。
- `agent-harness/cli_anything/medai/core/voxtell_student.py`：3D prompt student manifest / training handoff。
- `agent-harness/cli_anything/medai/core/label_merger.py`：merge / alias / skip 逻辑。

关键脚本：

- `scripts/audit_21_model_drive_alignment.py`：21-model Drive alignment。
- `scripts/audit_373_organ_routing.py`：373 target routing audit。
- `scripts/audit_teacher_readiness.py`：teacher command dry-run readiness。
- `scripts/run_real_teacher_subset_check.py`：真实 CT + 真实 teacher subset 验证。
- `scripts/train_voxtell_prompt_student.py`：VoxTell-style student training entry。
- `scripts/run_student_infer_then_round2.py`：student inference 并注入 Round2。
- `scripts/mine_student_failure_cases.py`：hard-case mining。

## 6. 当前建议的立即执行顺序

### 6.1 当前 sprint：先把老师要求的小闭环跑真实

目标不是一步冲到 50 CT × 373 organs，而是先完成一个真实可解释的小闭环：

```text
2-5 个真实 CT
  -> 5-10 个代表性 organs
  -> 2-4 个真实 teacher + 必要 mock comparator
  -> early Label Critic
  -> ShapeKit
  -> pseudo-label manifest
  -> VoxTell-style student real/smoke training
  -> student re-inference
  -> Round2 competition
  -> failure mining
  -> ITK-SNAP / patient trace 样例
```

当前优先任务：

1. 继续逐个扩大真实 teacher family 验证，优先把尚未 real subset 跑通的 21 模型补上。
2. 每新增一个 teacher，就选它在 373 target 中实际覆盖的代表性 organs，跑 1-case small E-step。
3. 对每次 E-step 检查 candidate list、selected_model、Label Critic metadata、ShapeKit status、review queue。
4. 修复发现的 wrapper、alias、merge、ShapeKit metadata 问题。
5. 扩展到 2-5 个优先带 tumor annotation 的 CT、10-20 个 organs、多 teacher candidate 的 small pseudo-label dataset。
6. 跑 VoxTell-style student small real training，而不只做 dry-run；如果真实训练命令不可用，先明确 `MEDAI_VOXTELL_TRAIN_CMD` blocker。
7. 用 small student 做 re-inference，并跑 Round2 student-vs-Round1 selection。
8. 跑 failure mining，输出 hard-case list。
9. 抽取典型成功/失败样例做 ITK-SNAP sanity check，并生成 patient trace / reasoning trace 样例。
10. 等 JHU 精标集到位后，补 true expert-label accuracy evaluation。
11. 最后再扩大到 formal 50 CT × 373 organs。

### 6.1.1 当前应该从哪里继续

结合当前审计状态，下一步不应该马上宣称 formal pipeline 完成，也不应该直接冲 50 CT × 373 organs。更稳的继续顺序是：

1. 先完成 `Label Critic` 真实决策质量排查：当前真实 VLM 链路可跑通，但多次返回 `uncertain`，需要判断这是模型确实不确定，还是 prompt/parser/dual-confirmation 设置导致无法稳定选出 Overlay 1/2。
2. 在不降低正式安全性的前提下，给 Label Critic 增加可配置的 prompt/parser ablation 或 non-dual confirmation 检查；如果仍 uncertain，就保持 conservative fallback + review queue。
3. 用 1 个 case、1 个 organ 的真实 A/B pair 做最小验证，再扩展到当前已跑通的 2-case、10-organ multi-teacher replay。
4. Label Critic 质量明确后，继续逐个扩大真实 teacher runnable 验证，优先补齐尚未真实跑过的小模型族和 TotalSegmentator 官方路径。
5. 当真实 teacher subset、Label Critic、ShapeKit、manifest 都稳定后，再跑 small pseudo-label dataset -> VoxTell student real ministep/longer training -> student inference -> Round2 competition -> failure mining。

这一顺序符合老师的重点：先把 teacher 输出后的候选选择位置和质量控制调对，再训练 student；否则 student 会学到未经筛选或不可追溯的伪标签，后面越迭代越难解释。

### 6.2 每次修改后的最低回归

每轮实际修改代码后，至少运行：

```bash
python scripts/audit_373_organ_routing.py
python scripts/audit_21_model_drive_alignment.py
python scripts/audit_teacher_readiness.py --models airrc,atm,cads551,cads552,cads553,cads554,cads555,cads556,cads557,cads558,cads559,daps,epai_20250421,lvp,moose666,moose888,nnunet_private,saros_nnunet,totalsegmentator,unest,vista3d,vsmtrans
python -m py_compile scripts/run_em_training.py scripts/run_real_teacher_subset_check.py scripts/run_student_infer_then_round2.py scripts/mine_student_failure_cases.py agent-harness/cli_anything/medai/medai_cli.py
```

如果改动涉及真实 wrapper 或 ShapeKit / LabelCritic / merge，还需要跑一个 1-case small E-step，并检查：

- `selection_metadata.json`
- `training_manifest.json`
- `vlm_decisions.jsonl`
- `review_queue.jsonl`
- `shapekit_status`
- candidate mask 是否存在且不是错误 alias。

### 6.3 与老师汇报前必须准备的证据

- 一张明确流程图，显示 teacher inference、candidate collection、Label Critic、ShapeKit、pseudo-label dataset、student training、Round2 competition、failure mining。
- 一张当前完成度表：21-model alignment、373 routing、real teacher runnable、Label Critic、ShapeKit、student、failure mining、JHU fine labels。
- 一个真实 case 的 patient trace，能解释某个 organ 为什么选这个 teacher。
- 一个 hard-case mining 表，哪怕先是 small subset，也要展示如何找 student 学不会的例子。
- 一段明确免责声明：当前没有 JHU 精标集，所以只报告 pseudo-label consistency，不报告 true accuracy。

## 7. 下次向老师汇报的建议结构

第一部分：说明我们已按老师意见改方向。

- 不再走 2D。
- 不再以 VISTA3D 127 类为核心。
- 当前 exact target 是 373 organs。
- teacher 只做 fixed inference。
- teacher 输出是 pseudo-label candidates。
- Label Critic 已前移到 candidate selection。
- 所有 selected outputs 正式流程中都过 ShapeKit。
- student 是 3D prompt-based。
- 暂无 JHU 精标集，所以当前只汇报 pseudo-label consistency。
- 账号已具备，下一步 blocker 是精标集访问/数据本身，不是账号申请。

第二部分：讲清楚新 pipeline。

```text
50 CT（优先有 tumor annotation）
  -> 多 teacher inference
  -> case-organ candidate collection
  -> single teacher default / multi teacher Label Critic
  -> ShapeKit
  -> pseudo-label dataset
  -> 3D prompt student training
  -> student re-inference
  -> student vs Round1 best comparison
  -> hard-case mining
  -> manual review
```

第三部分：说明当前已做和还缺。

- 已完成：373 target、routing audit、部分真实 teacher subset、Label Critic metadata、ShapeKit metadata、VoxTell dry-run/ministep、failure mining 初版。
- 正在补：21 个模型逐个真实 runnable、更多 teacher family、small pseudo-label dataset。
- 还缺：formal 50 CT × 373 organs、real Label Critic 大规模验证、student 长训练、Round2 大规模竞争、ITK-SNAP 展示复核、reasoning trace/VQA 旁线样例、JHU 精标集真实评估。

## 8. Definition of Done

本轮老师会议要求只有在以下全部满足时才算真正完成：

- 21 个模型资料已和 Drive / 官方使用方式对齐，状态逐一可审计。
- 373 exact organs 是当前默认 target，不再使用 VISTA3D 127 类作为系统上限。
- Teacher E-step 只做 fixed inference，不训练 teacher。
- 每个 case-organ 的 candidate masks 都被收集并记录。
- 多候选 organ 使用早期 Label Critic 选择最佳候选。
- 单候选 organ 被标记为 pseudo-label candidate，而不是 ground truth。
- 所有 selected outputs 正式流程中经过 ShapeKit 或记录 fallback。
- Pseudo-label dataset manifest 保留 source、candidate、Label Critic、ShapeKit、quality metadata。
- Student 是 3D prompt-based，不转 2D。
- Round2 能比较 student output 与 Round1 best pseudo label，并能保留更可靠输出。
- Student failure mining 能生成 hard-case / manual-review 列表。
- Failure mining 能按 Dice < 0.5、0.5-0.8 warning、空 mask、shape mismatch、体积异常等规则统计低质量 case-organ，并汇报低质量数量是否随迭代减少。
- Formal case selection 优先覆盖有 tumor annotation 的 CT；tumor annotation 作为上下文和旁线监督信息，不改变 373 organ target。
- 至少有一组 ITK-SNAP sanity-check 展示样例和 patient trace / reasoning trace 样例。
- 没有 JHU 精标集前，所有报告只说 pseudo-label consistency，不说 true accuracy。
- 拿到精标集后，能补充 true expert-label evaluation。
