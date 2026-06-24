# medical_agent 多模型封装 + 器官路由 实现方案（架构设计）

面向 384 器官、21 模型的 CT 分割融合系统。本文档供 code-writer 逐步落地。路径相对 `medical_agent/` 项目根。

## 0. 落地前必须知道的事实
1. **权重目录双层嵌套**：`checkpoints/CADS_series/CADS_series/Dataset55X_*`、`checkpoints/MOOSE_series/MOOSE_series/...`、`checkpoints/nnUNet_private/nnUNet_private/...`、`checkpoints/VSmTrans/VSmTrans/nnUNet_results/...`。`model_registry.yaml` 已用双层，但 `model_registry.py:_default_model_entry` 生成单层（不一致，需修）。
2. **MOOSE plans 真相**：run_MOOSE.sh 写 `nnUNetResEncUNetLPlans`，但本地 666/888 权重目录名与 plans.json 都是 **`nnUNetPlans`** → 以本地为准。
3. **ePAI dataset_id 错位**：registry=1017，本地权重在 Dataset1339_ePAI；推理走 `--model-folder` 绕开了 dataset_id，但字段应改 1339。ePAI plans=nnUNetPlans，26 标签。
4. CADS/MOOSE/private/VSmTrans 权重文件名 `checkpoint_final.pth`，fold=`fold_all`（VSmTrans=`fold_0`）。**DAPS 缺失**，下载后文件名是 `checkpoint_best.pth`（需 `-chk checkpoint_best`）。
5. **局部标签名 ≠ xlsx 全局器官名**（核心难点）：如 CADS551 用 `gallbladder`/`lung_upper_lobe_left`，xlsx 用 `gall_bladder`/`lung_upper_left_lobe`；MOOSE888 用 `portal_splenic_vein`，xlsx 用 `portal_vein_and_splenic_vein`；大小写不一。必须建「每模型局部标签名→全局器官名」别名表，不能纯字符串相等。
6. 可复用：`scripts/nnunetv2_predict_and_split.py`（输出 combined_labels.nii.gz + 拆分）、`scripts/{vista3d,unest,atlasnet}_predict_and_split.py`、`configs/checkpoint_dataset_catalog.json`。
7. 环境**无 rclone**（需装），有系统级 `nnUNetv2_predict`。
8. 以 `organ_routing_from_xlsx.json`(384) 为权威，`all_organs.json`(358) 废弃。

## 1. 总体架构（分层）
- **L0 下载/校验**：rclone(Drive root `1H11EMT83SyteAnh5DwaK5kkJpsr3UEbx`) → checkpoints/；`verify_checkpoints.py` 比对 manifest，仅补缺失/损坏（当前=DAPS）。
- **L1 单模型 CLI 封装**（一模型一入口）：`medai infer --model <key>` → registered_infer 按 `recipe` 分派（nnunet 配方 / vista3d / unest / totalseg）。每模型输出 `outputs/<case>/per_model/<model_key>/{combined_labels.nii.gz, local_labels.json, run_meta.json}`。各模型只产局部标签，不知全局空间。
- **L2 器官路由**：`organ_router.py` 读 `organ_routing_from_xlsx.json`，经 `routing_token_to_model.json` 归一化 token→model，过滤 disabled，产出 `organ→ranked[(model_key,subtask)]` + 本 case 需跑模型集。
- **L3 标签合并**：`label_merger.py`，用 `global_label_space.json`(384类固定id) + `model_label_aliases.json`(局部→全局)，按优先级合并、解决重叠冲突，重采样到输入 CT 网格。
- **L4 统一输出**：`outputs/<case>/{unified_labels.nii.gz, segmentations/<organ>.nii.gz, merge_report.json}`。
- 原则：全局对齐只在 L3；流程幂等，per_model 中间产物可缓存、单独重跑。

## 2. 单模型 CLI 封装规范
对外契约（每个 model_key 满足）：`medai infer --model <key> --image <ct> --output-folder <out>` → 产 `per_model/<key>/{combined_labels.nii.gz, local_labels.json, run_meta.json}`。

**四种 nnUNet 配方**（都复用 `nnunetv2_predict_and_split.py`，差异在 registry 字段）：
| 配方 | model_key | dataset_id | trainer | plans | folds | chk | nnunet-results |
|---|---|---|---|---|---|---|---|
| cads | cads551..559 | 551..559 | nnUNetTrainerNoMirroring | nnUNetResEncUNetLPlans | all | final | checkpoints/CADS_series/CADS_series |
| moose | moose666,moose888 | 666/888 | nnUNetTrainerNoMirroring | **nnUNetPlans** | all | final | checkpoints/MOOSE_series/MOOSE_series |
| type1 | nnunet_private(224),saros(1345),atm(1370),airrc(1380),lvp(1381),daps(1347) | 对应 | nnUNetTrainer | nnUNetResEncUNetLPlans | all | final/**best(DAPS)** | checkpoints/nnUNet_private/nnUNet_private |
| type2 | epai_20250421 | 1339 | nnUNetTrainer | nnUNetPlans | all | final | Dataset1339_ePAI(经--model-folder) |

**硬要求**：把单一 `cads` 拆成 `cads551..cads559` 九条独立条目；MOOSE 拆 `moose666`/`moose888`。否则无法按 CADS552→椎骨、CADS558→OAR 精确路由。

**三个特殊模型**：VISTA3D（`vista3d_predict_and_split.py`，MONAI bundle，everything_labels 全 117 类，local_labels=label_dict_127_abdomenAtlas3-1.json）；UNEST（`unest_predict_and_split.py`，肾亚结构 3 类）；VSmTrans（自带 nnunetv2 的 nnUNet，Dataset001_BDMAP，plans=nnUNetPlans，fold_0，`--workdir third_party/VSmTrans_lightweight/nnUNet`）。

**registry 新增字段**：`recipe`、`checkpoint_name`(默认 checkpoint_final)、`local_label_source`、`routing_aliases`(响应的路由 token 列表)；顶层加 `routing_token_to_model` 反查。

## 3. 器官路由 + 标签合并
**新增 3 个配置**：
- `configs/global_label_space.json`：384 类固定 id（1..384，0=背景），从 organ_routing keys 排序生成（organ_to_id / id_to_organ）。
- `configs/routing_token_to_model.json`：路由 token → {model, subtask, enabled, reason}。处理脏标签（Totalsegmentator 空格不一致 / MOOSE 缺 3.0 / 裸 CADS 按器官区分到 551/556/558）；ATLAS-Net 权重和真实 GPU 推理已验证，`Dataset001_ATLASNet` 启用为 22 个 373 腹部目标的辅助候选；Duke/缺权重 MOOSE 子任务/未下载 DAPS → enabled=false。归一化：去括号空格、统一大小写后匹配。
- `configs/model_label_aliases.json`：每模型「局部标签名→全局器官名」。先用归一化字符串相等自动对齐，**无法自动对齐的列入 unmapped 清单，阻断式提示人工确认，不得静默丢器官**。

**路由**（organ_router.py）：逐 token 经 (b) 解析→过滤 disabled→保留顺序得每器官有序候选；反查本 case 需跑模型集。

**合并**（label_merger.py）：
- **同器官多模型**（137 个）：遍历 ranked 候选，第一个产出非空 mask 的模型赢得该器官（first-non-empty-wins，与 xlsx `/` 靠前优先一致）。
- **跨器官像素重叠**：用全局 priority 数组，细粒度/专精器官优先（亚结构>整体）；同 rank 时体素更小者优先。**推荐：器官图与区域/composition 图分离输出**（unified_organs.nii.gz + unified_regions.nii.gz），把 body/skeletal_muscle/abdominal_cavity 等大区域分离，规避大多数跨器官冲突。
- **重采样**：统一以输入 CT 网格为参考，最近邻。nnUNet/VISTA3D 默认输出回原图；合并前断言 shape/affine 一致，不一致才 resample 并记录。
- **缺失处理**：候选为空→merge_report 记 missing；mask 全 0→记 empty；输出 coverage_summary（384 中产出 N，缺失列表 M）。

## 4. TotalSegmentator 接入推荐
路由表 121 个器官标 `Totalsegmentator (子任务)`，涉约 18 子任务。**推荐：全部走官方 pip 包 `totalsegmentator`**（按子任务分组调 `TotalSegmentator -i -o --task X`）。理由：官方覆盖全部子任务（CADS551-559 只有 9 个，补不齐 headneck/brain/oculomotor 等 100+ 器官）；license 清晰(Apache-2.0)；输出文件名即标准器官名，对齐成本最低。CADS551-559 仅用于路由表明确写 `CADS55X` 的器官（abdomen/vertebrae/OAR/SAROS），二者职责不重叠。实现：扩展 `totalseg_runner.py` 支持 `--task` + `configs/totalseg_subtask_organs.json`(子任务→全局器官子集)；需 `pip install totalsegmentator`，首跑自动下权重（注意联网/academic key）。

## 5. VISTA3D 接入
复用 `scripts/vista3d_predict_and_split.py`（按 lin-tianyu VISTA3D-Inference-Pipeline 的 MONAI bundle 方式）。权重 `checkpoints/VISTA3D-Inference-Pipeline-master/.../models/model.pt`（已在本地）。local_labels=label_dict_127_abdomenAtlas3-1.json。VISTA3D 的 kidney/lung 是整体类，与全局同名直接对齐。合并层只信任该 JSON（勿与 teacher_branch_map 的 VISTA3D_LABEL_MAP 双源）。

## 6. rclone 下载/校验
先装 rclone（非 sudo 装 ~/bin）。配只读 drive remote 指向 root id。**校验优先、按需更新**：`scripts/verify_checkpoints.py` 遍历期望清单（扩展 checkpoint_dataset_catalog.json，含期望文件+大小），仅对缺失/损坏用 `rclone copy --drive-root-folder-id <子id> --ignore-existing --checksum`。当前唯一缺失=DAPS（需先 `rclone lsd` 确认其 Drive 子文件夹 id）。manifest `configs/checkpoint_drive_manifest.json` 已含各子文件夹 folder id 可解析。产 `outputs/checkpoint_verify_report.json`。

## 7. registry 修正清单（逐条）
1. **CADS 拆 9 条** cads551..559（dataset 551..559，双层 checkpoint_path，各自 dataset_json，trainer=NoMirroring，plans=ResEncUNetLPlans，folds=all，recipe=cads）。
2. **MOOSE 拆** moose666(外周骨)/moose888(cardiac)，plans 确认=**nnUNetPlans**，trainer=NoMirroring。其余 MOOSE 子任务无权重→不造假条目（路由层 disabled）。
3. **ePAI dataset_id 1017→1339**；保留 --model-folder/--workdir third_party/ePAI-main/train/--output-label-mode all_organs。
4. **DAPS 新增** daps：recipe=type1，dataset 1347，trainer=nnUNetTrainer，plans=ResEncUNetLPlans，**checkpoint_name=checkpoint_best**，status=unavailable_until_downloaded（enabled=false 直到 verify 通过）。把旧 `dap` 模板改造为正式 daps，避免重复。
5. **type1 私有模型**对齐双层路径 checkpoints/nnUNet_private/nnUNet_private。
6. **VSmTrans** checkpoint_path=checkpoints/VSmTrans/VSmTrans/nnUNet_results，dataset 1，plans=nnUNetPlans，folds=0，--workdir。
7. **统一加** recipe/checkpoint_name/routing_aliases。
8. **忽略** atlasnet/duke/pedro/goacc 标 enabled=false/ignored。
9. **修 `model_registry.py:_default_model_entry`** 生成双层路径，与本地一致。

## 8. 分阶段落地计划
- **阶段0 配置基线（串行，无GPU）**：生成 global_label_space.json、routing_token_to_model.json；扩展 checkpoint_dataset_catalog.json（补 CADS552-559/MOOSE666/各 private/VSmTrans/UNEST/VISTA3D labels+期望文件）；`scripts/build_global_label_space.py`。
- **阶段1 registry 修正（串行，依赖0）**：改 model_registry.yaml + model_registry.py(§7)。
- **阶段2 单模型封装补强（可并行，依赖1）**：2a nnunetv2_predict_and_split.py 加 dump local_labels.json + per_model 目录；2b registered_infer.py 输出约定改 per_model/<key>/ + run_meta + 按 recipe 分派；2c totalseg_runner.py + totalseg_subtask_organs.json；2d vista3d/unest 真权重冒烟。
- **阶段3 下载/校验（与2并行）**：verify_checkpoints.py + 装 rclone。
- **阶段4 路由+合并（串行，依赖0/2）**：organ_router.py、label_merger.py、model_label_aliases.json、build_label_aliases.py、medai_cli.py 加 `segment-all`。
- **阶段5 端到端验证**：1 个真实 PanTS case 跑 segment-all，查类数/覆盖率/抽查 Dice。
- 并行：2、3 并行；0→1→(2,3)→4→5。

## 9. 风险与待确认
1. model_label_aliases 自动对齐覆盖不全→unmapped 清单阻断提示，不静默丢器官。
2. 跨器官像素重叠优先级 xlsx 未定义→默认亚结构>父结构 + 器官/区域分离输出，最终需 teacher 拍板。
3. MOOSE 缺权重→已决策 fallback；arms/legs/muscle_fat 接受丢失。
4. DAPS 在 Drive root 子文件夹 id 未知→rclone lsd 确认；若无则 DAPS 30 器官多数缺失。
5. TotalSegmentator 子任务权重首下需联网/可能需 academic key→确认服务器联网与 key。
6. VISTA3D/ePAI/VSmTrans 环境（MONAI 版本、自带 nnunetv2 冲突）→每模型独立 subprocess（registered_infer 已是）。
7. 裸 CADS token（pericardium→CADS556，submandibular→CADS558）按器官区分，勿一刀切 551。
8. VSmTrans 仅 fold_0 且自带 nnunetv2 版本需 real-run 验证。
