# VoxTell prompt-conditioned training：公开信息核验与源码团队一次性问题清单

核验日期：2026-07-02  
官方仓库：<https://github.com/MIC-DKFZ/VoxTell>  
论文：<https://arxiv.org/abs/2511.11450>  
当前官方 `main`：`ec517b79a19aa59b25789c878d808790326e9651`（2026-06-26）

## 结论先行

截至核验日期，**论文版 VoxTell 的完整 prompt-conditioned training
trainer、prompt sampler、统一数据清单和 1,087 concept-to-source-mask 映射尚未
公开**。不能把项目内 `train_voxtell_prompt_student.py` 描述成官方训练复现。

这里需要一个重要修正：**公开 prompt vocabulary 并非完全缺失**。Hugging Face
的 v1.1 模型仓库已经发布
`embeddings/voxtell_v1.1/labels.json` 和
`text_embeddings.npz`。实测可直接提取 **14,194 个 prompt 字符串**及对应
`14,194 × 2,560` 的 float16 Qwen embeddings。缺少的是这些扁平 prompt 到
canonical concept、源数据集 label/mask 和组合标签规则的结构化映射，而不是
prompt 文本本身。

但上次“官方完全没有训练代码”的结论已经过时：官方在 2026-06-19 加入、并于
2026-06-26 合并发布了 `voxtell-finetune`。这份代码只把 VoxTell 的
`encoder.*` 权重迁移到标准多类 nnU-Net，decoder 从头训练，deep supervision
关闭；它**不包含文本 encoder、prompt decoder、多尺度 vision-language fusion
或 positive/negative prompt sampling**。因此：

- 可以直接采用官方 `voxtell-finetune` 作为 encoder-transfer baseline；
- 不能用它替代本项目所需的 prompt-conditioned Student；
- 若最终 Student 必须接受自由文本并输出 prompt 对应 mask，仍需等待官方完整
  trainer，或明确标注为“按论文公开细节近似实现”。

## GitHub issue 核验

GitHub 当前 issue 总表显示 13 个历史 issue，其中 5 个 open、8 个 closed。
仓库禁止普通用户新建 issue（页面显示
`Issue creation is restricted in this repository`），所以新增问题应优先作为
一封邮件发送给 README 公布的两位联系人。

| Issue | 当前状态 | 已有内容/作者回答 | 对本项目的影响 |
|---|---|---|---|
| [#12 Dataset and Training Details/Logs release](https://github.com/MIC-DKFZ/VoxTell/issues/12) | Open，2026-05-14 | 请求 processed data、准确下载/预处理说明、training logs 和 hyperparameters；截至核验日期无作者回复 | 与我们最相关。证明完整训练数据和日志仍未公开；邮件中不重复泛问 logs，而只问论文/代码仍不能确定的字段 |
| [#7 Detailed linking of structures and datasets](https://github.com/MIC-DKFZ/VoxTell/issues/7) | Closed，2026-02-25 | 请求结构到源数据集的 spreadsheet；可见页面无作者文字回复 | 1,087 concepts 到数据集/公开标签的逐项映射仍不可得 |
| [#4 Missing details for data preprocessing](https://github.com/MIC-DKFZ/VoxTell/issues/4) | Closed，2026-01-20 | 询问固定 spacing、192³ patch 和 sliding-window inference；可见页面无作者文字回复 | 不重复泛问 preprocessing；论文只确认训练 patch 为 192³，README 明确推理不统一 resample |
| [#6 Reproducing Paper results](https://github.com/MIC-DKFZ/VoxTell/issues/6) | Closed，2026-02-17 | 用户报告大量低于论文的 Dice；可见页面无作者文字回复 | 说明仅按公开 inference API/类名 prompt 难以复现实验，不能把性能差简单归因于本地代码 |
| [#5 Reproducibility Issue (ReXGroundingCT)](https://github.com/MIC-DKFZ/VoxTell/issues/5) | Closed，2026-01-27 | ReXGroundingCT 复现问题 | 与 instance-specific prompt 评估相关；邮件只问训练/推理差异，不重复其具体 benchmark 报错 |
| [#9 Pediatric-CT-Seg / SKM-TEA preprocessing](https://github.com/MIC-DKFZ/VoxTell/issues/9) | Closed，2026-02-25 | 特定数据集 preprocessing/reproducibility | 与通用 prompt sampler 无直接答案 |
| [#10 Multi-sequence MRI input](https://github.com/MIC-DKFZ/VoxTell/issues/10) | Closed，2026-03-13 | 多序列 MRI 是合并还是分开输入 | 与 CT prompt-conditioned Student 无直接关系 |
| [#13 Poor ReXCT performance](https://github.com/MIC-DKFZ/VoxTell/issues/13) | Open，2026-06-06 | ReXCT 测试性能问题 | 再次提示官方 checkpoint 的输入、prompt 和 benchmark pipeline 必须严格对齐 |
| [#2 Timeline for code release](https://github.com/MIC-DKFZ/VoxTell/issues/2) | Open，2025-11-18 | 询问代码发布时间；无可见回复 | inference 和 encoder-transfer fine-tuning 后来已发布，但完整 prompt training 仍未发布 |
| [#1 Release artifacts](https://github.com/MIC-DKFZ/VoxTell/issues/1) | Closed，2025-12-29 | HF 团队邀请发布模型/数据；模型 checkpoint 后来已发布 | checkpoint 已可直接使用；聚合后的 4 TB 训练语料没有作为完整可训练数据发布 |

其余 #3（比较中加入 Medal S）、#8（web interface）、#11
（ReXGroundingCT data）不回答本项目的 prompt training 问题。

注意：closed 不等于“作者给出了答案”。上述 #4、#6、#7 等当前可见页面没有作者
解释，不能把关闭状态当作技术细节已公开。

## 论文已经明确回答的训练细节

以下内容来自论文正文第 3、5 节及 Supplementary A.1–A.3、C，不应再向作者
重复提问。

| 问题 | 公开答案 | 对本项目代码的直接影响 |
|---|---|---|
| Positive/negative ratio | 每张 training image 同时 query **3 个 prompt：2 positive + 1 negative** | 应实现为每个 image/step 的固定 2:1，而不是仅在全局 prompt pool 维持统计比例 |
| Positive 定义 | 对应 volume 中 present 的 structures | 正样本选择先基于 volume 可见类别；随后 patch sampler 通过 foreground oversampling 提高有目标 patch 的概率 |
| Negative 定义 | 对应 volume 中 absent structure，目标为空 mask | 可以确认不是“任意错误文本”；但论文没有说明 crop 后消失的 volume-positive 类是否也临时当 negative |
| Foreground sampling | 以 **85% 概率 oversample foreground structures**，增加包含 target 的 patch | patch sampler 应显式记录 85% foreground oversampling |
| Prompt expansion | 最终 1,087 unified concepts、9,682 rewritten labels；语义标准化、改写和层级聚合由 LLM 构建 | expansion 是预先构建的 vocabulary，不是训练时在线调用 LLM |
| 文本 embedding | frozen Qwen3-Embedding-4B；**所有 embedding 训练前预计算** | 可离线缓存 embedding；无需在每 step 运行 Qwen |
| Qwen instruction | `Instruct: Given an anatomical term query, retrieve the precise anatomical entity and location it represents. Query: [text input].` | embedding 生成必须使用同一 instruction template |
| 名称/改写采样 | default name 25%，rephrased variant 75% | 本地采样器不应把 canonical name 和 synonyms 均匀采样 |
| Deep supervision | 使用；五个 decoder scales 全部监督 | prompt Student 不能照搬官方 encoder-transfer trainer 的 `deep supervision OFF` |
| Loss | 每尺度 Dice + binary cross-entropy（BCE） | 自由文本是一 prompt 一二值 mask；不应误用标准多类 CE 作为论文等价项 |
| Scale weights | nnU-Net 默认 `[1, 1/2, 1/4, 1/8, 1/16]` | 每尺度 Dice+BCE 按该权重求和；论文未说明是否再归一化 |
| Patch / schedule | 192³ patch；2,000 epochs；250 iterations/epoch；SGD；初始 LR `1e-4`；polynomial decay | 作为论文对齐配置写入 manifest；资源不足时必须标注缩小实验 |
| Batch / compute | ablation：1×A100、batch 2；final：64×A100 40GB、global batch 128、约 6 天 | 本地小 batch 结果不可宣称完整复现 final checkpoint |
| Augmentation | 标准 nnU-Net augmentation，但禁用 left-right mirroring | laterality 类别必须关闭左右镜像 |
| Prompt-conditioned deep supervision | 每个尺度均做 vision-language fusion，再用下采样 GT 监督 | deep supervision 不是独立于 prompt loss 的第二个 loss；它是同一 prompt-conditioned segmentation objective 的多尺度加权和 |

论文公式写作 `Dice + cross-entropy`，Supplementary A.3 进一步明确为
`Dice + BCE`；实现时应采用后者。

## 官方当前源码到底能直接用什么

官方 [`voxtell/training/voxtell_trainer.py`](https://github.com/MIC-DKFZ/VoxTell/blob/main/voxtell/training/voxtell_trainer.py)
和 [`run_finetuning.py`](https://github.com/MIC-DKFZ/VoxTell/blob/main/voxtell/training/run_finetuning.py)
提供：

- 从官方 `checkpoint_final.pth` 严格加载 `encoder.*`；
- 固定 ResEnc-L 六层 encoder；
- 标准 nnU-Net 多类 decoder，从头训练；
- batch size 2、50 epoch linear warmup、之后 PolyLR；
- `VoxTellTrainer_noMirroring`；
- resume/validation 和官方 CLI `voxtell-finetune`。

它们没有提供：

- 2-positive/1-negative sampler；
- 9,682 prompt vocabulary manifest 或 1,087 concept mapping；
- frozen Qwen embedding 预计算训练数据；
- prompt decoder 的训练 dataloader；
- 五尺度 prompt-conditioned loss/training loop；
- 论文 final model 的 2,000 epoch、`1e-4` 配置；
- 训练 logs 或聚合数据预处理 pipeline。

此外，官方 encoder-transfer trainer 明确设置 `enable_deep_supervision=False`，
初始 LR 为 `1e-3`。这与论文 prompt-conditioned final training 的五尺度 deep
supervision 和 `1e-4` 不矛盾，因为二者是不同任务；但绝不能把前者标成后者的
官方训练实现。

## “完整 prompt trainer / 1,087 类映射 / prompt vocabulary”分别是什么

这三个概念不能混成一句“源码没给”：

### 1. Prompt-conditioned trainer

它不是模型结构本身，而是把训练数据送入已公开 `VoxTellModel` 的训练控制层：

```text
image volume
  + 该 volume/patch 的可用 mask 集合
  + 2 个 present-structure prompt
  + 1 个 absent-structure prompt
  + prompt 对应的 binary target masks
  -> VoxTellModel(image, three_text_embeddings)
  -> 五个尺度、每尺度三个 binary logits
  -> weighted Dice+BCE
  -> optimizer update
```

官方 `VoxTellModel` 已经完整公开了最难的网络部分：ResEnc-L encoder、文本投影、
六层 transformer prompt decoder、五阶段 mask-embedding fusion、五个 segmentation
heads，以及 deep-supervision 输出开关。官方 checkpoint 也包含这些层的全部权重：
1095 个 tensor，其中包括 448 个 encoder、508 个 decoder、110 个 transformer
decoder 参数和所有 text/image projection 参数。

真正没有独立公开的是：如何从多数据集 annotation 中安全判断 present/absent、
如何为同一 image 组织三 prompt、如何处理 partial labels、如何取 patch，以及
loss/optimizer/分布式 sampler 的 trainer 代码。论文给了大部分规则，因此**可以
合理重建**，但无法逐行证明与作者内部 trainer 相同。

### 2. 1,087 类映射

1,087 不是普通多类网络的 1,087 个固定输出 channel。VoxTell 对每条文本动态输出
一个 binary mask；这里的 1,087 是作者把 158 个数据集的原始标签经过合并、拆分、
冲突消解和专家审核后得到的 **canonical semantic concepts**。

精确映射类似：

```text
dataset X label 2 + label 3
  -> canonical concept "kidneys"
  -> ["kidneys", "bilateral kidneys", "renal organs", ...]
  -> 合并 dataset X 的 mask 2 和 mask 3 作为监督 target
```

论文 Appendix C 给出了生成流程和例子，但公开文件没有逐项列出：

```text
source dataset -> raw label IDs -> canonical concept ID
               -> allowed label combinations -> prompt variants
```

这个映射对“精确复现作者 158/190 数据集训练”很重要；对“在我们自己的
pseudo-mask 数据上继续训练 VoxTell”不是必需的。我们的每条记录只要有可靠的
`image + binary mask + canonical prompt + variants`，就可以直接训练，不需要先
复原全部 1,087 ontology。

### 3. Prompt vocabulary

论文 v1.0 使用 1,087 concepts 和 9,682 rewritten labels。官方当前推荐的 v1.1
扩展到 190 datasets、约 68,500 volumes，并公开了一个更大的扁平 embedding bank：

- `labels.json`：14,194 个小写 prompt；
- `text_embeddings.npz/labels`：同样的 14,194 个 prompt；
- `text_embeddings.npz/embeddings`：shape `(14194, 2560)`、float16；
- 官方 loader 会将 prompt key 统一小写；
- embedding 使用公开 `wrap_with_instruction()` 和 last-token pooling。

所以 vocabulary **能提取、能直接用于推理，也能作为训练 embedding cache**。
但是文件是扁平列表，没有 `concept_id` 或 synonym-group 字段。因此不能仅凭它
精确知道 `"renal organ"` 在作者训练时对应哪个源 mask、是否与 `"kidney"` 完全
同组，或某个组合 prompt 需要合并哪些原始 label。可以用 embedding 相似度聚类
做近似分组，但那会重新引入猜测，不能冒充官方 mapping。

## 逐文件源码检查结果

| 公开文件/制品 | 实际包含内容 | 能否直接用于 prompt training |
|---|---|---|
| `voxtell/model/voxtell_model.py` | 完整 prompt-conditioned network forward；支持一次输入 N prompts 和五尺度输出 | 能，是本项目应复用的核心 |
| `voxtell/model/transformer.py` | 六层 text-query/image-memory transformer decoder | 能 |
| `voxtell/utils/text_embedding.py` | 官方 instruction wrapping 与 last-token pooling | 能 |
| `voxtell/utils/embedding_bank.py`（最新 main） | 下载/读取官方 prompt bank | 能 |
| `voxtell/inference/predictor.py` | network 构造参数、Z-score、nonzero crop、192³ sliding window、Gaussian aggregation、sigmoid 0.5 | 能复用模型构造；无训练 sampler/loss |
| `voxtell/training/voxtell_trainer.py` | 只做 encoder-transfer 多类 nnU-Net | 只能作 baseline，不能训练自由文本路径 |
| `voxtell_v1.1/plans.json` | ResEnc-L 结构、batch 2、192³、ZScore、no-resampling | 可补齐构造/预处理 |
| `checkpoint_final.pth` | 只有一个顶层键 `network_weights`，含完整模型权重 | 能初始化全模型；不含 optimizer、epoch、trainer config、prompt bank 或 concept mapping |
| HF `labels.json` / `text_embeddings.npz` | 14,194 prompt 及预计算 embedding | 能直接提取和复用，但没有 synonym/concept/source-mask 分组 |
| 论文 Supplementary A/C | sampler 比例、85% foreground、25/75 文本采样、Dice+BCE、DS weights、训练 schedule、vocabulary 构建流程 | 足以形成高可信 paper-guided trainer，但仍有 partial-label/negative 边界缺口 |

遍历官方 Git 历史中的所有文件后，没有发现隐藏或改名的 prompt dataloader、
loss、training step、1,087 mapping 或数据 manifest；不是“文件名没看出来”，而是
这些逻辑确实不在已发布 Git 对象里。

## 仍需源码团队一次性确认的问题

已经删除论文明确回答的问题，只保留会改变本项目实现的缺口：

1. 是否会发布**完整 prompt-conditioned training code**（不是当前
   encoder-to-nnU-Net fine-tuning）及可复现实验的 prompt vocabulary/manifest？
2. “absent structure”是相对整张 volume 判定，还是相对当前 192³ patch 判定？
   若 structure 在 volume 中存在但不在采样 patch 中，仍作为 positive 的空 crop、
   被排除，还是重标为 negative？
3. 2:1 是否严格按每 image 固定三 prompt（论文措辞如此），分布式训练时是否还有
   跨 batch/class balancing？
4. 对 dataset 未标注的器官如何避免 false negative？只有明确已知在该 volume
   缺失的类别才能作为 negative，还是所有非该数据集标签都可采样？
5. 能否发布 1,087 unified concepts ↔ 9,682 rewritten labels ↔ source dataset
   label 的映射，或至少 checkpoint 使用的 prompt bank？
6. 五尺度 `[1,1/2,1/4,1/8,1/16]` 的 loss 最后是否按权重和归一化？空 mask
   negative 上 Dice 项和 BCE 项的具体实现/平滑常数是什么？
7. 官方 inference 是否包含未公开的 connected components、ROI restriction、
   anatomy containment 或针对 small/tubular structures 的后处理？当前公开 predictor
   看起来直接 threshold logits。
8. 从 v1.1 checkpoint 继续做 prompt-conditioned fine-tuning 时，推荐加载全模型
   （encoder + prompt decoder + image decoder），还是只加载 encoder？是否有
   optimizer/scheduler reset 的官方做法？

## 建议发送的完整邮件（英文）

收件人：`maximilian.rokuss@dkfz-heidelberg.de`,
`moritz.langenberg@dkfz-heidelberg.de`  
建议主题：`VoxTell prompt-conditioned training: consolidated reproducibility questions`

```text
Dear Dr. Rokuss and Dr. Langenberg,

Thank you for releasing VoxTell, its checkpoints, and the recent
encoder-transfer fine-tuning code. We are building a prompt-conditioned 3D
segmentation student and would like to follow the official VoxTell training
procedure rather than reimplementing it from assumptions.

We reviewed the paper and supplementary material, the current main branch
(including voxtell-finetune), and the existing GitHub issues, especially #12,
#7, #6, and #4. We therefore understand the following and do not need
clarification on these points:

- each training image uses two positive prompts and one absent-structure
  negative prompt;
- foreground patches are oversampled with 85% probability;
- the canonical name is sampled with 25% probability and a rewritten variant
  with 75%;
- Qwen3-Embedding-4B is frozen and embeddings are precomputed using the
  instruction in Supplementary A.2;
- Dice+BCE is applied with deep supervision at five decoder scales, using
  weights [1, 1/2, 1/4, 1/8, 1/16];
- final training uses 192^3 patches, 2,000 epochs x 250 iterations, SGD with
  initial LR 1e-4 and polynomial decay, without left-right mirroring.

Could you please clarify the remaining implementation-critical points in one
reply?

1. Do you plan to release the full prompt-conditioned training code and
   dataloader/sampler (distinct from the released encoder-to-nnU-Net
   fine-tuning code), and the prompt vocabulary/manifest used for training?
2. Is an “absent structure” defined relative to the entire volume or the
   sampled 192^3 patch? If a structure exists in the volume but is outside the
   sampled patch, is it retained as a positive prompt with an empty cropped
   target, excluded, or treated as a negative?
3. Is the 2:1 positive/negative ratio strictly instantiated as three prompts
   per image at every step, or is any additional class/prompt balancing applied
   across batches or distributed workers?
4. How do you avoid false negatives for partially labeled datasets? Are
   negatives sampled only from structures known to be absent from that volume,
   or from all labels not annotated by the source dataset?
5. Could you release the mapping between the 1,087 unified concepts, the 9,682
   rewritten labels, and source-dataset labels (or at least the checkpoint's
   prompt bank)?
6. Are the five deep-supervision losses normalized by the sum of their weights?
   For all-zero negative targets, which Dice smoothing/empty-target convention
   and Dice-vs-BCE weighting are used?
7. Does the official evaluation/inference pipeline apply any unpublished
   connected-component filtering, ROI/anatomical restriction, containment, or
   other post-processing, especially for small or tubular structures?
8. For prompt-conditioned fine-tuning from the v1.1 checkpoint, do you
   recommend loading the entire network or only the image encoder, and should
   optimizer/scheduler state be reset?

Even a brief answer or a pointer to code/config files would help us avoid an
inaccurate reimplementation. We will clearly distinguish the official
encoder-transfer baseline from any paper-guided approximation in our work.

Best regards,
[Name / affiliation]
```

## 项目当前应采用的表述与行动

在作者回复或完整源码发布前，报告中使用：

> We use the official VoxTell v1.1 checkpoint and model components. The
> released `voxtell-finetune` path is used only as an official
> encoder-transfer nnU-Net baseline. Because the full prompt-conditioned
> training pipeline and vocabulary mapping are not public, our
> prompt-conditioned Student is a paper-guided approximation implementing the
> disclosed 2:1 prompt sampling, 85% foreground oversampling, 25/75 prompt-name
> sampling, offline text embeddings, and five-scale Dice+BCE supervision.

近期代码动作优先级：

1. 更新 `third_party/VoxTell` 到官方 `ec517b7`，保留其源码不作本地修改；
2. 将官方 `voxtell-finetune` 作为 baseline，不作为 prompt Student；
3. 校正本地近似 trainer：每 image 固定 2+1 prompt、85% foreground、
   canonical/rephrase 25/75、五尺度 Dice+BCE；
4. 在未获作者确认前，不把“patch 外 target”擅自定义为官方 negative，也不把
   containment/post-processing 描述成 VoxTell 官方方法；
5. 邮件经老师/项目负责人确认署名后一次性发出，避免重复 issue。

## 本地 trainer 与论文的已确认差异（代码审计）

`scripts/train_voxtell_prompt_student.py` 已正确采用官方模型组件、冻结并缓存
Qwen embedding、默认 2:1 runtime sampling、默认 deep supervision、SGD
`1e-4` 和 polynomial LR；但仍有以下不等价项：

| 本地当前实现 | 论文公开配置 | 处理建议 |
|---|---|---|
| `--foreground-prob` 默认 `0.7` | `0.85` | 默认值改为 `0.85` |
| 每个 optimizer step 只抽一个 positive 或 negative item，以循环维持 2:1 | 每张 image 同时 query 2 positive + 1 negative | 需要改为同一 image 的三 prompt 训练单元；至少在完成前明确记录为近似 sampler |
| 可从 volume-positive mask 主动裁出 `absent_in_crop` 并作为 derived negative | negative 是 volume-absent structure；patch 外目标规则未公开 | 暂不能宣称论文一致；等待作者回答，formal run 建议默认关闭该派生策略 |
| prompt variants 是 manifest candidate rows，未见 canonical 25% / rewrite 75% 的显式分层采样 | canonical 25%，rewrite 75% | 按 prompt family 两阶段采样，避免 variant 数量改变概率 |
| BCE 对 positive patch 使用最高 100 的动态 `pos_weight` | 论文只写 Dice+BCE，未写 foreground-balanced BCE | 作为项目改动单独报告；论文对齐 run 应提供 unweighted BCE ablation |
| 五尺度 loss 除以权重和 | 论文给权重但未说明是否归一化 | 保留为待确认项，并在 manifest 记录 `normalized_by_weight_sum=true` |
| 默认 epoch 为 1、每 epoch 约一遍 manifest | 2,000 epochs × 250 iterations | smoke/pilot 可以保留；正式论文对齐配置必须显式覆盖并标注算力差异 |
