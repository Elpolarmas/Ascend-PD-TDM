# Ascend-PD-TDM 多模态补充实验方案

> 状态：实验前设计稿，2026-08-20。本文中的 `TBD` 表示尚待实测；“预期结果”仅是
> 可证伪的实验假设和论文决策规则，不是已经观测到的结果。任何实际数字只能在原始
> JSON/JSONL、配置和分析脚本完成审计后进入论文。

## 1. 执行摘要

本补充实验拟采用 **Qwen2.5-VL-7B-Instruct + VisionArena-Bench v0.1**，在与主实验
相同的 **2×Ascend 910B3** 平台上，对比 **vLLM-Ascend Chunked Prefill（CP）** 与
**PD-TDM** 的在线多模态生成性能。实验不评估视觉问答准确率，而把真实图像和用户问题
作为 serving workload，测量视觉编码、Prefill 和 Decode 共同作用下的端到端延迟与
联合 SLO Goodput。

该实验只承担一个论文角色：验证 PD-TDM 能否从纯文本 LLM serving 迁移到包含真实图像
信号和可变视觉 token 成本的多模态生成服务。它不取代 Qwen3-8B/Azure 主结果，不扩张为
视觉编码器调度贡献，也不支持音频、视频或跨 GPU 平台的普适性结论。

### 1.1 拟锁定方案

| 项目 | 拟采用设置 | 状态 |
|---|---|---|
| 模型 | `Qwen/Qwen2.5-VL-7B-Instruct` | 已选定 |
| 数据集 | `lmarena-ai/vision-arena-bench-v0.1` | 已选定 |
| 测量请求 | 固定 200 条单图、单轮请求 | 已选定 |
| Warm-up | 与测量集不重叠的 10 条请求 | 已选定 |
| 平台 | 2×Ascend 910B3，TP=2，BF16 | 拟锁定，需 smoke 验证 |
| Baseline | vLLM-Ascend 原生 CP | 已选定 |
| Ours | PD-TDM，2,048-token Prefill bound | 已选定 |
| 到达过程 | Poisson，3 个配对 arrival seed | 已选定 |
| QPS | 以 CP 实测容量为锚点的 5 个绝对 QPS | pilot 后锁定 |
| 主输出 | greedy，自然 EOS，最多 128 token | 已选定 |
| 主指标 | 联合 SLO attainment 与有效输出 token Goodput | 已选定 |
| 机制指标 | visual tokens、encoder time、TTFT/TPOT 分解 | 需确认可观测性 |

## 2. 研究问题与可证伪假设

### 2.1 研究问题

**RQ1：**在加入视觉编码和可变视觉 token 后，PD-TDM 是否仍能在相同负载下改善联合
TTFT/TPOT SLO？

**RQ2：**收益是否主要出现在中高负载和视觉 Prefill 压力较高的请求，而非低负载基础
延迟？

**RQ3：**当输出变长、Decode 压力增加时，PD-TDM 的收益是否缩小，保持与文本实验一致
的适用边界？

### 2.2 实验前假设

- **H1（低负载）：**低负载下两种方案接近，PD-TDM 不应依靠明显增加基础开销换取收益。
- **H2（中高负载）：**接近 CP 饱和区时，PD-TDM 预计降低 TTFT tail，并提高联合 SLO
  attainment/Goodput；TPOT 不应出现足以抵消 TTFT 收益的退化。
- **H3（视觉成本）：**按 visual-token 数量分组后，高视觉成本组的 TTFT/Goodput 改善
  预计更明显。
- **H4（长输出边界）：**把目标输出提高到 512 token 后，Decode 压力增加，PD-TDM 的
  净收益预计缩小；若 TPOT 变成主瓶颈，结果可能持平或反转。

这些假设只规定要验证的方向。若结果不符合，论文应收缩或删除多模态 claim，而不是重新
筛选数据或 SLO 以制造正结果。

## 3. 模型与平台配置

### 3.1 模型选择

| 配置项 | 设置 |
|---|---|
| 模型 | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 参数规模 | 约 7B 语言模型规模 |
| 数值精度 | BF16 |
| 模态 | 单图 + 文本问题 → 文本生成 |
| 解码 | greedy，`temperature=0` |
| 最大上下文 | `max_model_len=16384` |
| 每请求模态限制 | `image=1, video=0` |
| 图像 token 范围 | 256--1,280 visual tokens/image（拟锁定） |
| Processor 像素范围 | `min_pixels=256*28*28`，`max_pixels=1280*28*28` |

选择理由：

1. 7B 规模与主实验 Qwen3-8B 接近，减少模型规模差异。
2. vLLM-Ascend v0.11.0rc1 的支持矩阵明确列出 Qwen2.5-VL。
3. vLLM-Ascend 已给出该模型与 VisionArena 的在线 serving 配方，实施风险低。
4. 模型包含视觉编码、语言 Prefill 和 Decode，足以验证跨模态适用性。
5. 相比 Audio/Omni/Video 模型，不需要新增不成熟的在线输入链路，适合 ICASSP deadline
   和四页正文预算。

Qwen2.5-VL 默认允许每张图产生 4--16,384 个视觉 token，跨度过大会把图像分辨率、OOM
风险和调度机制混在一起。模型卡建议用 256--1,280 token 范围平衡成本与效果；本实验拟
显式固定该范围，并对两种调度器使用完全相同的 processor 配置。

### 3.2 硬件与公共服务参数

| 配置项 | CP | PD-TDM | 公平性要求 |
|---|---:|---:|---|
| NPU | 2×Ascend 910B3 | 2×Ascend 910B3 | 完全一致 |
| Tensor parallelism | TP=2 | TP=2 | 完全一致 |
| 精度 | BF16 | BF16 | 完全一致 |
| vLLM-Ascend | 0.11.0rc1 | 0.11.0rc1 + PD-TDM | 同一基础版本 |
| `max_model_len` | 16384 | 16384 | 完全一致 |
| `gpu_memory_utilization` | 0.9（拟） | 0.9（拟） | smoke 后锁定 |
| Prefix caching | 关闭 | 关闭 | 防止复用污染 |
| MM processor cache | 0 GiB | 0 GiB | 或每 run 重启 server |
| `max_num_seqs` | TBD | 相同值 | smoke 后记录精确值 |
| Swap space | 16 GiB（如启用） | 同值 | 不允许仅一侧启用 |

### 3.3 两种调度配置

| 维度 | vLLM-Ascend CP | PD-TDM |
|---|---|---|
| 调度形态 | mixed iteration | bounded pure-P / pure-D iteration |
| Prefill bound | `max_num_batched_tokens=2048` | `prefill_chunk_tokens=2048` |
| 外层 token budget | 2048 | 8192（拟沿用现有 m31-fix） |
| 初始 P/D ratio | 不适用 | 0.3 |
| slice 范围 | 不适用 | `min=2, max=8` iterations |
| selector | 原生 CP scheduler | token bucket，`bucket_cap=4` |
| controller | 不适用 | 当前 `c2_tdm_m31_2048` bundle |

这里“公平”指两者具有相同的**有效 Prefill 工作上限 2,048 token**，不要求两个实现使用
同一个外层参数。PD-TDM 的全局 budget 需要高于独立的 Prefill chunk bound；CP 则通过
`max_num_batched_tokens` 实现 chunking。由于视觉 token 的预算核算可能不同于纯文本，
smoke/pilot 必须验证两边实际每次 Prefill 的 token bound，而不能只比较配置文件字面值。

论文只把上述配置视为两个完整调度 bundle 的端到端对比，不把差值单独归因给 controller、
pure phase 或某个 kernel。

## 4. 数据集与测试数据构造

### 4.1 数据集

采用 `lmarena-ai/vision-arena-bench-v0.1`。公开 train split 有 500 条数据，每条包含
一个 `question_id`、一个语义类别 `cluster_name`、一轮用户问题和一张图像。任务覆盖描述、
OCR、图表分析、物体识别、代码截图、艺术分析和开放式生成。

选择理由：

1. 数据来自真实用户-VLM 交互，比 TextVQA 等短答案准确率集更接近在线服务输入。
2. vLLM/vLLM-Ascend 已直接支持其字段和 OpenAI Chat 请求格式。
3. 问题类型和图像复杂度具有自然异质性，适合观察视觉输入成本对调度的影响。
4. 500 条规模足以固定一个 200 请求的可复现子集，不需要人为构造问题。

### 4.2 冻结数据 manifest

先对全部 500 条数据完成 processor profiling，再以固定 `dataset_seed=0` 打乱：

- 前 10 条作为 warm-up，不计入指标；
- 随后 200 条作为正式 measurement set；
- 其余数据保留为替补，不因实验结果重新筛选；
- 下载并保存本地图像，记录数据集 revision/hash；
- 损坏图像只允许在首次冻结前按统一规则剔除，并记录 ID 和原因。

冻结后比较 200 条子集与全部 500 条在 text tokens、visual tokens、total input tokens 和
图像像素数上的分位数/经验 CDF。若随机子集明显不具代表性，必须在任何性能实验前按
visual-token 四分位和 `cluster_name` 做预先定义的分层抽样；不能根据 CP/PD 性能结果
重新选数据。

建议 manifest 字段：

```text
request_id, question_id, cluster_name, image_path, image_sha256,
prompt, image_width, image_height, image_bytes,
text_tokens, processed_visual_tokens, target_output_tokens
```

客户端 tokenizer 给出的文本长度不包含完整视觉 token 成本，因此不得把它直接写成论文的
“multimodal input length”。正式报告至少给出以下分布的 mean、p50、p95 和 max：

- 文本 token；
- 图像宽高和像素数；
- 处理后的 visual tokens；
- 端到端 prompt tokens（若服务端 usage 可可靠返回）；
- 实际输出 token。

### 4.3 必须先完成的数据特征分析

#### 已有公开量化结果与边界

不能说 VisionArena 完全没有被量化分析；更准确的说法是，公开材料尚未给出本文所需的
**全量、模型相关、可复现的服务工作负载画像**：

| 来源 | 已公开的量化信息 | 对本实验的局限 |
|---|---|---|
| VisionArena 原论文及补充材料 | 500 条 Bench 提示的模型排名，以及 Chat/Battle 的语言、任务类别、轮次和回答风格/长度分析 | 不是 Qwen2.5-VL processor 下的 500 条 Bench 输入长度分析；没有视觉 token 与 E/P/D 时间分解 |
| vLLM-Ascend 当前性能文档 | Qwen2.5-VL-7B、VisionArena-Bench、10 条请求：总输入 7,191 token、总输出 951 token，即均值约 719.1/95.1 token | 样本仅 10 条，只给总量；没有 p50/p95/max、图文拆分、图像分辨率和计数口径说明 |
| vLLM-Ascend v0.7.3 性能记录 | 同模型/数据集运行 200 条请求，并给出不同 QPS 下的吞吐、TTFT、TPOT、ITL | 可作系统性能参照，但未公开 200 条的输入长度分布、视觉 token 分布和 encoder/prefill/decode 服务时间 |

##### vLLM-Ascend 当前文档：10-request serving 示例

来源：<https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/performance_and_debug/performance_benchmark.md>

公开配置为 Qwen2.5-VL-7B-Instruct、BF16、每条最多一张图片，数据集
`lmarena-ai/vision-arena-bench-v0.1`，`openai-chat` 后端、非流式返回，10 条请求同时到达；
文档没有固定 dataset/model revision、抽样 seed，也没有说明该示例使用的具体 NPU 型号。

| 指标 | 公开结果 |
|---|---:|
| 成功/失败请求 | 10 / 0 |
| 测试时长 | 4.89 s |
| 总输入 token | 7,191 |
| 平均输入 token/request（推导） | 719.1 |
| 总生成 token | 951 |
| 平均输出 token/request（推导） | 95.1 |
| 请求吞吐 | 2.05 request/s |
| 输出吞吐 | 194.63 token/s |
| 峰值输出吞吐 | 290.00 token/s |
| 峰值并发请求 | 10 |
| 总 token 吞吐 | 1,666.35 token/s |
| TTFT mean / median / p99 | 722.22 / 589.81 / 1,377.02 ms |
| TPOT mean / median / p99 | 44.13 / 34.58 / 124.72 ms |
| ITL mean / median / p99 | 33.14 / 28.01 / 182.28 ms |

该示例证明同一模型与数据集已能在 vLLM-Ascend 跑通，并提供了输入/输出均值的初始参照；
但 10 条数据无法代表 500 条全集的分布，且 `Total input tokens` 没有拆分 text/visual
tokens。因此本项目不得用它替代 V0 profiling，也不得把这些系统性能值作为本文 baseline。

##### vLLM-Ascend v0.7.3：200-request 发布性能记录

来源：<https://github.com/vllm-project/vllm-ascend/issues/776>

公开配置为 Qwen2.5-VL-7B-Instruct、TP=1、`max_model_len=16384`、VisionArena-Bench
固定随机 seed 抽取 200 条，在线请求率为 1、4、16 和 `inf`；有限 QPS 使用固定 seed 的
Poisson 到达。下表先列出与本文更接近的原生 vLLM-Ascend 结果。该记录将 TTFT、TPOT、
ITL 报告为中位数。

| 目标 QPS | 实际吞吐 (req/s) | 输出吞吐 (tok/s) | TTFT (ms) | TPOT (ms) | ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.993584 | 109.046 | 345.717 | 40.9481 | 32.1247 |
| 4 | 3.66915 | 402.579 | 321.537 | 86.7003 | 42.4031 |
| 16 | 6.07012 | 668.412 | 8,580.35 | 164.858 | 71.5467 |
| inf | 5.26736 | 579.831 | 14,326.2 | 208.966 | 73.1931 |

同一记录还给出了接入 MindIE Turbo 后的结果，作为公开实现版本差异的参考：

| 目标 QPS | 实际吞吐 (req/s) | 输出吞吐 (tok/s) | TTFT (ms) | TPOT (ms) | ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.997964 | 109.247 | 374.364 | 30.8042 | 22.8870 |
| 4 | 3.75650 | 411.938 | 301.918 | 67.4147 | 31.0699 |
| 16 | 6.20110 | 673.533 | 8,744.60 | 158.046 | 64.3575 |
| inf | 5.11639 | 557.508 | 17,186.9 | 195.048 | 64.0484 |

由 `output throughput / request throughput` 推算，这些在线结果的平均实际输出长度稳定在
约 110 token/request，但该值是从聚合吞吐反推的近似量，不是公开的逐请求长度统计。
原生 vLLM-Ascend 的离线吞吐测试还报告：200 条请求、203,673 total tokens、69.9566 s、
2.85892 request/s 和 2,911.42 token/s，即平均约 1,018.4 total tokens/request；原记录没有
明确给出 total tokens 的输入/输出拆分，因此不能把 1,018.4 当作平均输入长度。记录中的
4,128.36 ms 平均离线延迟来自另一个固定 `32 input / 128 output / batch=8` 测试，也不是
VisionArena 的真实长度画像。

这组数据说明该 workload 在约 6 request/s 附近开始出现明显排队：目标 QPS 从 4 提升至
16 时，原生结果的 TTFT 从 321.537 ms 增至 8,580.35 ms。不过，硬件型号、图像缩放参数、
processor revision 与逐请求输入分布未完整披露，故它只能帮助选择本项目的 pilot QPS，
不能作为跨版本的绝对性能对照。

因此，719.1 input / 95.1 output 可以作为 pilot 前的一个**弱先验**，不能当作数据集的
正式 workload signature，也不足以直接认定 Qwen3 的 C4 是匹配单元。尤其是 Qwen2.5-VL
允许通过 `min_pixels/max_pixels` 改变每幅图像的视觉 token 数；输出长度也随模型、解码参数
和 `max_tokens` 改变。所谓“VisionArena 的输入/输出长度”并不是脱离 processor 与运行配置
后仍唯一成立的常数。

本项目的增量不是再做语义类别统计，而是固定模型 revision、processor、图像缩放策略和
解码配置后，给出全部 500 条的图像尺寸、文本 token、视觉 token、总输入 token，以及冻结
200 条测量子集的自然输出长度与 E/P/D 成本画像。

带图数据可以且必须分析，但需区分三个长度/成本层次：

1. **文本长度：**对完整 chat template 编码，记录 system/user/template 的 text tokens。
2. **视觉长度：**由 Qwen2.5-VL processor 的 `image_grid_thw` 计算 encoder 输入 patch 数，
   并统计处理后 `input_ids` 中的 image placeholder 数作为 post-merge visual tokens。
3. **完整 Prefill 长度：**`total_input_tokens=len(processor_output.input_ids)`，这是映射到
   Qwen3 Prompt 长度时最接近的 token-space 指标。

对 Qwen2.5-VL，若 `spatial_merge_size=m`，可同时记录：

```text
premerge_patches = temporal_grid * height_grid * width_grid
postmerge_visual_tokens = premerge_patches / (m * m)
total_input_tokens = text/template tokens + postmerge_visual_tokens
```

最终以 processor 实际 `input_ids` 中的 image token 数为权威值，公式只用于交叉核验。

VisionArena 没有参考答案，因此输出长度需单独 profiling：

- 在 CP 低负载/顺序模式下，对冻结的 200 条请求执行 greedy generation；
- 第一次使用 `max_tokens=128`，记录实际 output tokens 和 `output==128` 的 censor rate；
- 若 censor rate 较高，再以 `max_tokens=512` 运行一次，估计未截断的自然输出分布；
- 保存每条请求的生成长度和输出 hash，不把生成文本当作准确率 ground truth；
- 正式 CP/PD 测试使用同一输出规则，并核验工作量分布一致。

数据特征报告至少包含：

| 特征 | 统计量 |
|---|---|
| 原始 width/height/pixels | mean、p50、p90、p95、max |
| pre-merge image patches | mean、p50、p90、p95、max |
| post-merge visual tokens | mean、p50、p90、p95、max |
| text/template tokens | mean、p50、p90、p95、max |
| total multimodal input tokens | mean、p50、p90、p95、max |
| natural output tokens | mean、p50、p90、p95、max、censor rate |
| token-space `P/O` | mean、p50、p90、p95 |
| encoder/prefill/decode service time | mean、p50、p95 |

仅用 `total_input_tokens/output_tokens` 映射 Qwen3 仍不充分，因为图像 encoder 计算不等价
于语言模型 Prefill token。最终应同时报告两个 workload signature：

```text
token-space signature = (text tokens, visual tokens, total input tokens, output tokens)
cost-space signature  = (encoder_ms + prefill_ms) / decode_service_ms
```

先在现有 F5/F5d 的 `(Prompt, Output)` 单元中找到 token-space 邻近单元，再用服务时间比
检查是否真的接近。只有两种映射大致一致时，才能把某个 Qwen3 单元称为 VisionArena 的
“closest text workload”。

若没有现有单元同时匹配输入、输出和服务时间比，不应强行选择 C4/C6。条件允许时，可按
VisionArena 的经验 `(total_input_tokens, natural_output_tokens)` 分布生成一组 Qwen3
matched synthetic workload，作为“同 token 分布的文本对照”。该对照仍包含 Qwen3 与
Qwen2.5-VL 的模型差异；要严格隔离视觉 encoder，需另用同代 Qwen2.5 文本模型或
Qwen2.5-VL text-only 路径，因此不把跨模型差值归因成纯视觉开销。

### 4.4 输出长度

**主 workload：**

```text
temperature=0, max_tokens=128, ignore_eos=False
```

该设置与 vLLM 的 VisionArena loader 对齐：数据集没有参考答案，loader 默认只为每个请求
指定 128-token 上限。模型可提前输出 EOS，因此必须报告实际输出长度，并确认两种调度下
同一请求的输出 token 数和文本 hash 一致或近似一致。

**公平性控制：**若两边实际输出工作量出现系统性差异，增加：

```text
temperature=0, max_tokens=128, ignore_eos=True
```

**边界敏感性（条件性）：**

```text
temperature=0, max_tokens=512, ignore_eos=True
```

只在一个中负载点和一个近饱和点运行，用于检验 Decode 压力增加后的收益边界。

## 5. 可参考的官方 benchmark

vLLM-Ascend 已提供如下在线测试：

```text
模型        Qwen2.5-VL-7B-Instruct
数据        vision-arena-bench-v0.1
请求数      200
QPS         1, 4, 16, inf
有限 QPS    Poisson arrival
输出上限    128 token
指标        throughput, TTFT, TPOT, ITL
接口        /v1/chat/completions, streaming
```

参考命令：

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name hf \
  --dataset-path lmarena-ai/vision-arena-bench-v0.1 \
  --hf-split train \
  --num-prompts 200 \
  --hf-output-len 128 \
  --request-rate <QPS> \
  --burstiness 1 \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed
```

### 5.1 我们继承与修改的部分

| 项目 | 官方 benchmark | 论文实验 |
|---|---|---|
| 模型/数据/API | 原样采用 | 原样采用 |
| 请求数 | 200 | 200 + 独立 warm-up |
| 数据抽样 | `--seed` 临时 shuffle | 固定本地 manifest |
| 到达 seed | 与数据 seed 共用 | 与数据分离、CP/PD 配对 |
| QPS | 1/4/16/inf | 按本机 CP 容量校准 5 点 |
| 输出 | 最多 128 | 主实验相同，另有条件性控制 |
| 统计 | 常规 latency/throughput | 加三 seed、联合 SLO Goodput、机制分解 |
| 图像 cache | 未强调 | 禁用或每 run 重启 |

修改原因是官方配置主要用于功能和通用性能回归，固定 `1/4/16` 可能在本机上分别落入完全
空闲和完全过载区域，无法公平展示调度曲线；论文还需要配对随机性、tail latency、Goodput
和视觉成本分解。

## 6. 实验流程与矩阵

### Phase V0：数据 profiling 与文本 workload 映射（必须）

1. 下载并固定数据集 revision；
2. 用正式 processor 配置分析全部 500 条图像和问题；
3. 冻结并验证 200 条 measurement set 的代表性；
4. 顺序生成 128-token cap 的自然输出，必要时补 512-token uncensored profile；
5. 测量 encoder/prefill/decode 服务时间；
6. 根据 token-space 与 cost-space signature 确认最接近的 Qwen3/F5 单元。

V0 未完成前，C4/C6 只能称为候选参照，不能用于锁定正式 QPS、SLO 或预期收益。

### Phase V1：Correctness smoke（必须）

```text
2 systems × 20 requests × {sequential, small burst}
```

检查：

- 20/20 请求成功，无 OOM 和 scheduler exception；
- PD-TDM 能正确完成 image encoder、Prefill 和 Decode；
- 相同请求的 greedy 输出一致；
- 服务端 usage、TTFT、TPOT 和 encoder telemetry 可获得；
- 两边实际 Prefill bound 接近拟设的 2,048 token。

若需要改变调度语义或新增 encoder scheduler 才能跑通，则触发 V0 No-Go，不进入正式实验。

### Phase V2：容量 pilot（必须）

先以 64 条固定请求执行 `request_rate=inf` 和粗粒度 QPS sweep，得到 CP 的近似容量
`lambda_cap`。正式 QPS 拟设为：

```text
0.25, 0.50, 0.75, 1.00, 1.25 × lambda_cap(CP)
```

pilot 后将其转换成固定的绝对 QPS；CP 与 PD-TDM 不能分别按各自容量归一化。

公开的 vLLM-Ascend v0.7.3 同模型/同数据/TP=1 结果显示，系统在目标 QPS=16 时实际
吞吐约为 6.1 request/s，已经进入明显过载区。因此首次 pilot 可直接采用：

```text
QPS = {1, 2, 4, 6, 8, inf}
```

这只是减少盲扫的初始网格；正式 V3 仍根据当前 v0.11.0rc1、TP=2 的 CP 容量从中选取或
插值得到 5 个固定点。

### Phase V3：主实验（必须，V1/V2 通过后）

| 维度 | 数量/取值 |
|---|---|
| 调度器 | CP、PD-TDM |
| QPS | 5 个固定点 |
| arrival seed | 0、1、2 |
| 请求数 | 200/run |
| 到达 | Poisson，`burstiness=1` |
| 输出 | natural EOS，最多 128 token |
| 总请求 | `2×5×3×200 = 6,000` |

同一个 `(QPS, seed)` 下，两种调度器使用相同请求顺序和相同到达时间。建议交替运行 CP/PD，
并在每个 run 前完成相同 warm-up；若不禁用多模态 processor cache，则必须重启 server。

### Phase V4：边界敏感性（条件性）

若 V3 显示明确正向趋势，再运行：

```text
2 systems × 2 QPS × 3 seeds × 200 requests, output=512 exact
```

如果 V3 已经持平或为负，不继续扩张矩阵，而是记录 No-Go 结论。

## 7. 指标定义

### 7.1 端到端延迟

- `TTFT_i = first_token_time_i - submit_time_i`。
- `E2EL_i = finish_time_i - submit_time_i`。
- 当输出 token 数 `o_i > 1` 时，
  `TPOT_i = (E2EL_i - TTFT_i) / (o_i - 1)`。
- ITL 保留为诊断指标，不作为主要 claim。

每项报告 mean、median、p95、p99；p99 必须同时给出三 seed 分布或 bootstrap 置信区间，
不能只解读一次 200 请求中的最慢两个样本。

### 7.2 联合 SLO attainment

对预先锁定的 `S_TTFT` 和 `S_TPOT`：

```text
A = count(TTFT_i < S_TTFT and TPOT_i < S_TPOT) / N
```

SLO 不能根据最终 CP/PD 曲线事后挑选。拟采用以下锁定流程：

1. 预注册 `S1=500/100 ms`、`S2=1000/100 ms`、`S3=2000/150 ms`；
2. pilot 只查看 CP 低负载/单请求分布；
3. 选择低负载 attainment 仍不低于 90% 的最严格一档作为主档；
4. 在 V3 前将选择和毫秒值写入 manifest，再运行/分析 PD-TDM；
5. 两个系统、全部 QPS 和 seed 使用相同阈值，其他档只作 sensitivity。

主 SLO 档当前为 `TBD`，需与 T0/T1 的全文 SLO 口径一并冻结。

### 7.3 联合 SLO Goodput

沿用当前论文的有效输出 token 定义。测量窗口长度为 `T`、请求输出 token 为 `o_i`：

```text
G_tok = sum(o_i * I[TTFT_i < S_TTFT and TPOT_i < S_TPOT]) / T
```

单位为有效输出 token/s，是主 Goodput 指标。另报告：

```text
G_req = count(SLO-qualified requests) / T
```

单位为 request/s，用于与 vLLM CLI 和其他 serving 工作对齐。两种 Goodput 口径不得混写。

### 7.4 资源与机制指标

- 实际 arrival QPS、completed QPS、错误率；
- raw output token/s 和 total token/s；
- 原始/处理后图像尺寸和 visual tokens；
- image preprocessing、encoder、queueing、Prefill、Decode 时间；
- 每轮 phase、iteration duration、scheduled token 数；
- visual-token 分组后的 TTFT/TPOT/Goodput。

端到端 TTFT 包含视觉链路，但论文不能把全部 TTFT 差值归因给 P/D scheduler。只有在 encoder
时间可观测、且两边 encoder 工作量匹配时，才能讨论 scheduler 部分的机制。

## 8. 结果记录模板

### 8.0 可用于定量锚定的公开结果

vLLM-Ascend 官方在 v0.7.3 发布说明中给出了与本方案最接近的一组公开数字：

```text
模型：Qwen2.5-VL-7B-Instruct
数据：vision-arena-bench-v0.1
请求：200
TP：1
max_model_len：16384
到达：QPS=1/4/16/inf；有限 QPS 为 Poisson
```

其原生 vLLM-Ascend 结果如下。原页面把 TTFT、TPOT 和 ITL 描述为 median：

| 目标 QPS | 实际 req/s | 输出 tok/s | TTFT ms | TPOT ms | ITL ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.994 | 109.0 | 345.7 | 40.9 | 32.1 |
| 4 | 3.669 | 402.6 | 321.5 | 86.7 | 42.4 |
| 16 | 6.070 | 668.4 | 8,580.4 | 164.9 | 71.5 |
| inf | 5.267 | 579.8 | 14,326.2 | 209.0 | 73.2 |

同一页面还报告了启用 MindIE Turbo 的结果：

| 目标 QPS | 实际 req/s | 输出 tok/s | TTFT ms | TPOT ms | ITL ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.998 | 109.2 | 374.4 | 30.8 | 22.9 |
| 4 | 3.757 | 411.9 | 301.9 | 67.4 | 31.1 |
| 16 | 6.201 | 673.5 | 8,744.6 | 158.0 | 64.4 |
| inf | 5.116 | 557.5 | 17,186.9 | 195.0 | 64.0 |

这些数字可支持三个实验前判断：

1. 该 workload 的实际平均输出约为 110 token，和 128-token 上限一致，不是极短答案集。
2. 单 TP=1 公开结果的服务容量约为 5--6 request/s；QPS=4 尚可服务，QPS=16 已过载。
3. 低负载 TTFT 约 0.3--0.4 s、TPOT 约 31--41 ms，过载时 TTFT 可升到 8--17 s。

因此可以把下面的数值作为 **pilot 候选**，但不能预填为论文结果：

```text
初始 QPS sweep：{1, 2, 4, 6, 8, inf}
S1 严格候选：TTFT < 500 ms AND TPOT < 100 ms
S2 中等候选：TTFT < 1000 ms AND TPOT < 100 ms
S3 宽松敏感性：TTFT < 2000 ms AND TPOT < 150 ms
```

结合 Qwen3 C4 的 TTFT tail 与约 0.25 s 视觉前端开销，`500/100 ms` 可能穿过非饱和区
的 tail，而 `1000/100 ms` 位于公开数据的可服务区和过载区之间。主档采用预先规定的
选择规则：**只查看 CP pilot，选择低负载 attainment 仍不低于 90% 的最严格一档**；随后
冻结 SLO 并运行/分析 PD-TDM。S3 只检查结论是否由单一阈值制造。

上述结果只能作数量级参考，原因包括：公开页面未明确写出 NPU 具体型号；它使用
vLLM-Ascend v0.7.3、TP=1，而本实验使用 v0.11.0rc1、2×910B3、TP=2；MindIE Turbo 也不
属于当前 PD-TDM 公平配置。报告和论文不得把这些数字标成“our baseline”。

作为跨平台 sanity reference，vLLM 官方 recipe 还报告过 4×A100、DP=4、
VisionArena-Chat、128 个同时到达请求：13.09 request/s、1,455 output token/s、median
TTFT 4.75 s、p99 TTFT 7.58 s、median TPOT 45.3 ms、p99 TPOT 354.2 ms。由于数据集、
并行方式、硬件和到达模式均不同，这组数值只说明量级会随并发显著变化，不用于设置 NPU
性能目标。

#### 8.0.1 结合现有 Qwen3-8B 数据形成实验先验

项目已有 Qwen3-8B、2×910B3、TP=2 数据能够提供同平台的**相对变化规律**。在正式数据
profiling 之前，基于拟设的 256--1,280 visual-token cap，候选邻近单元是 F5d-C4：约
1,043-token Prompt、245-token Output、Poisson arrival、CP 与 PD-TDM 共用 2,048-token
Prefill bound。该组为 single seed，但 5 个 QPS 均无 client error：

| QPS | CP/PD TTFT mean ms | CP/PD TTFT p99 ms | CP/PD request-mean TPOT p99 ms | Goodput |
|---:|---:|---:|---:|---|
| 0.5 | 140.8 / 106.9 | 197.9 / 186.4 | 28.6 / 29.5 | 相同，均 100% attainment |
| 1.0 | 111.3 / 69.6 | 179.7 / 137.2 | 32.4 / 30.2 | 相同，均 100% attainment |
| 1.5 | 98.5 / 58.1 | 198.1 / 181.9 | 33.0 / 34.9 | 相同，均 100% attainment |
| 2.0 | 94.3 / 50.6 | 200.0 / 141.1 | 43.2 / 41.3 | 相同，均 100% attainment |
| 2.5 | 91.4 / 48.2 | 231.9 / 179.7 | 46.4 / 42.7 | 相同，均 100% attainment |

这里的 TPOT 列不是“所有逐 token 间隔的 p99”，而是先对每个请求计算
`(E2EL-TTFT)/(output_tokens-1)`，再对这些**每请求平均 TPOT**取跨请求 p99。该口径与本文
联合 SLO 的请求级 TPOT 定义一致，但会平滑请求内部偶发的长 token stall，因此原表简称
“TPOT p99”容易被误解。以 QPS=2.5 为例：

| 口径 | CP | PD-TDM |
|---|---:|---:|
| 每请求平均 TPOT 的跨请求 p99 | 46.4 ms | 42.7 ms |
| 每请求 ITL-p99 的跨请求 median | 77.4 ms | 63.7 ms |
| 每请求 ITL-p99 的跨请求 p99 | 197.3 ms | 247.0 ms |

因此 28--46 ms 的数值本身没有算错，但它只说明平均生成节奏较快，不能说明 token tail
同样只有几十毫秒。F5d-C4 还是 Qwen3 文本模型、2×910B3/TP=2、最高仅 2.5 target QPS
的非深度过载区；它比公开的 Qwen2.5-VL、TP=1、QPS=4/16 场景 TPOT 更小是合理的。

原始 `500/200 ms` SLO 对该单元过于宽松。更紧 SLO 的 post-hoc attainment/Goodput 不再
单列，而是在下面加入视觉开销后的统一表中与绝对 TTFT/TPOT 放在同一行，避免脱离延迟
分布解读 Goodput。

该数据进一步对应：

- TTFT mean 降低约 24%--47%；
- TTFT p99 降低约 6%--30%；
- 每请求平均 TPOT 的跨请求 p99 变化约在 `-8%` 到 `+6%` 内；
- 由于使用 `TTFT<500 ms AND TPOT<200 ms`，两边全部达标，Goodput 没有差异。

它直接说明两个问题：PD-TDM 的相对优势主要表现为 TTFT shift；只有 SLO 穿过两条延迟
分布之间时，这个 shift 才会转化成 Goodput。对未来 VLM 正式实验，可以预先设置严格、
中等和宽松三档并报告二维 SLO sensitivity；但 headline 档必须在查看 PD-TDM 结果之前，
依据公开服务要求或 CP-only pilot 冻结，不能在两条结果曲线之间事后挑阈值。

VisionArena 在显式限制到 256--1,280 visual tokens/image、输出约 110 token 后，会比 C4
更加 Prefill-leaning；但 VLM 还多出视觉处理和 encoder 开销。公开 vLLM-Ascend v0.7.3
在 QPS=1 时，Qwen2.5-7B 文本 TTFT 为 97.9 ms、Qwen2.5-VL-7B 为 345.7 ms；QPS=4 时
分别为 104.4 ms 和 321.5 ms。虽然两边数据集不同，约 0.22--0.30 s 的差值可以作为固定
视觉前端开销的粗略数量级，不能当成精确 encoder time。

若用 `E_vision=250 ms` 做最简单的保守估算，并假设 PD-TDM 只改善 C4 中的调度部分：

```text
QPS=1 mean: CP ~= 250+111 = 361 ms, PD ~= 250+70 = 320 ms, delta ~= -11.5%
QPS=2 p99 : CP ~= 250+200 = 450 ms, PD ~= 250+141 = 391 ms, delta ~= -13.1%
```

将同一计算展开到五个 QPS，并在每条逐请求 TTFT 上加 250 ms 后重新执行联合 SLO
判定，得到下面的统一先验表。原 Qwen3 的 `200/40` 和 `150/50 ms` 分别平移为机械 VLM
估算下的 `450/40` 和 `400/50 ms`。TTFT/TPOT 单位均为 ms，斜线前后为 mean/p99；箭头
表示 `CP -> PD-TDM`；`G` 为有效 output token/s：

| SLO | QPS | TTFT mean/p99：CP -> PD | TPOT mean/p99：CP -> PD | Attainment：CP -> PD | G：CP -> PD | Delta G |
|---:|---:|---:|---:|---:|---:|---:|
| 450/40 | 0.5 | 390.8/447.9 -> 356.9/436.4 | 25.6/28.6 -> 26.8/29.5 | 98.8% -> 98.8% | 132.6 -> 132.5 | -0.1% |
| 450/40 | 1.0 | 361.3/429.7 -> 319.6/387.2 | 27.0/32.4 -> 26.3/30.2 | 99.4% -> 100.0% | 258.1 -> 259.4 | +0.5% |
| 450/40 | 1.5 | 348.5/448.1 -> 308.1/431.9 | 28.2/33.0 -> 29.0/34.9 | 99.1% -> 99.1% | 383.0 -> 383.2 | +0.1% |
| 450/40 | 2.0 | 344.3/450.0 -> 300.6/391.1 | 30.3/43.2 -> 30.1/41.3 | 94.6% -> 97.8% | 515.4 -> 531.7 | +3.2% |
| 450/40 | 2.5 | 341.4/481.9 -> 298.2/429.7 | 32.5/46.4 -> 31.1/42.7 | 84.8% -> 90.0% | 583.6 -> 624.0 | +6.9% |
| 400/50 | 0.5 | 390.8/447.9 -> 356.9/436.4 | 25.6/28.6 -> 26.8/29.5 | 68.3% -> 91.5% | 93.1 -> 122.7 | +31.8% |
| 400/50 | 1.0 | 361.3/429.7 -> 319.6/387.2 | 27.0/32.4 -> 26.3/30.2 | 80.5% -> 98.7% | 208.3 -> 255.9 | +22.8% |
| 400/50 | 1.5 | 348.5/448.1 -> 308.1/431.9 | 28.2/33.0 -> 29.0/34.9 | 86.8% -> 97.9% | 330.8 -> 378.7 | +14.5% |
| 400/50 | 2.0 | 344.3/450.0 -> 300.6/391.1 | 30.3/43.2 -> 30.1/41.3 | 90.8% -> 99.4% | 487.9 -> 538.1 | +10.3% |
| 400/50 | 2.5 | 341.4/481.9 -> 298.2/429.7 | 32.5/46.4 -> 31.1/42.7 | 91.0% -> 97.5% | 627.4 -> 670.2 | +6.8% |

合并后可以直接看到：`450/40 ms` 在最低负载下仍约 98.8% 达标，是较可辩护的严格档；
`400/50 ms` 虽然产生更大的 Goodput 差异，却使 CP 在最低负载只有 68.3% 达标，不能作为
正常可服务的主 SLO。该表不是拟设“最终结果”：固定 250 ms 忽略 encoder 排队和视觉
token 离散度，TPOT 又直接继承文本模型，因此偏乐观；原始数据还是 single-seed post-hoc
sensitivity。QPS 增加时 TTFT mean 反而下降也来自 C4 的 batching/样本窗口效应，不能
解释成一般负载规律。

公开的 10-request VLM serving 示例可以提供另一侧的**并发压力锚点**。其 CP 实测为
TTFT mean/p99 `722.2/1377.0 ms`、TPOT mean/p99 `44.1/124.7 ms`。若只把 Qwen3 中较
保守的 TTFT `10%--20%` 改善和 TPOT `±10%` 变化用于规划，PD-TDM 的拟设区间为：

| 锚点 | 系统 | TTFT mean ms | TTFT p99 ms | TPOT mean ms | TPOT p99 ms |
|---|---|---:|---:|---:|---:|
| 10 requests concurrently | 公开 CP 实测 | 722.2 | 1,377.0 | 44.1 | 124.7 |
| 同压力下的规划区间 | PD-TDM 先验 | 577.8--650.0 | 1,101.6--1,239.3 | 39.7--48.5 | 112.2--137.2 |

该公开页面没有逐请求 TTFT/TPOT 联合分布，只给聚合统计，因此不能从 mean/p99 反推出任意
SLO 下的联合 attainment 或 Goodput；把它强行并入上面的 SLO 表会制造不存在的数据。
该区间只是把“公开 VLM 绝对量级”和“本项目 Qwen3 相对变化”显式组合起来，不能作为
验收目标；公开数据与本项目在 NPU 数量、TP、软件版本、图像处理配置和请求集合上均不
一致。正式结果表仍必须填入本项目逐请求测得的 TTFT/TPOT mean、p50、p95 和 p99。

基于上述两个绝对锚点，再归纳完成 Phase V0 之前的 **暂定实验规划先验**：

| 区域 | TTFT 相对改善先验 | TPOT 先验 | Goodput 先验 |
|---|---:|---:|---:|
| 低负载 | 0%--10% | 接近，`±10%` | 基本相同，SLO ceiling |
| 中负载 | 10%--20% | 接近，`±10%` | 0%--15%，取决于 SLO |
| 近饱和 | 15%--30% | 可能小幅退化或改善 | 10%--30% 可作为有价值区间 |
| 深度过载 | 不作数值预测 | 两者均可能恶化 | 不作为 headline |

这些范围不是拟造的实验结果，而是把“同平台文本相对差值”与“公开 VLM 固定开销量级”相加
得到的资源规划先验。实际结果可能更好（若 PD-TDM 同时减少 encoder 排队）或更差（若视觉
encoder 完全主导 TTFT，或 multimodal scheduler 路径无法受益）。

已有其他 Qwen3 结果只用于检查方向，不用于预测具体幅度：

- F5 balanced（约 512/512）中 PD-TDM 降低 TTFT、TPOT 基本接近，并在严格 SLO 下提高
  Goodput，但该组为 single seed；
- T6 Azure burst 的 conv/code 为 3-seed 主证据，但负载比 VisionArena 更 Prefill-heavy、
  到达也更 bursty，较大的 attainment/Goodput 差值不能搬到 VLM；
- MaaS 长输入短输出在 post-hoc `1.2×` SLO 下有约 `+29.2pp` attainment 和约 `+50.6%`
  Goodput，只能说明长输入短输出方向有利，不能作为预期 VLM 数字。

按照 VisionArena 平均约 110 输出 token，若系统能跟上到达率，则 raw output throughput 的
理论量级约为：

```text
QPS 1/2/4/6 -> 约 110/220/440/660 output token/s
```

联合 SLO Goodput 等于上述 raw throughput 乘以 attainment。例如 QPS=4 实测 raw output
throughput 若为 400 token/s，CP/PD attainment 分别为 80%/90%，则 Goodput 为
320/360 token/s、相对提升 12.5%。该例只演示如何把 attainment 换算成结果表中的 Goodput，
不是对实际 attainment 的预测。

### 8.1 数据与正确性

| 项目 | CP | PD-TDM | 判定 |
|---|---:|---:|---|
| 成功请求率 | TBD | TBD | 目标均为 100%，正式 run 至少 99% |
| 输出长度 mean/p95 | TBD | TBD | 应基本一致 |
| Greedy 输出 hash match | TBD | TBD | smoke 目标 100% |
| visual tokens mean/p95 | TBD | TBD | 输入完全一致 |
| 实际 Prefill bound | TBD | TBD | 均应约为 2,048 token |

### 8.2 主结果表

| QPS | 系统 | TTFT mean/p99 ms | request-mean TPOT mean/p99 ms | Attainment % | `G_tok` tok/s | Error % |
|---:|---|---:|---:|---:|---:|---:|
| Q1 | CP | TBD | TBD | TBD | TBD | TBD |
| Q1 | PD-TDM | TBD | TBD | TBD | TBD | TBD |
| Q2 | CP | TBD | TBD | TBD | TBD | TBD |
| Q2 | PD-TDM | TBD | TBD | TBD | TBD | TBD |
| Q3 | CP | TBD | TBD | TBD | TBD | TBD |
| Q3 | PD-TDM | TBD | TBD | TBD | TBD | TBD |
| Q4 | CP | TBD | TBD | TBD | TBD | TBD |
| Q4 | PD-TDM | TBD | TBD | TBD | TBD | TBD |
| Q5 | CP | TBD | TBD | TBD | TBD | TBD |
| Q5 | PD-TDM | TBD | TBD | TBD | TBD | TBD |

其中 TPOT p99 表示“每请求平均 TPOT”的跨请求 p99，不表示所有 token gap 的 p99。主表
优先给 mean/p99 以同时展示平均体验与 tail；p50/p95、每请求 ITL-p99 的跨请求
median/p99、单 seed 数值、离散度和 paired delta 放入附录或 artifact。每个主表值报告三
seed 聚合值，并明确聚合顺序，避免把 pooled-request p99 与 per-seed p99 median 混用。

### 8.3 视觉成本分组

| Visual-token 组 | 请求数 | CP TTFT p99 | PD TTFT p99 | TTFT delta | Goodput delta |
|---|---:|---:|---:|---:|---:|
| Low（Q1） | TBD | TBD | TBD | TBD | TBD |
| Medium（Q2-Q3） | TBD | TBD | TBD | TBD | TBD |
| High（Q4） | TBD | TBD | TBD | TBD | TBD |

该表只作机制支持；若组内样本过少或 visual-token 计量不可靠，则不进入论文。

### 8.4 长输出边界

| Output | QPS | CP `G_tok` | PD `G_tok` | Goodput delta | TPOT p99 delta |
|---:|---:|---:|---:|---:|---:|
| 128 natural | Q-mid | TBD | TBD | TBD | TBD |
| 128 natural | Q-near-sat | TBD | TBD | TBD | TBD |
| 512 exact | Q-mid | TBD | TBD | TBD | TBD |
| 512 exact | Q-near-sat | TBD | TBD | TBD | TBD |

## 9. 拟设结果与论文决策规则

以下不是实测数字，而是实验前的预期形态和 Go/No-Go 规则。

### 9.1 预期曲线

| 负载区间 | 拟设结果 | 解释 |
|---|---|---|
| 低负载 | CP 与 PD-TDM 接近 | 几乎无排队，调度自由度有限 |
| 中负载 | PD-TDM TTFT tail 开始下降，TPOT 接近 | 显式阶段服务机会减少 Prefill 等待 |
| 近饱和 | PD-TDM attainment/Goodput 优势最明显 | CP 的 mixed scheduling 更易形成 tail |
| 深度过载 | 两者退化，或 PD 仅延后崩溃点 | 不能把过载失败请求包装成收益 |
| 512-token 输出 | PD 优势缩小，TPOT 压力提高 | 与 deep-decode 边界一致 |

### 9.2 内部 Go 标准

满足以下条件时，结果可作为正文约 0.2--0.3 页的 portability 证据：

1. 无 correctness 差异和系统性错误率差异；
2. 至少两个中高负载点的 paired 结果方向一致；
3. PD-TDM 的联合 SLO Goodput 或 TTFT p99 有具有实际意义的改善，例如相对改善达到约
   10% 量级，同时 TPOT p99 没有同量级反向恶化；
4. 三 seed 结果不由单个异常请求驱动；
5. 结果能由 visual-token/phase telemetry 支持，或至少不与机制叙事冲突。

“10%”只作为是否值得占用四页正文的内部门槛，不是显著性定义，也不能在实验后调整。

### 9.3 三种可能结论

**A. 明确正向：**

> On Qwen2.5-VL-7B with real-world VisionArena requests, PD-TDM improves
> joint-SLO Goodput over vLLM-Ascend CP near saturation while preserving
> comparable TPOT, demonstrating applicability beyond text-only serving.

只能在实际数据支持后填入具体百分比。

**B. 条件性正向：**只有高 visual-token 或 128-token 输出场景获益，512-token 输出持平/
反转。正文应写成“适用于视觉 Prefill 压力较高且 Decode 有余量的区域”，这与主论文边界
叙事一致，仍然有价值。

**C. 持平或负向：**不把多模态实验放进 headline。可在 limitation 中说明视觉 encoder
开销或当前 scheduler 路径限制收益；若正文空间不足则完全省略。No-Go 不影响 ICASSP
投稿资格，venue relevance 由问题 framing 和 EDICS 选择建立。

## 10. 论文呈现建议

若结果为 A/B，正文只使用：

- Evaluation setup 中一句模型/数据/平台说明；
- 一个两行紧凑表：CP vs PD-TDM 的近饱和 TTFT p99、TPOT p99 和 Goodput；
- 一句边界说明；
- 不新增完整多模态章节，不报告 VQA accuracy。

建议 claim：

> PD-TDM 的调度收益能够扩展到包含视觉编码和异构视觉 token 成本的多模态生成服务，
> 但收益仍取决于 Prefill/Decode 服务压力和双 SLO 余量。

禁止 claim：

- 提升视觉问答准确率；
- 对所有多模态、音频或视频模型普遍有效；
- 优于 GPU Sarathi-Serve；
- TTFT 的全部改善来自 encoder 调度；
- Qwen2.5-VL 的结果可直接代表 Qwen3-8B 的容量变化。

## 11. 执行清单

- [ ] 确认 Qwen2.5-VL-7B 权重、本地 processor 和 BF16 可加载。
- [ ] 固定 dataset/model revision，并 profile 全部 500 条的图像、文本和 visual tokens。
- [ ] 冻结 10 warm-up + 200 measurement 的数据 manifest。
- [ ] 验证 200 条子集相对 500 条全集的代表性。
- [ ] 顺序生成自然输出，统计 output 分布与 128-token censor rate。
- [ ] 测量 encoder/prefill/decode service-time signature，并映射 Qwen3/F5 单元。
- [ ] 完成 CP/PD-TDM 各 20 请求 correctness smoke。
- [ ] 核验两边实际 Prefill bound 与 encoder 计量边界。
- [ ] 完成容量 pilot，冻结绝对 QPS 与 SLO。
- [ ] 生成 3 个 paired Poisson arrival traces。
- [ ] 执行 V3 的 30 个主 run，并保存详细 per-request 数据。
- [ ] 聚合三 seed、paired delta 和置信区间。
- [ ] 根据 Go/No-Go 规则决定是否执行 512-token V4。
- [ ] 将所有最终数字加入 `paper/data/manifest.json`，再进入论文正文。

## 12. 参考入口

- vLLM-Ascend 支持模型：
  `docs/source/user_guide/support_matrix/supported_models.md`
- vLLM-Ascend Qwen2.5-VL serving benchmark：`benchmarks/README.md`
- vLLM 0.11.0 VisionArena loader：`vllm/benchmarks/datasets.py` 中
  `VisionArenaDataset`
- VisionArena-Bench 数据：
  <https://huggingface.co/datasets/lmarena-ai/vision-arena-bench-v0.1>
- vLLM-Ascend v0.7.3 同模型/同数据公开性能：
  <https://github.com/vllm-project/vllm-ascend/issues/776>
- vLLM Qwen2.5-VL GPU recipe：
  <https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen2.5-VL.md>
- 当前论文任务门：`paper/TASKS.md` 的 V0/T5
- 当前 Goodput 定义：`paper/sections/02_background.tex`
