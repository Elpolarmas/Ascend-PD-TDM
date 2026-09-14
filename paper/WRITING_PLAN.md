# PD-TDM ICASSP 2027 四页写作蓝图

> 状态：D-020，2026-09-07。9/7开始、9/9 22:00交付完整英文R0；具体任务与验收见
> `TASKS.md`，缺口审查见 `READINESS_AUDIT_2026-09-07.md`。本文只定义结构和篇幅。

## 1. 一句话主线

> vLLM-Ascend Chunked Prefill 限制单次 Prefill 工作量，但跨 iteration 的 P/D 服务份额
> 仍隐式依赖 mixed batch。PD-TDM 保留 bounded Prefill，并以 phase-level allocator
> 显式配置阶段服务机会；后续 adaptive extension 根据 SLO pressure 更新这一倾向。
> 当前结果在 Prefill 压力高且 Decode 有 TPOT 余量时提高联合 SLO Goodput，同时在
> deep-decode 区域明确让出优势。

## 2. 直接对比范围

- 直接 baseline：`vLLM-Ascend Chunked Prefill (vLLM-Ascend CP)`。
- 同平台条件：2× Ascend 910B3、Qwen3-8B、TP=2、同一 vLLM-Ascend 版本。
- Sarathi-Serve：只在 Related Work 中作为 Chunked Prefill 技术来源，不作性能对比。
- 论文性质：Ascend 平台上的细粒度调度设计与边界评估，不声称跨 GPU 普适性。
- ICASSP venue-fit：官方 scope 已包含 Machine Learning and Generative AI 与 Applied
  Signal Processing Systems；多模态模型是条件性增强证据，不是进入会议 scope 的前提。
- 若 V0 通过，Qwen2.5-VL-7B 仅作为真实图像信号输入下的一行 portability/applicability
  结果，不取代 Qwen3-8B/T6 主证据，也不扩张为视觉 encoder 调度贡献。

## 3. 三条 claim

1. **设计：**PD-TDM 在一个共享 TP 副本内结合 bounded chunk continuation、pure-phase
   iteration 和显式阶段服务份额，保持权重/KV 本地且不依赖空间分区；adaptive update
   作为待验证扩展。
2. **主结果：**在 Azure Prefill-pressure burst workload 的严格联合 SLO 下，PD-TDM
   相对同平台 vLLM-Ascend CP 提高 attainment/Goodput。
3. **边界：**收益由 Prefill/Decode 服务压力和双 SLO 余量共同决定；宽松 SLO 出现
   ceiling，deep-decode/TPOT-tight 区域收益缩小或反转。

## 4. 四页页面预算

### Page 1：问题、gap 与贡献（约 0.9 页）

- 资源只够一个完整 TP 副本，物理 P/D 分离和设备内空间分区不可用。
- Chunked Prefill 的价值以及其“chunk bound 不等于 phase share”的缺口。
- PD-TDM 一句话方案、适用条件、三条贡献。
- Fig. 1 上半或右侧放 mixed CP vs bounded pure-phase 时间线。

### Page 2：方法（约 1.0 页）

- 两个独立旋钮：`prefill_chunk_tokens` 与 phase service share。
- pure-P/pure-D invariant、partial Prefill continuation、queue-empty fallback。
- ratio 在当前结果中只写 configured phase tendency；adaptive extension 可写设计公式，
  但不写在线最优控制或已验证收益；PID 不作为当前贡献。
- 最多保留一个简单约束或公式，不放长算法伪代码。

### Page 3：实验设置与主结果（约 1.2 页）

- 同平台公平性表述和联合 SLO/Goodput 定义。
- T6 Azure conv/code 三 seed 主结果。
- Fig. 2 以 attainment pp 为视觉中心，不用巨大相对百分比制造 headline。
- Telemetry 只证明运行行为，不归因固定 mixed tax。

### Page 4：支持、边界与结论（约 0.9 页）

- MaaS TTFT mean/P99 shift；原始宽松 SLO ceiling。
- 一个干净的 deep-decode 方向性负对照。
- Related Work 一个紧凑段落：CP、物理分离、空间/时间复用。
- 局限：单平台/模型、旧版本、F5 single seed、post-hoc MaaS sensitivity。
- adaptive-vs-fixed 数字补齐后，替换 provisional evaluation paragraph；在此之前明确其
  不支撑当前 headline。
- 两三句 Conclusion。

### Page 5：参考文献

只放会议允许的参考文献及其他非技术内容，不把实验、附录或补充讨论挤入该页。

## 5. 图表预算

1. **Fig. 1：机制图。**同一 chunk bound 下，vLLM-Ascend CP 的 mixed iteration 与
   PD-TDM 的 bounded pure-P/pure-D iteration；同时标出 chunk bound 和 phase share。
2. **Fig. 2：证据图。**主面板为 T6 conv/code 严格 SLO；小面板或紧凑表容纳 MaaS
   TTFT shift 与 deep-decode 边界。
3. **Table 1（可选）：**平台、公平配置和三个代表结果。若 Fig. 2 caption 已覆盖，则删除。

不进入正文：完整 F5 heatmap、c4 1P1D 全曲线、历史演进图、PID ablation、kernel profile。

## 6. 摘要五句结构

1. 双 SLO 与单 TP 副本资源约束。
2. Chunked Prefill 只控制单次工作量、阶段份额仍隐式的 gap。
3. PD-TDM 的 bounded pure-phase 设计。
4. T6 attainment pp 与 MaaS TTFT shift 两组数字。
5. deep-decode 边界和限定范围。

## 7. 写作红线

- 不写“PD-TDM 优于 Sarathi-Serve”；只比较 vLLM-Ascend CP。
- 不写“首次提出 temporal P/D multiplexing”。
- 不写固定 mixed-batch tax、variable-query kernel 根因或旧 FIA 消融结论。
- 不写 PID/在线控制器已经自动找到最优 ratio；adaptive extension 在结果补齐前只能作为
  待验证方法设计。
- 不把 MaaS `1.2×` post-hoc sensitivity 当成原始预设 SLO。
- 不把 F5 single-seed 负区精确百分比推广成稳定容量结论。
- 不以 `+403%/+619%` 作为摘要 headline；优先用 absolute attainment pp。

## 8. 初稿完成标准

- 严格围绕三条 claim，任一段落都能说明自己支撑哪一条。
- 所有数字来自 `paper/data/manifest.json`。
- 英文技术内容最终不超过 4 页；参考文献置于允许的第 5 页。
- 主图在双栏缩放后仍可读。
- 无旧 baseline 名称、旧控制器故事或超出现有证据的因果表述。
