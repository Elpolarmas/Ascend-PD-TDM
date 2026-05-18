# PD-TDM 项目记忆

> 新 session 从这里开始读。其它文档按需读。

## 项目一句话

在 single-node multi-NPU(2 张 Ascend 910B3,TP=2,锁 vllm-ascend v0.11.0rc1)上实现 PD-TDM(P/D Temporal Multiplexing)调度 paradigm,提供 goodput-centric improvement under SLO constraints。目标 CCF-B/C 论文。

**当前 thesis 版本:D-011(2026-05-18)**,完整 thesis 重定位记录在 `DECISIONS.md` D-011。

## 当前状态(2026-05-18 末)

- **本 session 做完(W1 第 1-2 步):**
  - **D-011 thesis 完整重定位:**
    - 双维度协同 thesis 删除(P1.9d/e 实证证伪)
    - 新 thesis = PD-TDM paradigm + SLO-aware fast-loop selector(单维度)
    - 慢回路 PID + Starvation tpot override + Graph-aware 全部退出 contribution(代码保留 internal)
    - **Paper scope = Framing C hybrid**:conceptual scope = "accelerators lacking SM partitioning support"(包括 NPU / older GPUs / 未来 ASIC,镜像 MuxWise scope 的 complement set);evaluation scope = Ascend 910B3 NPU
    - **Paper primary metric = goodput at SLO**(max sustainable QPS at SLO target ≥ X%,DistServe-compatible);secondary = SLO meet% at fixed QPS;主图 = Pareto frontier
    - **4-way paradigm comparison**:Unified TP=2 / Chunked Prefill (Sarathi) / PD-TDM (我们) / PD-Disagg 1P1D(降级 budget reference point)
    - **Interference shifting framework** 作为 conceptual unified lens
    - **MuxWise (ASPLOS'26) cite + explicit complementary positioning**(他们 explicit 排除 NPU,我们填 gap)
  - **PROJECT.md 完整重写** — 反映 D-011 新 thesis + 不能偏离的原则全更新 + 禁区扩展(慢回路 PID 不进 paper / 不实装 mixed_mode / Title 不锁 NPU)+ Paper 9-Section outline + W1-W4 timeline
  - **完整 Reviewer attack defense framework**(A1-E2 10 个 attack)在 D-011 内详细记录

- **下次 session 接 W1 第 3 步(实验启动):**
  - **#1 QPS sweep**(D-011 primary metric 数据):QPS [2,4,6,8,10,12,14,16] × 4 configs (C1/C3/M3.1/c4_pd) × 2 traces (conv/code) × 3 seeds × Qwen3-8B。1-2 天 wall,后台跑
  - **#2 跨模型 sweep**(Qwen3-4B QPS sweep,同上 matrix):2-3 天 wall。Qwen3-0.6B 系统中有但不进 paper,只 4B + 8B
  - **#3 c4_pd 补跑**(Azure trace + Qwen3-8B,跟主 evaluation 框架对齐,E3 之前是 Qwen3-4B):1-2 天
  - **#4 BurstGPT trace 加入**:写 loader + sweep,1-2 天
  - **#5 Mixed workload trace 生成 + sweep**(conv+code 同时到达):1-2 天,decide 是否扩 winning regime 到第 2 个
  - **W2 起 paper writing**(Section 1-4 先写,实验数据 ready 后写 Section 5)

- **完全删除的(D-011 退出 contribution):**
  - 慢回路 PID 闭环反馈(P1.9d B-A ≈ 0 / P1.9e -2.68pp 全证伪)
  - 双维度协同(P1.9d D ≈ max(B,C),协同效应不存在)
  - Starvation tpot override(P1.7c 触发率 0.5%,效果 ±1pp noise)
  - Graph-aware 模块(D-005 三选一定为 F only)
  - Mixed_mode 开关(D-010 立场:paradigm-level Δ 不归因单机制,不需要这个 ablation,不实装)

- **本 session 加的文件:**
  - `memory/DECISIONS.md`:加 D-011 完整记录
  - `memory/PROJECT.md`:完整重写(thesis + 原则 + 架构 + 代码地图 + 禁区 + paper outline + timeline)
  - `memory/README.md`:本文,反映 D-011

## 不能偏离的几条原则(D-011 修订)

任何工作方向跟下面冲突 → 先在 `DECISIONS.md` 写一条新决策再做。

1. **thesis = PD-TDM paradigm + SLO-aware fast-loop selector**(单维度,SLO-aware)。**不能**再 claim「双维度协同」或「闭环反馈控制器」。慢回路 PID 代码保留 internal,paper 主体不提
2. **主报 metric = goodput at SLO**(max sustainable QPS at SLO ≥ X%),Pareto frontier 为主图。SLO meet% 是 secondary
3. **Paper scope = Framing C hybrid**:conceptual = "accelerators lacking SM partitioning support";evaluation = Ascend 910B3 NPU。**Title 不锁 NPU**
4. **对手 = 4-way paradigm**:C1 / C3 / M3.1 / c4_pd(1P:1D 是 budget reference,不 claim 普遍反优势)。MuxWise 在 Related Work cite,但 scope explicit complementary 不直接对比
5. **vllm-ascend 锁 v0.11.0rc1**(D-007)
6. **kernel 速度不作为论文主图**(D-009,同 kernel ablation 已实证贡献 ≈ 0)
7. **实验:QPS sweep + 校准 SLO grid + Qwen3-4B/8B cross-model + Azure/BurstGPT/mixed cross-trace**

## 最近 3 条决策

- **D-011**(5/18 本次)thesis 完整重定位:PD-TDM paradigm + SLO-aware fast-loop selector + Framing C scope + goodput-centric metric + 4-way paradigm comparison + interference shifting framework + MuxWise complementary positioning。完整 reviewer attack defense framework 在 D-011 内
- **D-010**(5/15)phase Δ 归因层级修正:paradigm-level Δ 不归因单机制,bundle vs bundle 总差;C3 降为外部 reference point。Mixed_mode 开关需求被 D-011 撤销(D-011 立场延续 D-010,不需要单变量 isolation)
- **D-009**(5/14)dedicated kernel 完全移出 contribution,phase-pure 双向化

完整决策史见 `DECISIONS.md`。

## 文档导航

| 想知道 | 读哪里 |
|---|---|
| 项目核心 / 架构 / Paper outline / 禁区 / W1-W4 timeline | `PROJECT.md` |
| **代码地图 / TDM 插件位置 / 关键文件** | `PROJECT.md` §5 |
| **config 缩写(C1/C3/c4_pd/M3.1)+ M 家族演化** | `EXPERIMENTS.md` §2、§4 |
| 17+ 个 `results/` 目录用途 / 实验代次时间线 | `EXPERIMENTS.md` |
| Finding 链 / 设计与实测的差距 / 已废弃路径 | `FINDINGS.md` |
| Thesis 调整历史(D-011 当前)| `DECISIONS.md` |
| 慢回路 PID 细节(internal,paper 不写)| `design/slo_pid.md` |
| 采样模块(降级为 evaluation methodology)| `design/sampling.md` |
| Chunking 机制 + p1_chunk_scan 实测 | `design/chunking.md` |
| 状态采集接口(infrastructure,paper Section 4)| `design/interfaces.md` |
| 论文叙事 / 对照矩阵 / threats to validity | `design/paper.md`(W2 起重写)|

## 做完事要更新哪些文档

- **跑完实验:** `EXPERIMENTS.md` §2 主表加一行(目录 / 代次 / 主 finding / thesis 用途)
- **实测推翻原设计:** `FINDINGS.md` § 设计与实测的差距 加一条 Gap-X,并写一条 `DECISIONS.md` D-XXX
- **改 thesis 方向 / 不能偏离的原则:** **必须先**写 `DECISIONS.md` D-XXX,才能动其他文档
- **删 / 废弃实验目录:** `EXPERIMENTS.md` § 已废弃实验加一行
- **接口 / 代码地图变化:** `PROJECT.md` §5 + 相关 `design/*.md`

---

**给新 session 的关键 hand-off:**
1. 读 `PROJECT.md` 看 thesis(D-011)+ Paper outline + W1-W4 timeline
2. 读 `DECISIONS.md` D-011 看 reviewer attack defense 完整 framework
3. W1 第 3 步是启动实验(QPS sweep / c4_pd 补跑 / 跨模型 / BurstGPT / mixed workload),具体在 PROJECT.md §9 或 README 当前状态节
4. **不要回退 D-011 的任何决定**(慢回路 / 双维度协同 / Mixed_mode 等)— 这些都是 P1.9d/e 实证 + reviewer attack 系统分析后决定的
