# PD-TDM Related Work 核验记录

> 2026-09-07关键最近邻复核：EcoServe的OSDI2026正式条目与现有BibTeX一致；
> Sarathi-Serve继续只作技术来源，直接实验baseline为vLLM-Ascend CP。
> 本轮补充与novelty风险见READINESS_AUDIT_2026-09-07.md；不代表已穷尽新增文献。

更新时间：2026-08-04。本文档区分“综述提供的候选线索”和“已核验的一手来源”。

## 1. 来源与使用原则

- 候选文献主要来自用户撰写的《大语言模型推理服务中的 Prefill-Decode 调度机制综述》：
  `/home/north/workspace/PD综述（计算机研究与发展）.pdf`。
- 综述用于构建检索集合；论文中的事实和 BibTeX 以会议官网、ACM/IEEE DOI 页面或 arXiv
  最新版本为准。
- 综述 PDF 的文档元数据标题误写为“一种新的图像置乱算法”，不影响正文内容，但后续正式
  归档时建议修正元数据。

## 2. 与 PD-TDM 的关系矩阵

| 路线 | 代表工作 | P/D 是否同时执行 | 资源边界 | 与 PD-TDM 的核心区别 |
|---|---|---:|---|---|
| mixed iteration | Sarathi-Serve | 同一 forward 内混合 | token budget | PD-TDM 保留 chunk bound，但拆成相邻 pure-P/pure-D iteration |
| 物理 P/D 分离 | Splitwise、DistServe、Mooncake | 是，不同设备/池 | 机器或资源池 | 需要 KV 迁移与多副本；PD-TDM 在单 TP 副本内本地保存状态 |
| 设备内空间复用 | MuxWise、semi-PD、Nexus | 是 | SM/计算分区 | 依赖 GPU 可控分区和并发执行；PD-TDM 任一时刻只运行一个阶段 |
| 时空联合复用 | Bullet、DuetServe | 条件性/动态并发 | SM + 时间策略 | 以空间隔离或并发为核心机制之一；PD-TDM 是纯时间分相 |
| 多模型时空复用 | MuxServe | 视放置方案而定 | 多模型、多 GPU | 优化多模型放置；PD-TDM 聚焦单模型副本内部 P/D |
| 同实例时间分离 | EcoServe/PaDG | 单实例内否；跨实例滚动 | 长阶段窗口 + macro-instance | 最近邻；EcoServe 用多实例滚动保障 Prefill 可用性，PD-TDM 在单 TP=2 副本内用 bounded、iteration-level slices |

## 3. 关键新颖性结论

1. 不得声称“首次提出 P/D 时分复用”或“首次在同一实例上分时执行 P/D”。EcoServe 已明确
   将 P/D 角色映射到同一实例的不同时间窗口。
2. PD-TDM 当前最有防御力的差异是三项组合：
   - 额外设备和设备内空间分区不可得时，单个完整 TP 副本的双 SLO 配置问题；
   - bounded chunk continuation、phase-pure iteration 与显式阶段份额的结合；
   - 同时报告 Prefill-pressure 胜区、SLO ceiling 和 deep-decode 反例的边界研究。
3. EcoServe 面向 30B/70B 商品 GPU 集群，刻意延长 P/D 阶段并依赖多实例 rolling
   activation；其论文报告单实例时 PaDG 会退化为频繁切换的 NoDG。PD-TDM 恰在 8B
   单 TP 副本上研究细粒度分相；当前证据支持端到端 SLO 收益，但尚不支持把收益归因为
   固定的 NPU mixed-execution tax。

## 4. 已核验条目

`references.bib` 当前包含 17 项已核验条目：Orca、vLLM、DeepSpeed-FastGen/SplitFuse、
Sarathi-Serve、TetriInfer、Splitwise、DistServe、P/D-Serve、DéjàVu、Mooncake、MuxServe、
MuxWise、semi-PD、Bullet、Nexus、DuetServe 和 EcoServe。正文采用“系统名/论文名 +
作者年份引用”的文字形式，避免只在段末堆叠编号。投稿前仍需检查 arXiv 工作是否出现
正式会议版本，并按目标会议模板更新条目。
