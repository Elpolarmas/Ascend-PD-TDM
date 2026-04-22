# Current Tasks

> 更新日期：2026-04-22
> 本周目标（2026-04-22 → 2026-04-28）：把"专利对齐后的论文叙事 + V1 原型"一起跑通，给导师一份带 7B Qwen + V1 初步数据的增量报告。
> 详细专利进度见 `patent_task_context.md`（已完成至第十四轮，基本定稿）。

## 本周任务（对应 TaskList 中的 Day 1–7）

### ✅ Day 1 (2026-04-22) — 专利反灌 + 文件清理
- [x] 专利问题/三模块定位反灌到 `idea_proposal.md`（§1.2–§1.3、§2.2、§5.1、§6.1、§9）
- [x] 同步到 `meeting_outline.md`（§一、§四、§五小结）与 `work_report.md`（§一、§二、§五）
- [x] 同步 `tdm_related_work.md` 的独特定位与一句话概括
- [x] 文件清理：移走 cc_guide、删 results/legacy、scratch/ 清空、重复 plot 脚本与旧 json 删除（共 8 文件/目录、~600KB）
- [x] 更新本文件（task.md）为新周计划

### Day 2 (2026-04-23) — Qwen3-7B 环境 + Exp B
- [ ] Qwen3-7B 权重就位，vllm-ascend TP=2 起服验证
- [ ] 重跑 Exp B 资源画像（prefill-heavy / decode-heavy / mixed）
- [ ] 更新 `figures/fig1_exp_b_resource_profile.png`，对照写入 notes.md

### Day 3 (2026-04-24) — Exp D + Exp E3（7B 版）
- [ ] Exp D：并发 1→64 扫 TPS/TPOT/TTFT
- [ ] Exp E3：Unified TP=2 / Unified+CP / Phased / Disagg 1P1D 四方案公平对比
- [ ] 更新 fig2/fig6/fig7，对照 notes.md

### Day 4 (2026-04-25) — AscendScheduler 源码精读
- [ ] 精读 `vllm_ascend/core/scheduler.py::schedule()` 全流程
- [ ] 梳理 `_build_attn_state()` 的 P/D 判断逻辑
- [ ] 产出"V1 切入点+接口草图"写入 notes.md

### Day 5 (2026-04-26) — Phase Switching V1 原型
- [ ] 在 AscendScheduler 里实现 V1 规则版（P/D/M 三模式 + 防饿死硬约束）
- [ ] 加 `--enable-tdm` CLI 开关便于 A/B
- [ ] 单测：SLO slack 触发切换 / 防饿死生效

### Day 6 (2026-04-27) — V1 端到端基准
- [ ] V1 vs Unified+CP 在 6 种 workload × SLO 档位下的公平对比
- [ ] 收集 Goodput / SLO attainment / P99 延迟 / graph hit rate
- [ ] 产出 `results/results_v1_bench.json` + 对比图

### Day 7 (2026-04-28) — 收尾与周报
- [ ] 离线 (batch_size, phase) → latency 画像表采集（Ascend Profiler，为 V2 准备）
- [ ] 精读 PDM/Drift，补 `paper_template.md`，对照检查"跨多迭代生效的固定比例"论据是否精确
- [ ] 更新 `work_report.md` 为增量周报，带上 V1 初步数据和专利对齐后的新叙事

## Backlog（后续周期）
- V2 AIMD 自适应比例控制实现
- DCMI AICore 在线采样集成到 V2 状态反馈
- 消融实验（SLO-aware / Graph-aware / DCMI feedback 逐步移除）
- Workload 敏感性（ShareGPT / LMSYS-Chat 真实 trace）
- 大模型 / 多卡 TP 扩展性（导师 4-08 指示：当前降低优先级）
- vllm-ascend 社区 PR

## 已完成（历史快照，保留用于查阅）
- [x] 项目目录与 memory 结构搭建
- [x] vllm / vllm-ascend 代码架构初步探索
- [x] 环境安装（torch + vllm + vllm-ascend）
- [x] Baseline 实验（Qwen3-0.6B / 4B）
- [x] Exp A–F（ACL Graph / P/D 画像 / DCMI / 并发 / Phased vs Unified / Graph-Aware）
- [x] 12 篇核心论文精读 + 5 篇 TDM 通用工作对比
- [x] idea_proposal 首轮撰写 + 两轮导师讨论（4-03 / 4-08）
- [x] 专利交底书 14 轮迭代修订（含问题陈述精炼、三模块重定位）
