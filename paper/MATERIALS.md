# PD-TDM 论文材料索引

> D-020审查提示（2026-09-07）：最终结构/时间以TASKS与WRITING_PLAN为准。
> 下文Sarathi数据标签指本地vLLM-Ascend CP；历史六图候选不等于最终图表预算。
> F5统一离线绘图已存在于experiments/plot_paper_figures_offline.py，但仍待统计审计；
> paper/data/figure_metrics.json不能替代manifest。MaaS时长与分母风险详见本轮审查报告。

本文档回答三个问题：哪些材料能直接用于论文、哪些材料仍需处理、哪些材料只用于
理解项目历史。论文中的数字应以这里列出的原始结果和后续 `manifest` 为准，而不是
从 memory 中手工复制。

## 1. 事实优先级

当不同文档发生冲突时，采用以下顺序：

1. 当前代码、原始 JSON/JSONL 和可复现分析脚本；
2. `results/maas_replay/V12_REPORT.md`、F5b/F5c/F5d 等较晚产物；
3. `memory/FINDINGS.md` 最新条目和 `DECISIONS.md` D-015/D-014/D-013；
4. `memory/T6_FINDINGS.md`；
5. `memory/PROJECT.md`、`memory/README.md`；
6. `memory/design/` 仅作为设计演进记录，不作为当前论文口径。

## 2. A 级：可作为论文主要证据

| 主题 | 原始材料 | 当前可支持内容 | 尚缺处理 |
|---|---|---|---|
| T6 主结果 | `results/m31fix_validate/`、`results/c3_chunk2048_supplement/`、`results/phase_2_t6_burst_goodput/` | Azure burst 下 Vanilla CB、Sarathi、PD-TDM 的 Pareto、SLO attainment 和 latency | 审核聚合脚本与最终图，生成 paper manifest |
| T6 聚合 | `results/phase_2_post/m31fix_phase1_pointwise.json` | 3-seed 中位数、4-way 对比 | 逐项校验 config 名和 SLO 定义 |
| Iteration telemetry | `results/c3_telemetry/`、`results/m31fix_validate/*/tdm_trace/` | mixed composition、pure-P/pure-D interval、phase invariant | 只作阶段行为证据；样本并非等工作量，不估计固定 mixed tax |
| Workload 边界 | `results/f5_decode_heavy_synth/`、`results/f5b_decode_heavy_long_prompt/` | balanced 胜区和 deep-decode 反例 | 统一 workload/SLO 口径 |
| 二维矩阵 | `results/f5c_prefill_balanced_supp/`、`results/f5d_workload_matrix/` | prompt/output 二维覆盖；F5d 为 8 workload × 2 paradigm × 5 QPS | 尚无正式 post-hoc、heatmap 和多 seed |
| MaaS v12 | `results/maas_replay/maas_{pdtdm,sarathi}_rs0.10.json`、`V12_REPORT.md` | 35,441 请求/范式、0 error；TTFT 分布稳定左移 | 生成 CDF；明确原始 ×3 SLO 与 post-hoc ×1.2 sensitivity |

## 3. B 级：支持性证据

| 材料 | 用途 | 论文中的限制 |
|---|---|---|
| `results/phase_a_micro_ideal/` | 建立 relative-to-ideal SLO | 必须说明 model/workload/hardware 条件 |
| `results/c4_pd_supplement/` | 2-NPU 预算下的 1P1D reference | 不能推广成“disaggregation 普遍更差” |
| `results/c4_pd_steady_smoke/` | 排除 burst 是 c4_pd 唯一失败原因 | 仅 3 个单 seed cell |
| Vanilla CB 数据 | 复现长 prompt 上 chunked prefill 的价值及短 prompt 反例 | PD-TDM 相对 Vanilla 的收益必须给 Sarathi chunking 正确归功 |
| 早期硬件 profile figures | 解释 P compute-bound、D memory-bound | 原始数据已删，不适合作为核心定量结果 |

## 4. C 级：仅用于项目历史和写作背景

- `results/azure_main/`、`results/azure_p15/`：早期严格 SLO 和旧 framing；仅用于理解研究演进。
- PID、urgency/starvation、graph-aware 系列：用于解释为何这些机制不进入贡献。
- `memory/design/paper.md`：保留大量 D-011/D-012 旧叙事，不能直接移入新论文。
- C1 baseline：可用于内部诊断，不进入论文主 baseline 组。

## 5. D 级：禁止用于论文结论

- 所有带 `chunked_schedule` waiting-loop bug 的旧 m31 结果；论文只能引用
  `results/m31fix_validate/` 中的修复版 T6 数据。
- “PID 是性能来源”“双回路协同”“tail 换 mean”“优势只由 prompt:output ratio
  决定”“dedicated kernel 是主要收益来源”等已被后续实验推翻的表述。
- 把约 15% 差值表述为已证实的 mixed-iteration tax，或归因于 variable-query attention
  kernel；当前证据还没有完成这种单变量隔离。
- 把 MaaS ×1.2 post-hoc goodput 当作原始 ×3 SLO 实验结果。

## 6. 代码到论文章节的映射

| 代码/资产 | 论文内容 |
|---|---|
| `vllm_ascend/core/tdm/scheduler.py`、`chunked_schedule.py` | phase-pure scheduling 与 parent scheduler 集成 |
| `vllm_ascend/core/tdm/chunking.py` | bounded prefill work |
| `vllm_ascend/core/tdm/selector.py` | token bucket 与 TTFT urgency |
| `monitor.py`、`tracker.py`、`telemetry.py` | 实现与测量基础设施 |
| `controller.py` | internal implementation；不列为贡献 |
| `experiments/run_qps_sweep_all.py`、`qps_sweep.py` | evaluation harness |
| `experiments/lib/workload.py` | Azure、synthetic、MaaS workload generation/replay |

## 7. 最终图表候选

| 编号 | 内容 | 数据源 | 状态 |
|---|---|---|---|
| Fig. 1 | Mixed batching、PD-TDM、spatial/disagg 范式图 | 设计图 | 未制作 |
| Fig. 2 | Chunk bound、ratio 与双 SLO 的机制关系 | design/config | 未制作 |
| Fig. 3 | T6 goodput/SLO Pareto | T6 aggregate | 有旧 PNG，待审核并转 PDF |
| Fig. 4 | prompt × output 二维 winning-region heatmap | F5--F5d | 未分析 |
| Fig. 5 | MaaS TTFT/TPOT CDF | MaaS v12 raw JSON | 未制作 |
| Fig. 6 | MaaS SLO sensitivity | MaaS v12 post-hoc | 报告有表，未制作 |
| Table 1 | 平台、模型、trace、SLO、baseline setup | 多处 | 未冻结 |
| Table 2 | 关键主结果和反例 | T6/F5b/MaaS | 未冻结 |

## 8. 待建立的权威产物

- `paper/data/manifest.json`：每个最终数字的来源、commit、seed、SLO 和分析规则。
- `experiments/posthoc_f5_matrix.py`：统一处理 F5/F5b/F5c/F5d。
- `experiments/posthoc_iteration_telemetry.py`：复算阶段组成和 pure-D inter-arrival，不进行固定 tax 估算。
- `paper/figures/*.pdf`：仅存最终论文图。
- 当前紧凑结果表直接维护在 `sections/05_evaluation.tex`；若后续改为脚本生成，来源与
  命令必须写入 manifest。

## 9. 服务器维修期间的材料边界

当前论文只使用仓库内已有数据，不以任何新 NPU run 为前置条件。原计划中的 chunk
sensitivity、多 seed、Qwen3-4B 和 controlled mechanism isolation 统一列为 deferred。

初稿应让证据层级可见：T6 是 3-seed 主证据；F5 系列是单 seed exploratory boundary；
MaaS v12 是大规模但单 seed、聚合 trace 重建的 supporting evidence。无法补强的部分通过
claim 限定和 limitations 处理，不引用已失效或带 bug 的历史实验代替。

## 10. Related Work 来源

- 用户综述：`/home/north/workspace/PD综述（计算机研究与发展）.pdf`。用于建立 PD 耦合、
  P/D 分离、设备内动态混部及代表系统的候选集合，不直接替代 primary-source citation。
- 已逐项核验的论文元数据与范式判定记录在 `paper/LITERATURE_AUDIT.md`；BibTeX 只收录
  已核对论文主页、会议页面、DOI 或 arXiv 最新版本的条目。
- 当前最接近的时分工作是 EcoServe（OSDI 2026）：同实例 temporal disaggregation，
  依赖多实例 rolling activation。PD-TDM 不主张发明时间分相，而强调 Ascend 平台动机、
  单 TP 副本的迭代级 pure-phase scheduling 和 workload boundary。
