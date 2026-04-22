# System Context

## Project Overview
论文创新项目：在 vllm-ascend 基础上实现单张 NPU 卡上的 Prefill-Decode (PD) 混合部署。
- 混合部署包括：时分复用（time-division multiplexing）和空分复用（space-division multiplexing）
- 当前聚焦：时分复用，因为 NPU 环境上还没有成熟的单卡内空间切分技术
- 目标：算法改进 + 论文发表
- 近期目标：3 天内（2026-04-03 前）完成可讨论的 idea 方案

## Key Repositories
- `/vllm-workspace/vllm` — vLLM 主仓库（上游）
- `/vllm-workspace/vllm-ascend` — vLLM Ascend NPU 插件（华为昇腾后端）

## Environment
- Hardware: Ascend NPU (Atlas series)
- Python: >= 3.9, < 3.12
- CANN: >= 8.3.rc1
- PyTorch: 2.7.1 + torch-npu 2.7.1

## Research Direction
- PD 分离/混合部署是 LLM 推理优化的热点方向
- 业界方案多针对多卡/多节点场景（如 DistServe, Splitwise, TetriInfer）
- 本项目创新点：单卡 NPU 上的时分复用 PD 混合部署
- 老师要求先聚焦时分复用

## Key Code Paths
- AscendScheduler: `vllm_ascend/core/scheduler.py`
- AscendSchedulerConfig: `vllm_ascend/core/schedule_config.py`
- NPUModelRunner: `vllm_ascend/worker/model_runner_v1.py`
- AscendAttentionState: `vllm_ascend/attention/attention_v1.py`
- 上游 Scheduler: `vllm/v1/core/sched/scheduler.py`
