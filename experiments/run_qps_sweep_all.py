"""Phase 1 编排脚本：一键跑完 (config × qps) 全套 QPS-sweep。

工作流（每个 config）：
  1. cleanup 该 config 上次的 tracker jsonl
  2. 起 vllm serve（注入对应 TDM additional_config + run_id）
  3. 等 /health ready
  4. 顺序对 5 个 QPS 点调 qps_sweep.py
  5. SIGTERM 停 server，等进程退干净
切换下个 config 重复，全部跑完聚合写 summary。

用法：
  python run_qps_sweep_all.py                          # 默认全套
  python run_qps_sweep_all.py --dry-run                # QPS=2 / 20s / 单 config 验编排
  python run_qps_sweep_all.py --configs c2_tdm         # 只跑某个 config
  python run_qps_sweep_all.py --qps 4,16               # 自定义 QPS 点

设计要点：
- 每 config 一个 server（5 个 QPS 共享），同 config 内 tracker 记录靠 server_req_id
  精确归属到不同 QPS 点（详见 lib/metrics.py），不依赖时间窗
- driver 通过 subprocess 调用，独立进程，确保 asyncio 循环、httpx 连接池干净
- try/finally 保证 server 异常时被 SIGKILL，不留僵尸 NPU 占用
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

# 让 `from lib.workload import ...` 可用（与 qps_sweep.py 同级 import）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.workload import (  # noqa: E402
    DEFAULT_PROMPT_PROFILE,
    PROMPT_PROFILES,
    resolve_prompt_profile,
)

EXPERIMENTS_DIR = Path("/vllm-workspace/Ascend-PD-TDM/experiments")
RESULTS_ROOT = Path("/vllm-workspace/Ascend-PD-TDM/results")
TRACE_DIR = RESULTS_ROOT / "tdm_trace"
SERVER_LOG_DIR = Path("/tmp/vllm_server_logs")

MODEL_PATH = "/vllm-workspace/models/models/Qwen3-8B"
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"
PYTHON = sys.executable

# c4_pd（PD 分离）专用常量
PD_PREFILL_PORT = 9001
PD_DECODE_PORT = 9002
PD_KV_PORT = 20001
PD_RANKTABLE = "/vllm-workspace/Ascend-PD-TDM/ranktable.json"
PD_PROXY_SCRIPT = ("/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/"
                   "load_balance_proxy_server_example.py")
PD_HOST_IP = "172.17.0.3"  # eth0 IP，ranktable.json 里的 server_id
PD_NIC = "eth0"

DEFAULT_QPS_POINTS = [4, 8, 16, 32, 64]
DEFAULT_DURATION = 90.0
DEFAULT_WARMUP = 30.0
SLO_TTFT_MS = 500.0
SLO_TPOT_MS = 50.0

# workload 默认形状：output 维度沿用旧默认（median ~150 token），
# prompt 维度由 `--prompt-profile` 选定（默认 "short" 等价旧 4.5/1.0/10/1024）。
# 完整 profile 见 lib/workload.PROMPT_PROFILES。
DEFAULT_OUTPUT_MU = 5.0
DEFAULT_OUTPUT_SIGMA = 0.8
DEFAULT_OUTPUT_MIN = 16
DEFAULT_OUTPUT_MAX = 512


def build_additional_config(config_name: str, run_id: str,
                            slo_ttft_ms: float = SLO_TTFT_MS,
                            slo_tpot_ms: float = SLO_TPOT_MS,
                            slo_target_violation_rate: float | None = None,
                            slo_ratio_max: float | None = None) -> dict:
    # P1.6e/f/g: slo_*_ms + PID inner knobs are workload-conditional. They
    # propagate into TDMConfig via `common` so every M-class config inherits.
    common = {
        "telemetry_enabled": True,
        "telemetry_dir": str(TRACE_DIR),
        "run_id": run_id,
        "slo_ttft_ms": slo_ttft_ms,
        "slo_tpot_ms": slo_tpot_ms,
    }
    if slo_target_violation_rate is not None:
        common["slo_target_violation_rate"] = slo_target_violation_rate
    if slo_ratio_max is not None:
        common["slo_ratio_max"] = slo_ratio_max
    if config_name == "c1_baseline":
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {"enable_tdm": False, "passive_tracker": True, **common},
        }
    if config_name == "c2_tdm":
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                **common,
            },
        }
    if config_name == "c2_tdm_m1_chunk2048":
        # Ablation cell: M1 static ratio (no SLO PID feedback) + chunk=2048.
        # Tests whether the SLO PID controller (M2.7) adds value on top of
        # the chunking foundation, vs. just having phase-pure toggling with
        # static ratio.
        #   delta(m31_2048, m1_chunk2048) = SLO PID 增量价值 vs static ratio
        #   delta(m1_chunk2048, c1_baseline) = phase-pure toggling + chunk
        #                                       的总价值（无 PID）
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "static",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "prefill_chunk_tokens": 2048,
                **common,
            },
        }
    # 3a target_ratio static scan variants:M1 + chunk=2048 + 不同 static_ratio。
    # 用于验证 target_ratio 这个 actuator 的 trade-off 形状(U 型 / 单调 / 平台)。
    # 是 PID 设计成不成立的前提实验 — §3.2 元教训"做 dynamic-X 之前先 static-X
    # scan"应用到 controller 自己身上。c2_tdm_m1_chunk2048(r=0.3)是已有锚点。
    _m1_chunk2048_ratio_variants = {
        "c2_tdm_m1_chunk2048_r01": 0.1,
        "c2_tdm_m1_chunk2048_r05": 0.5,
        "c2_tdm_m1_chunk2048_r07": 0.7,
        # P1.9d 双维度协同 ablation (2026-05-17):r08 对应 M3.1 在 saturated
        # 上 PID 收敛到的稳态 ratio_max=0.8。是"只快(static + bucket)"的
        # C 配置锚点 — 跟 M3.1 比能看慢回路 dynamic 调整在稳态有无 Δ。
        "c2_tdm_m1_chunk2048_r08": 0.8,
        "c2_tdm_m1_chunk2048_r09": 0.9,
    }
    if config_name in _m1_chunk2048_ratio_variants:
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "static",
                "static_ratio": _m1_chunk2048_ratio_variants[config_name],
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "prefill_chunk_tokens": 2048,
                **common,
            },
        }
    # P1.9d 双维度协同 ablation cells (2026-05-17)。bucket_cap=1 把快回路
    # 的 burst 能力去掉,等价于"慢回路调 ratio,快回路退化为严格按 ratio 平
    # 滑分配"。组合:
    #   A 都没有:c2_tdm_m1_r08_cap1   (static + cap=1)
    #   B 只慢:  c2_tdm_m31_2048_cap1 (PID + cap=1)
    #   C 只快:  c2_tdm_m1_chunk2048_r08 (static + cap=4 = M1@ratio_max)
    #   D 慢+快: c2_tdm_m31_2048      (PID + cap=4 = M3.1 原版)
    # 协同 Δ = D - max(B, C),稳态 saturated 上预期 ≈ 0(P1.9a 实证 PID 钉
    # ratio_max,跟 static@0.8 等价)。这次实验是直接锁这个结构性结论。
    if config_name == "c2_tdm_m1_r08_cap1":
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "static",
                "static_ratio": 0.8,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "prefill_chunk_tokens": 2048,
                "bucket_cap": 1,
                **common,
            },
        }
    if config_name == "c2_tdm_m31_2048_cap1":
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": True,
                "slo_tpot_saturation_min_ticks": 2,
                "slo_pid_relu_err": True,
                "prefill_chunk_tokens": 2048,
                "bucket_cap": 1,
                **common,
            },
        }
    if config_name == "c2_tdm_m2":
        # M2: c2_tdm + SLO-adaptive PID controller. static_ratio=0.3 同时作为
        # PID 的 initial_ratio。其余 SLO knob 用 TDMConfig 默认（kp=0.5,
        # ema_alpha=0.3, deadband=0.02, ratio_min=0.05, ratio_max=0.8,
        # target_violation_rate=0.05, min_samples=16, slo_ttft_ms=500,
        # slo_tpot_ms=50）。
        # Starvation guard 显式禁用 (slo_starvation_min_ticks=10**9), backlog
        # 项 kp_q=0 显式禁用——保留 M2 旧 sweep 数据语义。
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": False,
                "slo_pid_relu_err": False,
                **common,
            },
        }
    if config_name == "c2_tdm_m21":
        # M2.1: c2_tdm_m2 + starvation guard 启用（starvation_min_ticks=3）。
        # release_margin=0.0 显式锁定 M2.1 旧行为（ratio 升回 floor 立即解除）。
        # kp_q=0 显式禁用 backlog 项，保留 M2.1 旧 sweep 数据语义。
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_release_margin": 0.0,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": False,
                "slo_pid_relu_err": False,
                **common,
            },
        }
    if config_name == "c2_tdm_m22":
        # M2.2: M2.1 + hysteresis release。release_margin=0.10 默认，guard
        # 一旦激活只有 ratio 升过 ratio_min+0.10 才解除，打破 M2.1 观察到的
        # activate↔release 快速振荡（M2.1 在 R3 实测 ttft p99 +113ms vs M2，
        # ttft p50 +29ms，根因是 PID 在 guard 解除后立即把 ratio 拉回 floor）。
        # kp_q=0 显式禁用 backlog 项（保留 M2.2 sweep 数据语义）。
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": False,
                "slo_pid_relu_err": False,
                **common,
            },
        }
    if config_name == "c2_tdm_m23":
        # M2.3: M2 PID + backlog-aware additive term。kp_q=0.5 启用，guard 显式
        # 禁用（starvation_min_ticks=10**9）以隔离 backlog 信号效果。M2.1/M2.2
        # 实测发现纯 PID 在稳态把 ratio 钉在 floor（≥94% iters 在 ratio_min=0.05），
        # 因为 SLO 违例率信号在稳态衰减到 0。backlog 项以"队首请求已用 SLO 预算
        # 的比例"为 leading indicator，让 ratio 在 SLO 真正违例前就抬起。
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_tpot_saturation_enabled": False,
                "slo_pid_relu_err": False,
                **common,
            },
        }
    if config_name == "c2_tdm_m24":
        # M2.4: M2 PID + tpot saturation detector。检测到 tpot SLO 物理不可达
        # （tpot_violation_rate > target 持续 5 个 update tick）→ 把 err_tpot
        # 归零，让 PID 不再徒劳压低 ratio。Hysteresis 释放（tpot 回到 target
        # 之下持续 5 tick）。M2.1/M2.2/M2.3 实测共同失败原因：tpot 是硬件天花板
        # 永远违例，PID 持续把 ratio 拉到 floor。M2.4 直接对症：识别"信号不可
        # 信"并屏蔽。
        # 隔离实验：guard 关，backlog 关，仅 saturation 检测启用。
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": True,
                "slo_pid_relu_err": False,
                **common,
            },
        }
    if config_name == "c2_tdm_m25":
        # M2.5 = M2.4 saturation + ReLU err clipping。M2.4 实测显示即使
        # saturation 关掉 err_tpot，ttft 表现宽裕时 err_ttft = -target 仍把
        # ratio 慢慢拉到 floor。ReLU 修正 err 公式让"满足的 SLO"不再产生反向
        # 推力。预期：tpot 物理饱和 + ttft 宽裕的稳态下，controller 优雅退化为
        # 静态 ratio（保留 initial=0.3，等价 M1 行为）；任一 SLO 真违例时正常
        # adaptive。
        # 隔离实验：guard 关，backlog 关，saturation 启用，ReLU 启用。
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": True,
                "slo_pid_relu_err": True,
                **common,
            },
        }
    if config_name == "c2_tdm_m26":
        # M2.6 = M2.5 + saturation_min_ticks=1: latch saturation on the very
        # first warm window (16 tpot samples) showing persistent violation.
        # Motivation: M2.5 trace shows PID decays ratio to floor in 2 update
        # ticks once warm, but default min_ticks=5 lets that decay finish
        # before saturation engages. min_ticks=1 latches before PID can pull
        # ratio off initial. Release hysteresis (5 ticks) unchanged.
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": True,
                "slo_tpot_saturation_min_ticks": 1,
                "slo_pid_relu_err": True,
                **common,
            },
        }
    if config_name == "c2_tdm_m27":
        # M2.7 = M2.5 + saturation_min_ticks=2: 2-tick confirmation before
        # latching. Compromise between M2.6 (immediate) and M2.5 (5-tick).
        # Expected: ratio decays one update tick to ~0.16 before saturation
        # locks it; better than M2.5's "decays to floor" but worse than
        # M2.6's "holds at 0.30".
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": True,
                "slo_tpot_saturation_min_ticks": 2,
                "slo_pid_relu_err": True,
                **common,
            },
        }
    # M3.1: M2.7 + prefill chunking via the chunked_schedule fork. The
    # *_<size> sister configs share everything except prefill_chunk_tokens,
    # used for the chunk-size scan. c2_tdm_m31 (no suffix) defaults to 512
    # for back-compat with the first M3.1 sweep run on 2026-05-08.
    M31_CHUNK_SIZES = {
        "c2_tdm_m31": 512,
        "c2_tdm_m31_1024": 1024,
        "c2_tdm_m31_2048": 2048,
        "c2_tdm_m31_4096": 4096,
        # Ablation: chunk_tokens > max_num_batched_tokens (8192). chunk_budget
        # never binds — fork code path runs but truncate semantics never fire.
        # Isolates the fork-path scheduling effect from the chunking effect:
        #   delta(m31_disabled, m27)    = fork-path effect
        #   delta(m31_2048, m31_disabled) = pure chunking effect
        "c2_tdm_m31_disabled": 100000,
    }
    if config_name in M31_CHUNK_SIZES:
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": True,
                "slo_tpot_saturation_min_ticks": 2,
                "slo_pid_relu_err": True,
                "prefill_chunk_tokens": M31_CHUNK_SIZES[config_name],
                **common,
            },
        }
    # c2_tdm_m31_fia: 跟 c2_tdm_m31_2048 完全一致,只是 attention 强制走 FIA
    # (_forward_v1_style)。用于 ablation 拆分:
    #   delta(m31_2048 - m31_fia)  = dedicated kernel 速度贡献
    #   delta(m31_fia - c3_cp)     = phase-pure 调度策略本身的贡献
    # 在 v0.11.0rc1 上预演 vllm-ascend v0.13+ 把 attention 统一到 FIA 的行为。
    if config_name == "c2_tdm_m31_fia":
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": True,
                "slo_tpot_saturation_min_ticks": 2,
                "slo_pid_relu_err": True,
                "prefill_chunk_tokens": 2048,
                "force_fia_attention": True,
                **common,
            },
        }
    # M3.2: M3.1 + P1.7 TTFT-urgency fast loop. Motivation: P1.5 cross-window
    # showed M3.1 vs M1+chunk delta collapsed to ~0 on saturated code traces
    # under strict TPOT — the slow PID loop cannot react inside a single SLO
    # budget. The urgency override rescues head-of-line waiting requests when
    # oldest_age_ms >= 0.8 * slo_ttft_ms. Threshold/charge knobs reuse defaults
    # from TDMConfig so tuning happens by varying these via separate config
    # names if needed.
    M32_CHUNK_SIZES = {
        "c2_tdm_m32_2048": 2048,
        # Sister scan slots reserved; add when threshold/charge variants needed.
    }
    if config_name in M32_CHUNK_SIZES:
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": True,
                "slo_tpot_saturation_min_ticks": 2,
                "slo_pid_relu_err": True,
                "prefill_chunk_tokens": M32_CHUNK_SIZES[config_name],
                # P1.7 urgency fast loop
                "urgency_ttft_enabled": True,
                "urgency_ttft_threshold": 0.8,
                "urgency_charge_bucket": True,
                **common,
            },
        }
    # M3.3: M3.1 + P1.7b bidirectional fast loop (urgency_ttft +
    # starvation_tpot). Companion to M3.2 (urgency only). Threshold variants
    # _starv{2,3,5} = starvation_tpot_threshold ∈ {2.0, 3.0, 5.0} (units of
    # slo_tpot_ms). Default _2048 uses 3.0.
    M33_VARIANTS = {
        "c2_tdm_m33_2048":       {"chunk": 2048, "starv": 3.0, "decouple": False},
        "c2_tdm_m33_2048_starv2": {"chunk": 2048, "starv": 2.0, "decouple": False},
        "c2_tdm_m33_2048_starv5": {"chunk": 2048, "starv": 5.0, "decouple": False},
        # P1.7b decouple ablation (2026-05-17). Drops the `tokens >= 1.0`
        # AND-gate in selector starvation check so silence>=threshold alone
        # fires rescue. Same starv values as the gated variants for direct
        # comparison; bucket still NOT debited on rescue.
        "c2_tdm_m33_2048_decouple":       {"chunk": 2048, "starv": 3.0, "decouple": True},
        "c2_tdm_m33_2048_decouple_starv2": {"chunk": 2048, "starv": 2.0, "decouple": True},
        "c2_tdm_m33_2048_decouple_starv5": {"chunk": 2048, "starv": 5.0, "decouple": True},
    }
    if config_name in M33_VARIANTS:
        v = M33_VARIANTS[config_name]
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
            },
            "tdm": {
                "enable_tdm": True,
                "controller_kind": "slo_pid",
                "static_ratio": 0.3,
                "min_slice_iters": 2,
                "max_slice_iters": 8,
                "kv_free_watermark": 0.05,
                "initial_phase": "prefill",
                "slo_starvation_min_ticks": 10**9,
                "slo_pid_kp_q": 0.0,
                "slo_tpot_saturation_enabled": True,
                "slo_tpot_saturation_min_ticks": 2,
                "slo_pid_relu_err": True,
                "prefill_chunk_tokens": v["chunk"],
                # P1.7 urgency_ttft + P1.7b starvation_tpot bidirectional
                "urgency_ttft_enabled": True,
                "urgency_ttft_threshold": 0.8,
                "urgency_charge_bucket": True,
                "starvation_tpot_enabled": True,
                "starvation_tpot_threshold": v["starv"],
                "starvation_decouple_bucket": v["decouple"],
                **common,
            },
        }
    if config_name == "c3_cp":
        # Chunked Prefill baseline: 仍走 TDMScheduler 但 passive 模式 +
        # enable_chunked_prefill=True。TDMScheduler.schedule() (passive)
        # → AscendScheduler.schedule() 见 chunked_prefill_enabled=True
        # 立即 fallback 到 vllm V1 Scheduler（真正的 chunked prefill 路径）。
        # 同时 passive_tracker 仍记录 per-req TTFT/TPOT，driver/metrics 无需改。
        return {
            "ascend_scheduler_config": {
                "enabled": True,
                "scheduler_cls": "vllm_ascend.core.tdm.scheduler.TDMScheduler",
                "enable_chunked_prefill": True,
            },
            "tdm": {"enable_tdm": False, "passive_tracker": True, **common},
        }
    raise ValueError(f"unknown config: {config_name}")


def cleanup_tracker(run_id: str) -> None:
    # P1 (2026-05-09): added ctrl/chunk JSONL streams alongside iter/req.
    for suffix in ("iter", "req", "ctrl", "chunk"):
        p = TRACE_DIR / f"{run_id}_{suffix}.jsonl"
        if p.exists():
            print(f"  [cleanup] rm {p}", flush=True)
            p.unlink()


def start_server(config_name: str, run_id: str,
                 max_model_len: int, max_num_batched_tokens: int,
                 slo_ttft_ms: float = SLO_TTFT_MS,
                 slo_tpot_ms: float = SLO_TPOT_MS,
                 slo_target_violation_rate: float | None = None,
                 slo_ratio_max: float | None = None,
                 ) -> tuple[list[subprocess.Popen], Path]:
    """启动单 server 或 PD 三进程组。返回 (procs, primary_log_path)。
    procs[0] 始终是健康检查的目标（PD 模式下是 proxy）。"""
    if config_name == "c4_pd":
        return start_pd_disagg(run_id, max_model_len, max_num_batched_tokens)
    additional_cfg = build_additional_config(
        config_name, run_id,
        slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms,
        slo_target_violation_rate=slo_target_violation_rate,
        slo_ratio_max=slo_ratio_max)
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SERVER_LOG_DIR / f"server_{run_id}.log"
    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--tensor-parallel-size", "2",
        "--max-model-len", str(max_model_len),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--gpu-memory-utilization", "0.85",
        "--port", str(PORT),
        "--additional-config", json.dumps(additional_cfg),
    ]
    log_f = open(log_path, "w")
    # cwd=/tmp 绕开 vllm-workspace/vllm namespace 撞包
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        cwd="/tmp", start_new_session=True,
    )
    print(f"  [server] pid={proc.pid} log={log_path}", flush=True)
    return [proc], log_path


def _pd_kv_config(role: str) -> str:
    return json.dumps({
        "kv_connector": "LLMDataDistCMgrConnector",
        "kv_buffer_device": "npu",
        "kv_role": role,
        "kv_parallel_size": 1,
        "kv_port": str(PD_KV_PORT),
        "engine_id": "0",
        "kv_connector_module_path":
            "vllm_ascend.distributed.llmdatadist_c_mgr_connector",
    })


def _start_pd_engine(role: str, device_id: int, port: int, rpc_port: int | None,
                     run_id: str, max_model_len: int,
                     max_num_batched_tokens: int) -> tuple[subprocess.Popen, Path]:
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "prefill" if role == "kv_producer" else "decode"
    log_path = SERVER_LOG_DIR / f"server_{run_id}_{suffix}.log"
    # 严格按 exp_e2_pd_disagg.py 已跑通的 offline 配置：
    # 仅 ASCEND_RT_VISIBLE_DEVICES + DISAGGREGATED_PREFILL_RANK_TABLE_PATH，
    # decode 端额外设 VLLM_ASCEND_LLMDD_RPC_PORT。
    # 不设 HCCL_IF_IP / *_SOCKET_IFNAME — 单机回环走 NPU HCCS 直连，
    # 多设了反而把 HCCL 引到 host eth0，导致 LLM_LINK_FAILED。
    env = os.environ.copy()
    env.update({
        "ASCEND_RT_VISIBLE_DEVICES": str(device_id),
        "DISAGGREGATED_PREFILL_RANK_TABLE_PATH": PD_RANKTABLE,
    })
    if rpc_port is not None:
        env["VLLM_ASCEND_LLMDD_RPC_PORT"] = str(rpc_port)
    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--max-model-len", str(max_model_len),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",  # 跳过 graph capture，PD 阶段先求能跑通
        "--kv-transfer-config", _pd_kv_config(role),
    ]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        cwd="/tmp", env=env, start_new_session=True,
    )
    print(f"  [pd-{suffix}] pid={proc.pid} dev={device_id} port={port} "
          f"log={log_path}", flush=True)
    return proc, log_path


def start_pd_disagg(run_id: str, max_model_len: int,
                    max_num_batched_tokens: int
                    ) -> tuple[list[subprocess.Popen], Path]:
    """启动 prefill(NPU0) + decode(NPU1) 两 engine + proxy(port 8000)。

    返回 (procs, proxy_log_path)，procs[0]=proxy（健康检查目标），
    procs[1]=prefill, procs[2]=decode。stop 顺序按 procs 倒序。
    """
    proxy_log = SERVER_LOG_DIR / f"server_{run_id}_proxy.log"

    # 1. 并行起 prefill + decode 两个 engine
    # exp_e2 验过的设置：prefill 用默认 RPC port，decode 显式设 6634
    prefill_proc, _ = _start_pd_engine(
        "kv_producer", device_id=0, port=PD_PREFILL_PORT, rpc_port=None,
        run_id=run_id, max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    decode_proc, _ = _start_pd_engine(
        "kv_consumer", device_id=1, port=PD_DECODE_PORT, rpc_port=6634,
        run_id=run_id, max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    # 2. 等两个 engine 各自 /health 起来再起 proxy
    print(f"  [pd] waiting prefill@{PD_PREFILL_PORT} + decode@{PD_DECODE_PORT} "
          f"to /health...", flush=True)
    if not _wait_url(f"http://127.0.0.1:{PD_PREFILL_PORT}/health", timeout_s=300):
        print(f"  [pd] prefill /health TIMEOUT", file=sys.stderr)
    if not _wait_url(f"http://127.0.0.1:{PD_DECODE_PORT}/health", timeout_s=300):
        print(f"  [pd] decode /health TIMEOUT", file=sys.stderr)

    # 3. 起 proxy
    # 子进程单独清掉 proxy env vars：proxy 走本地 loopback，
    # httpx 默认 trust_env=True 会拿宿主的 SOCKS proxy 配置但本环境缺
    # socksio 包导致 lifespan 启动失败。这里只清子进程 env，不动宿主 shell。
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    proxy_env = os.environ.copy()
    for k in ("http_proxy", "https_proxy", "all_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        proxy_env.pop(k, None)
    proxy_env["NO_PROXY"] = "*"
    proxy_log_f = open(proxy_log, "w")
    proxy_cmd = [
        PYTHON, PD_PROXY_SCRIPT,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--prefiller-hosts", "127.0.0.1",
        "--prefiller-ports", str(PD_PREFILL_PORT),
        "--decoder-hosts", "127.0.0.1",
        "--decoder-ports", str(PD_DECODE_PORT),
    ]
    proxy_proc = subprocess.Popen(
        proxy_cmd, stdout=proxy_log_f, stderr=subprocess.STDOUT,
        cwd="/tmp", env=proxy_env, start_new_session=True,
    )
    print(f"  [pd-proxy] pid={proxy_proc.pid} port={PORT} log={proxy_log}",
          flush=True)
    return [proxy_proc, prefill_proc, decode_proc], proxy_log


def _wait_url(url: str, timeout_s: float) -> bool:
    """通用 HTTP 健康轮询。trust_env=False 忽略 http_proxy。"""
    deadline = time.time() + timeout_s
    transport = httpx.HTTPTransport(retries=0)
    with httpx.Client(timeout=2.0, transport=transport, trust_env=False) as c:
        while time.time() < deadline:
            try:
                r = c.get(url)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2.0)
    return False


def wait_health(timeout_s: float = 240.0,
                health_path: str = "/health") -> bool:
    """轮询 PORT 上的 health 端点。c4_pd 用 /healthcheck，其他用 /health。"""
    return _wait_url(f"{BASE_URL}{health_path}", timeout_s)


def stop_server(procs: list[subprocess.Popen], grace_s: float = 30.0) -> None:
    """倒序 SIGTERM 所有进程（c4_pd 下: proxy → prefill → decode）。"""
    for proc in reversed(procs):
        if proc.poll() is not None:
            continue
        print(f"  [server] SIGTERM pid={proc.pid}", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            continue
    for proc in reversed(procs):
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            print(f"  [server] SIGKILL pid={proc.pid} after {grace_s}s",
                  flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    print("  [server] all procs exited", flush=True)


def run_one_qps(
    config_name: str,
    qps: float,
    duration_s: float,
    warmup_s: float,
    run_id: str,
    out_dir: Path,
    workload: dict,
    seed: int = 0,
    arrival_mode: str = "poisson",
    burst_period: float = 10.0,
    burst_high_qps: float = 64.0,
    burst_low_qps: float = 4.0,
    burst_high_frac: float = 0.2,
    trace_file: str | None = None,
    trace_time_scale: float = 1.0,
    trace_start_offset: float = 0.0,
    trace_max_rows: int | None = None,
    trace_max_prompt_tokens: int | None = None,
    trace_max_output_tokens: int | None = None,
    num_samples: int = 50,
    slo_ttft_ms: float = SLO_TTFT_MS,
    slo_tpot_ms: float = SLO_TPOT_MS,
) -> dict | None:
    out_path = out_dir / f"{config_name}_qps{qps}.json"
    tracker_path = TRACE_DIR / f"{run_id}_req.jsonl"
    cmd = [
        PYTHON, str(EXPERIMENTS_DIR / "qps_sweep.py"),
        "--qps", str(qps),
        "--duration", str(duration_s),
        "--warmup", str(warmup_s),
        "--base-url", BASE_URL,
        "--model", MODEL_PATH,
        "--config-name", config_name,
        "--out", str(out_path),
        "--tracker-jsonl", str(tracker_path),
        "--seed", str(seed),
        "--slo-ttft-ms", str(slo_ttft_ms),
        "--slo-tpot-ms", str(slo_tpot_ms),
        "--prompt-profile", str(workload.get("prompt_profile",
                                             "short")),
        "--prompt-mu", str(workload["prompt_mu"]),
        "--prompt-sigma", str(workload["prompt_sigma"]),
        "--prompt-min", str(workload["prompt_min"]),
        "--prompt-max", str(workload["prompt_max"]),
        "--output-mu", str(workload["output_mu"]),
        "--output-sigma", str(workload["output_sigma"]),
        "--output-min", str(workload["output_min"]),
        "--output-max", str(workload["output_max"]),
    ]
    if arrival_mode == "burst":
        cmd.extend([
            "--arrival-mode", "burst",
            "--burst-period", str(burst_period),
            "--burst-high-qps", str(burst_high_qps),
            "--burst-low-qps", str(burst_low_qps),
            "--burst-high-frac", str(burst_high_frac),
        ])
    elif arrival_mode == "trace":
        if not trace_file:
            raise ValueError("--arrival-mode=trace requires --trace-file")
        cmd.extend([
            "--arrival-mode", "trace",
            "--trace-file", str(trace_file),
            "--trace-time-scale", str(trace_time_scale),
            "--trace-start-offset", str(trace_start_offset),
        ])
        if trace_max_rows is not None:
            cmd.extend(["--trace-max-rows", str(trace_max_rows)])
        if trace_max_prompt_tokens is not None:
            cmd.extend(["--trace-max-prompt-tokens", str(trace_max_prompt_tokens)])
        if trace_max_output_tokens is not None:
            cmd.extend(["--trace-max-output-tokens", str(trace_max_output_tokens)])
    elif arrival_mode == "sequential":
        if not trace_file:
            raise ValueError(
                "--arrival-mode=sequential requires --trace-file"
            )
        cmd.extend([
            "--arrival-mode", "sequential",
            "--trace-file", str(trace_file),
            "--num-samples", str(num_samples),
        ])
        if trace_max_prompt_tokens is not None:
            cmd.extend(["--trace-max-prompt-tokens", str(trace_max_prompt_tokens)])
        if trace_max_output_tokens is not None:
            cmd.extend(["--trace-max-output-tokens", str(trace_max_output_tokens)])
    elif arrival_mode == "trace_sampled":
        if not trace_file:
            raise ValueError(
                "--arrival-mode=trace_sampled requires --trace-file"
            )
        cmd.extend([
            "--arrival-mode", "trace_sampled",
            "--trace-file", str(trace_file),
        ])
        if trace_max_prompt_tokens is not None:
            cmd.extend(["--trace-max-prompt-tokens", str(trace_max_prompt_tokens)])
        if trace_max_output_tokens is not None:
            cmd.extend(["--trace-max-output-tokens", str(trace_max_output_tokens)])
    elif arrival_mode == "trace_sampled_burst":
        if not trace_file:
            raise ValueError(
                "--arrival-mode=trace_sampled_burst requires --trace-file"
            )
        cmd.extend([
            "--arrival-mode", "trace_sampled_burst",
            "--trace-file", str(trace_file),
            "--burst-period", str(burst_period),
            "--burst-high-qps", str(burst_high_qps),
            "--burst-low-qps", str(burst_low_qps),
            "--burst-high-frac", str(burst_high_frac),
        ])
        if trace_max_prompt_tokens is not None:
            cmd.extend(["--trace-max-prompt-tokens", str(trace_max_prompt_tokens)])
        if trace_max_output_tokens is not None:
            cmd.extend(["--trace-max-output-tokens", str(trace_max_output_tokens)])
    # c4_pd 没 server tracker，必须开 streaming 让 client 自测 TTFT/TPOT
    if config_name == "c4_pd":
        cmd.append("--stream")
    print(f"  [qps={qps}] running...", flush=True)
    rc = subprocess.call(cmd, cwd=str(EXPERIMENTS_DIR))
    if rc != 0:
        print(f"  [qps={qps}] WARNING driver returned rc={rc} "
              f"(常见: client 端 PoolTimeout / n_err>0; summary 仍从 {out_path.name} 读回)",
              file=sys.stderr)
    if not out_path.exists():
        print(f"  [qps={qps}] no out file at {out_path} — skipping",
              file=sys.stderr)
        return None
    try:
        return json.loads(out_path.read_text())["summary"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [qps={qps}] failed to parse {out_path}: {e}", file=sys.stderr)
        return None


def run_config(
    config_name: str,
    qps_points: list[float],
    duration_s: float,
    warmup_s: float,
    out_dir: Path,
    health_timeout: float,
    max_model_len: int,
    max_num_batched_tokens: int,
    workload: dict,
    seed: int = 0,
    arrival_mode: str = "poisson",
    burst_period: float = 10.0,
    burst_high_qps: float = 64.0,
    burst_low_qps: float = 4.0,
    burst_high_frac: float = 0.2,
    trace_file: str | None = None,
    trace_time_scale: float = 1.0,
    trace_start_offset: float = 0.0,
    trace_max_rows: int | None = None,
    trace_max_prompt_tokens: int | None = None,
    trace_max_output_tokens: int | None = None,
    num_samples: int = 50,
    slo_ttft_ms: float = SLO_TTFT_MS,
    slo_tpot_ms: float = SLO_TPOT_MS,
    slo_target_violation_rate: float | None = None,
    slo_ratio_max: float | None = None,
) -> list[dict]:
    run_id = f"qps_sweep_{config_name}"
    print(f"\n{'='*70}\n=== Config: {config_name} (run_id={run_id})\n{'='*70}",
          flush=True)
    cleanup_tracker(run_id)
    procs, log_path = start_server(config_name, run_id,
                                   max_model_len, max_num_batched_tokens,
                                   slo_ttft_ms=slo_ttft_ms,
                                   slo_tpot_ms=slo_tpot_ms,
                                   slo_target_violation_rate=slo_target_violation_rate,
                                   slo_ratio_max=slo_ratio_max)
    summaries: list[dict] = []
    health_path = "/healthcheck" if config_name == "c4_pd" else "/health"
    try:
        if not wait_health(health_timeout, health_path=health_path):
            print(f"  [server] {health_path} timeout — see {log_path}",
                  file=sys.stderr)
            return summaries
        print(f"  [server] ready", flush=True)
        for qps in qps_points:
            s = run_one_qps(config_name, qps, duration_s, warmup_s, run_id, out_dir,
                            workload, seed=seed,
                            arrival_mode=arrival_mode,
                            burst_period=burst_period,
                            burst_high_qps=burst_high_qps,
                            burst_low_qps=burst_low_qps,
                            burst_high_frac=burst_high_frac,
                            trace_file=trace_file,
                            trace_time_scale=trace_time_scale,
                            trace_start_offset=trace_start_offset,
                            trace_max_rows=trace_max_rows,
                            trace_max_prompt_tokens=trace_max_prompt_tokens,
                            trace_max_output_tokens=trace_max_output_tokens,
                            num_samples=num_samples,
                            slo_ttft_ms=slo_ttft_ms,
                            slo_tpot_ms=slo_tpot_ms)
            if s is not None:
                summaries.append(s)
                w = s.get("window") or {}
                print(f"  [qps={qps}] goodput={w.get('goodput_tok_s')} "
                      f"ttft_p99={(w.get('ttft_ms') or {}).get('p99')} "
                      f"tpot_p99={(w.get('tpot_ms_mean') or {}).get('p99')} "
                      f"meet_slo={w.get('n_meet_slo')}/{w.get('n_ok')}",
                      flush=True)
    finally:
        stop_server(procs)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="c1_baseline,c2_tdm",
                        help="逗号分隔，可选: c1_baseline, c2_tdm, c2_tdm_m2, c2_tdm_m21, c2_tdm_m22, c2_tdm_m23, c2_tdm_m24, c2_tdm_m25, c2_tdm_m26, c2_tdm_m27, c2_tdm_m31, c2_tdm_m31_fia, c3_cp, c4_pd")
    parser.add_argument("--qps", default=None,
                        help="逗号分隔 QPS 点，默认 4,8,16,32,64")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP)
    parser.add_argument("--outdir", default=str(RESULTS_ROOT / "qps_sweep"))
    parser.add_argument("--health-timeout", type=float, default=240.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="只跑 c1_baseline + QPS=2 + 20s（验编排，~3 分钟）")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    # P1.6e: SLO targets are workload-conditional. Defaults match the legacy
    # hard-coded constants so existing callers behave identically; override
    # to sweep PID + evaluation target together.
    parser.add_argument("--slo-ttft-ms", type=float, default=SLO_TTFT_MS)
    parser.add_argument("--slo-tpot-ms", type=float, default=SLO_TPOT_MS)
    # P1.6f/g: PID inner knobs for sensitivity sweep. None = use TDMConfig
    # defaults (no override).
    parser.add_argument("--slo-target-violation-rate", type=float, default=None,
                        help="PID target violation rate (default TDMConfig 0.05)")
    parser.add_argument("--slo-ratio-max", type=float, default=None,
                        help="PID upper bound for ratio (default TDMConfig 0.8)")
    # prompt 形状：profile 给默认，个别 flag 显式传值时单点覆盖
    parser.add_argument("--prompt-profile",
                        choices=sorted(PROMPT_PROFILES),
                        default=DEFAULT_PROMPT_PROFILE,
                        help="prompt 形状预设（见 lib/workload.PROMPT_PROFILES）；"
                             "short=8K canonical (p50≈90); "
                             "long=TDM 主场 (p50≈1100, p99 clamp@4000)")
    parser.add_argument("--prompt-mu", type=float, default=None)
    parser.add_argument("--prompt-sigma", type=float, default=None)
    parser.add_argument("--prompt-min", type=int, default=None)
    parser.add_argument("--prompt-max", type=int, default=None)
    parser.add_argument("--output-mu", type=float, default=DEFAULT_OUTPUT_MU)
    parser.add_argument("--output-sigma", type=float, default=DEFAULT_OUTPUT_SIGMA)
    parser.add_argument("--output-min", type=int, default=DEFAULT_OUTPUT_MIN)
    parser.add_argument("--output-max", type=int, default=DEFAULT_OUTPUT_MAX)
    parser.add_argument("--seed", type=int, default=0,
                        help="Workload generation seed (passed to qps_sweep.py)")
    parser.add_argument("--arrival-mode",
                        choices=["poisson", "burst", "trace", "sequential",
                                 "trace_sampled", "trace_sampled_burst"],
                        default="poisson",
                        help="sequential = concurrent=1 micro-benchmark "
                             "(samples --num-samples rows from --trace-file, "
                             "qps_points forced to [0.0]). "
                             "trace_sampled = Poisson arrival at --qps with "
                             "lengths sampled from --trace-file (Phase 1 T5). "
                             "trace_sampled_burst = burst arrival (burst-* "
                             "params) + lengths from --trace-file (Phase 1 T5b).")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples for --arrival-mode=sequential.")
    parser.add_argument("--burst-period", type=float, default=10.0)
    parser.add_argument("--burst-high-qps", type=float, default=64.0)
    parser.add_argument("--burst-low-qps", type=float, default=4.0)
    parser.add_argument("--burst-high-frac", type=float, default=0.2)
    parser.add_argument("--trace-file", type=str, default=None,
                        help="Path to trace CSV (arrival-mode=trace)")
    parser.add_argument("--trace-time-scale", type=float, default=1.0,
                        help=">1 compress trace, <1 stretch")
    parser.add_argument("--trace-start-offset", type=float, default=0.0)
    parser.add_argument("--trace-max-rows", type=int, default=None)
    parser.add_argument("--trace-max-prompt-tokens", type=int, default=None,
                        help="Skip trace rows with prompt > limit (keeps "
                             "arrival rate honest); typically below "
                             "--max-model-len")
    parser.add_argument("--trace-max-output-tokens", type=int, default=None,
                        help="Skip trace rows with output > limit (avoid "
                             "marathon decodes blowing trace window)")
    args = parser.parse_args()

    if args.dry_run:
        configs = ["c1_baseline"]
        qps_points = [2.0]
        duration_s, warmup_s = 20.0, 0.0
    elif args.arrival_mode == "sequential":
        # Sequential mode = concurrent=1 micro-benchmark, no QPS sweep.
        # Single "qps=0" run per config; duration semantics flipped to
        # request-count (set in qps_sweep.py via --num-samples).
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        qps_points = [0.0]
        duration_s = args.duration
        warmup_s = args.warmup
    elif args.arrival_mode == "trace_sampled_burst":
        # Burst mode: qps controlled by --burst-* params, not qps_points.
        # Force qps_points = [0.0] (single run per config).
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        qps_points = [0.0]
        duration_s = args.duration
        warmup_s = args.warmup
    else:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        qps_points = ([float(x) for x in args.qps.split(",")]
                      if args.qps else [float(x) for x in DEFAULT_QPS_POINTS])
        duration_s = args.duration
        warmup_s = args.warmup

    # profile 给默认；个别 flag 显式传值时单点覆盖
    prof = resolve_prompt_profile(args.prompt_profile)
    workload = {
        "prompt_profile": args.prompt_profile,
        "prompt_mu": (args.prompt_mu if args.prompt_mu is not None
                      else prof["prompt_mu"]),
        "prompt_sigma": (args.prompt_sigma if args.prompt_sigma is not None
                         else prof["prompt_sigma"]),
        "prompt_min": (args.prompt_min if args.prompt_min is not None
                       else int(prof["prompt_min"])),
        "prompt_max": (args.prompt_max if args.prompt_max is not None
                       else int(prof["prompt_max"])),
        "output_mu": args.output_mu,
        "output_sigma": args.output_sigma,
        "output_min": args.output_min,
        "output_max": args.output_max,
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"configs={configs}  qps={qps_points}  "
          f"duration={duration_s}s warmup={warmup_s}s  outdir={out_dir}",
          flush=True)
    print(f"  max_model_len={args.max_model_len} "
          f"max_num_batched_tokens={args.max_num_batched_tokens}",
          flush=True)
    print(f"  workload: profile={workload['prompt_profile']} "
          f"prompt[mu={workload['prompt_mu']} sigma={workload['prompt_sigma']} "
          f"min={workload['prompt_min']} max={workload['prompt_max']}] "
          f"output[mu={args.output_mu} sigma={args.output_sigma} "
          f"min={args.output_min} max={args.output_max}]",
          flush=True)

    all_summaries: dict[str, list[dict]] = {}
    t0 = time.time()
    for cfg in configs:
        all_summaries[cfg] = run_config(
            cfg, qps_points, duration_s, warmup_s, out_dir,
            health_timeout=args.health_timeout,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            workload=workload,
            seed=args.seed,
            arrival_mode=args.arrival_mode,
            burst_period=args.burst_period,
            burst_high_qps=args.burst_high_qps,
            burst_low_qps=args.burst_low_qps,
            burst_high_frac=args.burst_high_frac,
            trace_file=args.trace_file,
            trace_time_scale=args.trace_time_scale,
            trace_start_offset=args.trace_start_offset,
            trace_max_rows=args.trace_max_rows,
            trace_max_prompt_tokens=args.trace_max_prompt_tokens,
            trace_max_output_tokens=args.trace_max_output_tokens,
            num_samples=args.num_samples,
            slo_ttft_ms=args.slo_ttft_ms,
            slo_tpot_ms=args.slo_tpot_ms,
            slo_target_violation_rate=args.slo_target_violation_rate,
            slo_ratio_max=args.slo_ratio_max,
        )

    summary_path = out_dir / "qps_sweep_summary.json"
    summary_path.write_text(json.dumps({
        "wall_time_s": round(time.time() - t0, 1),
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "qps_points": qps_points,
        "slo_ttft_ms": args.slo_ttft_ms,
        "slo_tpot_ms": args.slo_tpot_ms,
        "configs": all_summaries,
    }, indent=2))
    print(f"\n[done] {time.time()-t0:.1f}s wall, summary → {summary_path}",
          flush=True)

    # 简表
    def _fmt(v):
        return "-" if v is None else str(v)
    print(f"\n{'config':<14} {'qps':<6} {'goodput':<10} {'ttft_p99':<10} "
          f"{'tpot_p99':<10} {'meet_slo':<10}")
    for cfg, runs in all_summaries.items():
        for s in runs:
            w = s.get("window") or {}
            print(f"{cfg:<14} {str(s['qps_target']):<6} "
                  f"{_fmt(w.get('goodput_tok_s')):<10} "
                  f"{_fmt((w.get('ttft_ms') or {}).get('p99')):<10} "
                  f"{_fmt((w.get('tpot_ms_mean') or {}).get('p99')):<10} "
                  f"{_fmt(w.get('n_meet_slo'))}/{_fmt(w.get('n_ok'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
