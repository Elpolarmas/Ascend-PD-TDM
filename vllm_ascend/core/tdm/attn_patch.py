"""Force-FIA monkey-patch for c2_tdm_m31_fia ablation.

Purpose
-------
让 TDM 调度产出的 phase-pure batch 在 attention forward 时绕过 vllm-ascend
v0.11.0rc1 的 dedicated kernels (_npu_flash_attention_qlens 等),统一走
_forward_v1_style 即 npu_fused_infer_attention_score (FIA)。

为什么需要
----------
vllm-ascend v0.11.0rc1 的 attention dispatch (attention_v1.py:651-668) 是:

    PrefillNoCache  → _forward_prefill_no_cache  (dedicated qlens, 快)
    PrefillCacheHit → _forward_prefill_cache_hit (dedicated qlens, 快)
    DecodeOnly      → _forward_decode_only       (dedicated decode, 快)
    ChunkedPrefill  → _forward_v1_style          (FIA, 慢 2.7-3.4x)

C3 (chunked prefill baseline) 始终落到 ChunkedPrefill → FIA;
M3.1 (我们) phase-pure batch 落到前三种 → dedicated。

→ M3.1 vs C3 的优势包含两个不可拆的贡献:phase-pure 调度策略 + dedicated
kernel 速度。要拆出 phase-pure 自身的独立贡献,本 patch 强制让 M3.1 的
attention 也走 FIA,然后跟 C3 在同样 kernel 路径上比较。

这同时也是后续 vllm-ascend 版本的行为预演 —— v0.13+ 已经把所有 attn_state
统一到 FIA (dedicated kernels 被移除)。

实装方式
--------
不动 vllm-ascend 任何源文件。在 TDMScheduler.__init__ 通过 monkey-patch
替换 AscendAttentionBackendImpl.forward,在调用原 forward 前临时把
attn_metadata.attn_state 改为 ChunkedPrefill,这样原 dispatch 自然走到
_forward_v1_style (FIA)。forward 完恢复原值。

注意
----
- 只在 c2_tdm_m31_fia 这个 ablation config 启用 (TDMConfig.force_fia_attention=True)
- 进程级幂等,_PATCHED 全局 flag 防止重复 patch
- 不影响 C3 (C3 走 vLLM V1 Scheduler 不进 TDMScheduler 路径)
- 不影响其他 config (没启用 force_fia 时 patch 不会被加载)
"""
from __future__ import annotations

import logging

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
)

logger = logging.getLogger(__name__)

_PATCHED = False


def enable_force_fia() -> None:
    """启用 force-FIA。进程级幂等,重复调用 no-op。"""
    global _PATCHED
    if _PATCHED:
        return

    _orig_forward = AscendAttentionBackendImpl.forward

    def forward_force_fia(self, layer, query, key, value, kv_cache,
                          attn_metadata, output, *args, **kwargs):
        # 已是 ChunkedPrefill 或 metadata 为 None → 走原 dispatch (会落到 FIA)
        if attn_metadata is None or \
                attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            return _orig_forward(self, layer, query, key, value, kv_cache,
                                 attn_metadata, output, *args, **kwargs)

        # 临时改 attn_state,让原 dispatch 走 _forward_v1_style (FIA) 分支
        _orig_state = attn_metadata.attn_state
        attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        try:
            return _orig_forward(self, layer, query, key, value, kv_cache,
                                 attn_metadata, output, *args, **kwargs)
        finally:
            attn_metadata.attn_state = _orig_state

    AscendAttentionBackendImpl.forward = forward_force_fia  # type: ignore[assignment]
    _PATCHED = True
    logger.warning(
        "[TDM] force_fia_attention ENABLED: all attn_state routed through "
        "_forward_v1_style (FIA kernel). This is the c2_tdm_m31_fia ablation "
        "for isolating phase-pure scheduling contribution from dedicated "
        "kernel speed contribution. Expected: slower than M3.1 baseline.")
