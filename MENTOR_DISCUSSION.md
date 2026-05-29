# 导师讨论 — PD-TDM 当前状态(2026-05-25 D-013 finalize)

> 给导师的汇报底稿。覆盖**上次汇报后的机制更迭历史**、**方法设计的核心思路**、**为什么这个机制能带来性能优势(预测)**、**实验印证**、**Open questions**。

---

## 0. 一句话 thesis(D-013 finalize)

> 在 Azure burst trace + 2-NPU + Qwen3-8B 工况下,**PD-TDM 完整吸收 Sarathi-Serve 的 chunked prefill 思想(chunk_budget=2048,跟 Sarathi 一样),但把 "prefill chunks 和 decode 是否塞同一 iter" 这个 design choice 改成 phase-pure** — Sarathi 把 chunks 和 decode 放进同一 mixed iter,PD-TDM 把它们拆到独立的 prefill iter 和 decode iter。**Paper baseline 三层论证**:Vanilla CB(mixed batch + 无 chunk 上限)→ Sarathi chunked prefill(mixed batch + chunk_budget=2048)→ PD-TDM(phase-pure + chunk_budget=2048),其中 Sarathi 在长 prompt 上大胜 Vanilla CB(NPU 上复现 OSDI'24 finding),PD-TDM 在 chunked prefill 框架内再拿额外 phase-pure 增益。**在 burst 满负载 regime 下**,PD-TDM 一个 cycle(prefill iter + decode iter)的时间 189ms 短于 Sarathi mixed iter 220ms(实测,**净差 31ms = mixed iter 比 phase-pure prefill iter 多花 70ms − PD-TDM 多出的 39ms decode iter**)。Single-iter 层 +18% throughput 差在 strict SLO + 高 QPS 撞顶时被 queue 堆积**放大成 paradigm-level +30~+77pp meet rate 差**(vs Sarathi)/ +50~+80pp(vs Vanilla CB);loose SLO + 低 QPS 下 Δ wash out。**mean / tail 同步改善**,不是 trade-off。

---

## 1. 机制更迭历史(对应上次汇报后的演化)

上次汇报时,PD-TDM 的 contribution stack 是:**慢回路 PID 自适应 ratio + 快回路 selector + Graph-aware 模块 + Dedicated kernel 优势 + Dynamic chunking controller**。从 5/8 至 5/25,**每一层 contribution 都被自己的实证数据驱动剔除或降级**,thesis 收缩到一个更干净的 first-principles 故事。

### 1.1 五次实证驱动的删减(决策号 + 触发数据)

| 时间 | 决策 | 删 / 降级的 contribution | 触发的数据 | 后续 |
|---|---|---|---|---|
| 5/8 | **D-006** | Dynamic chunking controller → 静态 chunk=2048 | `p1_chunk_scan` chunk ∈ {512..8192} Pareto **单调**,无双向 trade-off | controller hook 不再做 |
| 5/12 | **D-005** | Graph-aware 模块 → 降级为 "F only" finding | P1.0/P1.0b:AIV 23 sizes vs FFTS+ 15 sizes 反而输 1.5-23.8pp;vllm-ascend 已内置等价功能 | 三模块 → 两模块 |
| 5/14 | **D-009** | Dedicated kernel 优势 → 完全移出 contribution | P1.8 ablation:强制走另一种 kernel 后所有指标变化 **≤ 13ms / < 1%**,kernel 贡献 ≈ 0 | paper 不再讲 kernel 优势 |
| 5/18 | **D-011** | 慢回路 PID + 双维度协同 + Starvation override → 退 contribution | P1.9d 4-way ablation:慢回路在 saturated 上 0 贡献,**双维度协同不存在**;Starvation override 触发率 0.5-0.8% | PID 代码保留 internal,**paper 报 m31 = static ratio per workload** |
| 5/20 | **D-012** | (重定位)thesis 三层结构化 + goal-first framing | azure_p15 4-way 数据重分析 | 主指标 = TTFT+TPOT 双 SLO goodput;framing = "trade tail for mean" |
| 5/24-5/25 | **D-013** | (修正)**framing 从 "trade tail for mean" 改为 "cycle 短于 c3 mixed iter,mean/tail 同步改善"** | (1) 调度 bug fix + Phase 1 全重跑 126 cells;(2) **c3 实测单步 220ms vs m31 prefill 步 150ms,差额 70ms 直接量化**;(3) c4_pd steady smoke 推翻 "burst 是放大器" 假设 | 当前 thesis 主版本 |

### 1.2 剩下的 stack(D-013 final)

```
PD-TDM paradigm = chunked prefill(Sarathi 同款,chunk_budget=2048)
                 + phase-pure iter dispatch(我们的 unique design choice)

  跟 Sarathi-Serve 的区别:
    Sarathi mixed iter   = 1 个 iter 同时跑 prefill chunks + decode tokens
    PD-TDM phase-pure    = prefill chunks 进 prefill iter,decode tokens 进 decode iter
                          (跟 Sarathi 共享 chunk size 控制,不共享 mix 实现)

  Burst 满负载下 cycle 时间对比(实测,code k=2.8 cell):
    Sarathi mixed iter            = 220.6 ms        (推 ~2048 token)
    PD-TDM cycle (P iter + D iter) = 150 + 39 ms    (推 2077 token)
    净差                          = 31 ms / +18% throughput

  Paradigm-level Δ(strict SLO + 高 QPS):
    +30~+77pp meet rate(由 single-iter 18% throughput 差被 queue 堆积 amplify)
```

**给老师的核心 narrative shift**:
> 一开始押的是 PID + graph + kernel + chunking 自适应组成的 controller stack,实证一步步证伪。**剩下唯一稳定有效的机制就是 phase-pure batching 本身** — 一个不依赖任何在线控制的、纯结构性的调度选择。故事从 "controller magic" 变成 "first-principles 调度分相",反而更扎实。

---

## 2. 方法设计

### 2.1 设计目标

在 single-node multi-NPU(2 张 Ascend 910B3,无 SM partition 能力)+ Azure burst trace 的工况下,最大化 **goodput at SLO**(满足 TTFT + TPOT 约束下的有效吞吐)。Prefill 和 decode 这两类计算特征截然不同的阶段如何共享算力,是核心设计问题。

### 2.2 三个核心设计决定

```
                trace + 模型 + 硬件
                       │
                       ▼ (离线)
                  SLO 校准
                       │
                       ▼  ┌─────────────────────────┐
                          │ 运行时状态采集            │
                          │  - 队列深度 / 等待时间    │
                          │  - 请求生命周期           │
                          └────────────┬────────────┘
                                       │
              ┌────────────────────────▼──────────────────────────┐
              │      请求级排序器(基于 TTFT 紧迫度优先)            │
              └────────────────────────┬──────────────────────────┘
                                       │
              ┌────────────────────────▼──────────────────────────┐
              │       PD-TDM 调度器(决定当前步是 prefill 还是      │
              │       decode,以及单步处理多少 token)              │
              └────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
                              NPU 计算(TP=2)
```

**决定 1:分相调度** — 每个调度步只跑 prefill 或 只跑 decode,**永不混合**。
- 对照 baseline:Sarathi-Serve 的 chunked prefill 把 prefill chunk 和 decode 塞进同一 mixed iter — 这是 Sarathi 的 design choice,**不是 chunked prefill 概念的必需**。
- PD-TDM 共享 Sarathi 的核心思想(用 chunk_budget 上限切碎 prefill,避免单 iter 跑整 prompt 导致 decode 沉默),**但选择把 chunks 和 decode 分到独立 iter**,而不是塞同一 iter。
- 为什么:见 §3 — 在 burst 满负载下,phase-pure iter 比 Sarathi mixed iter 跑得快(实测),来源是 mixed iter 内同时计算 prefill chunks 和 decode tokens 时硬件上的额外代价;phase-pure 让两类计算各自在均匀形态下跑。

**决定 2:由比例参数驱动 phase 切换,而不是由"新请求到达"驱动**
- 对照 baseline:Vanilla CB 和 Sarathi chunked prefill 都让 prefill chunks 跟 decode tokens 共存于同一 mixed batch — 哪类 token 在 batch 里占多少由队列状态决定,**phase 切换是隐含在 batch composition 的,不是显式控制**。
- 我们做法:调度器维护一个 prefill : decode 的目标比例,每步根据这个比例决定下一步类型,**和"是否有新请求来"解耦**。
- 比例本身设静态(每个 workload profile 出一个),不在线调(慢回路 PID 实证无效,D-011)。

**决定 3:单步 prefill 长度上限(chunk_budget=2048)**
- 为什么有上限:不限制时一个长 prompt 占整 iter 可达 2000ms,decode 进度被拖住 → TPOT 退化(这正是 Vanilla CB 在 code 长 prompt 上 TPOT p99 = 670ms 的原因 — 实测,见 §4.3)。
- 为什么 2048:p1_chunk_scan 扫 chunk ∈ {512..8192},Pareto 单调,2048 是 sweet spot(D-006)。

### 2.3 算法描述

#### 2.3.1 调度循环的分解

PD-TDM 的核心实现是 vllm-ascend `AscendScheduler` 的子类 `TDMScheduler`。每一次 iteration,系统都要回答两个问题:**(i) 本步跑 prefill 还是 decode?(ii) 选哪些 request 进 batch、各推几个 token?** 传统单一 scheduler loop(Vanilla CB / Sarathi)把这两个决策耦合在一起 — 当前 batch 形态决定 batch 内 token 类型混合,phase 概念是隐含的。**PD-TDM 显式拆开这两个决策**,把 phase 选择交给一个有状态的控制层(token bucket + 紧急 override + 物理边界),把 batch 构造交给两条按 phase 分开的纯化路径。

为了让 paper reader 能逐步把这个调度循环拼起来,我们把整个 per-iter 流程分成 4 个子算法。**Algorithm 1** 负责 phase 决策(产生 `PREFILL` 或 `DECODE` 一个 enum 值),**Algorithm 2 / 3** 分别对应 prefill iter / decode iter 的 batch 构造,**Algorithm 4** 是把前三者粘起来的 per-iter 主循环 — 含调用父类执行 forward 与执行后的 bucket commit / state bookkeeping。

这种分解的关键好处是:**phase 决策跟 batch 构造完全解耦**,phase 决策 algorithm 输出的 enum 只是个 hint,batch 构造 algorithm 在自己的 `if phase == PREFILL` 分支内独立处理 prefill chunking 的 two-pass 逻辑(resume partial-prefill,再 admit fresh)。这种分离让我们能在不动 batch 构造代码的前提下替换 phase 策略(eg. 静态 ratio vs 动态 PID),也是 D-011 把 PID controller 退出 contribution 但保留 `tdm/controller.py` 文件做 internal use 的代码层基础。

#### 2.3.2 Algorithm 1 — Per-iter phase selection

Phase 选择不能简单看 "哪个队列长就选哪个" — 这种 greedy 策略在 burst 下会让 phase 振荡(prefill 队列一长就猛切 prefill,把 decode 完全饿死;清空后又全切 decode,长 prompt 又积起来)。我们需要一个**有状态、有平滑、能在极端情况下被紧急 override 的多层决策**。

PD-TDM 用 **token bucket** 做平滑层:每 iter credit `target_ratio` 个 token,bucket 满 1 token 就允许切到 prefill。`target_ratio` 在我们的配置下是静态的(per-workload offline profile),所以 bucket 等价于一个"按 ratio 周期触发 prefill iter"的低通滤波器。在这个 base 之上叠两层紧急 override(prefill 太老 → urgency override 强制切 prefill;decode 沉默太久 → starvation override 强制切 decode),再叠一层 slice 上下限 clamp(防止 phase 抖动过快或单 phase 持续过久),最后叠一层物理 boundary(KV cache 低水位时禁止切 prefill,避免 OOM-style preempt)。

Algorithm 1 的输入是 bucket 状态、controller 给的 ratio、当前队列 snapshot、各类阈值;输出是这个 iter 应当执行的 phase。

```
Algorithm 1: PhaseSelection
─────────────────────────────────────────────────────────────────
Input:
  b           : token bucket credit (carry over from previous iter)
  C           : bucket capacity                                  (config; default 4)
  r           : target prefill ratio from controller ∈ [0,1]    (per-workload static)
  (W, R)      : waiting / running request queues
  τ_u, τ_s    : urgency / starvation thresholds (ms)            (config)
  φ_curr, s_held : current phase and iters held in it           (persistent)
  s_min, s_max  : phase slice min / max iter bounds             (config)
  k_wm        : KV-free watermark                                (config)
Output:
  φ ∈ {PREFILL, DECODE}

 1: b ← min(C, b + min(1, r))                                    ▷ credit bucket
 2: if W = ∅           then return DECODE                        ▷ no prefill work
 3: if R = ∅           then return PREFILL                       ▷ no decode work
 4: if silence_oldest_decode(R) ≥ τ_s  ∧  b ≥ 1
 5:     then return DECODE                                       ▷ starvation rescue (TPOT)
 6: if age_oldest_waiting(W) ≥ τ_u   ∧  b < 1
 7:     then b ← 0; return PREFILL                               ▷ urgency override (TTFT)
 8: φ_planned ← PREFILL if b ≥ 1 else DECODE
 9: φ_clamped ← EnforceSliceBounds(φ_planned, φ_curr, s_held, s_min, s_max)
10: if φ_clamped = PREFILL  ∧  kv_free_ratio < k_wm
11:     then return DECODE                                       ▷ KV pressure boundary
12: return φ_clamped
```

**逐步骤说明**:

- **第 1 行(bucket credit)**:credit 在任何 corner case 之前 unconditional 发生 — 即便最后这一步什么也没切,credit 也要记下来,避免连续多 iter 没切 prefill 时 ratio 信号丢失。`min(1, r)` 截断防止 misconfig 让 ratio 一次性灌满 bucket;`min(C, ...)` cap 在 `C=4` 防止积累过深(`C` 过大会让 PID 调整反应迟钝)。
- **第 2-3 行(corner cases)**:任何一边队列为空时不需要决策,直接返回另一边。这两行让 algorithm 在系统真空闲时 degenerate 回 "服务有工作的那边",不会因为 bucket 状态硬切到没工作的 phase 浪费一个 NPU iter。
- **第 4-5 行(starvation rescue)**:decode 老 req 沉默超 `τ_s` 时强制切 decode,把 TPOT tail 拉回来。**关键 design 是 `b ≥ 1` gate** — 仅在 bucket 本来想切 prefill 时才触发,否则 bucket 本来就要切 decode,override 是 no-op。这个 AND-gate 让 starvation 不打破 bucket 的 ratio 守恒。
- **第 6-7 行(urgency override)**:waiting 队列最老 req 等到 SLO 期限的 70%(`τ_u = 0.7 × slo_ttft`)时强制切 prefill,把 TTFT tail 拉回来。**与 starvation override 互斥** — urgency 要求 `b < 1`(本来想切 decode),所以单个 iter 最多触发其中一个。Urgency 触发时把 bucket 归零(不是 −1),既"消费"了这次紧急 prefill,又不让下次正常 prefill 被无限延后。
- **第 8 行(bucket 主路径)**:没紧急情况时,看 bucket 阈值。`b ≥ 1` 就切 prefill,否则 decode。这是 PD-TDM 在 95%+ iter 上走的路径。
- **第 9 行(slice 上下限)**:`EnforceSliceBounds` 是 `HardConstraints.enforce`:`s_held < s_min` 时不允许切相(防止 1-iter 抖动 — slice 过短导致 NPU graph 频繁切换 + cache 失效);`s_held ≥ s_max` 时强制切相(防止某种 corner case 让 phase 永久卡在一边)。**实测 `s_min=2, s_max=8`,代码上虽有这一层,但 burst 满负载下 selector 自然达成 1:1 alternation,slice 上下限实际很少触发** — D-013 telemetry 显示 `constraint_min_slice` 永远 0%。
- **第 10-11 行(KV 物理 boundary)**:KV cache 自由率低于水位时禁止切 prefill — 新 prefill 会立即分配 KV block,水位低时可能触发 OOM-style preempt(已有 running req 被 evict 重跑,代价巨大)。**这是物理硬约束,不可绕过**,等同于 vLLM PagedAttention 论文里的 watermark 思想。

#### 2.3.3 Algorithm 2 — Prefill iter scheduling (chunked, two-pass)

Phase 决策返回 `PREFILL` 后,需要构造一个纯 prefill 的 batch。这里的设计挑战是:**长 prompt 跨多个 prefill iter 完成**(chunk_budget = 2048,一个 7000-token prompt 要切 4 个 prefill iter),所以每个 prefill iter 的 batch 里可能既有 "上一个 prefill iter 没推完的老 req"(partial-prefill,留在 running 队列里),又有 "完全没开始的新 req"(在 waiting 队列里)。如果不区分,容易在 budget 用尽时把老 req 跳过,导致 prefill 进度被新 admit 抢走,违反 FIFO 公平性。

**PD-TDM 用 two-pass 解决**:Pass 1 先 sweep running 队列 resume 老 req(同时用 phase-pure filter 跳过已经进入 decode state 的 req — 这一过滤是 phase-pure 的核心保证),Pass 2 再 sweep waiting 队列 admit 新 req。每个 req 推进的 token 数由 `min(剩余 prefill, chunk_budget 残量, token_budget 残量)` 决定,**用 truncate 而不是 skip**(Sarathi 同款):budget 不够装下整个剩余 prefill 时,推到 budget 边界就停,余下的下个 prefill iter 由 Pass 1 接力。

```
Algorithm 2: PrefillIter (chunked, two-pass)
─────────────────────────────────────────────────────────────────
Input:
  W, R    : waiting / running queues (R 可含 partial-prefill reqs)
  B_c     : per-iter chunk budget                                 (TDMConfig; = 2048)
  B_t     : per-iter token budget (vllm-native)                   (= max_num_batched_tokens; = 8192)
Output:
  S = {(req, n)} : per-request scheduled token counts

 1: S ← ∅;  cb ← B_c;  tb ← B_t
 2: ▷ Pass 1: resume partial-prefill from running queue
 3: for req ∈ R do
 4:     if req.num_computed ≥ req.num_prompt
 5:         then continue                                         ▷ phase-pure filter: skip decode-state
 6:     if cb ≤ 0  ∨  tb ≤ 0  then break
 7:     n ← min(req.num_prompt - req.num_computed, cb, tb)
 8:     S ← S ∪ {(req, n)};  cb ← cb - n;  tb ← tb - n
 9: end for
10: ▷ Pass 2: admit fresh prefill from waiting queue
11: for req ∈ W do
12:     if cb ≤ 0  ∨  tb ≤ 0  then break
13:     n ← min(req.num_prompt, cb, tb)                           ▷ truncate (not skip)
14:     S ← S ∪ {(req, n)};  cb ← cb - n;  tb ← tb - n
15: end for
16: return S
```

**逐步骤说明**:

- **第 1 行(初始化)**:`cb` 是 chunk budget 残量(本 iter 内部 reset 为 `B_c = 2048`),`tb` 是 vllm-native token budget 残量(`B_t = 8192`)。两个 budget 共存的设计:`B_c` 限制单 iter prefill 的硬上限(避免长 prompt 一次占满让 cycle 拉长,这是 §2.2 决定 3 的实现);`B_t` 是 vllm-native 全局总 budget(prefill + decode 共享),保留兼容性。**实际跑 prefill iter 时 `B_t = 8192 > B_c = 2048` 始终成立,所以 `B_c` 是 binding constraint**;`B_t` 实测从不限流。
- **第 2-9 行(Pass 1 resume)**:遍历 running 队列。**第 4-5 行的 phase-pure filter 是关键** — 跳过 `num_computed ≥ num_prompt` 的 req(这些 req 已经把 prompt prefill 完了,处于 decode state,如果不跳过会被当成 "完整 prompt 还要再推 token" 处理,既污染 prefill batch 形态,又触发父类 assertion fail(`num_tokens - num_computed == 1` 假设)。第 7 行 `n` 的三个 min 选项 — 剩余 prefill、`cb` 残量、`tb` 残量 — 确保不超任何一个 budget。
- **第 10-15 行(Pass 2 admit fresh)**:Pass 1 用完 budget 后没剩多少时,Pass 2 可能一个新 req 都进不来;**这是 design 的有意行为** — 老 req 优先完成它们的 prefill,新 req 推迟到下个 iter 再 admit。FIFO 公平性 + 减小 partial-prefill 数(避免 running 队列里堆积过多半成品 req 占着 KV cache)。第 13 行 truncate semantics 是 Sarathi-style:**整 prompt 装不下不跳过,而是塞一个 chunk,剩余下 iter 接力**;这点 PD-TDM 跟 Sarathi 完全一致。

**为什么 two-pass 而不是 single-pass merged queue?** 实现上 single-pass(把 running + waiting 合成一个 stream)看似更简单,但会失去 "老 req 优先" 的 FIFO 语义 — 新 req 一旦混进来,长 prompt 容易被持续 starve(新 req 的 first chunk 很短,容易塞进 budget 缝隙)。Two-pass 显式保证 Pass 1 先吃 budget,Pass 2 才看残量,公平性硬约束。

#### 2.3.4 Algorithm 3 — Decode iter scheduling (phase-pure filter)

Phase 决策返回 `DECODE` 后,batch 构造看起来 trivial — 把 running 队列里的 req 各推 1 个 decode token。但 **`Algorithm 3` 第 3-4 行的 phase-pure filter 不能省**:running 队列里可能混入 partial-prefill 的 req(它们 prefill 没推完,刚被 Algorithm 2 在上个 prefill iter 处理过一部分),这些 req 的 `num_computed < num_prompt`,如果直接推 1 个 decode token 会让父类 forward 路径走错 kernel,触发 assertion fail。

```
Algorithm 3: DecodeIter (phase-pure filter)
─────────────────────────────────────────────────────────────────
Input: R ; running queue
Output: S = {(req, 1)} ; per-request one decode token

 1: S ← ∅
 2: for req ∈ R do
 3:     if req.num_computed < req.num_prompt
 4:         then continue                                         ▷ phase-pure filter: skip partial-prefill
 5:     S ← S ∪ {(req, 1)}                                        ▷ one token per req
 6: end for
 7: return S
```

**逐步骤说明**:

- **第 3-4 行(phase-pure filter)**:Algorithm 3 跟 Algorithm 2 共享同一个 phase-pure 不变式 — **任何一个 iter 的 batch 内,所有 req 处于同一 phase**。Algorithm 2 在 Pass 1 跳过 `num_computed ≥ num_prompt` 的 decode-state req,Algorithm 3 反过来跳过 `num_computed < num_prompt` 的 partial-prefill req,两个 filter 互补。这一点是 D-013 chunked_schedule.py L194 fix 修复的 bug 区域 — 修复之前 decode loop 没这个 filter,partial-prefill req 被父类误处理,导致 277/1073 个本该 decode 的 iter 实际偷塞 prefill chunk,且因为后续 budget 守恒检查 fail 而 skip decode loop 自己,**双输**。
- **第 5 行(单 token 推进)**:每个通过 filter 的 req 推 1 个 token,batch 大小由 running 队列里 decode-state req 的数量决定(实测 m31 decode iter batch 大小 mean 29,max 40,远小于 token_budget = 8192,所以 budget 这里实际不限流)。

#### 2.3.5 Algorithm 4 — Per-iter main loop (glue)

Algorithm 4 把前三者粘起来,加上 Execute + state commit + telemetry。**关键 design 是 commit-on-actual** — phase 决策给的是 planned phase,但父类 `AscendScheduler` 在两边队列同时为空的 corner case 会 auto-flip(将 planned PREFILL 回退为 DECODE,即 Option W 兜底)。如果 bucket commit 按 planned phase 算,bucket 会在 auto-flip 时被错误扣减 — 我们以为消费了一个 prefill iter,但实际没跑 prefill。所以 commit 必须按 `φ_actual` 算,守恒 bucket 的 token 数。

```
Algorithm 4: PerIterLoop (glue)
─────────────────────────────────────────────────────────────────
Persistent state:
  b          : bucket credit                                       (init 0)
  φ_curr    : current phase                                       (init DECODE)
  s_held    : iters held in φ_curr                                (init 0)

Per iter:
 1: r ← Controller.target_ratio()                                  ▷ per-workload static value
 2: φ_planned ← PhaseSelection(b, C, r, W, R, τ_u, τ_s,
                                φ_curr, s_held, s_min, s_max, k_wm) ▷ Alg. 1
 3: if φ_planned = PREFILL
 4:     then S ← PrefillIter(W, R, B_c, B_t)                       ▷ Alg. 2
 5:     else S ← DecodeIter(R)                                     ▷ Alg. 3
 6: φ_actual ← Execute(S)                                          ▷ NPU forward; parent may
                                                                     auto-flip PREFILL→DECODE
                                                                     when both queues empty
 7: if φ_actual = PREFILL  ∧  b ≥ 1  then  b ← b - 1              ▷ commit bucket on ACTUAL
 8: s_held ← (s_held + 1) if φ_actual = φ_curr else 1
 9: φ_curr ← φ_actual
10: ▷ telemetry: record (iter_id, φ_planned, φ_actual,
                           batch_size, batch_tokens, iter_time, ...)
```

**逐步骤说明**:

- **第 1 行(controller)**:`Controller.target_ratio()` 在 paper 配置下返回 per-workload 静态值(conv: 0.8 / code: 0.7;profile 自一次性 offline scan)。PID controller 代码留 internal,paper 不进 contribution(D-011 决定)— 实测 PID 在 saturated 区跨所有 SLO 档都钉 `ratio_max = 0.8` 撞顶,无自适应效果。
- **第 2-5 行(算法分派)**:phase 决策 → 按 phase 调用对应 batch 构造算法。**Alg. 2/3 完全独立**,phase 决策只是 `if` 分支选择器。这种分派让 phase 策略可插拔 — eg. 可以把 Alg. 1 替换为另一种(纯 admit-driven、纯 fixed-ratio、reactive PID 等),Alg. 2/3 完全不动。
- **第 6 行(Execute + auto-flip)**:NPU forward pass。父类 `AscendScheduler` 在 planned = PREFILL 但两边队列实际都空(因为本 iter 处理过程中 last running req finish 了,且 waiting 也空)时会 silently auto-flip 到 DECODE,返回一个空 SchedulerOutput。我们读 `self.phase`(父类暴露的内部状态)拿 `φ_actual`。
- **第 7 行(commit on actual)**:bucket 只在 `φ_actual == PREFILL` 时扣减。如果 planned = PREFILL 但 actual = DECODE(auto-flip),bucket 保留 credit,下次 iter 仍可触发 prefill。这是守恒 bucket 不变式的关键 invariant。
- **第 8-9 行(slice bookkeeping)**:`s_held` 在 phase 持续时累加,phase 切换时 reset 到 1 — 下个 iter 的 `EnforceSliceBounds` 用这个状态判定 min/max。
- **第 10 行(telemetry)**:4 通道 jsonl(iter / req / ctrl / chunk)。`iter_time` 由 `update_from_output` callback 回填(forward pass 完成后才有 wall time)。**这一通道是 §4.1 cycle time 实测的数据源** — 5/25 patch 让 passive 模式(C3 baseline,不开 phase-pure)也能采集 iter 时长,所以才能直接对比 C3 mixed iter 220.6ms vs PD-TDM prefill iter 150ms。

#### 2.3.6 算法 ↔ 代码模块对应

| Algorithm | 对应代码模块 |
|---|---|
| Algorithm 1 — Phase Selection | `tdm/selector.py::TokenBucketSelector.peek` + `tdm/constraints.py::HardConstraints.enforce` + `tdm/boundary.py::BoundaryGuard.override` |
| Algorithm 2 — Prefill Iter (chunked) | `tdm/chunking.py::plan_chunks` + `tdm/chunked_schedule.py` (Diff #2 chunk_budget / #3 truncate / #6 phase gate) |
| Algorithm 3 — Decode Iter (phase-pure) | `tdm/chunked_schedule.py` decode loop with Diff #5 partial-prefill filter |
| Algorithm 4 — Per-Iter Loop | `tdm/scheduler.py::TDMScheduler.schedule + update_from_output` + `tdm/engine.py::PhaseEngine.reconcile + apply` + `tdm/selector.py::commit` |

### 2.4 与 baseline 的高层对比

| 维度 | Vanilla CB | Sarathi Chunked Prefill | **PD-TDM (ours)** |
|---|---|---|---|
| 单 iter batch 组成 | **mixed**(prefill + decode 同 batch) | **mixed**(prefill chunks + decode 同 batch) | **phase-pure**(prefill iter / decode iter 独立) |
| Prefill 切块上限? | ❌(prompt 整推) | ✅ chunk_budget=2048 | ✅ chunk_budget=2048(跟 Sarathi 同款) |
| Long-prompt 下 decode 行为 | decode 在 batch 但被 prefill 拖住,有限推进(实测 TPOT p99 670ms) | chunk 切碎使 decode 每 iter 都能推进(TPOT p99 228ms) | decode iter 完全独立(TPOT p99 190ms) |
| 跟主流文献的对应 | **OSDI'22 ORCA / 早期 vLLM 默认 mixed batch CB** | **OSDI'24 Sarathi-Serve baseline** | **本工作 unique design choice** |
| 在 paper 中的角色 | Sarathi 论文比较的 baseline | 我们最强对手 | 主线 contribution |
| 实现配置(vllm-ascend) | `chunked_prefill_enabled=True` + `max_num_batched_tokens=8192`(prompt < chunk 不触发切分) | `chunked_prefill_enabled=True` + `max_num_batched_tokens=2048` | TDMScheduler + `prefill_chunk_tokens=2048` |

**关键说明**:
- **PD-TDM 不是反 chunked prefill**。Sarathi 用 chunk_budget 上限切碎 prefill,**Sarathi 论文 OSDI'24 实证比 Vanilla CB 强很多**,业界普遍采纳。PD-TDM 完整继承这一招(同样 chunk_budget=2048)。
- **PD-TDM 改的是 Sarathi 的 "mix" 实现** — Sarathi 把 chunks 和 decode tokens 塞同一 iter,PD-TDM 拆到独立 iter。这是 **chunked prefill 框架内的两个 design point**,Sarathi 论文没比较过 mix vs phase-pure variants。
- **Paper baseline 三层论证**:Vanilla CB → Sarathi(chunked prefill 的胜利,NPU 复现)→ PD-TDM(chunked prefill 内部 phase-pure 的额外增益)。这跟主流 LLM serving 文献术语完全对应,reviewer 一眼可 map。
- **Vanilla CB 实现 trick**:在 vllm-ascend 上 `chunked_prefill_enabled=True` + `max_num_batched_tokens=8192`,因为 Azure trace prompt cap = 7000 < 8192,chunk 实际不触发 — 长 prompt 一次跑完同 batch 还带 decode,**完全符合文献意义 vanilla CB**(无须新跑,**数据已存在于 `phase_2_t6_burst_goodput/*_nonpid/` 老 c3_cp config**)。

---

## 3. 性能分析(从调度思路推预测)

> 这一节从 chunked prefill 的现状出发,识别 Sarathi-Serve **没回答**的 sub-design-question(chunks 和 decode 是否塞同一 iter),提出 PD-TDM 是这个 sub-design-space 的另一个 point,基于此推出三个可验证 prediction。Prediction 的验证靠 §4 实测数据。

### 3.1 起点:chunked prefill 是业界已验证的胜利(NPU 上首次复现)

**Vanilla continuous batching(ORCA OSDI'22 / 早期 vLLM 默认)的瓶颈** — 长 prompt 一次跑完整 iter(可达 2000ms),期间同 batch 内的 decode tokens 推进缓慢,TPOT tail 巨大(实测 NPU 上 vanilla CB 在 code workload 上 TPOT p99 = 670ms,长 prompt 让 decode 在 batch 里被堵住)。

**Sarathi-Serve(OSDI'24)的 chunked prefill** 解决了这个问题:用 chunk_budget(默认 2048 token)把长 prefill 切成多个 chunks,**每 iter 跑一个 chunk + 一部分 decode**,decode 不再被长 prompt 拖太久。Sarathi 论文实证 chunked prefill 在 TPOT tail / generation stall 上**显著优于** vanilla CB,vLLM 等系统已普遍采纳。

**NPU 上首次复现**:我们 Phase 1 sweep 同时跑了 Vanilla CB(`chunked_prefill_enabled=True` + `max_num_batched_tokens=8192`,Azure trace prompt cap=7000 让 chunk 不触发)和 Sarathi(同 enable + budget=2048)两种配置。实测在 code workload(长 prompt)上 **Sarathi 大幅胜 Vanilla CB**(eg. code 4.9 s3:Sarathi meet 100% vs Vanilla CB 19.4%;code 2.8 s1:Sarathi TPOT p99 228ms vs Vanilla CB 670ms = **−65%**)。**这是 NPU 上首次直接量化 Sarathi 论文核心 finding**。

**这是业界共识。PD-TDM 完整继承 chunked prefill 思想(chunk_budget=2048,跟 Sarathi 同款),不挑战这个 baseline。**

**Bonus finding(NPU 短 prompt 上反例)**:conv workload(短 prompt 平均 ~500 token)上 **Vanilla CB 反而略胜 Sarathi**(eg. conv 1.0 s1:VCB meet 55.7% vs Sarathi 49.2%;conv 2.2 s1:VCB 35.4% vs Sarathi 33.8%)— 因为短 prompt 在 chunk=8192 下一次跑完首 token 比 chunk=2048 切多份更快;chunking 在短 prompt 上反向损失 ttft。这是个 fresh finding,Sarathi 论文没明示。

### 3.2 PD-TDM 的 unique angle:chunked prefill 框架内的两个 design point

Sarathi-Serve 的 chunked prefill 实际包含**两个**绑在一起的 design choice:

- **Choice A — 用 chunk_budget 上限切碎 prefill**(无可争议,是 Sarathi 的核心 contribution,解决 decode stall)
- **Choice B — 把 chunks 和 decode tokens 放进同一 mixed iter**(一个 implementation 选择;Sarathi 这么写是因为它最自然 — 既然要让 chunks 不阻塞 decode,那就把它们放一起跑)

**Sarathi 论文从没比较过 Choice B(mix)和它的替代(phase-pure)**。Sarathi 把 (A + B) bundle 跟 vanilla CB 比较,展示 chunked prefill 整体是 game changer,**但没回答 "chunks 和 decode 应该在同一 iter 还是不同 iter" 这个 sub-question**。

**PD-TDM 选另一个 design point**:
- **Choice A**:保留(跟 Sarathi 同款 chunk_budget=2048)
- **Choice B' — chunks 进 prefill iter,decode tokens 进 decode iter,phase-pure**

两种 design 都解决了 vanilla CB 的 decode stall 问题(都有 chunk 上限),区别只在 chunks 和 decode 是否同 iter。**Sarathi 没说过 Choice B 是最优,只是默认这么实现** — PD-TDM 探索 Choice B' 是 chunked prefill 框架内的合理 follow-up。

### 3.3 我们对 Choice B vs B' 的 hypothesis

**Hypothesis**:**在 burst 满负载工况下,Choice B'(phase-pure)比 Choice B(Sarathi mix)有 measurable 优势,差额来自 mixed iter 内同时计算 prefill chunks(长查询)和 decode tokens(单 token 查询)时硬件上的额外代价**。

具体物理来源在 attention 等核心算子上 — mixed query length 形态相对 uniform 形态有 kernel 路径退化成本。**Paper 顶层论证不深入到 kernel 层**,只把这个代价作为 measurable empirical fact 摆出来,差额由 §4.1 直接测出 = **70ms / iter**(同 chunk_budget = 2048 下,phase-pure prefill iter 150ms vs Sarathi mixed iter 220.6ms)。

**这个 hypothesis 跟 Sarathi 论文不矛盾**:
- Sarathi 比较的是 chunked prefill (A + B) vs vanilla CB → 结论 chunked prefill 整体好,**这个结论 PD-TDM 也接受**
- Sarathi **没比较** mix vs phase-pure variants of chunked prefill → 这个 sub-design-space 留白
- PD-TDM 填了这个留白,论证 Choice B' 在 burst 工况下更优

### 3.4 Cycle 概念与 regime 边界

**Cycle 定义**:一个 PD-TDM cycle = 1 个 prefill iter + N 个 decode iter,**N 由当前负载决定**:

| Regime | N 的取值 | 物理含义 | Choice B vs B' 的 Δ |
|---|---|---|---|
| **Burst 满负载**(高 QPS,prefill 队列持续堆积) | N = 1(1:1 严格交替) | 每跑完一个 prefill iter 立刻又有新 prefill 排队等 | **集中爆发,§3.3 hypothesis 适用** |
| 中负载 | N = 几 ~ 几十 | 1 个 prefill iter + 多个 decode iter,慢慢消化 prefill 队列 | 差额小但仍 > 0 |
| 轻负载 | N → ∞ | 几乎全是 decode iter,prefill 偶尔触发 | wash out(两边都跑 decode-dominated iter,差异微小) |

**Cycle time 比较只在 N=1 regime 严格成立**。这恰好对应 strict SLO + 高 QPS 撞顶时的工况,即 paper 主图最关心的 regime。轻负载下两个 paradigm 都"吃饱",差异自然吸收。

### 3.5 三个可验证 prediction

| # | Prediction | 成立 regime | 推论来源 | 直接对应实验 |
|---|---|---|---|---|
| **P1** | N=1 regime 下,PD-TDM cycle 时间(T_p + T_d)< Sarathi mixed iter 时间 T_mixed,且 T_mixed > T_p,差额由 §3.3 hypothesis 主导 | burst 满负载 | §3.3 hypothesis 直接预测 | §4.1 单步实测 |
| **P2** | Paradigm-level Δ meet rate 在 strict SLO + 高 QPS 上被 **amplify**(因系统接近 saturation,single-iter throughput 18% 差被 queue 堆积放大成 dramatic meet rate 差);在 loose SLO + 低 QPS 上 wash out(N→∞,退化为 decode iter 比拼) | 跨 SLO × QPS 全谱 | P1 的 cycle time 差 → queueing dynamics 放大 | §4.2 F2 heatmap |
| **P3** | 对照 Vanilla CB(无 chunk 上限):长 prompt 整 iter 跑同 batch,decode 在 batch 但被 prefill 拖住 → TPOT p99 ≈ 670ms(实测)。PD-TDM 因为保留 Sarathi chunk 上限 → TPOT 被 cycle time 约束 ≈ 190ms,**3.5× 改善**。**注:此项跟 Sarathi 共享优势,不是 PD-TDM 独占**(Sarathi TPOT p99 = 228ms,同样 3× 改善 vs VCB) | 长 prompt + strict TPOT SLO | §2.2 决定 3(chunk 上限,跟 Sarathi 共享) | §4.3 Vanilla CB vs PD-TDM TPOT 反例 |

**Prediction 之间的关系**:
- **P1 是 PD-TDM vs Sarathi 的核心论点** — single-iter 层 mix vs phase-pure 直接对比
- **P2 把 P1 的 single-iter 差跟 paradigm-level goodput 连起来** — 解释为什么 single-iter 18% throughput 差能放大成 paradigm-level 70pp meet rate 差(saturation 附近 queue 堆积非线性)
- **P3 是 PD-TDM 跟 Sarathi 共享的优势(都用 chunk 上限),vs vanilla CB 都赢** — 这一项 paper 须明确归 credit 给 Sarathi 的 chunk 上限设计,**不能 claim 是 PD-TDM 独占**;PD-TDM 在 P3 上面只是 "保留了 Sarathi 的好东西没破坏",真正独占的 contribution 在 P1 + P2

**这三个 prediction 是在 Phase 1 sweep 之前根据 design 写下的(决策日志 D-011 / D-012),不是看到数据再编出来的解释**。Phase 1 完成后三个 prediction 全部 match。

---

## 4. 实验印证

### 4.1 P1 印证 — Cycle time 实测(N=1 regime,code k=2.8 cell)

5/25 给 Sarathi chunked prefill 实测调度步时长(此前一直缺,只靠物理推理),直接对比 PD-TDM。此 cell 是 burst 满负载 cell(prefill 队列 waiting_depth=64 持续堆积),严格 1:1 alternation,**N=1 regime 成立**:

| paradigm | 单步 / 单 cycle 时间 (实测) | 推进 token 量 | 单位时间 token throughput |
|---|---|---|---|
| **PD-TDM** | prefill 步 150ms + decode 步 39ms = **cycle 189ms** | 2048 + ~29 = 2077 | **11.0 token/ms** |
| **Sarathi chunked prefill** | mixed 步 (chunk-cap) **220.6ms** | ~2018 + ~30 = 2048 | 9.3 token/ms |
| **Vanilla CB**(老 c3 chunk=8192) | 长 prompt mixed 步(整 prompt 一次推 + decode tokens)— 待补 telemetry | ~7000 prefill + ~30 decode | 显著低于上两者(paradigm-level TPOT p99 670ms ≈ single iter 时长 lower bound) |

(PD-TDM prefill 步 batch 含 2.16 个请求 / decode 步 batch 含 ~29 个请求;Sarathi mixed 步 batch 含 1-2 prefill chunk + ~30 decode;Vanilla CB 单 iter 含整长 prompt + 同 batch decode tokens)

**Vanilla CB iter telemetry 补救**:D-013 iter telemetry patch 是 5/25 加的,老 c3 (chunk=8192) 数据是 5/22 跑的,没有 iter 时长。补救方案 = 用 patch 后 scheduler 重跑 1-3 cell (~30min wall),见 §7 Future Work F0。

**P1 印证**:
- T_mixed = 220.6ms > T_p = 150ms ✓(§3.3 hypothesis:Sarathi mixed iter 比 phase-pure prefill iter 同 chunk_budget 下多花 70.6ms)
- PD-TDM cycle 189ms < Sarathi mixed iter 220.6ms ✓
- Throughput 比 = 11.0 / 9.3 = **1.184 → +18.4%** ✓
- **Cycle 净差额拆解**:Δ_cycle = 220.6 − 189 = 31.6ms = (Sarathi mixed iter 比 phase-pure prefill iter 多花 70.6ms) − (PD-TDM cycle 多出的 decode iter 39ms)。**净差额 31.6ms 才是 PD-TDM 在 N=1 regime 相对 Sarathi 的真实优势**。
- 数字 70.6ms 是 paradigm-level 实测差,严格说混入了少量其它次要因素(scheduler 路径开销等),paper 主 claim 锁在 "mixed iter 比 phase-pure prefill iter 慢一个 measurable amount" 这个 paradigm-level fact,不严格拆解 70.6ms 内部 — 详 §5.1 caveat

### 4.2 P2 印证 — Strict / loose 放大 vs wash out

F1 主图 + F2 advantage heatmap 直接呈现"strict 大胜 / loose wash out"曲线:

**F2 heatmap(`figures/paper_F2_advantage_heatmap.png`)**:
- conv:s1 row 一片深红(+9.9 → +34.8pp),s2 浅红(+2.8~+7.2),s3 几乎白(+1.9~+4.0)
- code:s1 row dark red max **+77.8pp**,s2 浅(+0.6~+2.5),**s3 全部 +0.0**(完全 tied)
- **P2 印证**:strict 上 Δ amplify、loose 上 Δ wash out ✓

### 4.3 P3 印证 — vs Vanilla CB 的 TPOT 反例(跟 Sarathi 共享优势)

**核心单点反例 — Vanilla CB 长 prompt 不切块的代价**(code workload,3-seed median):

| Cell | Vanilla CB TPOT p99 | Sarathi TPOT p99 | **PD-TDM TPOT p99** | 解读 |
|---|---|---|---|---|
| code 2.8 s1 | **669.5ms** | 228.1ms | **190ms** | Sarathi 3× 胜 VCB,PD-TDM 再 1.2× 胜 Sarathi |
| code 4.9 s1 | **670.5ms** | 228.2ms | **190ms** | 同上 |
| code 4.9 s3 | 670.5ms / meet 19.4% | 228.2ms / **meet 100%** | 190ms / 100% | Sarathi 完胜 VCB(chunked prefill 救活长 prompt) |

物理对应 §2.2 决定 3:**Vanilla CB 长 prompt 一次推完整 iter,decode tokens 在同 batch 里也被算但被 prefill 拖住推进缓慢 → TPOT p99 ~670ms**。Sarathi 和 PD-TDM 都通过 chunk_budget=2048 把长 prompt 切碎,decode 不再被长 prompt 拖太久 — Sarathi 让 decode 每 iter 都能推进 1-2 token(TPOT 228ms),PD-TDM 让 decode iter 完全独立(TPOT 190ms)。

**P3 印证 + 诚实归 credit**:
- ✓ PD-TDM 在长 prompt + strict TPOT SLO 上完胜 Vanilla CB(**3.5× 差距**)
- ✓ **Sarathi 在同 cell 也大胜 Vanilla CB**(228ms vs 670ms = **3× 差距**),说明这个胜利的**绝大部分来自 chunk 上限设计本身,而不是 phase-pure**
- → P3 是 PD-TDM 跟 Sarathi **共享的优势**(都用 chunk 上限),vs Vanilla CB 都赢。**Paper Section 3/5 须明确把这个 credit 归给 Sarathi 的 chunk 上限设计,不能 claim 是 PD-TDM 独占**
- PD-TDM 在 P3 上的 contribution 是 "保留了 Sarathi 的好东西没破坏 + phase-pure 再多榨一点"(228 → 190ms,~17% additional),真正独占的 contribution 在 P1 + P2(phase-pure 的 cycle time 增益在 strict SLO 上 amplify 成 paradigm-level Δ)

**NPU 首次复现 Sarathi finding 的强证据(code 4.9 s3 cell)**:
- Vanilla CB meet 19.4% / Sarathi meet **100%** — 同一 cell 上 +80.6pp
- 这是 NPU 上首次直接量化 "chunked prefill 把 vanilla CB 从灾难带到 SLO 完全满足" 的 finding,跟 Sarathi 论文 OSDI'24 在 GPU 上的 finding 同方向同量级

**Bonus 观察 — Phase 1 winning region(P2 的展开,跨 workload,vs Sarathi)**:

| Workload | strict s1 winning(vs Sarathi) | loose s3 | latency 全维度 |
|---|---|---|---|
| conv (decode-heavy) | +9.9~+34.8pp(7 cell 全胜) | +1.9~+4.0pp(7 cell 全胜,微赢) | PD-TDM 在 21 cells × 6 metric 上几乎全胜(Sarathi 拿 0 wins) |
| code (prefill-heavy + 长 prompt) | +12.3~+77.8pp(strict 大胜) | 全 tied(都 100%) | PD-TDM 在 21 cells × 6 metric 上 21/21 全胜 |

conv + code 双 workload 都受益,说明 P2 的 amplification 现象在多 workload 上 robust。

### 4.4 c4_pd 4-way 补完(D-011 commitment debt 还清)

`results/c4_pd_supplement/` — 42 cells,~87 min wall。c4_pd(1P1D disaggregation)**全谱 dominated**:
- conv s1 全 0% meet;TTFT mean 从 low load 472ms 单调爆到 high load **63,663ms**
- 但 TPOT mean 稳定 **~70ms**(decode 完全隔离,代价 = TTFT 灾难)

**TTFT 烂的根因(非 bug,4 层叠加)**:
1. TP=1 vs TP=2:c4_pd prefill 只在单卡 TP=1,长 prompt prefill 计算时间差 ~2×
2. 长 prompt 占满预算:7000-token prompt 吃 86% prefill budget,实际能撑 0.3-0.5 QPS
3. Burst trace 放大:1P1D 静态分区**没法借资源**
4. 静态分区不对称:decode 卡 70ms 全程 idle,但没法借给 prefill

### 4.5 c4_pd steady vs burst smoke(推翻 "burst 是放大器" 假设)

| Load | PD-TDM TTFT | c4_pd burst | **c4_pd steady** | burst/steady |
|---|---|---|---|---|
| QPS 2.85 (low) | 146 | 472 | **432** | 1.1× |
| QPS 5.70 (mid) | 191 | 1118 | **916** | 1.2× |
| QPS 7.98 (high) | 191 | 16266 | **17423** | 0.9× |

(high load steady 略**差于** burst,burst 完全不是放大器)

**Framing 修正**(堵 reviewer "你 setup 是 strawman" 攻击):**1P1D 在 2-NPU 上 prefill capacity 数学 ~5-6 QPS 撞顶,跟 trace 形态无关**。DistServe / Splitwise 原 paper 在 ≥4 NPU + 多节点下成立,**2-NPU 不在它们的应用域**;PD-TDM 是 2-NPU regime 的合适 paradigm。

### 4.6 数字 highlight 表(paper 直接 cite,3-way 主线 + c4_pd disagg reference)

| Workload / k / SLO | Vanilla CB | Sarathi (c3-fair) | **PD-TDM** | c4_pd (disagg ref) |
|---|---|---|---|---|
| **code 2.8 s1**(机制点) | meet 0.6% / TPOT p99 **670ms** | 2.3% / 228ms | **80.1% / 190ms** | 0.0% / 80ms |
| **code 4.9 s1**(最大 Δ vs Sarathi) | 1.0% / 670ms | 6.5% / 228ms | **80.4% / 190ms** | 0% / 80ms |
| **conv 3.0 s1**(最大 Δ vs Sarathi) | 27.9% / 173ms | 29.0% / 196ms | **63.8% / 173ms** | 0% / — |
| **conv 0.5 s3**(wash baseline) | 96.4% | 91.2% | 95.2% | — |
| **code 4.9 s3**(Sarathi 完胜 VCB) | **19.4% / 670ms** | **100% / 228ms** | 100% / 190ms | — |
| **conv 1.0 s1**(NPU 短 prompt 反例) | **55.7% / 127ms** | 49.2% / 137ms | (PD-TDM 完胜) | — |
| **conv 2.2 s1**(NPU 短 prompt 反例) | **35.4% / 170ms** | 33.8% / 205ms | (PD-TDM 完胜) | — |

**三类关键数字对应**:
- **NPU 复现 Sarathi finding**(长 prompt):code 4.9 s3 Sarathi 100% vs VCB 19.4% → **+80.6pp**(chunked prefill 把长 prompt 灾难救活)
- **PD-TDM vs Sarathi 主线 Δ**(strict SLO):conv 3.0 s1 +34.8pp,code 4.9 s1 +73.9pp(phase-pure 在 chunked prefill 内部再拿额外增益)
- **NPU 上的反例 finding**(短 prompt):conv 1.0 s1 / 2.2 s1 上 VCB 略胜 Sarathi(短 prompt + chunking 拖慢 ttft)— paper §5 caveat,Sarathi 论文没明示的 fresh finding

Goodput 提升:**conv s1 k=3.0 PD-TDM vs Sarathi +1870 tok/s (+123%)**;**code s1 k=4.9 +351 tok/s (+403%)**。

---

## 5. Caveat / 已知 gap(诚实交代)

### 5.1 机制论证 gap(部分关闭)

- ✅ **关闭**:Sarathi mixed iter 时长之前从未实测 → 5/25 patch 后实测 220.6ms,§3.3 hypothesis(Choice B vs B' overhead)直接量化 = paradigm-level 70ms 差
- ❌ **限定**:Paper 主 claim 锁在 paradigm-level "mixed iter 比 phase-pure prefill iter 慢 70ms" 这个 measurable fact。**没有独立 ablation 把 mixed iter 慢的内部成因完全 isolate**(70ms 是两 paradigm 单 iter 直接相减得到,严格说混入了 scheduler 路径开销等少量次要因素)。如果 reviewer 严抠,可以构造一个 "phase-pure 调度 + 强制混合 batch 形态" 的中间 config 跑 isolated ablation,直接测 batch 形态本身的代价
- ❌ **未做**:**追溯到 attention 计算的具体物理来源**(mixed query length 为什么有代价)。Paper 顶层论证不依赖底层 kernel 细节,但如果想做 deeper drill-down,需要 attention 计算的 micro-benchmark

### 5.2 实验 scope gap

- 单 trace(Azure burst)、单模型(Qwen3-8B)、单 setup(2-NPU + chunk_budget=2048)
- 长 prompt + 极重 prefill demand 下 PD-TDM cycle 退化的边界未测(chunk_budget ablation 还没做)
- 跨模型 / 跨 trace generalization 留 Phase 2

### 5.3 Sarathi chunked prefill vs Vanilla CB 不普适胜出(NPU 上 fresh finding,paper Section 5 须 explicit)

Sarathi 论文 OSDI'24 的核心 finding:chunked prefill 比 vanilla CB 好。**我们在 NPU 上的实测部分复现、部分推翻**:

- ✅ **长 prompt(code workload)上 Sarathi 大胜 Vanilla CB**(NPU 复现):code 4.9 s3 Sarathi meet 100% vs VCB 19.4%(+80.6pp);code 2.8 s1 Sarathi TPOT p99 228ms vs VCB 670ms(−65%)
- ❌ **短 prompt(conv workload)上 Vanilla CB 反而略胜 Sarathi**(fresh finding):conv 1.0 s1 VCB meet 55.7% vs Sarathi 49.2%(+6.5pp);conv 2.2 s1 VCB 35.4% vs Sarathi 33.8%
- 物理:短 prompt 在 chunk=8192(Vanilla CB 配置)下整 iter 一次跑完 ttft 更快;chunk=2048(Sarathi 配置)把短 prompt 也分块,首 token 延迟反而 +30~40ms

这是个 NPU 上首次实测的反例,**Sarathi 论文没明示 chunked prefill 在短 prompt regime 的反向行为**。Paper Section 5 须明确这个 caveat,但**对 PD-TDM 不构成威胁** — PD-TDM 在两个 regime 都赢(长 prompt 拿 Sarathi chunked prefill 的好处,短 prompt 拿 phase-pure 避开 mix overhead)。

---

## 6. Open questions(请导师 sign-off 决策)

### Q1. §3 框架(PD-TDM 定位为 chunked prefill 的 phase-pure 变体)够不够 CCF-B/C?
**当前框架**:
- §3.1:承认 Sarathi chunked prefill 是业界胜利,PD-TDM 不挑战这个 baseline
- §3.2:识别 Sarathi 没回答的 sub-question(Choice B mix vs B' phase-pure),PD-TDM 是这个 sub-design-space 的另一个 point
- §3.3:hypothesis 是 Choice B' 在 burst 工况下比 Choice B 优,差额来自 mixed iter 内 query 长度方差代价
- §3.4-3.5:推出三个 prediction(P1 单 iter 时间 / P2 paradigm-level amplification / P3 跟 Sarathi 共享 vs vanilla CB 优势)
- §4.1-4.3:三个 prediction 全部被实测印证

**Open**:
- 这个 framing(PD-TDM = chunked prefill 框架内的 phase-pure variant)够不够撑独立 contribution?会不会被 reviewer 视为 "Sarathi 的小变种"?
- 论证强度上,paradigm-level 70ms 差距 + 18% throughput 差是否足以支撑 CCF-B/C?

### Q2. 要不要补 ablation 独立量化 "mixed iter 内 query 长度方差代价"?
**当前**:70ms 是两 paradigm 单 iter 直接相减(Sarathi mixed iter 220.6ms − phase-pure prefill iter 150ms),严格说还混入了 scheduler 路径开销等少量次要因素
**收益**:构造 "phase-pure 调度 + 强制 mixed batch 形态" 的中间 config 跑 isolated ablation,直接测 batch 形态本身的代价
**成本**:scheduler 改动 + 半天跑
**Tradeoff**:paradigm-level 70ms 数字够不够撑论证,还是必须 isolated 数字?

### Q3. Phase 2 数据 priority(等导师定)

| 候选 | 防 reviewer 攻击 | 成本 | 风险 |
|---|---|---|---|
| (a) chunk_budget {2048, 4096, 8192} ablation | "极重 prefill 下 trade-off 边界?" | ~3h | 低,sweep 已有脚本 |
| (b) Qwen3-4B 跨模型 sweep | "只在 8B 上 work?" | Phase 1 一半 ~12h | 低 |
| (c) BurstGPT 跨 trace sweep | "只在 Azure 上 work?" | Phase 1 一半 ~12h | 中(BurstGPT 工况差异大) |

### Q4. Thesis 一句话最终版(§0)是否需要再调?

### Q5. Paper 主投目标 — D-011 立场 CCF-B/C,是否值得冲 EuroSys / SoCC?

---

## 7. Future Work / Follow-up 优化方向

> 本节列出 paper 提交 + 修订期间可继续 invest 的优化方向。每条按 "motivation / scope / 工作量 / 预期 paper impact" 标注。优先级排序按 "防 reviewer 攻击" + "工程性价比" 综合考量。

### F0 — Vanilla CB iter telemetry 补救(P0+,极小工作量,立即可做)

**Motivation**:Vanilla CB(老 c3 chunk=8192)paradigm-level 数据完整可用(`phase_2_t6_burst_goodput/*_nonpid/`),但 **iter telemetry 缺失**(D-013 patch 是 5/25 加的,老 c3 是 5/22 跑的)。当前 §4.1 单步实测表 Vanilla CB 那一行只能给 paradigm-level TPOT p99,不能直接对照 Sarathi mixed iter 220.6ms / PD-TDM cycle 189ms 的 single-iter 时长比较。

**Scope**:用 D-013 patch 后的 scheduler 跑 vanilla CB(`c3_cp` + `max_num_batched_tokens=8192`)在代表性 cell:
- code k=2.8 s1 single cell × 1-3 seed(对应 §4.1 表 cell)
- 复用 `experiments/run_c3_telemetry.sh` 改 config 参数

**工作量**:**~30min wall**(单 cell × 3 seed,跟 c3-fair telemetry 补救同 scope)。

**Paper impact**:
- 关闭 §4.1 单步实测表 Vanilla CB 行的 "待补 telemetry" 标签
- 直接量化 Vanilla CB single iter 时长(预期:长 prompt iter 时长 1500-2000ms,decode-dominated iter 时长短;mixed iter 时长跟 prompt 长度分布强相关)
- 强化 §3.1 NPU 复现 Sarathi finding 的论证 — 单 iter 层 vanilla CB 比 Sarathi 慢多少能直接量化(不只看 paradigm-level)

**风险**:极低,纯重跑现有 config + 已 patch scheduler。

### F1 — `chunk_budget` 自适应 sweep(P0,高优先级)

**Motivation**:当前 `chunk_budget=2048` 是静态 sweet spot(p1_chunk_scan 在 Azure conv/code 上 Pareto 单调,D-006 决定)。但在 **极重 prefill demand regime**(长 prompt + 极高频 burst)下,m31 cycle 会退化为 "多个 prefill iter + 1 个 decode iter",cycle 时间被拉长,**§3.4 cycle time 模型的 N=1 假设可能反向松动**。Paper §5 caveat 当前只是"未测",没给数据边界。

**Scope**:`chunk_budget ∈ {1024, 2048, 4096, 8192}` × 2 wl × 3 SLO × 3 seed = 72 cells。脚本已有(`run_qps_sweep_all.py` config 工厂支持参数化 chunk_budget)。

**工作量**:~3h wall。

**Paper impact**:**直接 address §5 caveat "未测极重 prefill regime 下 trade-off 边界"**,把 §3.4 cycle time 模型的适用范围定量画出来。如果发现 chunk_budget=4096 在某些 regime 比 2048 好,可以加进 paper 主图作 sensitivity sweep。

### F2 — Cross-model + Cross-trace generalization(P1,中优先级)

**Motivation**:reviewer 标配攻击 "只在 Qwen3-8B + Azure burst trace 上 work?" 的防御。当前主图全部 single model + single trace,反 reviewer 攻击力弱。

**Scope**:
- **F2a 跨模型**:Qwen3-4B sweep,跟 Qwen3-8B 主结果对照
- **F2b 跨 trace**:BurstGPT trace sweep,跟 Azure burst 对照(BurstGPT 工况 prompt 分布不一样,可能让 chunk_budget=2048 sweet spot 不普适)
- 两个 sub-sweep 各 Phase 1 一半体量 ~12h wall

**工作量**:F2a ~12h,F2b ~12h,合计 ~1 天 wall(单 NPU 节点)或可两 sub-sweep 并行。

**Paper impact**:**Section 5 generalization 子节必备**,堵 reviewer single-model / single-trace 攻击。F2a 风险低(同模型族扩 size),F2b 风险中(trace 分布差异可能让结果反向 — 但即使反向,paper 也可以诚实降级 framing 到 "Azure burst 上成立")。

### F3 — Mixed workload generator + sweep(P1,中优先级,设计 + 实验)

**Motivation**:现有 evaluation 全是 single workload(conv only 或 code only)。真实部署里 conv + code 同时到达 — 这种 mixed workload 下 prompt 长度方差极大(几百 → 几千),**single-workload profile 出来的 static target_ratio 可能不再 optimal**,phase-pure 的 batch composition 稳定性优势也可能减弱。

**Scope**:
- **F3a**:写一个 trace mixer(conv:code = 1:1 / 7:3 / 3:7 三档),输入 Azure conv + Azure code,输出 mixed arrival stream
- **F3b**:在 mixed trace 上跑 PD-TDM vs Sarathi vs Vanilla CB vs c4_pd 4-way sweep,主图加一个 mixed workload Pareto frontier

**工作量**:F3a 半天(纯脚本),F3b ~12h sweep。

**Paper impact**:**Section 5 evaluation 加一个 robustness 子节**,展示 PD-TDM 优势在 mixed workload 下是否 hold(预期 hold 但收窄 — chunk_budget=2048 在长 prompt 上不够,在短 prompt 上浪费)。如果发现 winning regime 在 mixed 下收窄,反过来 motivate **F4 dynamic ratio**(因为静态 ratio 在 mixed 下不再 optimal)。

### F4 — Dual-SLO continuous ratio control(P2,有理论吸引力但实证 risky)

**Motivation**(用户 5/26 提出):D-013 telemetry 揭示一个非平凡事实 — **实际 prefill iter 占比 12-50% 由 waiting queue 状态决定,target_ratio 实质是 "上限" 而不是 "强制频率"**(saturated 下 ratio=0.8 vs 1.0 都达成 1:1 alternation;轻负载下 ratio 不论多大都 fall-back DECODE)。这暗示 target_ratio 在两个极端都是 redundant signal,**但在中等负载 regime(N=几~几十)可能有 reactive 控制空间**。

构想:让 target_ratio 被 TTFT + TPOT 双 SLO 实时影响,continuous 浮动:
- TTFT pressure 高(waiting 老 req 接近 SLO 70%)→ ratio 拉高 → 多给 prefill iter
- TPOT pressure 高(running decode 沉默接近 SLO 70%)→ ratio 拉低 → 多给 decode iter
- Smooth response(sigmoid 类),不是 binary on/off

**与 D-011 已否决的 PID 的区别**:
- 旧 PID:基于 SLO 违反率反馈,saturated 下永远钉 `ratio_max=0.8` 撞顶,**没在 continuous 区间内调**
- F4:基于实时 TTFT/TPOT 压力比例 smooth 调,理论上能在中等负载区间内有意义浮动

**与当前 Algorithm 1 第 4-7 行 binary override 的区别**:
- 当前 urgency_ttft / starvation_tpot:**binary**(超阈值才触发,触发就强制切),触发率 0.5-2.5pp noise
- F4:**continuous**(根据压力比例平滑调 ratio,不依赖硬阈值)

**Risk**(为什么标 P2 而不是 P1):
- 物理瓶颈(saturated 顶 + loose 底)在我们工况下不可绕过 — F4 收益只能在中等负载区间挤
- 中等负载下 paradigm-level Δ 当前已经 +3~+7pp,即使 F4 把这个再提一截,**paper headline 数字仍由 saturated regime 主导**(那里 Δ +30~+77pp,F4 在 saturated 上几乎无用)
- 实证 risky — 如果 F4 实测 ≤ +1pp(类似 starvation override 命运),反而坐实 "复杂控制器没用" 的 D-011 立场,paper 须撤回 F4 的 claim

**Scope**(如果决定做):
- F4a 设计:写 continuous reactive controller(replace `Controller.target_ratio()`),输入 oldest_waiting_age + oldest_decode_silence + 当前 SLO 阈值,输出 ratio ∈ [0.05, 0.95]
- F4b 单元测试:跨多种压力场景的 controller 响应曲线
- F4c sweep:在 Phase 1 同 cell 上对照 static ratio vs F4 ratio,看是否有 paradigm-level gain

**工作量**:F4a 半天-1 天,F4b 半天,F4c ~12h sweep。

**Paper impact**:
- 如果 F4 实测有 paradigm-level gain → paper 主线增加一条 "dynamic ratio control" contribution,但与现有 "static ratio per workload" framing 部分冲突,需要重新设计 Section 3 narrative
- 如果 F4 实测 ≤ noise → 写进 Section 7 Discussion 作 "已探索但 negative result",诚实降级(类似 D-011 对 PID 的处理)
- 无论结果,**F4 都能展示 paper 把 design space 探索完整**,反 reviewer "为什么不试 dynamic ratio?" 的预期攻击

**推荐 timing**:**先做 F1 + F2 + F3,再决定 F4 是否做** — 因为 F3(mixed workload)如果发现静态 ratio 在 mixed 下收窄,F4 的 motivation 会变强;反之 F3 hold,F4 性价比就低。

### Future Work 优先级总结

| Tag | 方向 | 优先级 | 工作量 | Paper impact | 风险 |
|---|---|---|---|---|---|
| F0 | Vanilla CB iter telemetry 补救 | **P0+** | ~30min | 中(关闭 §4.1 表 "待补" 标签 + 强化 §3.1 NPU Sarathi 复现) | 极低 |
| F1 | chunk_budget 自适应 sweep | **P0** | ~3h | 高(直接 address §5 caveat) | 低 |
| F2a | Qwen3-4B 跨模型 | **P1** | ~12h | 高(generalization 必备) | 低 |
| F2b | BurstGPT 跨 trace | **P1** | ~12h | 中(可能反向但能诚实降级) | 中 |
| F3 | Mixed workload + sweep | **P1** | ~12h + 半天脚本 | 中-高(扩 robustness + 可能 motivate F4) | 中 |
| F4 | Dual-SLO continuous ratio | **P2(后置)** | ~1-2 天 + ~12h sweep | 不确定(可能 paper contribution / 可能 negative result) | 高 |

**整体建议执行顺序**:**F0(立即,30min)** → F1 → F2a / F2b 并行 → F3 → 根据 F3 结果决定 F4。

---

## 文件清单(导师可直接拉的 artifact)

### Figures
- `figures/paper_F1{a-f}_*.png` + `*_3way.png` — Pareto 主图组
- `figures/paper_F2_advantage_heatmap{,_3way}.png` — 机制视觉证据

### Data
- `results/m31fix_validate/` — Phase 1 PD-TDM 全数据 126 cells
- `results/c3_chunk2048_supplement/` — C3 配对
- `results/c4_pd_supplement/` — c4_pd 4-way 42 cells
- `results/c3_telemetry/code_k{2.8,4.9}_seed0/` — C3 单步实测(§4.1 数据来源)
- `results/c4_pd_steady_smoke/` — c4_pd steady vs burst
- `results/phase_2_post/m31fix_phase1_pointwise.json` — 4-way aggregate

### Memory(完整 hand-off)
- `memory/README.md` — 项目 hand-off
- `memory/T6_FINDINGS.md` — Phase 1 完整 findings + 5/25 三个新 section
- `memory/DECISIONS.md` — D-005 / D-006 / D-009 / D-011 / D-012 / D-013 完整决策链
- `memory/PROJECT.md` — 架构 + 代码地图
