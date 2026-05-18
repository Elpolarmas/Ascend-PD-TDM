# 决策日志

> 记录每次方向调整的来龙去脉。任何跟 `PROJECT.md` 「不能偏离的几条原则」抵触的修改 → 必须先在这里写一条新决策。时间倒序。

格式:每条独立,固定字段。状态 = active(当前生效)/ superseded(被新决策取代)/ pending(候选,未拍板)。

---

## D-011(2026-05-18)thesis 完整重定位:PD-TDM paradigm + SLO-aware fast-loop selector + Framing C scope

- **触发:** P1.9d + P1.9e + 完整 reviewer attack 分析三件事一起 force 重新定位 thesis。
  1. **P1.9d 4-way ablation 实证:** A 无 SLO 自适应 / B 只慢回路 PID / C 只快回路 / D 慢+快(M3.1 完整)。saturated 上 D-A ≈ 0 ~ +3.75pp(noise 内),**B-A ≈ 0(慢回路 saturated 上无贡献)**,**双维度协同 D ≈ max(B, C) 不存在**。conv 上 C-A = +1-2.5pp(快回路主贡献),code 上 C-A ≈ 0
  2. **P1.9e workload transition 实证:** M3.1 vs M1@r08 在 conv2code transition 上 -2.68pp,在 code2conv 上 -0.21pp(noise)。**慢回路 PID 在 transition 上也无 +Δ,saturated regime 唯一可能有 +Δ 的场景被实证证伪**
  3. **Reviewer attack 系统打磨**(用户 5/17-18 多轮讨论):
     - A 类 (novelty):"PD-TDM 是 Sarathi-Serve + DistServe 的 interpolation" / "Mechanism well-studied" → 需要新 angle defense
     - B 类 (scope/significance):"single-node scope 太窄" / "+1-2.5pp 太边际" → 当前 SLO meet% metric 跟 DistServe goodput metric 不可比较,被 attack 量级小
     - D 类 (specific finding):MuxWise (ASPLOS'26) 已经做了 GPU 上的 PD multiplexing,但**明确排除 NPU 平台**("supports intra-process spatial sharing")—— 这反而是 explicit gap
     - 用户指出 1P:1D disagg 不是 fair baseline(资源受限被迫配置),不能 anchor "disagg 反优势" 论点
     - 用户指出 thesis 原意是 goodput at SLO 提升,P1.x 只测 SLO meet% 是 methodological 缺口
- **旧 thesis(D-002 / D-009 / D-010 联合状态):**
  - 「分段控制器 + 双维度协同」:慢回路 PID 启动爬升 + 边界保护;快回路 selector iter 粒度双向预警
  - 论文 framing 隐含 "M3.1 stack vs C3 stack" paradigm bundle 对比(D-010),"M3.1 是另一个 regime 的方案"(D-009)
  - 主 metric = SLO meet% at fixed QPS(saturated regime)
- **新 thesis(D-011):**
  - **核心 claim:** PD-TDM 是 single-node multi-device accelerators(lacking SM partitioning support)上的 temporal P/D multiplexing paradigm,提供 goodput-centric improvement under SLO constraints。SLO-aware fast-loop selector 在 conv tight-TTFT regime 提供 mechanism-level 加成
  - **删除 claim:**
    - 慢回路 PID 不再作为 contribution(P1.9d/e 全证伪;代码保留 internal,paper 不提)
    - 双维度协同不再作为 thesis 核心(P1.9d D ≈ max(B,C) 实证证伪)
    - Starvation tpot override 删除(P1.7c 触发率 0.5-0.8% + ±1pp noise,无效;代码保留,paper 不提)
    - Graph-aware 模块 D-005 三选一定为 F only(不进 contribution,只在 Implementation 节一句 fairness setup)
  - **保留 claim:**
    - phase-pure paradigm 物理 fingerprint(iter trace decode silence p99 800-940ms,99.7% prefill 时 decode queue 非空)
    - paradigm-level Δ 双向 trade-off(conv tight TTFT M3.1 +7.58pp / conv relaxed C3 dominate / code 两者都瘫)
    - SLO-aware fast-loop selector(token bucket + urgency_ttft per-request sorting)= mechanism contribution
    - target_ratio = static config 设计选择(不动态调,paper Section 5.6 sweep + recommended config)
  - **新引入的 conceptual contribution:**
    - **Interference shifting framework** 作为 paradigm choice 的 unified lens:Chunked Prefill = mean-domain interference,PD-TDM = tail-domain interference,Spatial multiplexing (MuxWise) = avoid interference (requires SM partition),Disaggregation = eliminate interference (requires cheap KV transfer)
    - **Goodput-centric metric framework:** Primary metric = max sustainable QPS at SLO target ≥ X%(DistServe-compatible);Secondary metric = SLO meet% at fixed QPS(saturated quality view)。Pareto frontier 作为主图呈现
- **Paper scope strategy(Framing C hybrid):**
  - **Conceptual scope:** "Single-node multi-device accelerators **lacking SM partitioning support**"(包括 NPU / older GPUs / 未来 ASIC)。这个 scope 是 MuxWise paper "supports intra-process spatial sharing" scope 的 explicit complement set
  - **Evaluation scope:** Ascend 910B3 NPU 作为 representative platform(2 张 NPU,TP=2)
  - **Title:** 不锁 NPU,Section 1 scope statement 明确(待写作期定 title 具体措辞)
- **完整 paper structure(9 sections):**
  - S1 Intro / S2 Background+Motivation(完整 4-paradigm landscape) / S3 Design(PD-TDM + fast-loop) / S4 Implementation(state probe 三件套 + fairness setup) / S5 Evaluation(Pareto frontier + 4-way + ablation + cross-model + physical evidence) / S6 Discussion(interference shifting unified lens) / S7 Related Work(cite MuxWise + Semi-PD + Sarathi + DistServe) / S8 Limitations(single platform / single family / no online switching) / S9 Conclusion
- **实验需要补的(W1 启动):**
  - **QPS sweep**(8 QPS × 4 configs × 2 traces × 3 seeds):goodput-centric metric 数据,1-2 天 wall
  - **跨模型 sweep**(Qwen3-4B + Qwen3-8B 已确定,不加 Llama / Llama-3.2 / MoE):2-3 天 wall
  - **c4_pd 补跑**(Azure trace + Qwen3-8B,跟主 evaluation 框架对齐):1-2 天
  - **BurstGPT trace 加入**:1-2 天
  - **Mixed workload(conv + code 同时到达)**:1-2 天,decide 是否扩 winning regime 到第 2 个
- **Defense framework summary(完整 reviewer attack defense):**
  | Attack 类 | Defense anchor |
  |---|---|
  | A1 PD-TDM 是 interpolation | Interference shifting reframe + V 字形 trade-off 主图(非 monotone)+ mixed workload 数据 |
  | A2 Mechanism well-studied | Writing 上承认 primitive 不新,定位为 paradigm realization 一部分 |
  | B1 single-node scope 窄 | Resource utilization industry consensus + 撤回 disagg 反优势 anchor + 1P:1D 降级 reference point |
  | B2 +1-2.5pp 太边际 | Goodput-centric metric framework + Pareto frontier + paradigm Δ +7.58pp 作 headline |
  | C1 没跑 Sarathi official | vLLM-CP 是 Sarathi paradigm 的 open-source impl,vllm-ascend 已是 production stack |
  | C2 单一 hardware | Framing C scope generic("lacking SM partitioning")+ paradigm physical independence argument + cross-model 部分弥补 |
  | D1 disagg 反优势 Ascend-specific | B1 修订后已 defused,1P:1D 是 budget reference,不 claim 普遍反优势 |
  | D2 phase-pure trade-off 已 cite | **MuxWise (ASPLOS'26) explicit 排除 NPU,我们填 gap;Framing C scope 是 MuxWise scope 的 complement set;Interference shifting 是新 unified lens** |
  | E1 target_ratio hyperparameter | Paper 主动 disclaim,Section 5.6 sensitivity sweep |
  | E2 复现性 | 全 stack 开源(vllm-ascend Apache 2.0 + TDM 插件)+ Ascend 商业硬件 + 公开 trace |
- **影响文档:**
  - `PROJECT.md`:完整重写(本次 W1 第 2 步)
  - `EXPERIMENTS.md`:加 P1.9d / P1.9e 入主表,加 W1 待跑实验
  - `FINDINGS.md`:加 P1.9d/e 实证条目,interference shifting framework 作为 unified lens 记录
  - `design/paper.md`:完整重写 paper outline(W2 起做)
  - `design/slo_pid.md`:慢回路 PID 角色 final 收尾(代码留 internal,paper 不写)
  - `design/sampling.md`:Sampling 模块降级为 evaluation methodology(SLO grid 校准),不进 paper contribution
- **影响代码:**
  - 慢回路 PID(`SLOReactiveController`)代码保留 internal use,paper 不写
  - Starvation tpot override(`starvation_decouple_bucket` flag)代码保留 internal,paper 不写
  - Mixed_mode 开关(interfaces.md 缺口 #5)**不再做** —— 之前认为需要它做 phase-pure 单变量 ablation,现在 paradigm-level Δ 不归因到单机制(D-010 立场延续),不需要这个 ablation
  - Graph-aware 模块代码保持现状(D-005 已 settled F only)
- **风险:**
  - **QPS sweep 数据可能不利:** lower QPS 上 paradigm 差异可能消失或反向。如果 PD-TDM 在所有 SLO target 下都不能 sustain 比 C3 更高 QPS,B2 defense 弱化。Mitigation: 多个 SLO target reporting(60/70/80/90),Pareto frontier 让 reviewer 看完整 picture
  - **Mixed workload 可能不赢:** 如果 conv+code 混合到达上 PD-TDM 不能赢 C3,winning regime 仅限 conv tight-TTFT(narrow scope)
  - **Framing C scope claim 部分 hypothetical:** "older GPUs / future ASIC" 在 lacking SM partition class 内但我们没实测验证。Mitigation: Limitations 节诚实承认 + scope generalization 作为 conceptual/future-work
  - **MuxWise scope clause 解读:** 我们 explicit cite MuxWise 排除 NPU 的句子作为 motivation。如果 MuxWise reviewer 不同意我们解读,可能 attack。Mitigation: cite 原句,framework 镜像 MuxWise scope clause,leave interpretation to readers
  - **Top venue(EuroSys / SoCC)中签率 15-25%**(reviewer 仍可能觉得 contribution incremental + single platform)。Mitigation: 主投 B 中下(IPDPS / DSN / Middleware)+ fallback CCF-C(IISWC / ICPADS / HPCC)
- **状态:** active
- **Supersedes:**
  - D-002(分段控制器 + 双维度协同):双维度协同部分 superseded;分段控制器收缩为 fast-loop only
  - D-005(Graph 模块三选一 pending):final 定为 F only
  - PROJECT.md L17-21 不能偏离 §1(thesis 旧表述):被 D-011 新 thesis 取代

---

## D-010(2026-05-15)phase Δ 归因层级修正 + 双维度 ablation 结构

- **触发:** 用户 5/15 讨论指出 D-009 表述有两个方法论漏洞:
  1. 「phase-pure 贡献 86-101%」是把 paradigm-level 差异强行归因到 phase-pure 单变量。M3.1 vs C3 整套代码 stack 都不同(TDMScheduler / vLLM CP scheduler、chunk_tokens、max_num_batched_tokens、graph 配置),phase Δ 实际上是 bundle vs bundle 总差,内部机制不可单变量归因
  2. SLO 自适应本来就是双维度设计(D-002):慢回路 PID(tick 粒度,启动爬升 + 边界保护)+ 快回路 selector(iter 粒度,双向预警)。「PID 钉 ratio_max 因此 PID 没用」这种判断只看了慢回路一条腿,真正的 SLO 自适应力量在快回路(P1.7b 未实装)
- **旧:**
  - D-009 narrative:「M3.1 vs C3 差距 86-101% 来自 phase-pure 调度」(单机制归因)
  - 论文 ablation 隐含拿 C3 当 same-stack 对照
- **新:**
  - phase Δ 表述软化为 「TDM-paradigm vs CP-paradigm bundle 总差」,**不再分配到具体机制**。论文写法上明列两套 bundle 内容,不试图 isolate phase-pure
  - mechanism-level claim 改由**同 stack ablation**支撑:
    | claim | 同 stack 对照 | 状态 |
    |---|---|---|
    | phase-pure(P/D 时分复用)有用 | TDM mixed_mode on/off | **需要加 mixed_mode 开关** |
    | 慢回路 PID 真正角色 | M1 静态 ratio 扫到 ratio_max vs M3.1 | 不依赖新代码,可立即跑 |
    | 快回路双向预警有用 | M3.1(慢) vs M3.1+P1.7b(慢+快) | 依赖 P1.7b 实装 |
    | 双维度真协同 | 只快 vs 慢+快 | 同上 |
  - C3 在新结构里定位为 **外部 reference point**(end-to-end SOTA 对比),不当 ablation。代码 stack 差异这个 confound 在 reference 对比里是合法的,不需要消除
  - 论文 threats to validity 加两条:bundle 内不可拆 + TDMScheduler 跟 vLLM CP scheduler 实现质量差异 inherent confound
- **影响文档:** `design/paper.md`(§6 贡献声明 + §7 TtV + 新 ablation 矩阵),`design/slo_pid.md`(慢回路角色重定位),`design/interfaces.md`(加 mixed_mode 缺口),`FINDINGS.md`(P1.8 节注解 paradigm-level),`EXPERIMENTS.md`(三条 pending 实验)
- **影响代码:** TDMScheduler 加 `mixed_mode` 开关(中等工程量,几天)。M1 ratio_max 扫描只是配置改动,不动代码
- **launch 参数对齐(2026-05-15 收窄):** 既然接受 paradigm-level Δ 不归因到单机制,launch 参数对齐就不再是 D-010 根基,大部分参数本来就是 paradigm 各自的合理选择。只保留两件 sanity check:
  - c1/c3/m31 是否都开 graph capture(`enforce_eager` 一致)— 执行细节不一致是 bug
  - c3 默认 `chunk_tokens` 是多少 — 写论文时要知道 c3 跑在什么配置上;决定要不要补 c3@chunk=2048 做 sensitivity(不是对齐,是确认 c3 不在非最优点)
  - **不再要求** `max_num_seqs` / `max_num_batched_tokens` 等 paradigm-specific 参数对齐:强制对齐反而违反 paradigm 公平比较原则
- **风险:**
  - M1 ratio_max 扫描如果显示 M1@ratio_max ≈ M3.1,慢回路 PID 自适应贡献被实证弱化 → 接受,论文写"PID 提供启动爬升 + 边界保护"而非"在线 SLO 自适应"。这是诚实表述,不戳穿 thesis(thesis 第二条腿在快回路)
  - thesis 闭环更依赖 P1.7b。P1.7b 拖延 → mechanism-level claim 缺第二维度,paradigm-level reference 也不够
- **状态:** active

---

## D-009(2026-05-14)dedicated kernel 完全移出 contribution,phase-pure 双向化

- **触发:** P1.8 m31_fia ablation(`results/azure_m31_fia_ablation/`)。强制 M3.1 attention 走 FIA 后,所有指标变化 ≤ 13ms 或 < 1%,跟原 m31_2048 统计等价。M3.1 vs C3 总差距 86-101% 来自 phase-pure 调度本身,**kernel 贡献 ≈ 0**。同时 phase-pure 在 saturated 长 prompt 上 TPOT_p99 输 C3 992ms (M3.1 是 C3 的 2.5×),e2e 输 7.2s
- **旧:**
  - PROJECT.md 不能偏离 §5:「kernel 速度不作为论文主图,只解释 C3 反常」(隐含 kernel 仍是优势组件之一)
  - paper narrative 隐含「M3.1 vs C3 = phase-pure + dedicated kernel」两层叠加
- **新:**
  - kernel 完全移出 contribution。论文不再提 dedicated kernel 速度优势,vllm-ascend v0.13+ 删除 dedicated kernels 不威胁我们 narrative
  - phase-pure 显式双向化:**TTFT 紧 + 短 prompt → phase-pure 赢;TPOT 紧 + 长 prompt → mixed 赢**。M3.1 不再宣称单方面优于 C3
  - SLO-adaptive TDM 的卖点收紧为「根据 workload regime 动态选择 phase-pure / mixed」,这正好 motivate 分段控制器(D-002)+ 双向 selector(P1.7b)。论文 narrative 从「M3.1 全面胜出」改为「regime-aware 选择,在区分带 dominate」
- **影响文档:** `PROJECT.md` 不能偏离 §5(改写,kernel 移除)+ §其他论点连带核查,`design/paper.md`(主 narrative 重写 contribution section),`FINDINGS.md` Gap-5 已加,`EXPERIMENTS.md` 加 `azure_m31_fia_ablation/` 入主表
- **影响代码:** 无(`force_fia_attention` 字段 + `attn_patch.py` 保留作为可复现 ablation,默认 False 不影响主路径)
- **风险:** thesis narrative 从「M3.1 是更好的方案」变成「M3.1 是另一个 regime 的方案」,审稿要看 SLO-adaptive 部分是否真的能做出 regime-aware 切换效果 → P1.7b 双向 selector 必须 work,不然 thesis 论点缺第二条腿
- **状态:** active

---

## D-001(2026-05-14)P1.6h calibrated baseline 推迟到 P1.7b 阶段

- **触发:** 用户 5/14 要求 memory 整理 + 决定 P1.6f/g/h/i 脚本未跑就遗弃(见 D-004)。其中 P1.6h 跟其他三个不同:它是用调过的 SLO 档(conv=200ms)重跑全部 baseline 出主结果,跟「主报只在调过的档下报」直接相关
- **旧:** P1.6h 作为独立 sweep(6 calls × 22min)单独跑
- **新:** 脚本删除,实验推迟。P1.7b 阶段实装快回路 selector 时,baseline 顺便用调过的档跑一次,合并产出主结果
- **影响文档:** `EXPERIMENTS.md`(列为 pending,等 P1.7b),`PROJECT.md`「不能偏离 §2」(临时承认 `azure_main` strict 档作为临时主结果),`FINDINGS.md` 未决问题节
- **影响代码:** 无(脚本已随 D-004 删除)
- **风险:** 如果 P1.7b 拖延,主结果用 strict 档时间被拉长,审稿可能质疑。可接受
- **状态:** active

---

## D-002(2026-05-13)thesis 改为「分段控制器 + 双维度协同」

- **触发:** P1.6e 跨 4 档 SLO sweep(`results/azure_p16e/`)conv ratio 钉 max 80%+ 跨所有 SLO 档,`err_tpot_eff` 非零率最高 3.6%(PASS 标准是 >20%)。PID 在边界几乎必然单输入,**不是闭环反馈**
- **旧:** thesis claim = 「SLO-aware 闭环反馈控制 P/D 比例,tpot/ttft 双 SLO 作 runtime 受控变量」
- **新:** thesis claim = 「分段控制器:慢回路 PID 负责启动期把 ratio 爬升到合理值 + 饱和稳态做边界保护;快回路 selector 负责 iter 粒度 TTFT/TPOT 双向预警(P1.7b 待实装)」
- **影响文档:** `PROJECT.md` 不能偏离 §1,`design/slo_pid.md` 修正节,`design/paper.md` narrative,`FINDINGS.md` Gap-1
- **影响代码:** 无(代码层面慢回路 PID 已经是这个行为;快回路待 P1.7b 实装)
- **状态:** active(待 P1.7b 验证补全第二条腿)

---

## D-003(2026-05-13)Sampling 模块定位转向

- **触发:** P1.6e finding:SLO 档校准让 M2.4 屏蔽 99.5% → 5.4%(对反馈链有巨大影响),但 ratio 仍钉 max。SLO 校准**必要但不充分**
- **旧:** Sampling 模块 claim = 「提供 SLO target 让 PID 在双输入工作区工作」
- **新:** Sampling 模块的三个真实价值:
  1. 定义系统可服务边界(code 类工作负载无论 SLO 怎么调违例都极高 → 出 admission control 范畴)
  2. 提供主报结果的目标档(thesis 报这个档下 meet_slo%)
  3. 校准 PID 的「启动期爬升目标」,让爬升收敛到工程意义点而非物理不可达档
- **影响文档:** `design/sampling.md` 修正节,`PROJECT.md` § 三模块的 Sampling 节,`FINDINGS.md` Gap-2
- **影响代码:** 暂无(模块未实装)
- **状态:** active

---

## D-004(2026-05-14)P1.6f/g/i sensitivity sweep 全部遗弃

- **触发:** 用户 5/14 评估认为方向已定(P1.7b 是 thesis 第二条腿),P1.6f(target_violation_rate sweep)/ P1.6g(ratio_max sweep)/ P1.6i(chunk on azure)这类 sensitivity 实验价值降级
- **旧:** 三个 sensitivity sweep 脚本就位,各 ~2-2.4h wall,准备一夜串联跑
- **新:** 全部删除(P1.6f / P1.6g / P1.6h / P1.6i 共 9 文件:4 run sh + 4 posthoc py + 1 execute_guide)。P1.6h 单独推迟(D-001),其他三个永久遗弃
- **影响文档:** `EXPERIMENTS.md` 已废弃实验节,`FINDINGS.md` 已废弃路径节
- **影响代码:** 删除 `experiments/run_p16{f,g,h,i}*.sh`、`experiments/posthoc_p16{f,g,h,i}*.py`、`memory/p16_sweep_execute_guide.md`
- **状态:** active

---

## D-005(2026-05-12)Graph 模块降级为论文 finding

- **触发:** P1.0(FFTS+ baseline)+ P1.0b(AIV 23 sizes vs FFTS+ 15 sizes)。AIV 反而比 FFTS+ 输 -1.5~-23.8pp,**capture 数量非单调**。vllm-ascend 已内置 cudagraph_capture_sizes + decode 自动 pad
- **旧:** Graph 自适应模块是 TDM 三大控制模块之一(SLO + Sampling + Graph)
- **新:** Graph 模块没有算法空间,降级为论文 finding。三模块变两模块(Sampling + SLO 控制器)。论文形态待论文写作前三选一:
  - **G+F**:作为独立章节写入论文
  - **D**:合并到 chunking 节
  - **F only**:只留作 finding 引用,不进 contribution
- **影响文档:** `PROJECT.md` § 三模块,`design/paper.md`,`FINDINGS.md` Gap-4
- **影响代码:** 无(模块本来就没独立实装,只在 chunking 节有 capture sizes 对齐逻辑)
- **状态:** pending(三选一,论文写作前定)

---

## D-006(2026-05-08)chunking 不做 dynamic controller,收敛为静态 chunk=2048

- **触发:** p1_chunk_scan 实测,chunk_tokens ∈ {512, 1024, 2048, 4096, 8192} pareto 单调,chunk=2048 是 sweet spot。没有双向 trade-off,无 dynamic 设计空间
- **旧:** M3.4 设计是 SLOReactiveController 加 chunk_tokens 输出维度(运行时调)
- **新:** chunking 静态 chunk_tokens=2048,不做 dynamic controller。M3 的 thesis 价值收敛为「推进区分带边界」(让区分带从 ttft∈[500,1500] × tpot∈[100,200] 扩到更紧的档)
- **影响文档:** `design/chunking.md`、`design/paper.md`,`FINDINGS.md` Gap-3
- **影响代码:** 无(M3.4 controller hook 未实装,留 `prefill_chunk_tokens` 配置字段静态用)
- **状态:** active

---

## D-007(2026-05-07)锁 vllm-ascend v0.11.0rc1

- **触发:** vllm-project/vllm-ascend [PR #4623](https://github.com/vllm-project/vllm-ascend/pull/4623) 在 v0.13.0 删除 AscendScheduler(我们继承的父类),没有 phase-aware 替代接口。v0.13.0+ 强制走上游 V1 Scheduler + chunked prefill + FIA kernel(等价于 C3)。我们环境受限不能跑并行 v0.18 supplementary
- **旧:** 跟随上游版本升级
- **新:** 所有实验 / 论文锁 v0.11.0rc1。论文 §implementation 主动写明锁定理由,把 narrative 升级为「v0.11.0rc1 是最后一个保有 phase-aware AscendScheduler 的版本,恰好是研究 phase-aware 调度价值的最后窗口」
- **影响文档:** `PROJECT.md` § 版本锁,`design/paper.md` § threats to validity
- **影响代码:** 实验编排都锁这个版本
- **状态:** active(不可逆)

---

## D-008(2026-05-07)长 prompt sweep 双 pivot

- **触发:** P0-3 长 prompt SLO grid sweep(`results/long_prompt_sweep/`)发现:
  - strict 档 (ttft<500 / tpot<50):所有 4 个 config 趴地板 < 5%
  - loose 档 (ttft>2000 / tpot>300):所有 4 个 config 100%
  - 区分带 ttft∈[500, 1500] × tpot∈[100, 200]:M2.7 一致胜出
- **旧:**
  1. strict SLO 档作为 evaluation 主档
  2. M3 chunking 计划做在线 controller 调 chunk_tokens
- **新:**
  1. Sampling 模块从设计文档升格为实装目标(workload-conditional SLO 框架)
  2. M3 chunking 收敛为「推进区分带边界」(进一步发展为 D-006)
- **影响文档:** `design/sampling.md`、`design/chunking.md`、`PROJECT.md` 不能偏离 §2
- **影响代码:** 无
- **状态:** superseded by D-003(Sampling 模块进一步转向)+ D-006(chunking 收敛静态)

---

# 候选 / 暂未拍板

(写在这里的是脑暴中的方向,尚未变成 active 决策。落地前讨论)

- **(无候选)**——P1.7b 双向 selector 设计还在 `design/slo_pid.md` 快回路节讨论,未到决策阶段
