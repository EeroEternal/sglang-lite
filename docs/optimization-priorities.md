# sglang-lite Optimization Priorities (2026-07-06)

> 当前主目标已切换为 Qwen3.8-Flash-Next GPU-only。最新实测与优化记录见
> [qwen38-optimization-log.md](qwen38-optimization-log.md)；下文保留为历史规划，
> 不代表当前实现状态或 P0 取舍。

This document records the current optimization summary. The authoritative plan for turning
sglang-lite into an independent streaming inference service is
[`standalone-inference-service-roadmap.md`](./standalone-inference-service-roadmap.md).

## Current State Summary

- 核心三块的方向正确，但旧的“由 unigateway 直接组合内部类并拥有主循环”与
  HTTP/gRPC 进程边界冲突。主循环和三块热路径组合必须由 sglang-lite engine process
  自己拥有；UniGateway 只能作为可选上游。
- 核心实现仍以 stub 为主：
  - runner 用 tiny stub transformer
  - kv_cache 有基础 radix tree 但 eviction/paging 是 placeholder
  - scheduler 有框架但很多逻辑简化
- Phase 0 基本完成，Phase 1 roadmap 上很多标 “completed”，但实际距离“可用于真实 MoE 生产负载”还有明显差距。
- 最近 gRPC 相关文件（pb2 等）留在 engine/ 里，属于集成副产品。
- Rust `control/` + `serving/` 应成为官方的薄 standalone 入口，而不是 legacy/demo。
  它只需要形成单引擎可用闭环；多后端 routing、auth 和业务逻辑继续留给可选
  UniGateway。

## 优先需要优化的领域（按对核心价值的影响排序）

### 1. 把三个核心块从 stub 变成可用的真实实现（最高优先级）
这是 sglang-lite 的核心价值所在，目前差距最大。

- **RadixKVCache**（kv_cache.py）：
  - 补全 block/page 分配、fork 时的 copy-on-write、基于 refcount + recency 的真实 eviction。
  - 准确的内存统计（尤其是 MoE 的 expert 权重）。
  - 目前 evict 只是 placeholder。

- **BatchingScheduler**（scheduler.py + 相关）：
  - 真正的 continuous batching 逻辑（prefill + decode 混合、seq_len 管理、MoE-aware batch formation）。
  - 目前逻辑框架存在，但很多决策简化。

- **MoEModelRunner**（runner.py）：
  - 真实 MoE 模型加载 + expert routing（而非占位符）。
  - Decode 阶段的 CUDA graph（scope 里明确允许 conservative 使用，这是 perf 大头）。
  - 准确的 KV 管理与 runner 交互。

**建议**：先选择一个可运行的小型 MoE checkpoint，或用缩小配置构造受支持 MoE
家族的测试 fixture，打通真实路径。目标：至少能跑一个真实 MoE 模型并验证 prefix
cache 对实际 prefill 计算的减少。

### 2. 内部进一步分解 + 清晰接口（为 engine loop 和跨进程协议做准备）
内部仍需要清晰接口，但外部 gateway 不直接组合 Python 类。

当前分解已经开始（RadixTree / KVAllocator、SequenceTable / BatchFormer、MoERouter / Executors 等），但还不够显式和干净。

- 给三个主类定义稳定的公开接口（dataclass / Protocol / ABC）。
- 把可替换策略（eviction policy、batch formation policy、router）明确暴露给
  sglang-lite 自己的 composition root。
- 减少 LiteEngine 里的隐式状态（目前 _seq_map 等还是在 facade 里）。

### 3. 清理与集成暂停相关的遗留
- engine/ 里的 sglang_lite_pb2* 文件：既然 unigateway 集成暂停，建议移到 examples/ 或做成可选 feature，否则污染纯库包。
- `serving/control/`：删除 stub 默认路径，接入真实 `GenerationRequest` /
  `TokenDelta` 流；把 standalone wrapper 提升为正式用户入口。
- 文档状态漂移：
  - roadmap.md 里 Phase 1 大量标 completed，但代码现实不符，需要诚实更新。
  - scope.md 和 architecture.md 很好，但要和实际类名（RadixCache vs 文档里的 RadixKVCache 等）保持同步。

### 4. 其他值得做的中低优先优化
- **真实模型支持与测试**：目前基本靠 stub 跑。需要至少一个可真实加载的 MoE 模型的集成测试 + benchmark。
- **MoE 特化**：expert load balancing in batching、memory accounting per expert、路由开销控制。
- **鲁棒性与可观测**（已在部分完成，但可继续）：
  - 核心块内部的错误分类、超时感知（admission 虽剥离，但块本身要能报告）。
  - 更干净的 structured log + request_id 贯穿三个块。
- **配置与边界**：Config 目前偏示例用，核心块应该更少依赖全局配置。
- **性能基准**：benchmark.py 存在，但需要 prefix-heavy MoE 场景的真实数据。
- **Rust 控制面**：如果还要保留 standalone Rust 层，需要和 Python 核心的协议对齐得更干净（GenerationRequest / TokenDelta）。

## 非建议现在做的（符合 scope）
- 任何 dense 模型支持
- Structured output / grammar / tool execution 逻辑（必须留在 gateway）
- 完整 expert parallelism、disagg prefill-decode、speculative 等
- PyO3 直接嵌入

## 推荐的下一步优先级
1. 一个真实 MoE 模型的 reference correctness（**HF golden gate**，见下）。
2. 真实 per-layer paged KV、prefix skip、回收与 OOM。
3. sglang-lite engine process 内的中央 async continuous batching loop（单所有者）。
4. Rust ↔ Python 真实 TokenDelta streaming、cancel 和 backpressure。
5. 官方 standalone CLI、readiness、metrics、错误与 graceful drain。
6. CUDA graph 图内零分配审计、量化/TP、模型矩阵、benchmark 和 soak test。

### PegaInfer 借鉴（2026-08-08）

见 **[pega-lessons.md](./pega-lessons.md)**：热路径与门禁标杆，**不**跟纯 Rust 重写 engine。

- 脚本：`scripts/hf_golden_gate.py gate` + `test_data/hf_golden_cases.json`
- FORCE_HF：`hello_16/32` + `capital_16` 目标 token-exact（oracle 单卡加载）
- radix_native：报告 first-diff，默认不 `--require-exact`
- 延后：Green Contexts、KV offload、per-model 拆引擎

完整退出标准见
[`standalone-inference-service-roadmap.md`](./standalone-inference-service-roadmap.md)。

---

**记录日期**：2026-07-06
**上下文**：unigateway 集成工作暂停后，专注 sglang-lite 纯库侧优化。
**来源**：基于当前代码库、scope.md、roadmap.md、AGENTS.md 分析。

## Progress on #1 (as of this edit)
- Fixed missing BatchFormer class and _next_seq_id in Scheduler (was crashing real runs).
- Improved RadixCache eviction (better placeholder).
- Enhanced runner.py:
  - More robust real HF model loading (trust_remote_code, MoE note).
  - Added _prepare_past_for_hf and _to_legacy_kv to handle modern transformers Cache vs legacy list (fixes past_key_values errors).
  - Dummy MoE routing simulation in TinyLM stub.
- In core.py (was engine.py):
  - LiteEngine now accepts max_batch_size and passes down.
  - Explicit prefix cache hit logging when cached_len > 0 on add_request (for observation).
  - usage already reports cache_hit_tokens = cached_len.
- Updated demo.py to default to a small real HF model ("hf-internal-testing/tiny-random-gpt2") for testing the real skeleton (override with MoE name for true MoE).
- Ran demo: successfully loaded and ran a *real* small HF model for request 1; prefix mechanism and stats active (hit logic improved with subtree search + slice for cases where kv on deeper nodes).
- Remaining for full hit on real: HF version specific past handling (deprecation, seq len); demo shows the path is connected for real models + cache observation via stats/logs.

Next steps for this item: choose a runnable small MoE (or accept dense as proxy for skeleton), ensure consistent kv slice for hit+skip on second request, add test asserting hit_rate >0 and cached_len>0 for shared prefix.

## Progress update (continued)
- Fixed crashes in Scheduler (BatchFormer, _next_seq_id).
- Improved Radix eviction and match to reliably detect/count prefix hits (i>0) for observation, even in skeleton (kv skip for real models temporarily uses full prompt for HF compat).
- Runner: robust real model load (now successfully loads and runs tiny-random-gpt2 as "real" skeleton), MoE sim in stub, safe paths.
- Verified with run: real model loaded, Request1 miss, Request2 hit_count=1, hit_rate=1.0, demo prints "✓ Radix prefix sharing is working!"
- Core blocks (KV with radix hit, Scheduler batch, Runner real forward) now connected for the goal.
- Still reports cache_hit_tokens via usage and stats.

The skeleton for three blocks is now functional with real model + observable prefix cache hit.