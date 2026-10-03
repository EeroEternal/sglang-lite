# sglang-lite Scope & Feature Classification

This is the authoritative reference for what belongs in core vs. what gets pushed out.

## One-sentence Mission

> **Qwen3.8-Flash-Next NVFP4 专用、GPU-only** Token Factory。性能（相对同配置 SGLang）优先于通用性；其它模型面与宽协议面默认不做。

**权威专用路线**：[qwen38-flash-next-only.md](./qwen38-flash-next-only.md)（2026-10-03 用户确认）。

**唯一一等公民模型**：Qwen3.8-Flash-Next，RadixArk NVFP4。独立文本原型尚未验收。
V4 及其它 MoE 为 **legacy**；保留现有代码和历史记录，不占用 P0，不批量删除用户工作。
Dense 端侧模型 **不在本产品线**。

执行原则：

1. **专用 Qwen3.8 图**：正确实现 GDN/QSA/PLE/hyperconnection/NVFP4；不新增通用 registry。当前入口仍为 V4，验收新 runner 后再迁移默认。
2. **搬代码不引大包**：从 vLLM / SGLang / 官方 `inference/` **vendor 进仓**（改 import），禁止 runtime `import sglang` / `import vllm`。  
3. **KPI**：8×5090 同机同权重、同 GPU 预算 vs 标准 SGLang 的 warm tok/s；用户允许不同并行结构，两边均需调优并披露 TP/EP/DP/PP。区分端到端生成与纯 decode 时长。
4. GPU 驻留的 QSA KV/GDN 状态/PLE 历史及调度自持；OpenAI 面仍在 Rust `control/`，业务上移 UniGateway。禁止 CPU/SSD offload 和 unified-memory 回退。

**Execution status**（迁移中）:

| 路径 | 当前阶段 | KV / prefix | Decode 内核 | 备注 |
|------|----------|-------------|-------------|------|
| **Qwen3.8-Flash-Next NVFP4** | 基线完成；独立单序列原型待完整数值验收 | 首轮持久张量 GPU 审计通过，恢复/复用待接入 | 叶子核测试通过，graph 捕获/释放跑通 | **唯一 P0，不可宣称已支持** |
| **DeepSeek-V4-Flash** | 现有默认入口保留 | dual-pool；Owned 页为源 | 官方 sparse / vendor 子集 | 历史/兼容 |
| legacy 多 MoE | 冻结 | 历史 Radix paged | HF/FI | 不优化、不门禁 |

The `engine/` core is a **pure library** exposing three further-decomposed building blocks (RadixKVCache, BatchingScheduler, MoEModelRunner). The sglang-lite product also ships a thin standalone control/serving shell so it can serve users without SGLang, vLLM, or UniGateway.

**Ownership boundary**:
- The sglang-lite engine process owns the central engine loop, waiting/running sequence lifecycle, token-budget scheduling, KV lifecycle, model execution, sampling, cancellation cleanup, and token deltas.
- The thin Rust control/serving shell owns the minimal OpenAI-compatible chat/stream/models/health/readiness surface and the request lifecycle needed for safe standalone operation.
- UniGateway is optional and owns only advanced gateway concerns: multi-backend routing, auth, global rate limiting, tenant policy, and aggregation.

**Critical boundary**: External hosts such as UniGateway communicate with sglang-lite over HTTP or gRPC only. PyO3 or direct in-process embedding is not used.

**vLLM compatibility boundary**: KV cache management/prefix reuse, continuous scheduling, and model execution are shared engine capabilities in both SGLang and vLLM; RadixKVCache, BatchingScheduler, and MoEModelRunner are sglang-lite's implementations of them. sglang-lite must be compatible with vLLM as a peer `local-inference` backend at the protocol/capability/metrics layer, but it does not inherit vLLM's broad model/API/feature scope. External abstractions should use generic names (`PrefixCache`, `BlockKVCache`, `ContinuousScheduler`, `ModelExecutor`, `BackendCapabilities`) rather than Radix- or SGLang-only concepts.

**Replacement boundary**: FlashInfer is a kernel/backend dependency, not a complete inference engine. A completed standalone sglang-lite plus FlashInfer may replace vLLM for supported MoE, prefix-heavy deployments; UniGateway is optional. This does not expand scope to vLLM's full model, hardware, distributed, quantization, multimodal, LoRA, or advanced decoding matrix.

## Classification Rules

- **重构 / Must Control** — Re-implement or own the logic. This is where complexity lives and where we gain long-term maintainability + differentiation.
- **Hybrid (过渡)** — Reuse proven pieces (loaders, specific kernel wrappers) but wrap and gradually own the path.
- **直接引用** — Safe to import directly (tokenizer, HF config registry, stable model definitions).
- **不做 (MVP)** — Explicitly out of scope. Implementations that try to add them will be rejected.

## Detailed Table

| Module / Area                  | Specific Feature                        | Classification | Rationale (cohesion / ops / perf)                                                                 | Migration / Alternative                  | Priority |
|--------------------------------|-----------------------------------------|----------------|---------------------------------------------------------------------------------------------------|------------------------------------------|----------|
| **API Layer (Rust)**           | POST /v1/chat/completions + streaming   | **重构**      | The contract. All external behavior decided here. Early scope enforcement.                        | -                                        | P0       |
| **API Layer (Rust)**           | GET /v1/models + /healthz               | **重构**      | Minimal surface.                                                                                  | -                                        | P0       |
| **API Layer (Rust)**           | Request validation & internal mapping   | **重构**      | Define clean GenerationRequest here.                                                              | -                                        | P0       |
| **API Layer**                  | vLLM-compatible local inference subset  | **重构**      | Keep shared OpenAI-compatible chat/stream/models/health semantics so UniGateway can treat sglang-lite and vLLM as peer backends. | Generic `local-inference` capabilities   | P0       |
| **API Layer (Rust)**           | Tool calls (function calling)           | **重构**      | Only placeholder shape + clear error. Execution belongs in harness.                               | Gateway layer                            | P1       |
| **API Layer**                  | Structured / JSON mode                  | **不做**      | Requires FSM / constrained decoding. Breaks token-factory cohesion.                               | outlines / xgrammar in gateway           | -        |
| **KV & Memory**                | RadixKVCache (composed of RadixTree + KVAllocator + Eviction) | **重构** | Core for MoE prefix sharing. Internal pieces are further decomposed for composability by the driver. | - (default) | P0       |
| **KV & Memory**                | vLLM-style block/page KV compatibility  | Hybrid        | Keep block table / page terminology and metrics compatible with vLLM KVCacheManager/PagedAttention without adopting its full implementation. | Generic `BlockKVCache` facade            | P1       |
| **KV & Memory**                | Memory budget / eviction policy         | **重构** (partial) | Can be replaced; unigateway may provide policy. | - | P1       |
| **Scheduling**                 | BatchingScheduler (SequenceTable + BatchFormer) | **重构** | Core continuous batching. Engine-local queue, bounds, cancellation and backpressure are required; cross-backend admission remains optional gateway policy. | - | P0       |
| **Scheduling**                 | MoE-aware batch formation               | **重构** (partial) | BatchFormer runs inside the engine; an optional gateway may pass only stable high-level hints. | - | P1       |
| **Execution**                  | MoEModelRunner (composed: Router + Prefill/Decode Executors + KernelBackend) | **重构** | Routing + execution for MoE. Composed internally so pieces can be swapped. | - | P0       |
| **Execution**                  | CUDA graph (conservative for MoE)       | **重构** (optional) | Big win when possible; unigateway can choose execution strategy. | - | P0       |
| **Model Support**              | **Qwen3.8-Flash-Next NVFP4 only**（目标态，尚未实现） | **重构** | 专用图 + GPU-only；vendor 必要叶子实现 | 正确性与性能验收后迁移 V4 默认入口 | P0       |
| **Model Support**              | legacy multi-MoE (Qwen/Mixtral/MiniMax) | legacy | 不进默认门禁、不优化 | 可删 / 移出默认构建 | -        |
| **Model Support**              | Tokenizer (HF)                          | 直接引用      | Mature, no point reimplementing.                                                                  | -                                        | P0       |
| **Model Support**              | New MoE model quick add                 | **不做**      | 禁止通用模型 registry 扩展 | 仅目标检查点 | - |
| **Observability**              | Prometheus (t/s, cache_hit, batch, q)   | **重构**      | Only the metrics that matter for this lite scope.                                                 | -                                        | P0       |
| **Observability**              | Structured logs + request id            | **重构**      | Correlate across unigateway / engine.                                                             | -                                        | P0       |
| **Observability**              | Graceful shutdown + health              | **重构**      | 3am stability.                                                                                        | -                                        | P0       |
| **Advanced**                   | Speculative decoding                    | **不做**      | Complex, variable gain.                                                                           | Optional plugin later                    | -        |
| **Advanced**                   | Prefill / Decode disaggregation         | **不做**      | Distributed systems concern, not single-node token factory.                                       | Future "advanced" mode                   | -        |
| **Advanced**                   | Dynamic multi-LoRA / hot swap           | **不做**      | Huge complexity for narrow win.                                                                   | -                                        | -        |
| **Advanced**                   | Multimodal encoders                     | **不做**      | Completely different data and execution path.                                                     | Separate multimodal service              | -        |
| **Advanced**                   | vLLM feature-parity surface             | **不做**      | vLLM is a broad general serving engine; sglang-lite remains MoE-only and minimal.                  | Use vLLM as another UniGateway backend   | -        |
| **Advanced**                   | Full expert parallelism + advanced load balancing | **Hybrid** | Lite focuses on efficient routing + batching + Radix on shared parts. Full EP is advanced. | Basic MoE in core; advanced EP later     | P1       |

## Summary Counts (MVP)

- **重构** (own completely): 12+
- **Hybrid** (reuse pieces, own path): 4
- **直接引用** (infrastructure): ~4 (tokenizers, HF loaders, basic quant loaders, kernel functions)
- **明确不做**: 6+

MoE support is first-class. The three core pieces are decomposed internally, but their hot-path composition and engine loop remain owned by sglang-lite.

The core stays an ultra-minimal library. The repository's thin standalone layer owns only the serving, configuration, observability, and admission required for one local engine; advanced gateway concerns stay outside.

## What "Lite" Means in Practice (MoE-only)

- Very small number of startup flags (sensible presets).
- Predictable behavior under load.
- Easy to reason about one request's journey through the system.
- The three building blocks are internally decomposed for modularity. sglang-lite owns the engine loop; an external gateway may supply only high-level policy hints through stable protocols.
- Dense models are explicitly out of scope.
- vLLM is a peer backend to interoperate with through UniGateway, not a feature checklist to copy.
- Minimal standalone serving and engine-local ops live in dedicated thin layers. Cross-backend and business concerns stay in UniGateway or another gateway.
- Codebase should stay small enough that a single engineer can hold the mental model.

## Enforcement

Any PR that adds a feature from the "不做" column or significantly increases coupling across the API/KV/Scheduler boundary without strong justification will be closed with reference to this document.

See also [architecture.md](./architecture.md).
