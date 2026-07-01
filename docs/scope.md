# sglang-lite Scope & Feature Classification

This is the authoritative reference for what belongs in core vs. what gets pushed out.

## One-sentence Mission

> An extremely cohesive production token generator primarily for MoE models. Everything not required to reliably turn requests into token streams at high throughput stays outside.

The engine focuses exclusively on popular MoE architectures (Mixtral-style, DeepSeek-style, Qwen-MoE, etc.). Dense models are out of scope. MoE support is first-class.

sglang-lite is a **pure library** exposing three further-decomposed building blocks (RadixKVCache, BatchingScheduler, MoEModelRunner). The orchestrator and most policy logic live in the driver.

**unigateway as backend driver** (driver code lives in the unigateway repository):
- Uses the three building blocks directly (unigateway owns the main loop and admission control).
- Handles all serving, routing, auth, rate-limit, advanced config, metrics export, graceful shutdown, etc.
- sglang-lite only owns the high-cohesion core pieces (internal decomposition is allowed for modularity).

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
| **API Layer (Rust)**           | Tool calls (function calling)           | **重构**      | Only placeholder shape + clear error. Execution belongs in harness.                               | Gateway layer                            | P1       |
| **API Layer**                  | Structured / JSON mode                  | **不做**      | Requires FSM / constrained decoding. Breaks token-factory cohesion.                               | outlines / xgrammar in gateway           | -        |
| **KV & Memory**                | RadixKVCache (composed of RadixTree + KVAllocator + Eviction) | **重构** | Core for MoE prefix sharing. Internal pieces are further decomposed for composability by the driver. | - (default) | P0       |
| **KV & Memory**                | Memory budget / eviction policy         | **重构** (partial) | Can be replaced; unigateway may provide policy. | - | P1       |
| **Scheduling**                 | BatchingScheduler (SequenceTable + BatchFormer) | **重构** | Core continuous batching. Admission/queueing peeled to unigateway driver. | - | P0       |
| **Scheduling**                 | MoE-aware batch formation               | **重构** (partial) | BatchFormer can be supplied by unigateway. | - | P1       |
| **Execution**                  | MoEModelRunner (composed: Router + Prefill/Decode Executors + KernelBackend) | **重构** | Routing + execution for MoE. Composed internally so pieces can be swapped. | - | P0       |
| **Execution**                  | CUDA graph (conservative for MoE)       | **重构** (optional) | Big win when possible; unigateway can choose execution strategy. | - | P0       |
| **Model Support**              | Popular MoE (DeepSeek, Qwen-MoE, Mixtral 等) | 直接引用 | HF + proven loading paths. MoE is first-class (dense models explicitly out of scope).             | Register approved MoE families only      | P0       |
| **Model Support**              | Tokenizer (HF)                          | 直接引用      | Mature, no point reimplementing.                                                                  | -                                        | P0       |
| **Model Support**              | New MoE model quick add                 | **重构**      | Registry + loader hook only. Support for common MoE patterns.                                     | Simple config + extension point          | P1       |
| **Observability**              | Prometheus (t/s, cache_hit, batch, q)   | **重构**      | Only the metrics that matter for this lite scope.                                                 | -                                        | P0       |
| **Observability**              | Structured logs + request id            | **重构**      | Correlate across unigateway / engine.                                                             | -                                        | P0       |
| **Observability**              | Graceful shutdown + health              | **重构**      | 3am stability.                                                                                        | -                                        | P0       |
| **Advanced**                   | Speculative decoding                    | **不做**      | Complex, variable gain.                                                                           | Optional plugin later                    | -        |
| **Advanced**                   | Prefill / Decode disaggregation         | **不做**      | Distributed systems concern, not single-node token factory.                                       | Future "advanced" mode                   | -        |
| **Advanced**                   | Dynamic multi-LoRA / hot swap           | **不做**      | Huge complexity for narrow win.                                                                   | -                                        | -        |
| **Advanced**                   | Multimodal encoders                     | **不做**      | Completely different data and execution path.                                                     | Separate multimodal service              | -        |
| **Advanced**                   | Full expert parallelism + advanced load balancing | **Hybrid** | Lite focuses on efficient routing + batching + Radix on shared parts. Full EP is advanced. | Basic MoE in core; advanced EP later     | P1       |

## Summary Counts (MVP)

- **重构** (own completely): 12+
- **Hybrid** (reuse pieces, own path): 4
- **直接引用** (infrastructure): ~4 (tokenizers, HF loaders, basic quant loaders, kernel functions)
- **明确不做**: 6+

MoE support is first-class. The three core pieces are now further decomposed internally (RadixKVCache, BatchingScheduler, MoEModelRunner) so that unigateway (as driver) can own composition and policy.

sglang-lite is an ultra-minimal pure library. Serving, config, observability, admission, etc. are peeled to unigateway.

## What "Lite" Means in Practice (MoE-only)

- Very small number of startup flags (sensible presets).
- Predictable behavior under load.
- Easy to reason about one request's journey through the system.
- The three building blocks are internally decomposed for modularity. unigateway (the driver) owns the main loop and higher-level policies.
- Dense models are explicitly out of scope.
- Serving, ops, and cross-cutting concerns are peeled to unigateway or dedicated thin layers. The core is a pure library.
- Codebase should stay small enough that a single engineer can hold the mental model.

## Enforcement

Any PR that adds a feature from the "不做" column or significantly increases coupling across the API/KV/Scheduler boundary without strong justification will be closed with reference to this document.

See also [architecture.md](./architecture.md).
