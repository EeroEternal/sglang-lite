# sglang-lite

**A minimal, production-grade LLM inference engine focused on MoE models.** High-cohesion "Token Factory" focused on:

- RadixAttention (prefix sharing for chat/agent workloads)
- Continuous batching scheduler
- Low-overhead decode with CUDA graph
- OpenAI-compatible API (the critical control point)

> "The engine is only responsible for reliably and efficiently producing tokens." — Push agent logic, structured output, and multimodal handling up to the gateway/harness layer (unigateway).

## Mission

A minimal, production-grade LLM inference engine focused exclusively on popular MoE models. It provides stable, high-throughput, low-latency token generation for MoE architectures. The core revolves around continuous batching + efficient KV cache (Radix) + expert-routing-aware execution, exposing a minimal and stable OpenAI-compatible interface.

It does **not** do: agent runtime, multimodal, built-in constrained decoding, speculative decoding, prefill-decode disaggregation, dynamic multi-LoRA, or complex expert parallelism scheduling.

## Why sglang-lite?

SGLang and vLLM are powerful, but their **accidental complexity** has become overwhelming. Many real production workloads (multi-turn chat, RAG, tool calling) do not need the full feature set, yet they pay the price in configuration hell, debugging difficulty, and instability caused by monthly changes.

sglang-lite follows a **high-cohesion** design:
- KV Cache full lifecycle + Scheduler + Execution must be tightly coupled as the core.
- Everything else (serving, routing, auth, advanced observability, business logic) is peeled to unigateway or thin layers.

The goal is to achieve stable 80%+ of theoretical throughput in target scenarios (prefix-heavy chat/agent) with extremely low operational burden.

References: nano-vLLM (~1.2k LOC teaching version), mini-sglang.

**Scope note**: After re-evaluation, sglang-lite supports **only popular MoE models** (DeepSeek, Qwen-MoE, Mixtral-style, etc.). Dense models are explicitly out of scope. MoE routing and batching are first-class considerations, while keeping the overall design lightweight.

## Target Workloads (Priority)

1. Multi-turn dialogue / Agent traffic (RadixAttention advantage is largest due to prefix sharing)
2. RAG + Tool Use (structured logic handled at gateway layer using outlines/xgrammar)
3. High-concurrency batch inference (evaluation, synthetic data generation)

## Non-Goals (Firmly out of scope for MVP)

- Multimodal (Vision/Audio/Video)
- Built-in Structured Output / JSON mode / Grammar (pushed to gateway)
- Speculative Decoding
- Prefill-Decode Disaggregation
- Dynamic Multi-LoRA
- Complex MoE expert parallelism scheduling and advanced load balancing (lite provides basic MoE routing + batching support)
- Diffusion / non-Transformer models
- Built-in SGLang-style frontend language

## Feature Take/Keep Table (High Cohesion)

| Category    | Feature                              | Category     | Rationale / Strategy                              | Alternative / Push Up     |
|-------------|--------------------------------------|--------------|---------------------------------------------------|---------------------------|
| API         | OpenAI /v1/chat/completions + Streaming | **Rewrite** | Critical control point, fully owned               | -                         |
| API         | /v1/models + Health                  | **Rewrite** | Minimal implementation                            | -                         |
| KV Cache    | RadixAttention (prefix tree)         | **Rewrite** | Core component, must own for cohesion             | - (default)               |
| KV Cache    | PagedAttention                       | **Rewrite** | Pluggable alternative                           | Config switch             |
| Scheduling  | Continuous Batching + dynamic batch  | **Rewrite** | Deeply coupled with KV Cache                      | -                         |
| Scheduling  | Queue, priority, timeout             | **Rewrite** | Production robustness                           | -                         |
| Execution   | Heavy CUDA Graph for decode          | **Rewrite** | Key to reducing CPU overhead and high decode throughput | -                    |
| Execution   | Basic Quant (BF16/FP8/AWQ)           | Hybrid       | Loader reusable, own the path                     | SGLang loader pieces      |
| Generation  | Basic Sampling (temp/top_p/...)      | Reuse/Hybrid | Standard logic                                    | -                         |
| Model       | Popular MoE model loading (DeepSeek, Qwen-MoE, Mixtral, etc.) | Direct reuse | Lowest day-0 support cost (MoE only)            | HF + registry snippets    |
| Model       | Tokenizer                            | Direct reuse | transformers is mature                            | -                         |
| Ops         | Prometheus metrics + /healthz + graceful | **Rewrite** | Production requirement, only expose lite-relevant metrics | -                  |
| Structured  | JSON mode / Grammar                  | **No**       | Breaks cohesion                                   | Gateway (xgrammar)        |
| Multimodal  | Vision etc.                          | **No**       | Very low cohesion                                 | Separate service          |
| Advanced    | Speculative / Disagg / Multi-LoRA    | **No** (MVP) | Complex and non-core                              | Later plugin              |

**Classification summary**: ~12+ full control/rewrite, small amount direct reuse (infrastructure), Hybrid transition, several out of scope.

## Recommended Architecture (Rust + Python Hybrid) — Library First

sglang-lite is a **pure library**. Serving and cross-cutting concerns are peeled to unigateway or thin wrappers.

```
Client / unigateway
        ↓ (OpenAI + routing/auth)
[Rust API Layer] (axum, optional)  <--- Thin control point
  - Minimal validation + early reject
  - Streaming + thin call to engine
        ↓ (HTTP/gRPC or PyO3)
[Python Core Engine] (pure library)
  - KVCacheManager (Radix)
  - ContinuousBatchingScheduler (MoE-aware)
  - ModelRunner (routing + execution)
        ↑ token stream + usage
```

Serving, metrics export, config, graceful shutdown live in unigateway or dedicated thin server (see examples/).

**Why this combination?**

- Rust (optional thin layer): Type safety for the API surface and easy early rejection.
- Python + Triton: Mature kernels and fast iteration for the core engine.
- Library first: The Python engine is a pure library. Serving, ops, and advanced features are peeled to unigateway or examples/. This keeps sglang-lite minimal and high-cohesion.

**Communication (MVP)**: Start with local HTTP (loose coupling, easy to debug), or gRPC. Tighten with PyO3 later.

## Tech Stack

- **Rust API**: axum, tokio, serde, serde_json, tracing, reqwest (to Python), uuid
- **Python Core**: torch, transformers (for initial loading/tokenizer), triton (kernels), flashinfer (optional)
- **Interface**: Minimal OpenAI-compatible + clean internal request/response
- **Deployment**: Single-node + simple tensor parallelism. Complex distributed serving later.

## MVP Roadmap (Phases)

**Phase 0 (Core Verification)**

- Rust: Full /v1/chat/completions (stream + non-stream) + /v1/models + /healthz
- Define internal GenerationRequest / Token protocol
- Python side: Working MoE-aware engine with basic routing + batching (switchable to real HF MoE models)
- Able to run popular MoE models (e.g. Mixtral, DeepSeek, Qwen-MoE class)
- Basic continuous batching + simple Radix skeleton (even if imperfect at first)
- Metrics: tokens/s, TTFT, cache hit (even if mocked)

**Phase 1 (Production Ready)**

- Real Radix KV Cache + deep Scheduler integration
- CUDA graph decode path
- Prometheus metrics, structured logging, graceful shutdown, request timeouts
- Mainstream MoE model support (validate throughput at 70B+ scale)
- Configuration presets ("lite" defaults)

**Phase 2**

- Optional Paged strategy
- Limited model extension points
- Optionally migrate scheduler/KV hotspots to Rust
- Deep integration validation with unigateway (KV affinity, etc.)

## Quick Start (current stub state)

```bash
# Rust API (stub)
cargo run -p sglang-lite-api -- serve --port 8000

# or Python stub
python -m sglang_lite.server
```

Then:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 128,
    "stream": true
  }'
```

See [docs/roadmap.md](docs/roadmap.md) and [docs/architecture.md](docs/architecture.md).

## Contribution and Scope Discipline

Please strictly follow the scope. Before adding any new feature, ask: Does it belong to the high-cohesion "Token Factory" core? If it is business logic, please push it upward.

## License

Apache-2.0 (to be confirmed)

## Status

Phase 1 in progress (MoE-focused production shell + metrics + robustness).
See v0.1.0 for the last Phase 0 release (pre-MoE scope realignment). Not for production use yet.

See git history and docs for detailed design discussions.
