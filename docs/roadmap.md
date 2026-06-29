# sglang-lite Roadmap

## Phase 0 — Core Verification (Current Focus)

Goal: A runnable system that proves the architecture and can serve real (small) models with basic continuous batching + Radix skeleton.

**Deliverables**
- [x] Project skeleton + docs (README, architecture, scope, roadmap)
- [ ] Rust axum OpenAI server
  - POST /v1/chat/completions (non-stream + stream via SSE)
  - GET /v1/models
  - GET /healthz
  - Strict minimal request model + early reject
  - Clean internal GenerationRequest sent to backend
- [ ] Internal protocol (GenerationRequest / deltas) defined in both Rust and Python (shared or duplicated with tests)
- [ ] Python execution package with clear interfaces:
  - `KVCacheManager` (Radix skeleton + allocate/evict)
  - `Scheduler` (waiting + running, step())
  - `ModelRunner` stub + simple real path (HF generate for tiny model on CPU/CUDA when available)
- [ ] Integration: Rust can talk to Python (HTTP first)
- [ ] Basic metrics emitted (even if stubbed): queue depth, batch size, fake t/s
- [ ] Can run a full chat roundtrip for a small open model (e.g. Qwen2.5-0.5B or TinyLlama) and observe logs
- [ ] CI skeleton (cargo check + fmt + clippy, python lint + basic test)

**Success criteria**
- `curl .../v1/chat/completions` returns valid OpenAI shaped response (streaming works)
- Code for core pieces is readable and < a few thousand LOC total for engine
- Easy to point at a real model dir and get tokens

**Non goals in Phase 0**
- 70B perf
- Real Radix tree with GPU pages working end to end
- Production metrics
- Any multimodal or structured output

## Phase 1 — Production Shell + Real Core

- Full RadixAttention implementation with block / page management, prefix matching, eviction policy
- Scheduler that actually does continuous batching (add prefill, promote decode sequences, manage seq lens)
- CUDA graph capture around decode forward (the big win)
- Real quant support (at least AWQ + FP8 path with correct memory accounting)
- Good model support matrix for popular MoE models only (DeepSeek, Qwen-MoE, Mixtral 等; up to 70B+ class). Dense models are out of scope.
- Prometheus + /metrics, structured logging, graceful drain
- Configuration system with "lite" preset (very few flags)
- Request timeout, max concurrent, queue limits
- Error normalization that matches OpenAI behavior closely
- Basic benchmark script (share prefix chat workload)

**Stretch**
- Optional Paged strategy selectable at startup
- Simple model registry extension point

## Phase 2 — Polish + Integration + Hardening

- Deep integration tests with unigateway (semantic routing + KV affinity hints)
- Rust control plane enhancements (more early filters, intent metadata passthrough)
- Optional: move hot scheduler or cache management bits to Rust via PyO3
- Performance parity target vs SGLang on prefix-heavy chat workloads (within 10-15% on same hardware)
- Packaging: docker images, helm chart skeleton
- Documentation: operator guide, "when to choose sglang-lite vs full SGLang"
- Formal support policy for model families

## Later / Optional

- gRPC control plane (alternative to HTTP)
- Disaggregated mode (advanced)
- Speculative as plugin
- Rust-only hot path experiment (full control)

## Non-Roadmap (永远不做 in core)

See scope.md "不做" list. If a feature is desired by many users, the correct answer is "put it in the gateway/harness layer or as a separate sidecar".

## How to Track Progress

- Issues / milestones will mirror the phases.
- Every merged PR should update this file or link the changed scope.

Current status: **Phase 1 in progress — MoE-only scope** (reassessed 2026-06-28)

**Phase 0 complete** — v0.1.0

Reassessed boundary: sglang-lite now targets **MoE models only**. Dense models are explicitly out of scope. Primary focus on popular MoE (DeepSeek, Qwen-MoE, Mixtral-style, etc.). The engine is a pure library. Serving and cross-cutting concerns are peeled to unigateway or thin layers. The engine remains high-cohesion and "lite".

## Phase 1 Deliverables (all completed)
- [x] Prometheus + /metrics (Python + Rust)
- [x] Structured logging + request_id
- [x] Config system ("lite" preset + env/CLI)
- [x] Robustness: timeouts, max_concurrent, queue limits
- [x] Benchmark script
- [x] Scheduler improvements (batching, eviction)
- [x] OpenAI error normalization
- [x] Graceful shutdown via server runtimes

Non-goals for Phase 1: CUDA graph, 70B perf, full paged attention.
