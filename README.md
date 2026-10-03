# sglang-lite

**Qwen3.8-Flash-Next NVFP4、GPU-only** Token Factory（性能优先于通用性）。

**迁移中**：2026-10-03 经用户确认切换主目标。Qwen3.8 独立文本原型尚未验收，
真实吞吐尚未验收；当前默认入口仍是保留的 V4 路径，不能直接加载新模型。

- 自持 QSA KV / GDN 状态 / PLE 历史与 continuous batching（目标态）
- **Vendor** 必要叶子实现（引擎不 `import sglang` / `import vllm`）
- GPU-only decode 与 CUDA graph 优化（待真实模型验证）
- 薄 Rust OpenAI 控制面

> "The engine is only responsible for reliably and efficiently producing tokens." — 业务与宽协议上移 UniGateway。

**产品宪章**：[docs/qwen38-flash-next-only.md](docs/qwen38-flash-next-only.md)
**优化实录**：[docs/qwen38-optimization-log.md](docs/qwen38-optimization-log.md)，含原始时长、失败实验和未通过的验收项。

## Mission

主目标只服务 **Qwen3.8-Flash-Next，RadixArk NVFP4**。KPI：8×5090 同机同权重、
同 GPU 预算、两边分别调优后的 warm tok/s **超过标准 SGLang**，先验收正确性及 GPU 驻留。
允许不同 GPU 并行结构，但必须披露 TP/EP/DP/PP，并保留同配置受控对照。
禁止 CPU/SSD offload 或 unified-memory 回退。V4 和其它 MoE 为历史/兼容路径；
Dense / 多模态 / 投机 / PD disagg **不做**。文档中的原有模型示例和数字不代表 Qwen3.8 验收结果。

## Why sglang-lite?

SGLang and vLLM are powerful, but their **accidental complexity** has become overwhelming. Many real production workloads (multi-turn chat, RAG, tool calling) do not need the full feature set, yet they pay the price in configuration hell, debugging difficulty, and instability caused by monthly changes.

sglang-lite follows a **high-cohesion + composable** design:
- The three core building blocks (RadixKVCache, BatchingScheduler, MoEModelRunner) are the only deeply coupled pieces.
- Each block is further decomposed internally (e.g. RadixTree + KVAllocator, BatchFormer, MoERouter + Executors).
- The Python engine core remains a small library, while this repository also ships an official thin standalone service.
- The sglang-lite engine process owns its orchestration loop, KV lifecycle, continuous batching, model execution, and token streaming.
- **UniGateway is optional**. It can sit above sglang-lite for multi-backend routing, auth, global admission, and policy, but sglang-lite does not require UniGateway, SGLang, or vLLM to serve requests.

A formal requirements document has been prepared for the UniGateway team:

**→ docs/unigateway-sglang-lite-requirements.md**

The goal is to achieve stable 80%+ of theoretical throughput in target scenarios (prefix-heavy chat/agent) with extremely low operational burden.

References: nano-vLLM (~1.2k LOC teaching version), mini-sglang.

**Scope note**: The primary target is Qwen3.8-Flash-Next NVFP4 only, not a general
MoE registry. The dedicated runner is under preparation. Dense models and other
model-family optimizations are outside the current scope.

## vLLM Positioning Compatibility

sglang-lite is **not** a SGLang-only design and should not become a mini vLLM. It is a narrow MoE token factory that must remain compatible with vLLM at the **gateway/protocol/capability** layer:

- The three core capabilities are engine-neutral and apply equally to SGLang and vLLM: KV/prefix-cache management, continuous scheduling, and model execution. SGLang uses RadixAttention-oriented implementations; vLLM uses KVCacheManager/APC/PagedAttention, the V1 Scheduler, and GPUModelRunner/Worker.
- Capability equivalence is not product equivalence: FlashInfer supplies kernels, not vLLM's complete engine and serving ecosystem. `sglang-lite + FlashInfer` can be a targeted standalone vLLM alternative for the supported MoE/prefix-heavy envelope; UniGateway is an optional upstream gateway, not a runtime dependency.
- UniGateway should treat both sglang-lite and vLLM as `local-inference` backends.
- Shared compatibility target: OpenAI-compatible chat completions, streaming, models, health, request-id passthrough, and generic prefix-cache metrics such as `usage.cache_hit_tokens`.
- Internal concepts should map to generic names (`PrefixCache`, `BlockKVCache`, `ContinuousScheduler`, `ModelExecutor`) instead of leaking Radix/SGLang-specific terms into gateway core abstractions.
- vLLM-specific broad features (Responses API, multimodal, LoRA, spec decode, disaggregation, structured output backends) remain outside sglang-lite core.

See **docs/vllm-positioning-compatibility.md** for the detailed comparison and compatibility strategy.

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

## Recommended Architecture — Independent Backend, Optional Gateway

The engine core is a pure MoE Token Factory library. The product also includes a thin Rust control/serving shell so users can run it directly.

**sglang-lite owns the complete local inference loop:**
- Engine loop, request/sequence lifecycle, KV allocation, continuous batching, model execution, sampling, and token deltas.
- Minimal OpenAI-compatible chat/stream/models/health/readiness/metrics surface.
- Backpressure, cancellation, timeout propagation, graceful drain, and engine error normalization required for safe standalone use.

```
Clients
  ↓ OpenAI
sglang-lite-serving (thin Rust control plane)
  ↓ internal HTTP/gRPC token stream
sglang-lite engine process
  • Engine loop
  • RadixKVCache + BatchingScheduler + MoEModelRunner
  • FlashInfer / Triton / CUDA kernels
```

For deployments that need a full gateway:

```
Clients → UniGateway → (sglang-lite | vLLM | SGLang)
```

UniGateway owns cross-backend routing, auth, rate limits, global policy, and aggregation. It communicates with sglang-lite over HTTP or gRPC and never drives the engine's internal building blocks in-process.

**Why this combination?**

- Thin Rust service layer: Type safety for the API surface, early rejection, and SSE lifecycle control.
- Python + Triton: Mature kernels and fast iteration for the core engine.
- Library-first core: the Python engine remains small and explicit; the official serving shell stays thin and contains no model execution or gateway business logic.

**Communication**: Use HTTP or gRPC for integration with unigateway. Direct embedding (e.g. PyO3) is not used to keep unigateway as a general SDK.

## Tech Stack

- **Rust API**: axum, tokio, serde, serde_json, tracing, reqwest (to Python), uuid
- **Python Core**: torch, transformers (for initial loading/tokenizer), triton (kernels), flashinfer (optional)
- **Interface**: Minimal OpenAI-compatible + clean internal request/response
- **Deployment**: Single-node + simple tensor parallelism. Complex distributed serving later.

## MVP Roadmap (Phases)

**Phase 0 (Core Verification)** — largely done for Mixtral-class MoE

- Rust: Full /v1/chat/completions (stream + non-stream) + /v1/models + /healthz
- Internal GenerationRequest / TokenDelta protocol
- MoE-aware engine with continuous batching + Radix skeleton
- Metrics: tokens/s, TTFT, cache hit

**Phase 0b (Stabilize — current for DeepSeek-V4 Hybrid)** — done

- Incremental UTF-8 `detokenize_delta` + unit tests
- Prefill-final logits gate (`scripts/v4_logits_compare.py`; soft top5 / CVD near-tie)
- TP process + standalone SSE acceptance (dedicated CUDA thread)

**Phase 0c (Own KV)** — **done** (slices 1–4 + PRO6000 gates)

- Dual-pool packed + bf16 restore; dual-write / hit fork / append / page restore
- **0c-4 page-primary**: decode stages official buffers from pages; kernel still official
- Throughput regression §6.4.2: stage overhead ~1–2.5% warm (PASS)
- See [docs/deepseek-v4-flash-plan.md](docs/deepseek-v4-flash-plan.md) §8.2

**Phase 1 (Own kernels)** — optional leaf only; **not** default

- Production default: official TileLang `sparse_attn` (`DISABLE_FI_SPARSE=1`)
- FI SM120: numerical OK (Path A), e2e slower with Hybrid pack; FORCE only

**Phase 2 (Production hardening)** — slices 1–4 for stable deploy

- Prometheus: dual_pool/0c + TTFT / tok/s
- Graceful drain + structured `request_id` JSON logs
- **Soak**: `scripts/soak_stability.py` (blocks/oom/error gates)
- **MoE regression**: `scripts/moe_regression.py` (tiny Mixtral fixture by default)
- **Lite preset / Runbook**: `scripts/env_lite.sh`, [docs/runbook.md](docs/runbook.md)

**Phase 2**

- Optional Paged strategy / limited extension points
- Optional Rust migration of scheduler/KV hotspots
- UniGateway protocol/metrics integration only (no business logic in engine)

V4 “健全引擎”窄口径与验收表：[docs/deepseek-v4-flash-plan.md](docs/deepseek-v4-flash-plan.md) §8。
健全 ≠ vLLM 功能面对等。

## Quick Start (standalone MoE service)

Does **not** require SGLang, vLLM, or UniGateway.

```bash
pip install -e .
# Official entry: spawns Python engine process + Rust OpenAI control plane
cargo run -p sglang-lite-serving -- serve \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --device cuda \
  --port 8000

# Control-plane only (no real model; for API tests)
cargo run -p sglang-lite-serving -- serve --stub --port 8000
```

Engine process alone (internal GenerationRequest / TokenDelta NDJSON):

```bash
python -m sglang_lite.process --model mistralai/Mixtral-8x7B-Instruct-v0.1 --device cuda --port 9001
```

Then:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 128,
    "stream": true
  }'
```

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /healthz`, `GET /readyz`, `GET /metrics`.

Supported families (MoE only): Mixtral-style, Qwen-MoE, DeepSeek-MoE. Dense models are rejected.
See [docs/scope.md](docs/scope.md), [docs/architecture.md](docs/architecture.md), and
[docs/standalone-inference-service-roadmap.md](docs/standalone-inference-service-roadmap.md).

## Contribution and Scope Discipline

Please strictly follow the scope. Before adding any new feature, ask: Does it belong to the high-cohesion "Token Factory" core? If it is business logic, please push it upward.

## License

Apache-2.0 (to be confirmed)

## Status

**Execution path** (v0.2.x): standalone Rust→Python serve, true NDJSON/SSE deltas,
central scheduler loop, tensor-batched HF forward (same `cached_len` groups), and
paged KV as rebuildable attention state. Transformers cache extraction supports both
4.x (`to_legacy_cache`) and 5.x (`layers.keys` / `layers.values`); pin is
`transformers>=4.40,<6`.

Current limits (still not full SGLang/vLLM parity):
- Logits still produced by HuggingFace MoE modules; on CUDA, FlashInfer paged
  prefill/decode attention reads paged KV directly (no DynamicCache rebuild).
  FlashInfer MoE kernels remain P1.
- Prefill/decode batching requires equal sequence lengths within a group (otherwise serial).
- Popular hub MoE + CUDA: enable with GPU and optional `SGLANG_LITE_POPULAR_MOE=<id>`.
- Strict P0 DoD (OpenAI E2E on real MoE, concurrent batch evidence on clean install)
  requires re-audit after this DynamicCache fix; do not treat rust/ruff alone as P0 done.

P1 remains: FlashInfer attention in the hot path, CUDA graph decode, quant/TP.
Not a vLLM feature-parity substitute.

See git history and docs for detailed design discussions.

## Optional Integration with UniGateway

sglang-lite can be used directly or as a backend of [unigateway](https://github.com/EeroEternal/unigateway).

### Running for UniGateway (HTTP mode)

sglang-lite can be run as a standalone OpenAI-compatible server that unigateway can connect to.

Example:

```bash
# Using the example server
python -m sglang_lite.server \
  --port 8000 \
  --model "deepseek-ai/DeepSeek-V2-Lite-Chat" \
  --device cuda \
  --max-batch-size 8
```

Or use the Rust binary which proxies to a Python core:

```bash
SGLANG_LITE_PYTHON_CORE=http://localhost:9001 PORT=8000 ./target/debug/sglang-lite-serving
```

### Environment variables (for Config.from_env)

- SGLANG_LITE_MODEL
- SGLANG_LITE_DEVICE
- SGLANG_LITE_PORT
- SGLANG_LITE_MAX_BATCH_SIZE
- SGLANG_LITE_MAX_CONCURRENT
- SGLANG_LITE_REQUEST_TIMEOUT
- SGLANG_LITE_LOG_LEVEL

### UniGateway TOML example

```toml
[[providers]]
name = "my-moe"
provider_type = "sglang-lite"
base_url = "http://localhost:8000/v1"
# api_key can be empty for local
api_key = ""

[providers.model_policy]
default_model = "local-moe"
```

See unigateway docs for full details.

### Response format for cache metrics passthrough

The usage object in responses includes:

```json
"usage": {
  "prompt_tokens": 20,
  "completion_tokens": 10,
  "total_tokens": 30,
  "cache_hit_tokens": 12
}
```

This allows unigateway to extract cache hit information.
