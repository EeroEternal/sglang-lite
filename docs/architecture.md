# sglang-lite Architecture

## 当前主目标与迁移状态

2026-10-03 经用户确认，主目标切换为 **Qwen3.8-Flash-Next RadixArk NVFP4，
GPU-only**，见 [产品宪章](./qwen38-flash-next-only.md)。现有 V4 路径保留；
Qwen3.8 的独立单序列文本原型位于 `engine/qwen38_runner/`，尚未通过真实
正确性/驻留/吞吐验收，也未接入调度器；以下引擎/控制面边界不变。

新执行核心需要自持 QSA KV、GDN 状态和 PLE 历史的联合生命周期，并正确处理
FP8 PLE scale、hyperconnection、NVFP4 expert 分片。模型算子和持久状态留在 GPU，
禁止 CPU/SSD offload 或 unified-memory 回退。CPU tokenizer、加载与控制逻辑允许。
SGLang 只可用于 `scripts/` 的外部基线；独立 lite runner 不导入 SGLang/vLLM。
默认入口需待真实正确性与同机吞吐验收后迁移，不能仅禁用 V4 gate。

## Guiding Principle: High Cohesion + Composability

The `engine/` core remains a **pure library**, while sglang-lite as a product is an
independent MoE inference backend with a thin Rust control/serving shell. It does not require
SGLang, vLLM, or UniGateway. Its three core capabilities are not SGLang-specific; they are
the same engine-level responsibilities found in vLLM:

| Engine-neutral capability | sglang-lite / SGLang-oriented implementation | vLLM implementation |
| --- | --- | --- |
| KV lifecycle + prefix reuse | RadixKVCache / RadixAttention | KVCacheManager + APC + PagedAttention blocks |
| Continuous token scheduling | BatchingScheduler | vLLM V1 Scheduler |
| Model execution | MoEModelRunner + CUDA graph/kernel backend | GPUModelRunner / Worker + CUDA graph/compile |

The shared capability model does **not** imply implementation or internal API compatibility. Each backend owns its scheduler state, cache indexing, block lifecycle, and execution path. UniGateway should depend on capability and protocol contracts, not these internal classes.

It also does not imply complete product equivalence. FlashInfer can provide attention, KV, sampling, and related GPU kernels, but it does not provide the scheduler, cache policy, model registry/loaders, distributed executor, broad feature surface, or production compatibility matrix that make up vLLM as a complete engine. A completed standalone `sglang-lite + FlashInfer` can replace vLLM only inside sglang-lite's deliberately narrow supported envelope; UniGateway is optional.

sglang-lite realizes the three capabilities with these deeply coupled pieces:

1. **RadixKVCache** (was KVCacheManager)
   - Composed of: RadixTree + KVAllocator + MemoryBudget + EvictionPolicy
2. **BatchingScheduler** (was Scheduler)
   - Composed of: SequenceTable + BatchFormer + engine-local admission/backpressure
3. **MoEModelRunner** (was ModelRunner)
   - Composed of: ModelLoader + MoERouter + PrefillExecutor + DecodeExecutor + KernelBackend

The sglang-lite engine process owns:
- The central orchestration/engine loop
- Waiting/running sequence state and token-budget batch formation
- Engine-local queue bounds, cancellation, timeout cleanup, and OOM handling
- KV/model execution hot-path composition and internal metrics

An external gateway may own cross-backend admission and routing policy, but it never directly
drives the engine's internal Python classes.

## Layered Architecture — Independent Backend, Optional Gateway

The direct-use topology is:

```text
OpenAI Client
  ↓ HTTP/SSE
sglang-lite-serving (Rust executable)
  • minimal OpenAI validation and error shape
  • request id, true SSE, disconnect/timeout/cancel propagation
  • health, readiness, metrics, graceful drain
  ↓ internal HTTP or gRPC stream
sglang-lite engine process (Python)
  • central engine loop
  • RadixKVCache + BatchingScheduler + MoEModelRunner
  • sampling + tokenizer/detokenizer
  • FlashInfer / Triton / CUDA
```

For a full gateway deployment:

```text
Clients → UniGateway → (sglang-lite | vLLM | SGLang)
```

UniGateway is optional. It adds multi-backend routing, auth, tenant/global rate limits,
failover, policy, and metrics aggregation. It calls sglang-lite through HTTP or gRPC;
direct embedding is prohibited.

**Important note for the UniGateway team**:
- Please keep UniGateway's core abstractions (`ProviderDriver`, registry, routing, etc.) completely general.
- Do **not** introduce sglang-lite specific concepts into the core engine or protocol layers.
- All MoE/Radix-specific logic must stay inside the sglang-lite driver implementation.
- Treat sglang-lite the same way you would treat any other local or remote LLM backend, including vLLM.
- Prefer generic local-inference concepts (`PrefixCache`, `BlockKVCache`, `ContinuousScheduler`, `ModelExecutor`, `BackendCapabilities`) over SGLang/Radix-specific terms in UniGateway core.
- **Critical boundary**: PyO3, direct Python embedding, or any in-process library calls are explicitly not used. All communication must go through HTTP or gRPC. This boundary is to keep UniGateway as a general-purpose embeddable SDK.
- Detailed requirements document: `docs/unigateway-sglang-lite-requirements.md`
- vLLM positioning document: `docs/vllm-positioning-compatibility.md`

## vLLM-compatible positioning

sglang-lite's external position should be compatible with vLLM as another `local-inference` backend, while its internal implementation remains MoE-only and Radix-first.

The primary conclusion is that the original three sglang-lite core capabilities also apply to vLLM. Compatibility therefore starts from a shared capability model:

1. KV allocation, eviction, block lifecycle, and optional prefix reuse.
2. Continuous scheduling across prefill/decode token work.
3. Model execution, kernel selection, CUDA graph/compile, and MoE routing where supported.

RadixAttention is only one implementation of the first capability; it must not define the generic backend contract.

Compatibility target:

- **Protocol**: keep the minimal OpenAI-compatible chat completions surface aligned with vLLM's local server shape where in scope: `/v1/chat/completions`, streaming chunks, `/v1/models`, health, request-id passthrough, and OpenAI-shaped errors.
- **Capabilities**: expose generic backend capabilities (`chat_completions`, `streaming`, `prefix_cache`, `prefix_cache_metric`, `moe_optimized`) rather than backend-name-specific checks.
- **Metrics**: treat `usage.cache_hit_tokens` as a generic prefix-cache metric. It should be usable for both sglang-lite RadixKVCache and vLLM Automatic Prefix Caching.
- **KV abstraction**: keep RadixKVCache as the default internal structure, but make block/page terminology compatible with vLLM-style KVCacheManager/PagedAttention concepts.

Non-goals:

- Do not implement vLLM's broad API surface in sglang-lite core (`/v1/responses`, audio, multimodal, embeddings for non-core models).
- Do not expose vLLM-specific request parameters as stable sglang-lite contract unless they are part of the shared minimal sampling surface.
- Do not import vLLM internals or copy its scheduler/KV manager wholesale.
- Do not add dense, multimodal, LoRA, speculative decoding, or disaggregated serving to core for the sake of vLLM compatibility.

```
sglang-lite engine process
  • Owns engine loop, request/sequence state, token budget,
    cancellation cleanup, KV lifecycle, and model execution

  RadixKVCache
    ├── RadixTree
    ├── KVAllocator / MemoryManager
    └── EvictionPolicy (can be swapped)

  BatchingScheduler
    ├── SequenceTable
    └── BatchFormer

  MoEModelRunner
    ├── ModelLoader
    ├── MoERouter
    ├── PrefillExecutor / DecodeExecutor  (tensor-batched HF forward by cached_len)
    └── KernelBackend (paged rebuild → HF attn; FlashInfer append on CUDA when present)

  Execution note: paged K/V is the rebuildable attention store; equal-length groups
  share one model forward. Full FlashInfer attention + CUDA graph remain P1.

sglang-lite-control / serving
  • Own minimal OpenAI/SSE, readiness, engine-local limits and lifecycle

Optional UniGateway
  • Own cross-backend routing, auth, global policy and aggregation
```

### DeepSeek-V4 transition topology (Hybrid → Owned)

V4 currently runs a **Hybrid** path (official `Transformer.forward` + in-model
Attention/Compressor buffers). Prefix reuse is a CPU snapshot
(`engine/v4_prefix_cache.py`), not Radix dual-pool COW. Decode still calls
official `sparse_attn` unless FI SM120 probe is non-zero
(`SGLANG_LITE_V4_DISABLE_FI_SPARSE=1` by default). Stage definitions:
[deepseek-v4-flash-plan.md](./deepseek-v4-flash-plan.md) §8.

```
V4 Hybrid (Phase 0 / 0b)          Phase 0c Own KV              Phase 1 Own Kernels
─────────────────────────         ──────────────────           ───────────────────
official Attention buffer   →     Radix dual pool              KernelBackend decode
  kv_cache / kv_state /             compressed MLA pages         FI SM120 sparse MLA
  score_state                       + SWA / dsv4_packed(584)     (else official fallback)
CPU snapshot prefix hit     →     page restore / COW / evict   MoE GEMM: DeepGEMM SM120
clear_v4_kv_slot on final   →     release pages on finish      (MXFP4 e8m0); B12x 布局不兼容
LiteEngine CB + TP SSE      →     same scheduler contract      optional CUDA graph / grouped MoE
```

**MoE / FP4 GEMM（Hybrid，2026-08-08）**

| 组件 | 路径 | 说明 |
|------|------|------|
| DeepGEMM SM120 | `engine/v4_deep_gemm.py` + `vendor/deep_gemm_sm120` | 默认开启；替换 `kernel.fp4_gemm`；官方 MXFP4 e8m0 **1×32** |
| MoE 调度快路径 | `engine/v4_moe_fast.py` | 仅激活专家 + gate/up 共用 act_quant |
| B12x | `engine/v4_b12x.py` | 默认关：需 NVFP4 e4m3/sf16 + 无 EP；Hybrid 为 MXFP4 + TP/EP |

验收与开关见 [v4-flash-only.md](./v4-flash-only.md) §9–10。

**Phase 0c slices 1–3 (landed):**

1. Dual-write packed SWA+compressed pages after Hybrid prefill.
2. **Page ownership**: cache holds forked refs; seq holds allocate/hit forks.
3. Decode best-effort dual-append.
4. **0c-3 page restore**: bf16 restore pool writes back into official
   `kv_cache` rows on hit (`restore_dual_pool_to_model`); CPU snapshot is
   **slimmed** (`dual_primary`) to state tensors only when page-backed.
   Decode still runs official `sparse_attn` (pages are not yet the attention
   source).

Metrics: `dual_restore_count`, `prefix_dual_primary`.

**Phase 0c-4 (landed):** when `seq._v4_page_primary`, each decode **stages**
official `kv_cache` from dual-pool bf16 pages (`dual_stage_count`) then runs
official `sparse_attn`. Pages are SoT; module buffers are staging only.
FI remains opt-in FORCE (Path A numerical OK; e2e not faster yet).

**Phase 1:** leaf optional only; production default official. Throughput
baseline: plan §6.4.1.

Do not treat “FI attached” as Owned KV: FlashInfer is a leaf; Radix page
lifecycle and scheduler grouping remain the engine’s job.

**Kept outside sglang-lite and optionally provided by UniGateway:**
- Auth, rate limit, routing, semantic routing
- Cross-backend retry/failover and tenant policy
- Broad OpenAI/other-provider API normalization
- Advanced metrics aggregation and distributed tracing
- Agent/tool/structured-output business logic

Minimal chat serving, real streaming, health/readiness, request cancellation, local
backpressure, configuration, and graceful shutdown are required for standalone use and remain
inside the thin sglang-lite service shell.

## Internal Protocol (to be defined precisely in code)

**GenerationRequest** (from Rust → Python)

- request_id: string (for streaming correlation)
- model: string
- input_ids: Vec<u32>   // already tokenized (or raw prompt + do tokenization in python)
- sampling_params: { temperature, top_p, top_k, max_tokens, stop, ... }
- priority?
- created_ts

**GenerationResponse** (stream of)

- request_id
- token_id / text delta
- finish_reason?
- usage (on last)

## Key Data Structures (Python core sketch)

```python
# radix cache
class RadixCache:
    root: Node
    # each node: token seq slice, child map, refcount, kv pages

class Sequence:
    seq_id: int
    input_ids: list[int]
    output_ids: list[int]
    # cache hit length, allocated blocks, etc.

class Scheduler:
    waiting: deque
    running: list[Sequence]
    # radix cache ref

    def add_request(...)
    def step() -> Batch   # returns batch of tokens to run
```

## Communication Options (evolution)

| Stage   | Mechanism          | Pros                          | Cons                     | When                |
|---------|--------------------|-------------------------------|--------------------------|---------------------|
| MVP     | Local HTTP (FastAPI from Python) | Easy debug, separate process | Serialization + latency  | Now                 |
| Phase1  | gRPC (tonic + py)  | Typed, efficient, streaming  | More boilerplate         | When core stabilizes|
| Later (not planned) | Direct embedding (e.g. PyO3) | Avoided | Conflicts with unigateway generality | Not considered for unigateway driver |

## Model Support Strategy

- Only register popular **MoE** models (dense models are out of scope).
- Use HF `AutoModelForCausalLM` + `AutoTokenizer` initially for loading.
- Later: direct safetensors weight loading + custom modeling files for speed (like nano-vLLM style).
- Extension point: small model registry + loader trait.
- vLLM compatibility is handled through protocol/capability mapping, not by expanding sglang-lite model scope to match vLLM.

## Observability

Rust layer owns:
- Request lifecycle events
- External API metrics (latency, error rate)
- High level engine metrics forwarded from python (tokens/s, batch size, cache hit rate, queue depth)

Python emits structured counters via prometheus client or JSON over the control channel.

## Failure Philosophy

- Engine never "hides" errors. Clear 4xx for bad requests at Rust layer.
- 5xx only for true internal failures.
- Graceful degradation on OOM (evict, reject new).
- Always return OpenAI-compatible error shape.

See [scope.md](./scope.md) and [roadmap.md](./roadmap.md).
