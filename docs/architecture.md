# sglang-lite Architecture

## Guiding Principle: High Cohesion + Composability

sglang-lite remains a **pure library** whose job is to provide the three core building blocks for a MoE "Token Factory". These three must stay reasonably cohesive for performance, but we further decompose them internally so that unigateway (as the driver) can compose and control higher-level policies.

Core three (still the only deeply coupled pieces):

1. **RadixKVCache** (was KVCacheManager)
   - Composed of: RadixTree + KVAllocator + MemoryBudget + EvictionPolicy
2. **BatchingScheduler** (was Scheduler)
   - Composed of: SequenceTable + BatchFormer + (AdmissionController can be peeled to unigateway)
3. **MoEModelRunner** (was ModelRunner)
   - Composed of: ModelLoader + MoERouter + PrefillExecutor + DecodeExecutor + KernelBackend

unigateway (or any host) owns:
- The main orchestration loop (what used to be LiteEngine)
- Admission control, request queuing, timeouts
- Higher-level batching policy selection
- Metrics collection, logging context, lifecycle

sglang-lite only exports the fine-grained building blocks + a small default composition helper.

## Layered Architecture (MVP) — sglang-lite as pure engine, unigateway as driver

**sglang-lite is now a pure library** (the "Token Factory").

All serving, routing, auth, rate-limiting, configuration, advanced observability, and driver integration are peeled to **unigateway** (or thin dedicated layers).

unigateway acts as the **backend driver** for sglang-lite (the actual driver code lives in the unigateway repository):
- It loads and manages the sglang-lite engine (preferred: direct Python library import inside unigateway: `LiteEngine(model_name=..., device=...)`).
- It handles the OpenAI surface, streaming, validation, metrics, routing, auth, etc.
- Any "sglang-lite backend" registration and connection logic moves to unigateway.
- sglang-lite only exposes the minimal engine API.

**Important note for the UniGateway team**:
- Please keep UniGateway's core abstractions (`ProviderDriver`, registry, routing, etc.) completely general.
- Do **not** introduce sglang-lite specific concepts into the core engine or protocol layers.
- All MoE/Radix-specific logic must stay inside the sglang-lite driver implementation.
- Treat sglang-lite the same way you would treat any other local or remote LLM backend.
- Detailed requirements document: `docs/unigateway-sglang-lite-requirements.md`

```
Clients / unigateway (full gateway + driver)
  • Owns: HTTP server, routing, auth, metrics, admission control,
    request lifecycle, higher-level policies
  • Composes sglang-lite building blocks
        ↓ uses
sglang-lite (pure library — only the three building blocks)

  RadixKVCache
    ├── RadixTree
    ├── KVAllocator / MemoryManager
    └── EvictionPolicy (can be swapped)

  BatchingScheduler
    ├── SequenceTable
    └── BatchFormer (unigateway can supply custom policy)

  MoEModelRunner
    ├── ModelLoader
    ├── MoERouter
    ├── PrefillExecutor / DecodeExecutor
    └── KernelBackend

Everything else (serving, config, observability export, driver glue)
is in unigateway or host application.
```

**Key peelings to unigateway:**
- All HTTP serving and OpenAI surface
- Auth, rate limit, routing, semantic routing
- Advanced metrics, structured logging, tracing
- Configuration and presets
- Graceful shutdown, health, timeouts coordination
- Driver glue (how to load/call sglang-lite engine)

sglang-lite only owns the three high-cohesion pieces inside the MoE engine.

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
| Later   | PyO3 / in-process  | Zero copy, best perf         | Build complexity         | Hot paths only      |

## Model Support Strategy

- Only register popular **MoE** models (dense models are out of scope).
- Use HF `AutoModelForCausalLM` + `AutoTokenizer` initially for loading.
- Later: direct safetensors weight loading + custom modeling files for speed (like nano-vLLM style).
- Extension point: small model registry + loader trait.

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
