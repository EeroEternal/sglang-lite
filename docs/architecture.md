# sglang-lite Architecture

## Guiding Principle: High Cohesion

The engine is a **Token Factory**.

Only three things belong deeply coupled inside the core:

1. **KVCacheManager** — allocate / reuse (Radix prefix tree) / evict / memory budget
2. **Scheduler** — continuous batching, dynamic batch formation, fairness / timeout
3. **ModelRunner** — prefill + decode (heavy CUDA graph on decode path) + kernel calls

Everything else is pushed out or treated as thin adapter.

## Layered Architecture (MVP)

sglang-lite is a **pure library**. Serving, routing, auth, rate limiting, advanced observability, and most configuration are peeled to unigateway or thin dedicated layers.

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients / unigateway                  │
│  (OpenAI /v1/chat/completions + routing, auth, rate-limit)   │
│  (advanced metrics, config, graceful shutdown peeled here)   │
└───────────────────────────────┬─────────────────────────────┘
                                │ HTTP/gRPC or in-process
┌───────────────────────────────▼─────────────────────────────┐
│  Rust Control Plane (axum, optional) — Control Point         │
│  • Minimal OpenAI request models                             │
│  • Validation + early reject                                 │
│  • Streaming orchestration                                   │
│  • Thin client to Python engine (or direct)                  │
│  (most ops peeled to unigateway)                             │
└───────────────────────────────┬─────────────────────────────┘
                                │ protocol or PyO3
┌───────────────────────────────▼─────────────────────────────┐
│  Python Execution Core (pure library)                        │
│  ┌──────────────────────┐   ┌───────────────────────────┐   │
│  │   KVCacheManager     │◄──┤   RadixTree (token seq)   │   │
│  │   (pages / blocks)   │   │   prefix match + evict    │   │
│  └──────────────────────┘   └───────────────────────────┘   │
│  ┌──────────────────────┐   ┌───────────────────────────┐   │
│  │ Continuous Batching  │◄──┤   Scheduler               │   │
│  │   + Batcher          │   │   (add seq, step, retire) │   │
│  └──────────────────────┘   └───────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ModelRunner (torch + flash / triton)                 │   │
│  │   • MoE routing + expert execution                   │   │
│  │   • CUDA graph (optional)                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  (Tokenizer & HF model registry — thin reuse)               │
│  (hooks for metrics/logging; export peeled to gateway)      │
└─────────────────────────────────────────────────────────────┘
```

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
