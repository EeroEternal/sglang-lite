"""
LiteEngine (thin facade, mostly for examples).

In the recommended architecture, unigateway (the driver) owns the main loop and
directly composes the fine-grained pieces:
  - RadixKVCache
  - BatchingScheduler
  - MoEModelRunner

This class is kept for backward compatibility and simple standalone use.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Generator, List, Optional

# Prometheus metrics are optional (peeled to serving layer like unigateway when needed)
try:
    from prometheus_client import Counter, Gauge, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    Counter = Gauge = Histogram = lambda *a, **k: type('Noop', (), {'inc': lambda s, *a, **k: None, 'dec': lambda s, *a, **k: None, 'observe': lambda s, *a, **k: None, 'set': lambda s, *a, **k: None})()

# Config is kept for convenience in examples / thin servers.
# In production the driver (unigateway) controls parameters.
from .kv_cache import RadixCache
from .runner import ModelRunner
from .scheduler import Scheduler, Sequence

logger = logging.getLogger("sglang_lite")

def _log_structured(level: str, msg: str, **kwargs):
    log_data = {"level": level, "msg": msg, "ts": time.time(), **kwargs}
    if level == "INFO":
        logger.info(json.dumps(log_data))
    elif level == "WARNING":
        logger.warning(json.dumps(log_data))
    else:
        logger.error(json.dumps(log_data))

# Prometheus metrics (optional, peeled to unigateway/serving layer)
if HAS_PROMETHEUS:
    REQUESTS_TOTAL = Counter("sglang_lite_requests_total", "Total number of generation requests", ["model"])
    TOKENS_GENERATED = Counter("sglang_lite_tokens_generated_total", "Total tokens generated")
    ACTIVE_REQUESTS = Gauge("sglang_lite_active_requests", "Currently active requests")
    QUEUE_DEPTH = Gauge("sglang_lite_queue_depth", "Current waiting queue depth")
    BATCH_SIZE = Gauge("sglang_lite_current_batch_size", "Current batch size being processed")
    REQUEST_LATENCY = Histogram("sglang_lite_request_latency_seconds", "End-to-end request latency", buckets=(0.1, 0.5, 1, 2, 5, 10, 30))
else:
    REQUESTS_TOTAL = TOKENS_GENERATED = ACTIVE_REQUESTS = QUEUE_DEPTH = BATCH_SIZE = REQUEST_LATENCY = type('NoopMetric', (), {'inc': lambda *a,**k:None, 'dec':lambda *a,**k:None, 'observe':lambda *a,**k:None, 'set':lambda *a,**k:None, 'labels': lambda *a,**k: REQUESTS_TOTAL})()


class LiteEngine:
    def __init__(
        self,
        model_name: str = "stub",
        device: str = "cpu",
    ):
        # All admission, timeouts, concurrency, config, etc. are handled in the driver (unigateway).
        # sglang-lite only receives already-accepted work and a model name.
        self.model_name = model_name

        self.radix = RadixCache(max_tokens=65536)
        self.scheduler = Scheduler(self.radix, max_batch_size=4)
        self.runner = ModelRunner(model_name, device=device, max_batch=4)

        self._seq_map: Dict[str, Sequence] = {}   # request_id -> Sequence
        self._start_time: Dict[str, float] = {}   # for latency tracking

    def tokenize(self, text: str) -> List[int]:
        return self.runner.tokenize(text)

    def detokenize(self, ids: List[int]) -> str:
        return self.runner.detokenize(ids)

    def add_request(
        self,
        request_id: str,
        input_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Sequence:
        # Admission control, concurrency limits, and queueing are handled in the driver (unigateway).
        # The engine only sees already-accepted work.

        seq = self.scheduler.add_request(request_id, input_ids)
        seq.max_tokens = max_tokens          # attach generation params
        seq.temperature = temperature
        self._seq_map[request_id] = seq
        self._start_time[request_id] = time.time()

        if HAS_PROMETHEUS:
            REQUESTS_TOTAL.labels(model=self.model_name).inc()
            ACTIVE_REQUESTS.inc()
            QUEUE_DEPTH.set(len(self.scheduler.waiting))

        _log_structured("INFO", "request_added", request_id=request_id, prompt_len=len(input_ids), max_tokens=max_tokens)
        return seq

    def step(self, max_steps: int = 1) -> List[Dict]:
        """
        Run the engine for up to max_steps scheduler steps.
        Returns list of finished or intermediate results.
        """
        results = []
        for _ in range(max_steps):
            batch, is_prefill = self.scheduler.step()
            if not batch:
                break

            BATCH_SIZE.set(len(batch))
            _log_structured("INFO", "batch_step", batch_size=len(batch), prefill_count=sum(1 for p in is_prefill if p))

            next_tokens = self.runner.run_batch(batch, self.radix, is_prefill)

            for seq, tok, pre in zip(batch, next_tokens, is_prefill):
                if seq.finished or tok is None:
                    continue

                if pre:
                    # After prefill we usually emit the first real token
                    # and move to decode phase
                    self.scheduler.update_after_prefill(seq, [], seq.kv_state or [])
                    # We already got the first token from prefill in many cases
                    # For simplicity we treat the returned tok as first generated
                    self.scheduler.update_after_decode(seq, tok, seq.kv_state or [])
                else:
                    self.scheduler.update_after_decode(seq, tok, seq.kv_state or [])

                # Check stopping conditions
                if len(seq.output_ids) >= getattr(seq, "max_tokens", 128):
                    self.scheduler.mark_finished(seq, "length")
                    results.append(self._make_result(seq, finished=True))
                elif tok == self._get_eos_token():
                    self.scheduler.mark_finished(seq, "stop")
                    results.append(self._make_result(seq, finished=True))
                else:
                    # still running
                    results.append(self._make_result(seq, finished=False))

            # remove finished from running list
            self.scheduler.running = [s for s in self.scheduler.running if not s.finished]

        return results

    def generate(
        self,
        request_id: str,
        input_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Dict:
        """Blocking generation (convenient for /generate)."""
        start = time.time()
        seq = self.add_request(request_id, input_ids, max_tokens, temperature)

        # Run until this sequence finishes.
        # Timeouts and admission are enforced by the driver (unigateway) before calling the engine.
        while not seq.finished:
            self.step(max_steps=1)
            if len(seq.output_ids) > max_tokens + 10:
                break

        if not seq.finished:
            self.scheduler.mark_finished(seq, "length")

        result = self._make_result(seq, finished=True)

        # Record metrics
        duration = time.time() - self._start_time.pop(request_id, time.time())
        REQUEST_LATENCY.observe(duration)
        TOKENS_GENERATED.inc(result["usage"]["completion_tokens"])
        ACTIVE_REQUESTS.dec()
        QUEUE_DEPTH.set(len(self.scheduler.waiting))

        _log_structured("INFO", "request_finished", request_id=request_id, duration=duration, finish_reason=result.get("finish_reason"), usage=result.get("usage"))
        return result

    def generate_stream(
        self,
        request_id: str,
        input_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Generator[Dict, None, None]:
        """Yields token deltas one by one. Good for SSE."""
        seq = self.add_request(request_id, input_ids, max_tokens, temperature)

        last_len = 0
        while not seq.finished:
            self.step(max_steps=1)

            current_out = seq.output_ids
            new_tokens = current_out[last_len:]
            last_len = len(current_out)

            for t in new_tokens:
                text = self.detokenize([t])
                yield {
                    "request_id": request_id,
                    "token": t,
                    "text": text,
                    "finished": seq.finished,
                    "finish_reason": seq.finish_reason or None,
                }

            if seq.finished:
                break

        # final
        yield {
            "request_id": request_id,
            "token": None,
            "text": "",
            "finished": True,
            "finish_reason": seq.finish_reason or "stop",
            "usage": self._usage(seq),
        }

    def get_stats(self) -> Dict:
        cache_stats = self.radix.get_cache_stats()
        return {
            "waiting": len(self.scheduler.waiting),
            "running": len(self.scheduler.running),
            "cache": cache_stats,
            "model": self.runner.model_name,
            "device": self.runner.device,
        }

    # --- helpers ---

    def _make_result(self, seq: Sequence, finished: bool) -> Dict:
        text = self.detokenize(seq.output_ids)
        return {
            "request_id": seq.request_id,
            "text": text,
            "output_ids": list(seq.output_ids),
            "finish_reason": seq.finish_reason if finished else None,
            "finished": finished,
            "usage": self._usage(seq),
        }

    def _usage(self, seq: Sequence) -> Dict:
        prompt = len(seq.input_ids)
        completion = len(seq.output_ids)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def _get_eos_token(self) -> int:
        if self.runner.tokenizer is not None and hasattr(self.runner.tokenizer, "eos_token_id"):
            return self.runner.tokenizer.eos_token_id or 2
        return 2
