"""
Example thin server (peeled out of core).

sglang-lite core is a pure library. This example shows a minimal
HTTP server built on top.

Serving, routing, auth, advanced metrics, etc. belong in unigateway
or a dedicated thin server.

Run:
    python examples/sglang_lite_server.py --port 9001 --model ...
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

import sys
sys.path.insert(0, "python")

from sglang_lite.config import Config
from sglang_lite.engine.engine import LiteEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("sglang_lite_server")

def _log(msg: str, **kwargs):
    log_data = {"ts": time.time(), "msg": msg, **kwargs}
    logger.info(json.dumps(log_data))

app = FastAPI(title="sglang-lite Python Core (real engine)")

# Global engine
ENGINE: Optional[LiteEngine] = None

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:12]
    request.state.request_id = req_id
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    _log("http_request", request_id=req_id, method=request.method, path=str(request.url.path), duration=duration, status=response.status_code)
    response.headers["x-request-id"] = req_id
    return response


class GenRequest(BaseModel):
    request_id: Optional[str] = None
    model: str = "stub"
    messages: Optional[List[dict]] = None
    input_ids: Optional[List[int]] = None
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False


class GenResult(BaseModel):
    text: str
    finish_reason: Optional[str] = "stop"
    usage: dict


def _extract_input_ids(req: GenRequest) -> List[int]:
    if req.input_ids:
        return req.input_ids

    # crude extraction from messages (last user message)
    last_user = ""
    if req.messages:
        for m in reversed(req.messages):
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    last_user = content
                break
    if not last_user:
        last_user = "Hello"

    if ENGINE and ENGINE.runner.tokenizer:
        return ENGINE.runner.tokenize(last_user)
    return [hash(c) % 32000 for c in last_user[:80]]


@app.post("/generate", response_model=GenResult)
async def generate(req: GenRequest, request: Request):
    global ENGINE
    rid = req.request_id or getattr(request.state, "request_id", f"req-{uuid.uuid4().hex[:12]}")
    input_ids = _extract_input_ids(req)

    _log("generate_start", request_id=rid, max_tokens=req.max_tokens)
    result = ENGINE.generate(
        rid,
        input_ids,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    _log("generate_end", request_id=rid, finish_reason=result.get("finish_reason"))
    return GenResult(
        text=result["text"],
        finish_reason=result.get("finish_reason"),
        usage=result.get("usage", {}),
    )


@app.post("/generate_stream")
async def generate_stream(req: GenRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:12]}"
    input_ids = _extract_input_ids(req)

    async def event_generator():
        for delta in ENGINE.generate_stream(
            rid, input_ids, max_tokens=req.max_tokens, temperature=req.temperature
        ):
            import json
            yield f"data: {json.dumps(delta)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    stats = ENGINE.get_stats() if ENGINE else {}
    return {
        "status": "ok",
        "phase": "0",
        "core": "sglang-lite-python",
        "stats": stats,
    }


@app.get("/stats")
async def stats():
    return ENGINE.get_stats() if ENGINE else {}

@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--model", type=str, default="stub")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=32)
    args = parser.parse_args()

    cfg = Config.from_env("lite")
    # Override with CLI args if provided
    if args.model != "stub":
        cfg.model = args.model
    if args.device != "cpu":
        cfg.device = args.device
    cfg.port = args.port
    cfg.max_batch_size = args.max_batch_size
    cfg.max_concurrent = args.max_concurrent

    global ENGINE
    ENGINE = LiteEngine(config=cfg)

    print(f"[sglang-lite-core] Phase 1 engine starting on :{args.port}")
    print(f"  config: {cfg.to_dict()}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=cfg.log_level.lower())


if __name__ == "__main__":
    main()
