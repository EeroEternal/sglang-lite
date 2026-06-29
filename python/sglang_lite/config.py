"""
Configuration system for sglang-lite Phase 1.

Supports:
- Environment variables
- "lite" preset (sensible defaults for simple deployments)
- Basic validation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    model: str = "stub"
    device: str = "cpu"
    port: int = 9001
    max_batch_size: int = 4
    max_concurrent: int = 32
    max_tokens_default: int = 128
    request_timeout: float = 300.0  # seconds
    queue_timeout: float = 60.0
    log_level: str = "INFO"
    # Future: quant, etc.

    @classmethod
    def from_env(cls, preset: str = "lite") -> "Config":
        if preset == "lite":
            # Lite preset: minimal, stable, low resource
            base = {
                "max_batch_size": 4,
                "max_concurrent": 32,
                "request_timeout": 300.0,
                "queue_timeout": 60.0,
            }
        else:
            base = {}

        return cls(
            model=os.getenv("SGLANG_LITE_MODEL", base.get("model", "stub")),
            device=os.getenv("SGLANG_LITE_DEVICE", base.get("device", "cpu")),
            port=int(os.getenv("SGLANG_LITE_PORT", base.get("port", 9001))),
            max_batch_size=int(os.getenv("SGLANG_LITE_MAX_BATCH_SIZE", base.get("max_batch_size", 4))),
            max_concurrent=int(os.getenv("SGLANG_LITE_MAX_CONCURRENT", base.get("max_concurrent", 32))),
            max_tokens_default=int(os.getenv("SGLANG_LITE_MAX_TOKENS", base.get("max_tokens_default", 128))),
            request_timeout=float(os.getenv("SGLANG_LITE_REQUEST_TIMEOUT", base.get("request_timeout", 300.0))),
            queue_timeout=float(os.getenv("SGLANG_LITE_QUEUE_TIMEOUT", base.get("queue_timeout", 60.0))),
            log_level=os.getenv("SGLANG_LITE_LOG_LEVEL", base.get("log_level", "INFO")),
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}