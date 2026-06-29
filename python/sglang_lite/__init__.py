"""sglang-lite Python execution core.

Pure library for MoE-focused token generation.

Core components:
- LiteEngine (orchestrator)
- RadixCache (KV)
- Scheduler (continuous batching)
- ModelRunner (execution with MoE routing)

Serving layer (HTTP server) is peeled to examples/ or unigateway.
"""

__version__ = "0.1.0"

from .engine.engine import LiteEngine
from .config import Config

__all__ = ["LiteEngine", "Config"]
