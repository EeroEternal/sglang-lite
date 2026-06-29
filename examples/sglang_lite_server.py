"""
Example thin server for sglang-lite engine.

This is peeled out of the core library. In production, the serving layer
should live in unigateway or a dedicated thin server.

Run with: PYTHONPATH=. python examples/sglang_lite_server.py --model ...
"""

import sys
sys.path.insert(0, "python")

from server import main

if __name__ == "__main__":
    main()
