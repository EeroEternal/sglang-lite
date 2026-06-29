#!/usr/bin/env python3
"""
Basic benchmark script for sglang-lite (Phase 1)

Focus: prefix-heavy chat workload to measure cache hit benefit.

Usage:
    PYTHONPATH=python python benchmark.py --url http://localhost:9001 --num-requests 20 --prefix-len 80
"""

import argparse
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate(url: str, prompt: str, max_tokens: int = 32):
    payload = {
        "model": "stub",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.time()
    resp = requests.post(f"{url}/generate", json=payload, timeout=60)
    latency = time.time() - start
    data = resp.json()
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return latency, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:9001")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    # Build a shared prefix + different continuations
    prefix = "You are a helpful AI assistant. " * (args.prefix_len // 8)
    base = prefix + "User asks: What is the capital of "

    print(f"Benchmarking {args.num_requests} requests against {args.url}")
    print(f"Shared prefix length ~{args.prefix_len} tokens")

    start = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(args.num_requests):
            prompt = base + f"country number {i}?"
            futures.append(executor.submit(generate, args.url, prompt, args.max_tokens))

        results = []
        for f in as_completed(futures):
            try:
                lat, toks = f.result()
                results.append((lat, toks))
            except Exception as e:
                print("Error:", e)

    total_time = time.time() - start
    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print("\n=== Results ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests: {len(results)}")
    print(f"Total completion tokens: {total_tokens}")
    print(f"Avg latency: {sum(latencies)/len(latencies):.3f}s")
    print(f"Throughput: {total_tokens / total_time:.1f} tokens/s")
    print(f"P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.3f}s")


if __name__ == "__main__":
    main()
