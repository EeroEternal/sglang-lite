"""Argument and measurement guards for the separate Qwen3.8 baseline."""

import runpy
import json
import tempfile
import unittest
from pathlib import Path


MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "qwen38_sglang_baseline.py")
)


class Qwen38BaselineTests(unittest.TestCase):
    def test_incomplete_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "shard.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "missing shards"):
                MODULE["checkpoint_identity"](model)

    def test_checkpoint_identity_deduplicates_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "shard", "b": "shard"}})
            )
            (model / "shard").write_bytes(b"fixture")
            identity = MODULE["checkpoint_identity"](model)
            self.assertEqual(identity["shard_sizes"], {"shard": 7})
            self.assertEqual(len(identity["index_sha256"]), 64)

    def test_all_offload_is_explicitly_disabled(self):
        args = MODULE["engine_args"]("/model", 8)
        self.assertIs(args["ple_offload_embedding"], False)
        self.assertEqual(args["cpu_offload_gb"], 0)
        self.assertIs(args["enable_unified_memory"], False)
        self.assertIsNone(args["speculative_algorithm"])
        self.assertEqual(args["tp_size"], args["ep_size"])
        # SGLang 0.5.20 rejects this flag for Qwen4Exp, even for text requests.
        self.assertNotIn("language_model_only", args)

    def test_unsupported_parallel_size_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE["engine_args"]("/model", 1)

    def test_attention_dp_preserves_global_expert_group(self):
        args = MODULE["engine_args"]("/model", 8, 2)
        self.assertEqual(args["tp_size"], 8)
        self.assertEqual(args["ep_size"], 8)
        self.assertEqual(args["dp_size"], 2)
        self.assertIs(args["enable_dp_attention"], True)

    def test_unsupported_attention_dp_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE["engine_args"]("/model", 8, 4)

    def test_incomplete_generation_is_not_a_throughput_sample(self):
        class FakeEngine:
            def generate(self, _prompt, _params):
                return {"meta_info": {"completion_tokens": 3}}

        with self.assertRaisesRegex(RuntimeError, "expected 8"):
            MODULE["checked_generation"](FakeEngine(), "hello", 8)

    def test_fixed_length_greedy_sampling(self):
        class FakeEngine:
            def generate(self, _prompt, params):
                self.params = params
                return {"meta_info": {"completion_tokens": params["max_new_tokens"]}}

        engine = FakeEngine()
        MODULE["checked_generation"](engine, "hello", 8)
        self.assertEqual(engine.params["temperature"], 0)
        self.assertIs(engine.params["ignore_eos"], True)
