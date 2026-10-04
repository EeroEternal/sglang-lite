"""Timing accounting tests without a Torch dependency."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts" / "qwen38_lite_bench.py"
SPEC = importlib.util.spec_from_file_location("qwen38_lite_bench", PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class LiteBenchmarkTests(unittest.TestCase):
    def test_decode_counts_do_not_include_prefill_first_token(self):
        rows = [
            {"generation_s": t, "decode_s": 0.5, "ids": [1, 2], "text": "text"}
            for t in [3.0, 1.0, 2.0]
        ]
        result = BENCH.summarize(rows, 2)
        self.assertEqual(result["warm_median_s"], 2.0)
        self.assertEqual(result["warm_median_tok_s"], 1.0)
        self.assertEqual(result["decode_steps"], 1)
        self.assertEqual(result["warm_decode_tok_s"], 2.0)
        self.assertTrue(result["greedy_repeat_ids_match"])

    def test_single_token_has_no_decode_rate(self):
        result = BENCH.summarize(
            [{"generation_s": 1.0, "decode_s": 0.0, "ids": [1], "text": "text"}], 1
        )
        self.assertIsNone(result["warm_decode_tok_s"])

    def test_different_outputs_are_reported(self):
        rows = [
            {"generation_s": 1.0, "decode_s": 0.5, "ids": [i, i], "text": "text"} for i in [1, 2]
        ]
        self.assertFalse(BENCH.summarize(rows, 2)["greedy_repeat_ids_match"])

    def test_positive_lengths_required(self):
        self.assertEqual(BENCH.parse_lengths("64,128"), [64, 128])
        with self.assertRaises(ValueError):
            BENCH.parse_lengths("0,128")

    def test_short_output_is_not_a_speed_sample(self):
        with self.assertRaises(ValueError):
            BENCH.summarize([{"generation_s": 1.0, "decode_s": 0.5, "ids": [1], "text": "text"}], 2)

    def test_all_rank_traces_require_profiling(self):
        with (
            patch.object(
                sys,
                "argv",
                ["bench", "--model", "unused", "--out", "unused", "--profile-all-ranks"],
            ),
            self.assertRaises(SystemExit) as caught,
        ):
            BENCH.main()
        self.assertEqual(caught.exception.code, 2)
