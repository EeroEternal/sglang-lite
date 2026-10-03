"""Device-selection and benchmark option checks without a GPU dependency."""

import importlib.util
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v4_graph_thruput.py"


def _script():
    with patch.dict("os.environ", {}, clear=True):
        return runpy.run_path(str(SCRIPT))


class GraphBenchmarkTests(unittest.TestCase):
    def test_torchrun_selects_one_device_from_visible_list(self):
        select = _script()["_select_rank_device"]
        with patch.dict("os.environ", {"LOCAL_RANK": "2", "CUDA_VISIBLE_DEVICES": "3,5,7,9"}, clear=True):
            select()
            self.assertEqual(__import__("os").environ["CUDA_VISIBLE_DEVICES"], "7")

    def test_torchrun_selects_rank_when_devices_unspecified(self):
        select = _script()["_select_rank_device"]
        with patch.dict("os.environ", {"LOCAL_RANK": "3"}, clear=True):
            select()
            self.assertEqual(__import__("os").environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_torchrun_rejects_more_ranks_than_visible_devices(self):
        select = _script()["_select_rank_device"]
        with patch.dict("os.environ", {"LOCAL_RANK": "2", "CUDA_VISIBLE_DEVICES": "0,1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "exceeds 2 visible GPUs"):
                select()

    def test_graph_benchmark_rejects_missing_grouped_deep_gemm(self):
        main = _script()["main"]
        with patch.object(sys, "argv", [str(SCRIPT), "--deep-gemm", "0"]):
            with self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(exc.exception.code, 2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_graph_feedback_advances_tokens_and_position(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        from sglang_lite.v4_decode_fast import capture_decode, set_fast

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("_decode_pos", torch.zeros((), device="cuda", dtype=torch.long))
                self.register_buffer("classes", torch.arange(256, device="cuda"))

            def forward(self, tokens, _start_pos):
                next_id = (tokens[0, 0] + self._decode_pos + 1) % 256
                return -(self.classes - next_id).abs().float().unsqueeze(0)

        model = FakeModel()
        tokens = torch.zeros((1, 1), device="cuda", dtype=torch.long)
        try:
            set_fast(True)
            graph, _logits = capture_decode(model, tokens, autoregressive=True)
            tokens.zero_()
            model._decode_pos.zero_()
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(tokens.item(), 10)
            self.assertEqual(model._decode_pos.item(), 4)
        finally:
            set_fast(False)
