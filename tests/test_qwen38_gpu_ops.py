"""Narrow GPU math tests, runnable independently of model weights."""

import sys
import unittest
from pathlib import Path

try:
    import torch
    import triton  # noqa: F401, verifies the required GPU leaf dependency
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(), "requires CUDA Torch and Triton"
)
class QwenGpuOpsTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
        from qwen38_runner import ops

        self.ops = ops
        torch.manual_seed(7)

    def test_dense_attention_and_sparse_selected_slots(self):
        q = torch.randn((3, 256), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        for position, selected, budget in [
            (0, [0], 2048),
            (33, [0], 2048),
            (64, [63, 5, 2, -1], 32),
        ]:
            slots = torch.tensor(selected, device="cuda", dtype=torch.int32)
            actual = self.ops.attention(
                q, k, v, torch.tensor([position], device="cuda"), slots, budget
            )
            wanted = (
                list(range(position + 1)) if position < budget else [s for s in selected if s >= 0]
            )
            keys, values = k[wanted].float(), v[wanted].float()
            scores = (q.float() @ keys.T) / 16
            expected = (scores.softmax(-1) @ values).to(q.dtype)
            torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.004)

    def test_gdn_state_and_output(self):
        q = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn((6, 128), device="cuda", dtype=torch.bfloat16)
        z = torch.randn_like(v)
        a = torch.randn(6, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        bias = torch.randn(6, device="cuda")
        log_a = torch.randn(6, device="cuda")
        norm = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        state = torch.randn((6, 128, 128), device="cuda") * 0.1
        qf = torch.nn.functional.normalize(q.float(), dim=-1, eps=1e-12)
        kf = torch.nn.functional.normalize(k.float(), dim=-1, eps=1e-12)
        qf, kf = qf.repeat_interleave(3, 0), kf.repeat_interleave(3, 0)
        decay = (-log_a.exp() * torch.nn.functional.softplus(a.float() + bias)).exp()
        expected_state = state * decay[:, None, None]
        correction = (v.float() - (expected_state * kf[:, None, :]).sum(-1)) * b.float().sigmoid()[
            :, None
        ]
        expected_state = expected_state + correction[:, :, None] * kf[:, None, :]
        expected = ((expected_state * qf[:, None, :]).sum(-1) / (128**0.5)).to(v.dtype).float()
        expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
        expected = (expected * norm.float() * z.float().sigmoid()).to(v.dtype).reshape(1, -1)
        actual = self.ops.gdn_update(q, k, v, z, b, a, bias, log_a, state, norm)
        torch.testing.assert_close(state, expected_state, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    def test_greedy_candidates_match_full_argmax_including_ties(self):
        for tied in [False, True]:
            logits = torch.randn((8, 32), device="cuda")
            if tied:
                logits[1, 7] = 100
                logits[1, 9] = 100
                logits[5, 2] = 100
            candidates = torch.stack(
                [self.ops.local_greedy_candidate(logits[r], 32 * r) for r in range(8)]
            )
            actual = self.ops.greedy_from_candidates(candidates)
            torch.testing.assert_close(actual, logits.flatten().argmax().reshape(1))

    def test_cached_conv_weight_keeps_exact_eager_math(self):
        history = torch.randn((1280, 3), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((1280, 4), device="cuda", dtype=torch.bfloat16)
        cached_weight = weight.float()
        for _ in range(8):
            projected = torch.randn((1, 1280), device="cuda", dtype=torch.bfloat16)
            window = torch.cat([history, projected.reshape(-1, 1)], -1)
            expected = torch.nn.functional.silu((window.float() * weight.float()).sum(-1)).to(
                projected.dtype
            )
            actual = torch.nn.functional.silu((window.float() * cached_weight).sum(-1)).to(
                projected.dtype
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            history.copy_(window[:, 1:])

    def test_fp8_table_partition_does_not_read_outside_owner(self):
        table = torch.randn((32, 160), device="cuda").to(torch.float8_e4m3fn)
        ids = torch.tensor([9, 10, 41, 42], device="cuda")
        actual = self.ops.table_lookup(table, ids, 10)
        expected = torch.zeros((4, 160), device="cuda", dtype=torch.bfloat16)
        expected[1] = table[0].to(torch.bfloat16)
        expected[2] = table[31].to(torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_cutlass_up_gate_layout(self):
        from qwen38_runner.weights import cutlass_up_gate

        gate = torch.full((2, 4, 8), 3, device="cuda", dtype=torch.uint8)
        up = torch.full_like(gate, 7)
        packed = cutlass_up_gate(gate, up)
        torch.testing.assert_close(packed[:, :4], up)
        torch.testing.assert_close(packed[:, 4:], gate)

    def test_nvfp4_swizzle_preserves_blockscale_bytes(self):
        from flashinfer.fp4_quantization import nvfp4_block_scale_interleave

        scale = torch.randint(0, 256, (1280, 160), device="cuda", dtype=torch.uint8)
        actual = nvfp4_block_scale_interleave(scale).reshape_as(scale)
        expected = (
            scale.reshape(10, 4, 32, 40, 4).permute(0, 3, 2, 1, 4).contiguous().reshape_as(scale)
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
