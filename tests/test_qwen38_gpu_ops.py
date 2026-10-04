"""Narrow GPU math tests, runnable independently of model weights."""

import subprocess
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

    def test_gdn_recurrence_state_and_output(self):
        q = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn((6, 128), device="cuda", dtype=torch.bfloat16)
        z = torch.randn_like(v)
        a = torch.randn(6, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        beta = b.float().sigmoid().to(b.dtype)
        bias = torch.randn(6, device="cuda")
        log_a = torch.randn(6, device="cuda")
        norm = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        state = torch.randn((6, 128, 128), device="cuda") * 0.1
        qf = q.float() / torch.sqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        kf = k.float() / torch.sqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        qf, kf = qf.repeat_interleave(3, 0), kf.repeat_interleave(3, 0)
        decay = (-log_a.exp() * torch.nn.functional.softplus(a.float() + bias)).exp()
        expected_state = state * decay[:, None, None]
        correction = (v.float() - (expected_state * kf[:, None, :]).sum(-1)) * beta.float()[:, None]
        expected_state = expected_state + correction[:, :, None] * kf[:, None, :]
        expected = ((expected_state * qf[:, None, :]).sum(-1) / (128**0.5)).to(v.dtype).float()
        expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
        expected = (expected * norm.float() * z.float().sigmoid()).to(v.dtype).reshape(1, -1)
        actual = self.ops.gdn_update(q, k, v, z, beta, a, bias, log_a, state, norm)
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

    def test_fused_conv_sum_preserves_fp32_order_and_history(self):
        for channels in [1280, 2560, 5120]:
            history = torch.randn((channels, 3), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn((channels, 4), device="cuda")
            for _ in range(16):
                projected = torch.randn((1, channels), device="cuda", dtype=torch.bfloat16)
                window = torch.cat([history.clone(), projected.reshape(-1, 1)], -1)
                expected = (window.float() * weight).sum(-1)
                actual = self.ops.gdn_conv_sum(projected, history, weight)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(history, window[:, 1:], atol=0, rtol=0)

    def test_fused_conv_and_gate_preserve_rounding(self):
        for channels in (1280, 2560, 5120):
            history = torch.randn((channels, 3), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn((channels, 4), device="cuda")
            projected = torch.randn((1, channels), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((6,), device="cuda", dtype=torch.bfloat16) * 5
            window = torch.cat([history.clone(), projected.reshape(-1, 1)], -1)
            expected = (window.float() * weight).sum(-1)
            conv, beta = self.ops.gdn_conv_sum(projected, history, weight, b)
            torch.testing.assert_close(conv, expected, atol=0, rtol=0)
            torch.testing.assert_close(history, window[:, 1:], atol=0, rtol=0)
            torch.testing.assert_close(beta, b.float().sigmoid().to(b.dtype), atol=0, rtol=0)

    def test_fused_conv_activation_preserves_bf16_and_state(self):
        for channels in (1280, 2560, 5120):
            history = torch.randn((channels, 3), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn((channels, 4), device="cuda")
            for seed in range(16):
                projected = torch.randn((1, channels), device="cuda", dtype=torch.bfloat16) * (
                    seed + 1
                )
                b = torch.randn((6,), device="cuda", dtype=torch.bfloat16)
                window = torch.cat([history.clone(), projected.reshape(-1, 1)], -1)
                expected = torch.nn.functional.silu((window.float() * weight).sum(-1)).to(
                    projected.dtype
                )
                activated, beta = self.ops.gdn_conv_activated(projected, history, weight, b)
                torch.testing.assert_close(activated, expected, atol=0, rtol=0)
                torch.testing.assert_close(history, window[:, 1:], atol=0, rtol=0)
                torch.testing.assert_close(beta, b.float().sigmoid().to(b.dtype), atol=0, rtol=0)

    def test_packed_gdn_a_b_gemv_matches_separate_bf16(self):
        for channels in (2560, 10240):
            b_weight = torch.randn((6, channels), device="cuda", dtype=torch.bfloat16)
            a_weight = torch.randn_like(b_weight)
            packed = torch.cat([b_weight, a_weight])
            for seed in range(128):
                value = torch.randn((1, channels), device="cuda", dtype=torch.bfloat16) * (
                    seed % 11 + 1
                )
                b, a = torch.nn.functional.linear(value, packed).flatten().chunk(2)
                torch.testing.assert_close(
                    b, torch.nn.functional.linear(value, b_weight).flatten(), atol=0, rtol=0
                )
                torch.testing.assert_close(
                    a, torch.nn.functional.linear(value, a_weight).flatten(), atol=0, rtol=0
                )

    def test_packed_qsa_k_v_gemv_matches_separate_bf16(self):
        k_weight = torch.randn((256, 2560), device="cuda", dtype=torch.bfloat16)
        v_weight = torch.randn_like(k_weight)
        packed = torch.cat([k_weight, v_weight])
        for seed in range(128):
            value = torch.randn((1, 2560), device="cuda", dtype=torch.bfloat16) * (seed % 11 + 1)
            k, v = torch.nn.functional.linear(value, packed).chunk(2, -1)
            torch.testing.assert_close(
                k, torch.nn.functional.linear(value, k_weight), atol=0, rtol=0
            )
            torch.testing.assert_close(
                v, torch.nn.functional.linear(value, v_weight), atol=0, rtol=0
            )

    def test_shared_expert_side_stream_graph_matches_serial(self):
        from qwen38_runner.model import Layer

        layer = object.__new__(Layer)
        layer.shared = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.01
            for shape in ((640, 2560), (640, 2560), (2560, 640), (1, 2560))
        ]
        value = torch.randn((1, 2560), device="cuda", dtype=torch.bfloat16)
        result = torch.empty_like(value)
        side = torch.cuda.Stream()
        main = torch.cuda.current_stream()
        graph = torch.cuda.CUDAGraph()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            layer._shared_expert(value)
        main.wait_stream(side)
        torch.cuda.synchronize()
        try:
            with torch.cuda.graph(graph):
                seed = value + 0
                capture = torch.cuda.current_stream()
                side.wait_stream(capture)
                with torch.cuda.stream(side):
                    shared = layer._shared_expert(seed)
                capture.wait_stream(side)
                result.copy_(shared + 0)
            for _ in range(3):
                sample = torch.randn_like(value)
                value.copy_(sample)
                graph.replay()
                expected = layer._shared_expert(sample)
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
        finally:
            graph.reset()

    def test_cached_gemma_norm_scale_is_exact(self):
        for shape, group in [((3, 256), None), ((1, 128), None), ((1, 10240), 2560)]:
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
            scale = weight.float() + 1.0
            expected = self.ops.rms(x, weight, group=group)
            actual = self.ops.rms(x, scale, group=group, gemma=False)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_topk_normalization_and_cast_are_exact(self):
        for _ in range(32):
            logits = torch.randn((1, 512), device="cuda") * 3
            probabilities, ids = logits.softmax(-1).topk(10, -1)
            expected = probabilities / probabilities.sum(-1, keepdim=True)
            actual, actual_ids = self.ops.normalize_topk(probabilities, ids)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual_ids, ids.int(), atol=0, rtol=0)

    def test_qsa_compressed_write_only_on_complete_blocks(self):
        original = torch.randn((3, 128), device="cuda", dtype=torch.bfloat16)
        actual = original.clone()
        expected = original.clone()
        for pos in range(12):
            position = torch.tensor([pos], device="cuda")
            value = torch.randn((1, 128), device="cuda", dtype=torch.bfloat16)
            block = position // 4
            old = expected.index_select(0, block)
            expected.index_copy_(
                0, block, torch.where((position % 4 == 3).reshape(1, 1), value, old)
            )
            self.ops.store_qsa_compressed(actual, position, value)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_qsa_block_mean_matches_original_four_row_mean(self):
        for capacity in (8, 512):
            index_raw = torch.randn((capacity, 128), device="cuda", dtype=torch.bfloat16)
            for pos in (0, 1, 3, 4, 7, capacity - 4, capacity - 1):
                position = torch.tensor([pos], device="cuda")
                slots = (position // 4) * 4 + torch.arange(4, device="cuda")
                expected = (
                    index_raw.index_select(0, slots.clamp_max(capacity - 1))
                    .float()
                    .mean(0, keepdim=True)
                    .to(index_raw.dtype)
                )
                actual = self.ops.qsa_block_mean(index_raw, position)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_qsa_rope_preserves_bf16_rotation_and_remaining_channels(self):
        for rows, width in ((1, 128), (4, 128), (1, 256), (3, 256), (6, 256), (12, 256)):
            for seed in range(16):
                angles = torch.randn((1, 32), device="cuda") * (seed + 1)
                cos, sin = angles.cos(), angles.sin()
                x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16) * 3
                first, second = x[:, :32].float(), x[:, 32:64].float()
                expected = torch.cat(
                    [
                        torch.cat([first * cos - second * sin, second * cos + first * sin], -1).to(
                            x.dtype
                        ),
                        x[:, 64:],
                    ],
                    -1,
                )
                actual = self.ops.rope(x, cos, sin)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_qsa_fused_rope_and_kv_write_preserves_all_state(self):
        keys = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        expected_keys, expected_values = keys.clone(), values.clone()
        for pos in (0, 1, 3, 7, 63, 127):
            position = torch.tensor([pos], device="cuda", dtype=torch.int64)
            k = torch.randn((1, 256), device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            angles = torch.randn((1, 32), device="cuda")
            cos, sin = angles.cos(), angles.sin()
            expected_keys.index_copy_(0, position, self.ops.rope(k, cos, sin))
            expected_values.index_copy_(0, position, v)
            self.ops.store_qsa_kv(k, v, cos, sin, keys, values, position)
            torch.testing.assert_close(keys, expected_keys, atol=0, rtol=0)
            torch.testing.assert_close(values, expected_values, atol=0, rtol=0)

    def test_bounded_dense_attention_matches_full_kernel_exactly(self):
        q = torch.randn((3, 256), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        selected = torch.full((2048,), -1, device="cuda", dtype=torch.int32)
        dummy = selected[:1]
        for pos in [0, 3, 31, 32, 127, 255]:
            position = torch.tensor([pos], device="cuda")
            expected = self.ops.attention(q, k, v, position, selected)
            actual = self.ops.attention(q, k, v, position, dummy, dense_limit=256)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.ops.attention(q, k, v, position, dummy, dense_limit=257)

    def test_bounded_qsa_preserves_output_and_compressed_state(self):
        from qwen38_runner.model import QSA

        class FakeWeights:
            device = torch.device("cuda")

            def __init__(self):
                shapes = {
                    "q_proj.weight": (12288, 2560),
                    "k_proj.weight": (512, 2560),
                    "v_proj.weight": (512, 2560),
                    "o_proj.weight": (2560, 6144),
                    "q_norm.weight": (256,),
                    "k_norm.weight": (256,),
                    "indexer.index_qk_proj.weight": (640, 2560),
                    "indexer.q_layernorm.weight": (128,),
                    "indexer.k_layernorm.weight": (128,),
                }
                self.tensors = {}
                for name, shape in shapes.items():
                    self.tensors[name] = (
                        torch.randn(shape, device=self.device, dtype=torch.bfloat16) * 0.01
                    )

            def get(self, name, rows=None, cols=None):
                result = self.tensors[name.removeprefix("attn.")]
                if rows is not None:
                    result = result[rows]
                if cols is not None:
                    result = result[:, cols]
                return result.contiguous()

        weights = FakeWeights()
        full = QSA(weights, "attn", 0, 8, 8)
        bounded = QSA(weights, "attn", 0, 8, 8, execution_limit=8)
        self.assertEqual(full.slot_offsets.tolist(), [0, 1, 2, 3])
        self.assertEqual(bounded.slot_offsets.tolist(), [0, 1, 2, 3])
        slot_ptrs = (full.slot_offsets.data_ptr(), bounded.slot_offsets.data_ptr())
        angles = torch.randn((8, 32), device="cuda")
        cos, sin = angles.cos(), angles.sin()
        for pos in range(8):
            value = torch.randn((1, 2560), device="cuda", dtype=torch.bfloat16)
            position = torch.tensor([pos], device="cuda")
            args = (value, position, cos[pos : pos + 1], sin[pos : pos + 1], cos, sin)
            torch.testing.assert_close(bounded(*args), full(*args), atol=0, rtol=0)
            for name in ("keys", "values", "index_raw", "compressed"):
                torch.testing.assert_close(
                    getattr(bounded, name), getattr(full, name), atol=0, rtol=0
                )
        self.assertEqual((full.slot_offsets.data_ptr(), bounded.slot_offsets.data_ptr()), slot_ptrs)
        fallback = QSA(weights, "attn", 0, 8, 8, execution_limit=2049)
        self.assertEqual(fallback.dense_limit, 0)

    def test_invalid_execution_limit_fails_before_loading(self):
        from qwen38_runner.model import Qwen38Runner

        for limit in [0, -1, 4097, 0.5, True]:
            with self.assertRaises(ValueError):
                Qwen38Runner("/not-a-model", 0, 8, execution_limit=limit)

    def test_dense_execution_limit_traps_on_gpu(self):
        # A device assertion poisons its CUDA context, so isolate this negative test.
        code = f"""
import sys
sys.path.insert(0, {sys.path[0]!r})
import torch
from qwen38_runner import ops
q = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
k = torch.zeros((4, 256), device="cuda", dtype=torch.bfloat16)
position = torch.tensor([2], device="cuda")
selected = torch.tensor([-1], device="cuda", dtype=torch.int32)
ops.attention(q, k, k, position, selected, dense_limit=2)
torch.cuda.synchronize()
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dense attention execution limit exceeded", result.stdout + result.stderr)

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
