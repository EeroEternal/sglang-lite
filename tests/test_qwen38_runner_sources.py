"""Dependency boundary and scalar shape guards, no local Torch required."""

import ast
import unittest
from pathlib import Path


class RunnerBoundaryTests(unittest.TestCase):
    def test_no_sglang_or_vllm_runtime_dependency(self):
        root = Path(__file__).resolve().parents[1] / "engine" / "qwen38_runner"
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [v.name for v in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                self.assertFalse(
                    any(
                        n in {"sglang", "vllm"} or n.startswith(("sglang.", "vllm.")) for n in names
                    ),
                    path.name,
                )

    def test_shard_sizes_cover_without_overlap(self):
        path = Path(__file__).resolve().parents[1] / "engine" / "qwen38_runner" / "weights.py"
        tree = ast.parse(path.read_text())
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "row_slice"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102, trusted local function
        rows = [namespace["row_slice"](2560, r, 8) for r in range(8)]
        self.assertEqual(
            [(s.start, s.stop) for s in rows], [(r * 320, (r + 1) * 320) for r in range(8)]
        )
        with self.assertRaises(ValueError):
            namespace["row_slice"](3, 0, 8)

    def test_scalar_scales_are_loaded_without_slicing(self):
        path = Path(__file__).resolve().parents[1] / "engine" / "qwen38_runner" / "weights.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Weights")
        namespace = {}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102, trusted local class

        class Scalar:
            def get_shape(self):
                return []

            def to(self, **kwargs):
                return self

            def contiguous(self):
                return self

            def __getitem__(self, key):
                raise AssertionError("0-D tensors cannot be sliced")

        class File:
            def get_tensor(self, name):
                return scalar

        scalar = Scalar()
        loader = object.__new__(namespace["Weights"])
        loader.source = lambda name: scalar
        loader.index = {"scale": "shard"}
        loader.files = {"shard": File()}
        loader.device = "cuda"
        self.assertIs(loader.get("scale"), scalar)
        with self.assertRaises(ValueError):
            loader.get("scale", slice(0, 1))

    def test_config_rejects_changed_equations(self):
        path = Path(__file__).resolve().parents[1] / "engine" / "qwen38_runner" / "config.py"
        tree = ast.parse(path.read_text())
        namespace = {}
        exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102, trusted local config
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        assignment = next(
            n
            for n in function.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "expected"
        )
        text = ast.literal_eval(assignment.value)
        text["layer_types"] = (["linear_attention"] * 3 + ["full_attention"]) * 12
        text["rope_parameters"] = {
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        }
        quant = {"group_size": 16, "num_bits": 4, "dynamic": False, "type": "float"}
        config = {
            "model_type": "qwen4_exp",
            "text_config": text,
            "quantization_config": {
                "quant_algo": "NVFP4",
                "config_groups": {
                    "group_0": {"weights": quant.copy(), "input_activations": quant.copy()}
                },
            },
        }
        validate = namespace["validate_config"]
        self.assertIs(validate(config, 8, 4096), text)
        for world, capacity in [(4, 4096), (8, 4097), (8, 8192), (8, 0)]:
            with self.assertRaises(ValueError):
                validate(config, world, capacity)
        text["output_gate_type"] = "silu"
        with self.assertRaises(ValueError):
            validate(config, 8, 4096)
        text["output_gate_type"] = "sigmoid"
        config["quantization_config"]["config_groups"]["group_0"]["weights"]["dynamic"] = True
        with self.assertRaises(ValueError):
            validate(config, 8, 4096)
