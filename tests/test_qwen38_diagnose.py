import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class DifferenceTests(unittest.TestCase):
    def test_first_difference(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location("qwen38_diagnose", scripts / "qwen38_sglang_diagnose.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict("os.environ", {}, clear=True), patch.object(sys, "path", [str(scripts), *sys.path]):
            spec.loader.exec_module(module)
        compare = module.first_difference
        self.assertIsNone(compare([1, 2], [1, 2]))
        self.assertEqual(compare([1, 2], [1, 3]), 1)
        self.assertEqual(compare([1], [1, 2]), 1)
        self.assertEqual(compare([], [1]), 0)
