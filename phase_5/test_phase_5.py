#!/usr/bin/env python3
"""Focused unit tests for Phase 5 metric and classification invariants."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import bootstrap_mean_ci, common_parser, edit_alignment, normalize, resolve_args, text_metrics


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), HERE/name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Phase5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.errors = load_script("02_error_analysis.py")
        cls.scaling = load_script("05_scalability_estimate.py")

    def test_unicode_normalization(self):
        self.assertEqual(normalize("  తెలుగు\n  భాష "), "తెలుగు భాష")

    def test_metric_fixture(self):
        metric = text_metrics("అమ్మ", "అమ")
        self.assertAlmostEqual(metric["cer"], 0.5)
        self.assertFalse(metric["exact_match"])

    def test_alignment_operations(self):
        distance, operations = edit_alignment("abc", "adc")
        self.assertEqual(distance, 1)
        self.assertIn(("substitution", "b", "d"), operations)

    def test_error_categories(self):
        categories, _ = self.errors.classify_word("క్త", "కత")
        self.assertIn("virama_or_conjunct_error", categories)
        categories, _ = self.errors.classify_word("కా", "క")
        self.assertIn("diacritic_or_vowel_sign_error", categories)
        categories, _ = self.errors.classify_word("తెలుగు", "Telugu")
        self.assertIn("non_telugu_output", categories)
        categories, _ = self.errors.classify_word("తెలుగు", "")
        self.assertIn("missing_output", categories)

    def test_bootstrap_is_deterministic(self):
        self.assertEqual(bootstrap_mean_ci([1, 2, 3], 42, 100), bootstrap_mean_ci([1, 2, 3], 42, 100))

    def test_cost_formula(self):
        price = {"input_usd_per_million_tokens": 2.5, "output_usd_per_million_tokens": 10.0}
        self.assertAlmostEqual(self.scaling.estimate_api_cost(1_000_000, 1_000_000, price), 12.5)

    def test_output_must_be_under_data_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"data"; root.mkdir()
            args = common_parser("test", "phase_5").parse_args(["--data-root", str(root), "--manifest", str(Path(temp)/"manifest.csv"), "--output-dir", str(Path(temp)/"outside")])
            with self.assertRaisesRegex(ValueError, "beneath"):
                resolve_args(args)


if __name__ == "__main__":
    unittest.main()
