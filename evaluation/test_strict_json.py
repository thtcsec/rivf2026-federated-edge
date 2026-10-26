"""Strict-JSON export must not touch the frozen original summary.json."""
from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
import sys
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import export_strict_summary_json as strict_export

VER = ROOT / 'results' / 'verified'
ORIGINAL = VER / 'summary.json'
STRICT = VER / 'summary.strict.json'


class StrictSummaryJsonTests(unittest.TestCase):
    def test_original_hash_unchanged(self):
        digest = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
        self.assertEqual(digest, strict_export.ORIGINAL_SHA256)

    def test_original_contains_python_nan_tokens(self):
        text = ORIGINAL.read_text(encoding='utf8')
        self.assertIn('NaN', text)
        with self.assertRaises(ValueError):
            json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

    def test_strict_file_is_rfc8259(self):
        self.assertTrue(STRICT.exists())
        text = STRICT.read_text(encoding='utf8')
        self.assertNotIn('NaN', text)
        parsed = json.loads(text)
        json.dumps(parsed, allow_nan=False)
        self.assertIsNone(self._first_nonfinite(parsed))

    def test_finite_metrics_identical(self):
        original = strict_export.load_original(ORIGINAL)
        exported = json.loads(STRICT.read_text(encoding='utf8'))
        self.assertTrue(strict_export.finite_equal(original, exported))
        self.assertAlmostEqual(
            original['summary']['cross_trafficlabelling']['roc_auc']['mean'],
            exported['summary']['cross_trafficlabelling']['roc_auc']['mean'])
        self.assertAlmostEqual(
            original['summary']['cross_tl_counterfactual']['roc_auc']['mean'],
            exported['summary']['cross_tl_counterfactual']['roc_auc']['mean'])

    def test_exporter_refuses_to_overwrite_original(self):
        with self.assertRaises(SystemExit):
            strict_export.export(src=ORIGINAL, dst=ORIGINAL)

    @staticmethod
    def _first_nonfinite(obj):
        if isinstance(obj, float) and not math.isfinite(obj):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                found = StrictSummaryJsonTests._first_nonfinite(v)
                if found is not None:
                    return found
        if isinstance(obj, list):
            for v in obj:
                found = StrictSummaryJsonTests._first_nonfinite(v)
                if found is not None:
                    return found
        return None


if __name__ == '__main__':
    unittest.main()
