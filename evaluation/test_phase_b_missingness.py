"""Phase B.4 tests. Synthetic fixtures are not experimental evidence."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import OneHotEncoder

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import output_isolation as iso
import run_phase_b_missingness as miss
from run_phase_b_mechanism import UNAFFECTED_I, load_matched_sample
from run_phase_b_noniid_cross import REFERENCE_HASHES, SEEDS, validate_inputs
from run_verified_federated_baseline import FEATURES, met
from protocol_label_audit import PATHS

SRC = (EVAL / 'run_phase_b_missingness.py').read_text(encoding='utf8')
OUT = ROOT / 'results' / 'phase_b' / 'missingness'
B1 = ROOT / 'results' / 'phase_b'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_matches(path: Path, digest: str) -> bool:
    """Accept LF or CRLF of the same bytes; pandas CSVs were hashed on Windows."""
    data = path.read_bytes()
    lf = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    crlf = lf.replace(b'\n', b'\r\n')
    return digest in {hashlib.sha256(x).hexdigest() for x in (data, lf, crlf)}


class IsolationTests(unittest.TestCase):
    def test_no_foreign_main_calls(self):
        calls = []
        for node in ast.walk(ast.parse(SRC)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'main':
                    calls.append(True)
        self.assertEqual(calls, [])
        self.assertIn('assert_missingness_output', SRC)
        self.assertIn("'federated_hgb': False", SRC)

    def test_refuses_protected_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase_b = Path(tmp) / 'phase_b'
            verified = Path(tmp) / 'verified'
            miss_dir = phase_b / 'missingness'
            for p in (verified, miss_dir, phase_b / 'robustness', phase_b / 'mechanism'):
                p.mkdir(parents=True)
            iso.assert_phase_b_output(miss_dir / 'ok.json', allowed_root=miss_dir,
                                      forbidden_root=verified)
            for banned in (
                verified / 'summary.json',
                phase_b / 'per_seed_noniid_cross.csv',
                phase_b / 'robustness' / 'nonlinear_summary.json',
                phase_b / 'mechanism' / 'factorial_summary.json',
                miss_dir / '..' / 'robustness' / 'x.json',
            ):
                with self.assertRaises(iso.UnsafeOutputError):
                    iso.assert_phase_b_output(banned, allowed_root=miss_dir,
                                              forbidden_root=verified)


class ContractTests(unittest.TestCase):
    def test_input_hashes(self):
        if not all(p.exists() for p in PATHS.values()):
            self.skipTest('prepared CSVs absent')
        self.assertEqual(validate_inputs()['hashes'], REFERENCE_HASHES)

    def test_mask_probability_specified(self):
        self.assertEqual(miss.MASK_P, 0.5)
        self.assertIn('MASK_P = 0.5', SRC)
        self.assertNotIn('near-perfect on unseen source rows', SRC)

    def _vec(self, *vals):
        row = np.zeros(9)
        row[:len(vals)] = vals
        return row

    def test_leakage_audit_no_overlap(self):
        xtr = np.stack([self._vec(1), self._vec(1)])
        ytr = np.array([0, 0])
        xte = np.stack([self._vec(2)])
        yte = np.array([1])
        d = miss.leakage_audit(xtr, xte, ytr, yte)
        self.assertEqual(d['unique_exact_overlap'], 0)
        self.assertEqual(d['test_rows_matching_a_train_vector'], 0)
        self.assertEqual(d['test_overlap_frac'], 0.0)
        self.assertEqual(d['label_conflicts_in_this_split_overlap'], 0)
        self.assertIn('No identical feature vector', d['verdict'])
        self.assertNotIn('appear in both', d['verdict'])
        self.assertNotIn('inflates source AUC', d['verdict'])
        self.assertIn('does not measure source-AUC inflation', d['verdict'])

    def test_leakage_audit_identical_labels_overlap(self):
        xtr = np.stack([self._vec(1), self._vec(1), self._vec(3)])
        ytr = np.array([0, 0, 1])
        xte = np.stack([self._vec(1), self._vec(1)])
        yte = np.array([0, 0])
        d = miss.leakage_audit(xtr, xte, ytr, yte)
        self.assertEqual(d['unique_exact_overlap'], 1)
        self.assertEqual(d['test_rows_matching_a_train_vector'], 2)
        self.assertEqual(d['label_conflicts_in_this_split_overlap'], 0)
        self.assertEqual(d['train_internal_label_conflicts'], 0)
        self.assertEqual(d['test_internal_label_conflicts'], 0)
        self.assertIn('label-consistent', d['verdict'])

    def test_leakage_audit_cross_split_conflict(self):
        xtr = np.stack([self._vec(1), self._vec(2)])
        ytr = np.array([0, 1])
        xte = np.stack([self._vec(1)])
        yte = np.array([1])
        d = miss.leakage_audit(xtr, xte, ytr, yte)
        self.assertEqual(d['cross_split_conflicting_groups'], 1)
        self.assertEqual(d['label_conflicts_in_this_split_overlap'], 1)
        self.assertGreater(d['label_conflicts_in_this_split_overlap'], 0)
        self.assertNotIn('near-perfect', d['verdict'])
        self.assertIn('does not measure source-AUC inflation', d['verdict'])

    def test_leakage_audit_train_only_conflict(self):
        xtr = np.stack([self._vec(1), self._vec(1)])
        ytr = np.array([0, 1])
        xte = np.stack([self._vec(9)])
        yte = np.array([0])
        d = miss.leakage_audit(xtr, xte, ytr, yte)
        self.assertEqual(d['train_internal_label_conflicts'], 1)
        self.assertEqual(d['test_internal_label_conflicts'], 0)
        self.assertEqual(d['unique_exact_overlap'], 0)
        self.assertEqual(d['cross_split_conflicting_groups'], 0)

    def test_leakage_audit_test_only_conflict(self):
        xtr = np.stack([self._vec(9)])
        ytr = np.array([0])
        xte = np.stack([self._vec(1), self._vec(1)])
        yte = np.array([0, 1])
        d = miss.leakage_audit(xtr, xte, ytr, yte)
        self.assertEqual(d['test_internal_label_conflicts'], 1)
        self.assertEqual(d['train_internal_label_conflicts'], 0)
        self.assertEqual(d['unique_exact_overlap'], 0)

    def test_leakage_audit_multiple_labels_one_vector(self):
        xtr = np.stack([self._vec(1)] * 3)
        ytr = np.array([0, 1, 0])
        xte = np.stack([self._vec(1)] * 2)
        yte = np.array([1, 0])
        d = miss.leakage_audit(xtr, xte, ytr, yte)
        self.assertEqual(d['train_internal_label_conflicts'], 1)
        self.assertEqual(d['test_internal_label_conflicts'], 1)
        self.assertEqual(d['unique_exact_overlap'], 1)
        self.assertEqual(d['cross_split_conflicting_groups'], 0)
        self.assertEqual(d['test_rows_matching_a_train_vector'], 2)

    def test_leakage_audit_permutation_invariant(self):
        xtr = np.stack([self._vec(1), self._vec(1), self._vec(2), self._vec(3)])
        ytr = np.array([0, 1, 1, 0])
        xte = np.stack([self._vec(1), self._vec(2), self._vec(2), self._vec(9)])
        yte = np.array([1, 0, 0, 1])
        base = miss.leakage_audit(xtr, xte, ytr, yte)
        rng = np.random.default_rng(0)
        keys = (
            'unique_exact_overlap', 'test_rows_matching_a_train_vector',
            'test_overlap_frac', 'label_conflicts_in_this_split_overlap',
            'train_internal_label_conflicts', 'test_internal_label_conflicts',
            'cross_split_conflicting_groups', 'unique_train_vectors',
            'unique_test_vectors',
        )
        for _ in range(8):
            pi = rng.permutation(len(xtr))
            pj = rng.permutation(len(xte))
            got = miss.leakage_audit(xtr[pi], xte[pj], ytr[pi], yte[pj])
            for k in keys:
                self.assertEqual(got[k], base[k], k)

    def test_mask_deterministic_and_label_independent(self):
        a = miss.provenance_masks(20, 11)
        b = miss.provenance_masks(20, 11)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertTrue(a[0].any() and (~a[0]).any())
        self.assertFalse(np.array_equal(a[0], a[1]))

    def test_imputation_source_only(self):
        rng = np.random.default_rng(0)
        xtr = rng.normal(size=(30, 9))
        xtr[:, 0] = rng.choice([0, 6, 17], 30)
        xte = xtr.copy() + 99
        mode, median = miss.source_imputation_constants(xtr)
        self.assertIn(mode, (0.0, 6.0, 17.0))
        got = miss.apply_imputation(xte, mode, median)
        self.assertTrue(np.all(got[:, 0] == mode))
        self.assertTrue(np.all(got[:, 1] == median))
        np.testing.assert_array_equal(got[:, miss.UNAFFECTED_I], xte[:, miss.UNAFFECTED_I])

    def test_protocol_zero_not_missing_token(self):
        x = np.zeros((6, 9))
        x[:, 0] = [0, 6, 17, 0, 6, 17]
        mask_p = np.array([False, False, False, True, True, True])
        mask_s = np.zeros(6, dtype=bool)
        y = miss.apply_nan_missing(x, mask_p, mask_s)
        self.assertEqual(y[0, 0], 0.0)
        self.assertTrue(np.isnan(y[3, 0]))
        self.assertFalse(np.isnan(y[0, 0]))
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        enc.fit(x[:, [0]])
        z = miss.encode_lr_p4(x, enc, mask_p, mask_s)
        n_hot = enc.categories_[0].shape[0]
        # row 0: observed protocol 0, unavailable flag 0
        self.assertEqual(z[0, n_hot], 0.0)
        self.assertGreater(z[0, :n_hot].sum(), 0)
        # masked rows: all protocol dummies 0, flag 1
        np.testing.assert_array_equal(z[3:, :n_hot], 0)
        np.testing.assert_array_equal(z[3:, n_hot], 1)

    def test_p4_masked_rows_do_not_need_original_values(self):
        x = np.ones((4, 9)) * 3
        x[:, 0] = [0, 6, 17, 0]
        enc = miss.fit_protocol_encoder(x)
        full = np.ones(4, dtype=bool)
        a = miss.encode_lr_p4(x, enc, full, full)
        x[:, 0] = 99
        x[:, 1] = -123
        b = miss.encode_lr_p4(x, enc, full, full)
        np.testing.assert_allclose(a, b)

    def test_metrics_match_baseline(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        self.assertAlmostEqual(met(y, p)['roc_auc'], 1.0)
        self.assertIs(miss.met, met)

    def test_no_target_fit_in_source(self):
        self.assertNotRegex(SRC, r'\.fit\([^)]*y_pair')
        self.assertNotIn('early_stopping=True', SRC)
        self.assertIn('target_labels_used', SRC)

    def test_paired_rows_when_data_present(self):
        if not all(p.exists() for p in PATHS.values()):
            self.skipTest('prepared CSVs absent')
        tl, cf, y = load_matched_sample(11)
        np.testing.assert_array_equal(tl[:, UNAFFECTED_I], cf[:, UNAFFECTED_I])
        self.assertEqual(len(y), 30000)

    def test_hgb_not_called_federated(self):
        self.assertIn('not a federated model', SRC.lower())
        self.assertNotIn('federated Hist', SRC)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv = OUT / 'per_seed_policy_metrics.csv'
        cls.man = OUT / 'experiment_manifest.json'
        cls.summary = OUT / 'policy_comparison_summary.json'
        if not cls.csv.exists():
            raise unittest.SkipTest('Phase B.4 artifacts not present yet')
        import pandas as pd
        cls.df = pd.read_csv(cls.csv)
        cls.manifest = json.loads(cls.man.read_text(encoding='utf8'))
        cls.summary_obj = json.loads(cls.summary.read_text(encoding='utf8'))

    def test_feasibility_flags(self):
        p0 = self.df[self.df.policy == 'P0']
        self.assertFalse(bool(p0.mlcve_feasible.iloc[0]))
        self.assertNotIn('mlcve', set(p0.view))
        for pid in ('P1', 'P2', 'P3', 'P4'):
            self.assertTrue(bool(self.df[self.df.policy == pid].mlcve_feasible.iloc[0]), pid)

    def test_p2_p3_p4_tl_equals_cf(self):
        for model in miss.MODELS:
            for pid in ('P2', 'P3', 'P4'):
                for seed in SEEDS:
                    tl = self.df[(self.df.model == model) & (self.df.policy == pid)
                                 & (self.df.view == 'trafficlabelling') & (self.df.seed == seed)]
                    cf = self.df[(self.df.model == model) & (self.df.policy == pid)
                                 & (self.df.view == 'tl_counterfactual') & (self.df.seed == seed)]
                    self.assertAlmostEqual(float(tl.roc_auc.iloc[0]), float(cf.roc_auc.iloc[0]),
                                           places=12, msg=f'{model} {pid} seed {seed}')

    def test_reproduced_baselines(self):
        got = self.manifest['reproduction']
        self.assertAlmostEqual(got['fedavg P0 TL']['got'], 0.7357, places=4)
        self.assertAlmostEqual(got['fedavg P1 CF']['got'], 0.3545, places=4)
        self.assertAlmostEqual(got['fedavg P2 TL']['got'], 0.5075, places=4)

    def test_protected_freeze(self):
        frozen = self.manifest['phase_b_frozen_hashes']
        for rel, digest in frozen.items():
            path = B1 / rel
            self.assertTrue(path.exists(), rel)
            self.assertTrue(_sha_matches(path, digest), rel)
        blob = ''.join(frozen)
        self.assertNotIn('missingness', blob)

    def test_hgb_not_federated_in_artifacts(self):
        report = (OUT / 'missingness_scientific_report.md').read_text(encoding='utf8')
        self.assertIn('federated model', report.lower())
        self.assertIn('hgb is not a', report.lower())
        self.assertFalse(self.summary_obj.get('federated_hgb', True))
        self.assertEqual(self.manifest['mask_probability'], 0.5)


if __name__ == '__main__':
    unittest.main()
