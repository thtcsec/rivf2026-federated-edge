"""Phase B.2 tests. Synthetic fixtures are not experimental evidence."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import output_isolation as iso
from auc_pair_decomposition import decompose_roc_auc, pairwise_auc
from run_phase_b_mechanism import (
    CONDITIONS, PORT_I, PROTO_I, UNAFFECTED_I, build_factorial, effects_from_conditions,
)
from run_verified_federated_baseline import FEATURES

SRC = (EVAL / 'run_phase_b_mechanism.py').read_text(encoding='utf8')
MECH = ROOT / 'results' / 'phase_b' / 'mechanism'
VER = ROOT / 'results' / 'verified'
B1 = ROOT / 'results' / 'phase_b'


class IsolationTests(unittest.TestCase):
    def test_no_writes_to_protected_directories(self):
        tree = ast.parse(SRC)
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'main':
                    calls.append(ast.unparse(node.func) if hasattr(ast, 'unparse') else 'main')
        self.assertEqual(calls, [])
        self.assertIn('assert_mechanism_output', SRC)
        self.assertNotIn('OUT.mkdir', SRC)
        self.assertIn("target_training': False", SRC)

    def test_mechanism_dir_cannot_clobber_b1_or_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase_b = Path(tmp) / 'phase_b'
            verified = Path(tmp) / 'verified'
            mech = phase_b / 'mechanism'
            phase_b.mkdir(); verified.mkdir(); mech.mkdir()
            (phase_b / 'per_seed_noniid_cross.csv').write_text('seed\n11\n', encoding='utf8')
            iso.assert_phase_b_output(mech / 'ok.json', allowed_root=mech,
                                      forbidden_root=verified)
            with self.assertRaises(iso.UnsafeOutputError):
                iso.assert_phase_b_output(phase_b / 'per_seed_noniid_cross.csv',
                                          allowed_root=mech, forbidden_root=verified)
            with self.assertRaises(iso.UnsafeOutputError):
                iso.assert_phase_b_output(verified / 'summary.json',
                                          allowed_root=mech, forbidden_root=verified)
            with self.assertRaises(iso.UnsafeOutputError):
                iso.assert_phase_b_output(mech / '..' / 'per_seed_noniid_cross.csv',
                                          allowed_root=mech, forbidden_root=verified)


class FactorialConstructionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n, d = 20, 9
        tl = rng.normal(size=(n, d))
        tl[:, PROTO_I] = rng.choice([0.0, 6.0, 17.0], size=n)
        tl[:, PORT_I] = rng.integers(1, 5000, size=n).astype(float)
        cf = tl.copy()
        cf[:, PROTO_I] = np.where(tl[:, PROTO_I] == 17.0, 0.0, tl[:, PROTO_I])
        cf[:, PORT_I] = -1.0
        y = np.array([0, 1] * 10)
        self.tl, self.cf, self.y = tl, cf, y
        self.fac = build_factorial(tl, cf, y)

    def test_a_reproduces_tl(self):
        np.testing.assert_array_equal(self.fac['A'], self.tl)

    def test_d_reproduces_cf(self):
        np.testing.assert_array_equal(self.fac['D'], self.cf)

    def test_labels_preserved(self):
        np.testing.assert_array_equal(self.fac['y'], self.y)

    def test_unaffected_fields_preserved(self):
        for cond in CONDITIONS:
            np.testing.assert_array_equal(
                self.fac[cond][:, UNAFFECTED_I], self.tl[:, UNAFFECTED_I])

    def test_row_order_unchanged(self):
        np.testing.assert_array_equal(self.fac['A'][:, UNAFFECTED_I],
                                      self.fac['D'][:, UNAFFECTED_I])
        self.assertEqual(self.fac['A'].shape, self.tl.shape)

    def test_b_reads_cf_protocol_not_a_heuristic(self):
        np.testing.assert_array_equal(self.fac['B'][:, PROTO_I], self.cf[:, PROTO_I])
        np.testing.assert_array_equal(self.fac['B'][:, PORT_I], self.tl[:, PORT_I])
        self.assertIn('cf_x[:, PROTO_I]', SRC)
        self.assertNotIn('flags>0', SRC)

    def test_interaction_identity(self):
        fake = {c: {'roc_auc': v} for c, v in zip(CONDITIONS, [0.8, 0.4, 0.7, 0.35])}
        fx = effects_from_conditions(fake)['roc_auc']
        A, B, C, D = 0.8, 0.4, 0.7, 0.35
        self.assertAlmostEqual(fx['protocol_only_original_ports_B_minus_A'], B - A)
        self.assertAlmostEqual(fx['port_only_original_protocol_C_minus_A'], C - A)
        self.assertAlmostEqual(fx['combined_D_minus_A'], D - A)
        self.assertAlmostEqual(fx['interaction_D_minus_C_minus_B_plus_A'], D - C - B + A)

    def test_masking_not_conflated_with_replacement(self):
        self.assertIn('not_the_same', SRC)
        self.assertIn('variable', SRC.lower())
        self.assertIn('constant', SRC.lower())


class PairDecompositionTests(unittest.TestCase):
    def test_simple_fixture_matches_sklearn(self):
        y = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        groups = np.array([6, 6, 17, 17])
        d = decompose_roc_auc(y, scores, groups)
        self.assertAlmostEqual(d['reconstructed_auc'], roc_auc_score(y, scores))
        self.assertAlmostEqual(d['reconstructed_auc'], 1.0)
        cell = [c for c in d['cells'] if c['pos_group'] == 6 and c['neg_group'] == 17][0]
        self.assertTrue(cell['estimable'])
        self.assertAlmostEqual(cell['weight'], 1.0)
        within17 = [c for c in d['cells'] if c['pos_group'] == 17 and c['neg_group'] == 17][0]
        self.assertFalse(within17['estimable'])
        self.assertIsNone(within17['pairwise_auc'])
        self.assertEqual(within17['contribution'], 0.0)

    def test_exact_tie_handling(self):
        pos = np.array([1.0, 0.5])
        neg = np.array([0.5, 0.0])
        auc, n_pairs = pairwise_auc(pos, neg)
        # (1,0.5)=1, (1,0)=1, (0.5,0.5)=0.5, (0.5,0)=1 → 3.5/4
        self.assertEqual(n_pairs, 4)
        self.assertAlmostEqual(auc, 0.875)
        y = np.array([1, 1, 0, 0])
        scores = np.array([1.0, 0.5, 0.5, 0.0])
        groups = np.array([1, 1, 0, 0])
        d = decompose_roc_auc(y, scores, groups)
        self.assertAlmostEqual(d['reconstructed_auc'], roc_auc_score(y, scores))
        self.assertAlmostEqual(d['reconstructed_auc'], 0.875)

    def test_zero_weight_subgroup_pairs(self):
        y = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        groups = np.array([6, 6, 17, 17])
        d = decompose_roc_auc(y, scores, groups)
        empty = [c for c in d['cells'] if c['pos_group'] == 17]
        self.assertTrue(empty)
        for c in empty:
            self.assertEqual(c['weight'], 0.0)
            self.assertFalse(c['estimable'])
            self.assertIsNone(c['pairwise_auc'])
            self.assertEqual(c['contribution'], 0.0)

    def test_undefined_within_group_not_filled(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        groups = np.array([17, 17, 6, 6])
        d = decompose_roc_auc(y, scores, groups)
        cell = [c for c in d['cells'] if c['pos_group'] == 17 and c['neg_group'] == 17][0]
        self.assertFalse(cell['estimable'])
        self.assertIsNone(cell['pairwise_auc'])
        self.assertNotEqual(cell['pairwise_auc'], 0.5)
        self.assertNotEqual(cell['pairwise_auc'], 0)

    def test_global_reconstruction_mixed_groups(self):
        rng = np.random.default_rng(1)
        n = 40
        y = np.array([0, 1] * 20)
        scores = rng.random(n)
        groups = rng.choice([0, 6, 17], size=n)
        d = decompose_roc_auc(y, scores, groups)
        self.assertAlmostEqual(d['reconstructed_auc'], roc_auc_score(y, scores), places=12)

    def test_no_target_side_fitting_in_runner(self):
        self.assertNotIn('StandardScaler().fit', SRC)
        self.assertNotIn('partial_fit', SRC)
        self.assertIn('load_checkpoint', SRC)


class ExperimentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fact = MECH / 'factorial_per_seed.csv'
        cls.val = MECH / 'auc_decomposition_validation.json'
        if not cls.fact.exists() or not cls.val.exists():
            raise unittest.SkipTest('Phase B.2 artifacts not present yet')
        import pandas as pd
        cls.df = pd.read_csv(cls.fact)
        cls.val_doc = json.loads(cls.val.read_text(encoding='utf8'))
        cls.summary = json.loads((MECH / 'factorial_summary.json').read_text(encoding='utf8'))

    def test_replay_parity_recorded(self):
        self.assertEqual(self.val_doc['status'], 'PASS')
        self.assertEqual(self.summary['replay']['status'], 'PASS')
        self.assertEqual(self.summary['replay']['n_fail'], 0)

    def test_a_and_d_present_for_all_seeds_models(self):
        seeds = sorted(self.df.seed.unique())
        self.assertEqual(list(seeds), [11, 23, 42, 77, 101])
        for model in ('fedavg_iid', 'fedavg_noniid', 'centralized'):
            for cond in CONDITIONS:
                sub = self.df[(self.df.model == model) & (self.df.condition == cond)]
                self.assertEqual(len(sub), 5)

    def test_multi_seed_paired_consistency(self):
        for model in self.df.model.unique():
            for seed in self.df.seed.unique():
                sub = self.df[(self.df.model == model) & (self.df.seed == seed)]
                a = float(sub.loc[sub.condition == 'A', 'roc_auc'].iloc[0])
                d = float(sub.loc[sub.condition == 'D', 'roc_auc'].iloc[0])
                self.assertGreater(a, 0.5)
                self.assertLess(d, 0.5)

    def test_masking_replacement_flag_in_summary(self):
        self.assertTrue(self.summary['masking_versus_replacement']['not_the_same'])


if __name__ == '__main__':
    unittest.main()
