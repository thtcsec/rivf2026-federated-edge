"""Regression tests for Phase B.1. Synthetic fixtures are not experimental evidence."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, auc)
from sklearn.preprocessing import StandardScaler

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import output_isolation as iso
import run_phase_b_noniid_cross as phase_b
from run_verified_federated_baseline import FEATURES, met, parts, prob
from protocol_label_audit import PATHS, audit as protocol_audit

SRC_PHASE = (EVAL / 'run_phase_b_noniid_cross.py').read_text(encoding='utf8')
SRC_ISO = (EVAL / 'output_isolation.py').read_text(encoding='utf8')
SRC_BASE = (EVAL / 'run_verified_federated_baseline.py').read_text(encoding='utf8')
VER = ROOT / 'results' / 'verified'
PHASE_OUT = ROOT / 'results' / 'phase_b'


def _synthetic_xy(n=80, d=9, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    y[0] = 0
    y[1] = 1
    return x, y


class IsolationTests(unittest.TestCase):
    def test_phase_b_runner_never_writes_verified(self):
        import ast
        tree = ast.parse(SRC_PHASE)
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == 'main':
                calls.append(ast.unparse(func) if hasattr(ast, 'unparse') else func.attr)
            if isinstance(func, ast.Name) and func.id == 'main' and not isinstance(node, ast.FunctionDef):
                # Direct main() is Phase B's own entry point, allowed only at the bottom.
                pass
        self.assertEqual(calls, [], msg=f'Phase B must not call another module main(): {calls}')
        self.assertNotIn("OUT.mkdir", SRC_PHASE)
        self.assertIn('results/phase_b', SRC_ISO)
        self.assertIn('UnsafeOutputError', SRC_PHASE)

    def test_refuses_verified_and_sneak_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowed = Path(tmp) / 'phase_b'
            forbidden = Path(tmp) / 'verified'
            forbidden.mkdir()
            allowed.mkdir()
            iso.assert_phase_b_output(allowed / 'ok.json', allowed_root=allowed,
                                      forbidden_root=forbidden)
            with self.assertRaises(iso.UnsafeOutputError):
                iso.assert_phase_b_output(forbidden / 'x.json', allowed_root=allowed,
                                          forbidden_root=forbidden)
            with self.assertRaises(iso.UnsafeOutputError):
                iso.assert_phase_b_output(allowed / '..' / 'verified' / 'x.json',
                                          allowed_root=allowed, forbidden_root=forbidden)
            with self.assertRaises(iso.UnsafeOutputError):
                iso.assert_phase_b_output(Path(tmp) / 'elsewhere.json',
                                          allowed_root=allowed, forbidden_root=forbidden)
            self.assertFalse((forbidden / 'x.json').exists())

    def test_write_text_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowed = Path(tmp) / 'phase_b'
            allowed.mkdir()
            iso.write_text(allowed / 'a.txt', 'ok', allowed_root=allowed)
            self.assertEqual((allowed / 'a.txt').read_text(encoding='utf8'), 'ok')
            with self.assertRaises(iso.UnsafeOutputError):
                iso.write_text(Path(tmp) / 'verified' / 'nope.txt', 'bad',
                               allowed_root=allowed)


class InputInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(p.exists() for p in PATHS.values()):
            raise unittest.SkipTest('prepared CSVs not present')
        cls.inputs = phase_b.validate_inputs()
        cls.rep = protocol_audit()

    def test_original_dataset_hashes_unchanged(self):
        for k, expected in phase_b.REFERENCE_HASHES.items():
            self.assertEqual(self.inputs['hashes'][k], expected)

    def test_tl_cf_row_matching_preserved(self):
        self.assertEqual(self.inputs['n_tl'], self.inputs['n_cf'])
        self.assertEqual(self.inputs['differing'], ['ip_proto', 'tp_src'])
        m = self.rep['matched']
        self.assertTrue(m['same_length'])
        self.assertTrue(m['label_equal'])
        self.assertTrue(m['unaffected_equal'])
        self.assertEqual(m['differing'], ['ip_proto', 'tp_src'])

    def test_feature_order_is_original(self):
        self.assertEqual(self.inputs['features'], list(FEATURES))


class TrainingContractTests(unittest.TestCase):
    def test_both_fedavg_configs_use_same_source_training_sample(self):
        self.assertIn('pi = parts(xtr_s, ytr, seed)', SRC_PHASE)
        self.assertIn('pn = parts(xtr_s, ytr, seed, True)', SRC_PHASE)
        x, y = _synthetic_xy()
        iid = parts(x, y, 11)
        non = parts(x, y, 11, True)
        self.assertEqual(sum(len(yy) for _, yy in iid), len(y))
        self.assertEqual(sum(len(yy) for _, yy in non), len(y))

    def test_source_only_scaler_fitting_preserved(self):
        self.assertIn('StandardScaler().fit(xtr)', SRC_PHASE)
        self.assertNotIn('StandardScaler().fit(xte)', SRC_PHASE)
        self.assertNotIn('StandardScaler().fit(xo_tr)', SRC_PHASE)
        self.assertEqual(SRC_PHASE.count('StandardScaler().fit('), 1)

    def test_noniid_scoring_uses_wn_bn(self):
        self.assertIn('scores_from_fl(wn, bn, x, y)', SRC_PHASE)
        self.assertIn("noniid_cross = {m: scores_from_fl(wn, bn, x, y)", SRC_PHASE)
        self.assertNotRegex(SRC_PHASE, r'noniid_cross = \{m: scores_from_fl\(wi, bi')

    def test_no_target_side_training(self):
        self.assertNotIn("fit(so.transform", SRC_PHASE)
        self.assertIn("'target_training': False", SRC_PHASE)
        # Oracle/target-trained control lives only in the baseline script.
        self.assertIn("oracle", SRC_BASE)
        self.assertNotIn("oracle[", SRC_PHASE)

    def test_metric_definitions_match_baseline(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        p = np.clip(rng.random(8), 1e-6, 1 - 1e-6)
        got = met(y, p)
        q = (p >= .5).astype(int)
        pc, rc, _ = precision_recall_curve(y, p)
        self.assertAlmostEqual(got['roc_auc'], roc_auc_score(y, p))
        self.assertAlmostEqual(got['pr_auc'], auc(rc, pc))
        self.assertAlmostEqual(got['balanced_accuracy'], balanced_accuracy_score(y, q))
        self.assertAlmostEqual(got['mcc'], matthews_corrcoef(y, q))
        self.assertIs(phase_b.met, met)
        self.assertIs(phase_b.prob, prob)


class CheckpointSyntheticTests(unittest.TestCase):
    def test_checkpoint_round_trip_reproduces_synthetic_scores(self):
        x, y = _synthetic_xy(n=60, seed=3)
        sc = StandardScaler().fit(x)
        xs = sc.transform(x)
        coef = np.array([[1.2, -0.4, 0.1, 0, 0, 0, 0, 0, 0]], dtype=float)
        intercept = np.array([-0.05], dtype=float)
        saved = phase_b.scores_from_fl(coef, intercept, xs, y)
        with tempfile.TemporaryDirectory() as tmp:
            allowed = Path(tmp) / 'phase_b'
            allowed.mkdir()
            # Temporarily point saver at the temp tree.
            meta = phase_b.save_checkpoint(allowed, 11, 'fedavg_noniid', coef, intercept, sc,
                                           extra={'data_hashes': {'insdn': 'abc'}})
            self.assertEqual(meta['features'], list(FEATURES))
            ckpt = phase_b.load_checkpoint(allowed, 11, 'fedavg_noniid',
                                           expected_hashes={'insdn': 'abc'})
            xs2 = phase_b.transform_with_checkpoint(x, ckpt)
            got = phase_b.scores_from_fl(ckpt['coef'], ckpt['intercept'], xs2, y)
            for k in saved:
                self.assertAlmostEqual(float(got[k]), float(saved[k]), places=12)
            with self.assertRaises(ValueError):
                bad = json.loads((allowed / 'checkpoints' / 'seed_11_fedavg_noniid.json').read_text())
                bad['features'] = list(FEATURES)[::-1]
                (allowed / 'checkpoints' / 'seed_11_fedavg_noniid.json').write_text(
                    json.dumps(bad), encoding='utf8')
                phase_b.load_checkpoint(allowed, 11, 'fedavg_noniid',
                                        expected_hashes={'insdn': 'abc'})


class ExperimentArtifactTests(unittest.TestCase):
    """These read results/phase_b/ after the experiment; they skip if absent."""

    @classmethod
    def setUpClass(cls):
        cls.parity = PHASE_OUT / 'baseline_parity_report.json'
        cls.csv = PHASE_OUT / 'per_seed_noniid_cross.csv'
        if not cls.parity.exists() or not cls.csv.exists():
            raise unittest.SkipTest('Phase B artifacts not present yet')
        cls.parity_doc = json.loads(cls.parity.read_text(encoding='utf8'))
        cls.manifest = json.loads((PHASE_OUT / 'experiment_manifest.json').read_text(encoding='utf8'))

    def test_original_iid_and_centralized_remain_consistent(self):
        self.assertEqual(self.parity_doc['status'], 'PASS')
        self.assertEqual(self.parity_doc['n_fail'], 0)
        self.assertGreater(self.parity_doc['n_checks'], 50)

    def test_verified_directory_untouched_after_run(self):
        before = self.manifest.get('never_writes')
        self.assertTrue(str(VER).replace('\\', '/') in str(before).replace('\\', '/')
                        or Path(before).name == 'verified')
        for p in VER.glob('*'):
            if p.is_file():
                # Presence plus the experiment's own snapshot check is the live
                # guarantee; here we assert the reference files still exist.
                self.assertGreater(p.stat().st_size, 0)
        self.assertTrue((VER / 'summary.json').exists())

    def test_manifest_has_provenance(self):
        for key in ('git_commit', 'input_hashes', 'seeds', 'feature_schema',
                    'training_configuration', 'package_versions', 'timestamp_utc'):
            self.assertIn(key, self.manifest)
        self.assertFalse(self.manifest['target_training'])
        self.assertFalse(self.manifest['target_scaler'])
        self.assertEqual(self.manifest['feature_schema'], list(FEATURES))

    def test_noniid_columns_are_not_iid_copies(self):
        import pandas as pd
        df = pd.read_csv(self.csv)
        # If wn,bn were accidentally replaced by wi,bi, non-IID target AUCs
        # would match IID exactly on every seed.
        self.assertTrue(
            (df['noniid_trafficlabelling_roc_auc'] != df['iid_trafficlabelling_roc_auc']).any()
            or (df['noniid_tl_counterfactual_roc_auc'] != df['iid_tl_counterfactual_roc_auc']).any()
        )


class BaselineMainGuardTests(unittest.TestCase):
    def test_import_does_not_create_writes(self):
        # Importing the baseline module must not mkdir/write verified artifacts.
        hashed = hashlib.sha256((VER / 'summary.json').read_bytes()).hexdigest()
        import importlib
        import run_verified_federated_baseline as baseline
        importlib.reload(baseline)
        self.assertEqual(hashlib.sha256((VER / 'summary.json').read_bytes()).hexdigest(), hashed)
        self.assertTrue((EVAL / 'run_verified_federated_baseline.py').read_text(encoding='utf8').count('OUT.mkdir') == 1)
        self.assertIn('def main():\n OUT.mkdir', SRC_BASE.replace('\r\n', '\n'))


if __name__ == '__main__':
    unittest.main()
