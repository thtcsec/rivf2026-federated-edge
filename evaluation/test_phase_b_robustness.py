"""Phase B.3 tests. Synthetic fixtures are not experimental evidence."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import output_isolation as iso
import run_phase_b_robustness as rob
from run_phase_b_mechanism import CONDITIONS, PORT_I, PROTO_I, UNAFFECTED_I, build_factorial
from run_phase_b_noniid_cross import REFERENCE_HASHES, validate_inputs
from run_verified_federated_baseline import FEATURES, INSDN, SEEDS, met
from protocol_label_audit import PATHS

SRC = (EVAL / 'run_phase_b_robustness.py').read_text(encoding='utf8')
OUT = ROOT / 'results' / 'phase_b' / 'robustness'
B1 = ROOT / 'results' / 'phase_b'
MECH = B1 / 'mechanism'
VER = ROOT / 'results' / 'verified'
PROTECTED_B1 = [
    B1 / 'per_seed_noniid_cross.csv',
    B1 / 'experiment_report.md',
    B1 / 'experiment_manifest.json',
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha_matches(path: Path, digest: str) -> bool:
    """Accept LF or CRLF of the same bytes; pandas CSVs were hashed on Windows."""
    data = path.read_bytes()
    lf = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    crlf = lf.replace(b'\n', b'\r\n')
    return digest in {hashlib.sha256(x).hexdigest() for x in (data, lf, crlf)}


class IsolationTests(unittest.TestCase):
    def test_no_protected_writes_in_source(self):
        calls = []
        for node in ast.walk(ast.parse(SRC)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'main':
                    calls.append(True)
        self.assertEqual(calls, [])
        self.assertIn('assert_robustness_output', SRC)
        self.assertNotIn('OUT.mkdir', SRC)
        self.assertIn("'federated': False", SRC)
        self.assertIn("target_training': False", SRC)

    def test_refuses_verified_mechanism_and_b1(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase_b = Path(tmp) / 'phase_b'
            verified = Path(tmp) / 'verified'
            mech = phase_b / 'mechanism'
            rob_dir = phase_b / 'robustness'
            for p in (verified, mech, rob_dir):
                p.mkdir(parents=True)
            (phase_b / 'per_seed_noniid_cross.csv').write_text('x', encoding='utf8')
            iso.assert_phase_b_output(rob_dir / 'ok.json', allowed_root=rob_dir,
                                      forbidden_root=verified)
            for banned in (
                verified / 'summary.json',
                phase_b / 'per_seed_noniid_cross.csv',
                mech / 'factorial_summary.json',
                rob_dir / '..' / 'mechanism' / 'x.json',
            ):
                with self.assertRaises(iso.UnsafeOutputError):
                    iso.assert_phase_b_output(banned, allowed_root=rob_dir,
                                              forbidden_root=verified)

    def test_robustness_dir_constant(self):
        self.assertEqual(iso.ROBUSTNESS_DIR, B1 / 'robustness')
        self.assertEqual(rob.ROBUSTNESS_DIR, iso.ROBUSTNESS_DIR)


class ContractTests(unittest.TestCase):
    def test_input_hashes(self):
        if not all(p.exists() for p in PATHS.values()):
            self.skipTest('prepared CSVs absent')
        got = validate_inputs()['hashes']
        self.assertEqual(got, REFERENCE_HASHES)

    def test_source_partition_reuse(self):
        self.assertIn("test_size=.2", SRC)
        self.assertIn("stratify=yi", SRC)
        self.assertIn("random_state=seed", SRC)
        self.assertIn("load(INSDN", SRC)
        if not INSDN.exists():
            self.skipTest('InSDN CSV absent')
        a = rob.source_partition(11)
        b = rob.source_partition(11)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(len(a[0]), 24000)
        self.assertEqual(len(a[1]), 6000)
        self.assertEqual(len(a[2]), 24000)
        self.assertEqual(len(a[3]), 6000)
        self.assertEqual(set(a[2].tolist()) | set(a[3].tolist()), {0, 1})

    def test_tl_cf_paired_rows(self):
        if not all(p.exists() for p in PATHS.values()):
            self.skipTest('prepared CSVs absent')
        from run_phase_b_mechanism import load_matched_sample
        tl, cf, y = load_matched_sample(11)
        self.assertEqual(tl.shape, (30000, 9))
        self.assertEqual(cf.shape, tl.shape)
        self.assertEqual(len(y), 30000)
        np.testing.assert_array_equal(tl[:, UNAFFECTED_I], cf[:, UNAFFECTED_I])
        self.assertFalse(np.array_equal(tl[:, [PROTO_I, PORT_I]], cf[:, [PROTO_I, PORT_I]]))

    def test_protocol_index_from_schema(self):
        self.assertEqual(rob.PROTOCOL_INDEX, FEATURES.index('ip_proto'))
        self.assertEqual(FEATURES[rob.PROTOCOL_INDEX], 'ip_proto')
        self.assertEqual(rob.PROTOCOL_INDEX, PROTO_I)
        self.assertIn('categorical_features=[PROTOCOL_INDEX]', SRC)

    def test_categorical_and_no_scaler(self):
        tree = ast.parse(SRC)
        imported = []
        call_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.extend(a.name for a in node.names)
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    call_names.append(func.id)
                elif isinstance(func, ast.Attribute):
                    call_names.append(func.attr)
        self.assertNotIn('StandardScaler', imported)
        self.assertNotIn('StandardScaler', call_names)
        self.assertEqual(rob.HGB_HYPERPARAMETERS['early_stopping'], False)
        self.assertIsNone(rob.HGB_HYPERPARAMETERS['class_weight'])
        self.assertEqual(rob.HGB_HYPERPARAMETERS['loss'], 'log_loss')
        self.assertEqual(rob.HGB_HYPERPARAMETERS['learning_rate'], 0.1)
        self.assertEqual(rob.HGB_HYPERPARAMETERS['max_iter'], 100)
        self.assertEqual(rob.HGB_HYPERPARAMETERS['max_leaf_nodes'], 15)
        self.assertEqual(rob.HGB_HYPERPARAMETERS['min_samples_leaf'], 20)
        self.assertEqual(rob.HGB_HYPERPARAMETERS['l2_regularization'], 1.0)
        self.assertIn('categorical_features',
                      inspect.signature(HistGradientBoostingClassifier.__init__).parameters)
        self.assertNotIn('GridSearchCV', call_names)
        self.assertNotIn('RandomizedSearchCV', call_names)

    def test_code_zero_not_missing(self):
        x = np.zeros((4, 9))
        x[:, rob.PROTOCOL_INDEX] = [0, 6, 17, 0]
        y = rob.as_hgb_matrix(x)
        self.assertFalse(np.isnan(y[:, rob.PROTOCOL_INDEX]).any())
        self.assertEqual(set(y[:, rob.PROTOCOL_INDEX]), {0.0, 6.0, 17.0})
        known = rob.protocol_support(y, y)
        self.assertEqual(known['n_unknown_protocol'], 0)
        unseen = y.copy()
        unseen[:, rob.PROTOCOL_INDEX] = 99
        miss = rob.protocol_support(y, unseen)
        self.assertEqual(miss['n_unknown_protocol'], 4)
        self.assertEqual(miss['unknown_codes'], [99])

    def test_attack_class_probability(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 9))
        x[:, 0] = rng.choice([0, 6, 17], size=40)
        y = np.array([0, 1] * 20)
        m = rob.make_hgb(0)
        m.fit(rob.as_hgb_matrix(x), y)
        p = rob.attack_proba(m, rob.as_hgb_matrix(x))
        self.assertEqual(list(m.classes_), [0, 1])
        self.assertEqual(p.shape, (40,))
        np.testing.assert_allclose(p, m.predict_proba(rob.as_hgb_matrix(x))[:, 1])
        self.assertTrue(bool(m.is_categorical_[rob.PROTOCOL_INDEX]))
        self.assertFalse(bool(np.any(np.delete(m.is_categorical_, rob.PROTOCOL_INDEX))))

    def test_metrics_match_baseline_helper(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        self.assertIs(rob.met, met)
        got = met(y, p)
        self.assertAlmostEqual(got['roc_auc'], 1.0)
        self.assertEqual(got['tp'], 2)
        self.assertEqual(got['tn'], 2)

    def test_factorial_a_d_and_unaffected(self):
        rng = np.random.default_rng(1)
        tl = rng.normal(size=(12, 9))
        tl[:, 0] = rng.choice([0.0, 6.0, 17.0], 12)
        tl[:, 1] = np.arange(12, dtype=float)
        cf = tl.copy()
        cf[:, 0] = np.where(tl[:, 0] == 17.0, 0.0, tl[:, 0])
        cf[:, 1] = -1.0
        y = np.array([0, 1] * 6)
        fac = build_factorial(tl, cf, y)
        np.testing.assert_array_equal(fac['A'], tl)
        np.testing.assert_array_equal(fac['D'], cf)
        np.testing.assert_array_equal(fac['B'][:, PORT_I], tl[:, PORT_I])
        np.testing.assert_array_equal(fac['B'][:, PROTO_I], cf[:, PROTO_I])
        np.testing.assert_array_equal(fac['C'][:, PROTO_I], tl[:, PROTO_I])
        np.testing.assert_array_equal(fac['C'][:, PORT_I], cf[:, PORT_I])
        for cond in CONDITIONS:
            np.testing.assert_array_equal(fac[cond][:, UNAFFECTED_I], tl[:, UNAFFECTED_I])

    def test_determinism_synthetic(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(50, 9))
        x[:, 0] = rng.choice([0, 6, 17], 50)
        y = np.array([0, 1] * 25)
        x = rob.as_hgb_matrix(x)
        p1 = rob.attack_proba(rob.make_hgb(7).fit(x, y), x)
        p2 = rob.attack_proba(rob.make_hgb(7).fit(x, y), x)
        np.testing.assert_allclose(p1, p2)

    def test_source_only_training_in_source(self):
        self.assertIn('model.fit(xtr, ytr)', SRC)
        tree = ast.parse(SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'HistGradientBoostingClassifier':
                    keys = [k.arg for k in node.keywords]
                    self.assertNotIn('class_weight', keys)
                    self.assertIn('categorical_features', keys)
                    self.assertIn('early_stopping', keys)
        self.assertFalse(rob.HGB_HYPERPARAMETERS['early_stopping'])
        # Target matrices are scored, never fitted.
        self.assertNotRegex(SRC, r"\.fit\(\s*tg")


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv = OUT / 'nonlinear_per_seed.csv'
        cls.fact = OUT / 'nonlinear_factorial.csv'
        cls.man = OUT / 'nonlinear_experiment_manifest.json'
        cls.summary = OUT / 'nonlinear_summary.json'
        cls.report = OUT / 'nonlinear_scientific_report.md'
        if not cls.csv.exists():
            raise unittest.SkipTest('Phase B.3 artifacts not present yet')
        import pandas as pd
        cls.df = pd.read_csv(cls.csv)
        cls.fac = pd.read_csv(cls.fact)
        cls.manifest = json.loads(cls.man.read_text(encoding='utf8'))
        cls.summary_obj = json.loads(cls.summary.read_text(encoding='utf8'))
        cls.report_text = cls.report.read_text(encoding='utf8')

    def test_a_equals_tl_and_d_equals_cf(self):
        for seed in SEEDS:
            tl = float(self.df[(self.df.seed == seed) & (self.df.view == 'trafficlabelling')].roc_auc.iloc[0])
            cf = float(self.df[(self.df.seed == seed) & (self.df.view == 'tl_counterfactual')].roc_auc.iloc[0])
            a = float(self.fac[(self.fac.seed == seed) & (self.fac.condition == 'A')].roc_auc.iloc[0])
            d = float(self.fac[(self.fac.seed == seed) & (self.fac.condition == 'D')].roc_auc.iloc[0])
            self.assertAlmostEqual(a, tl, places=12)
            self.assertAlmostEqual(d, cf, places=12)

    def test_manifest_contract(self):
        self.assertFalse(self.manifest['federated'])
        self.assertFalse(self.manifest['target_training'])
        self.assertFalse(self.manifest['target_calibration'])
        self.assertFalse(self.manifest['early_stopping'])
        self.assertEqual(self.manifest['protocol_column_index'], FEATURES.index('ip_proto'))
        self.assertEqual(self.manifest['input_hashes'], REFERENCE_HASHES)
        self.assertEqual(self.manifest['seeds'], SEEDS)
        self.assertEqual(self.manifest['feature_schema'], list(FEATURES))
        self.assertIsNone(self.manifest['scaler'])
        self.assertIsNone(self.manifest['class_weight'])
        self.assertEqual(self.manifest['threshold'], 0.5)
        self.assertIn('evaluation', self.manifest)
        self.assertEqual(self.manifest['hyperparameters']['max_leaf_nodes'], 15)

    def test_not_called_federated(self):
        self.assertIn('Not a federated model', self.report_text)
        self.assertNotIn('federated HistGradient', self.report_text.lower())
        self.assertFalse(self.summary_obj['federated'])

    def test_protected_phase_b_files_unchanged_from_manifest_freeze(self):
        frozen = self.manifest['phase_b_frozen_hashes']
        for rel, digest in frozen.items():
            path = B1 / rel
            self.assertTrue(path.exists(), rel)
            self.assertTrue(_sha_matches(path, digest), rel)
        self.assertNotIn('robustness', ''.join(frozen))

    def test_unknown_protocol_fraction_zero(self):
        tgt = self.df[self.df.view.isin(['trafficlabelling', 'tl_counterfactual', 'mlcve'])]
        self.assertTrue((tgt.n_unknown_protocol == 0).all())
