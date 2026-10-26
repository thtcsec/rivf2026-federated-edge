"""Regression tests for Phase A invariants. Does not retrain models.

Prepared CSVs are optional for GitHub clones; tests that need them skip.
Committed feature_shift_audit.csv and evaluation source are always required.
"""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'evaluation'
import sys
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VER = ROOT / 'results' / 'verified'
PREPARE = (EVAL / 'prepare_public_data.py').read_text(encoding='utf8')
RUN = (EVAL / 'run_verified_federated_baseline.py').read_text(encoding='utf8')


def _csvs():
        from protocol_label_audit import PATHS
        return {k: p for k, p in PATHS.items() if p.exists()}


class PreparedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = _csvs()
        if len(cls.paths) < 4:
            raise unittest.SkipTest('prepared CSVs not present')
        from protocol_label_audit import audit
        cls.rep = audit()

    def test_tl_cf_row_match(self):
        m = self.rep['matched']
        self.assertTrue(m['same_length'])
        self.assertTrue(m['label_equal'])
        self.assertTrue(m['unaffected_equal'])
        self.assertEqual(m['differing'], ['ip_proto', 'tp_src'])
        self.assertEqual(m['cf_protocol_values'], [0, 6])
        self.assertEqual(m['cf_source_port_unique'], [-1])
        self.assertEqual(m['cf_sentinel_rows'], self.rep['cf']['n'])

    def test_tl_ddos_all_code6_but_portscan_has_code0(self):
        s = self.rep['tl_attack_codes']
        self.assertTrue(s['ddos_all_code6'])
        self.assertGreater(s['n_portscan_code0'], 0)
        self.assertEqual(s['n_portscan_code0'] + s['n_portscan_code6'], s['n_portscan'])
        self.assertEqual(s['portscan_codes'], [0, 6])
        self.assertTrue(s['code0_attacks_are_portscan'])
        self.assertEqual(s['n_code0_attack'], s['n_portscan_code0'])
        # Table IV 3.7% on TL code 0 is exactly those PortScan rows.
        self.assertAlmostEqual(100 * s['code0_attack_rate'], 3.7, places=1)

    def test_protocol_transition_is_udp_to_zero(self):
        t = self.rep['transition']
        self.assertEqual(t['changed_pairs'], [(17, 0)])
        self.assertEqual(t['udp_to_zero'], t['n_changed'])
        self.assertEqual(t['udp_to_zero'], t['tl_udp_n'])
        self.assertEqual(t['tl_udp_attack_rate'], 0.0)

    def test_insdn_code0_is_portless_and_attack_associated(self):
        z = self.rep['insdn_code0']
        self.assertTrue(z['src_port_all_zero'])
        self.assertTrue(z['dst_port_all_zero'])
        self.assertGreater(z['share'], 0.35)
        self.assertLess(z['share'], 0.36)
        self.assertGreater(z['attack_rate'], 0.99)

    def test_checksums_match_readme(self):
        readme = (ROOT / 'data' / 'README.md').read_text(encoding='utf8')
        from protocol_label_audit import PATHS
        for p in PATHS.values():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertIn(h, readme, msg=p.name)


class MetadataTests(unittest.TestCase):
    def test_cf_summary_describes_cf_not_tl(self):
        p = ROOT / 'data' / 'cicids2017_mirror_counterfactual' / 'dataset_summary.json'
        if not p.exists():
            self.skipTest('CF metadata absent')
        meta = json.loads(p.read_text(encoding='utf8'))
        self.assertEqual(meta.get('protocol_values'), [0, 6])
        self.assertTrue(meta.get('row_matched'))
        self.assertEqual(sorted(meta.get('columns_changed', [])), ['ip_proto', 'tp_src'])
        n = meta.get('source_port_sentinel_rows')
        self.assertIsNotNone(n)
        self.assertGreater(n, 0)
        cf = ROOT / 'data' / 'cicids2017_mirror_counterfactual' / 'test.csv'
        if cf.exists():
            self.assertEqual(n, len(pd.read_csv(cf)))


class MaskingSignTests(unittest.TestCase):
    def test_source_port_masking_improves_tl_auc(self):
        audit = pd.read_csv(VER / 'feature_shift_audit.csv')
        delta = audit.loc[
            (audit.mirror == 'trafficlabelling') & (audit.feature == 'tp_src'),
            'mirror_auc_delta'].mean()
        self.assertGreater(delta, 0.0)
        self.assertAlmostEqual(delta, 0.0165, places=3)

    def test_protocol_masking_sign_flips(self):
        audit = pd.read_csv(VER / 'feature_shift_audit.csv')
        g = audit.groupby(['mirror', 'feature'])['mirror_auc_delta'].mean()
        self.assertLess(g.loc[('trafficlabelling', 'ip_proto')], 0.0)
        self.assertGreater(g.loc[('tl_counterfactual', 'ip_proto')], 0.0)

    def test_constant_source_port_does_not_change_cf_ranking(self):
        audit = pd.read_csv(VER / 'feature_shift_audit.csv')
        delta = audit.loc[
            (audit.mirror == 'tl_counterfactual') & (audit.feature == 'tp_src'),
            'mirror_auc_delta'].mean()
        self.assertAlmostEqual(delta, 0.0, places=6)


class MappingAndLeakageTests(unittest.TestCase):
    def test_duration_converted_from_microseconds(self):
        self.assertIn("/1e6", PREPARE)
        self.assertIn("'Flow Duration'", PREPARE)

    def test_named_source_columns(self):
        for col in ['Src Port', 'Dst Port', 'Protocol', 'Tot Fwd Pkts',
                    'Pkt Size Avg', 'Source Port', 'Destination Port',
                    'Total Fwd Packets', 'Average Packet Size']:
            self.assertIn(col, PREPARE)

    def test_zero_shot_scaler_is_source_only(self):
        self.assertIn("StandardScaler().fit(xtr)", RUN)
        self.assertNotIn("StandardScaler().fit(xte)", RUN)
        # Target views are transformed with the source-fitted scaler; a second
        # fit on target features would be leakage.
        self.assertEqual(RUN.count("StandardScaler().fit("), 2)  # source train + oracle


class PaperSignTests(unittest.TestCase):
    def test_manuscript_does_not_invert_source_port_masking(self):
        tex = (ROOT / 'rivf2026.tex').read_text(encoding='utf8')
        self.assertNotIn('true IANA', tex)
        bad = re.compile(r'costs \$0\.0165\$ ROC-AUC on (?:\\textsc\{tl\}|TL)', re.I)
        self.assertIsNone(bad.search(tex))
        self.assertNotRegex(tex, r'masking it costs \$0\.0165\$')
        self.assertNotIn('every DDoS and PortScan row has code 6', tex)


if __name__ == '__main__':
    unittest.main()
