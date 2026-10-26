"""Assert that every number quoted in rivf2026.tex matches results/verified/.

Run after run_verified_federated_baseline.py. Exits non-zero on the first
mismatch so that a stale table cannot survive a rerun unnoticed. It also
re-checks the structural claims the paper makes about the counterfactual
mirror, since those are what license the causal reading of Table III.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / 'rivf2026.tex').read_text(encoding='utf8')
VER = ROOT / 'results' / 'verified'
SUM = json.loads((VER / 'summary.json').read_text(encoding='utf8'))
AUDIT = pd.read_csv(VER / 'feature_shift_audit.csv')
SCR = pd.read_csv(VER / 'provenance_screen.csv')

VIEWS = [('trafficlabelling', 'TL'), ('tl_counterfactual', 'CF'), ('mlcve', 'MLCVE')]
MIRROR_DIR = {'trafficlabelling': 'cicids2017_mirror',
              'tl_counterfactual': 'cicids2017_mirror_counterfactual',
              'mlcve': 'cicids2017_mirror_mlcve'}
PROVENANCE_FIELDS = ('ip_proto', 'tp_src')
failures: list[str] = []


def q(x: float) -> str:
    """Render a value the way the paper's tables print it: four decimals with a
    leading zero, and a math minus rather than a hyphen when negative."""
    return f'$-${abs(x):.4f}' if x < 0 else f'{x:.4f}' 


def check(label: str, expected: str, *, where: str | None = None) -> None:
    """Assert `expected` occurs verbatim in the paper (or a named substring)."""
    hay = where if where is not None else TEX
    if expected not in hay:
        failures.append(f'{label}: paper is missing {expected!r}')


def table_block(tex_label: str) -> list[str]:
    """Return the lines of the table environment carrying \\label{tex_label}."""
    blocks = [b for b in re.split(r'\\begin\{table\*?\}', TEX)
              if f'\\label{{{tex_label}}}' in b]
    if len(blocks) != 1:
        failures.append(f'{tex_label}: expected exactly one table environment')
        return []
    return blocks[0].splitlines()


def split_at(block: list[str], marker: str) -> tuple[list[str], list[str]]:
    """Split a table body at the line containing `marker` (for two-panel tables)."""
    for i, line in enumerate(block):
        if marker in line:
            return block[:i], block[i:]
    failures.append(f'expected a panel marked {marker!r}')
    return block, []


def table_row(label: str, block: list[str]) -> str:
    """Return the body of the unique row in `block` beginning with `label&`."""
    pat = re.compile(r'^\s*' + re.escape(label) + r'\s*&(.*?)\\\\')
    hits = [m.group(1) for m in map(pat.match, block) if m]
    if len(hits) != 1:
        failures.append(f'{label}: expected exactly one table row, found {len(hits)}')
        return ''
    return hits[0]


# --- structural claims about the counterfactual -----------------------------
# The paper says CF is the same rows as TL with only two columns overwritten.
# Everything downstream depends on that, so verify it against the files.
tl_csv = ROOT / 'data' / 'cicids2017_mirror' / 'test.csv'
cf_csv = ROOT / 'data' / 'cicids2017_mirror_counterfactual' / 'test.csv'
_HAVE_PREPARED_CSV = tl_csv.exists() and cf_csv.exists()
if _HAVE_PREPARED_CSV:
    a, b = pd.read_csv(tl_csv), pd.read_csv(cf_csv)
    if list(a.columns) != list(b.columns):
        failures.append('counterfactual has a different column set from TL')
    elif len(a) != len(b):
        failures.append(f'counterfactual has {len(b)} rows against {len(a)} for TL')
    else:
        differing = sorted(c for c in a.columns if not a[c].equals(b[c]))
        if differing != sorted(PROVENANCE_FIELDS):
            failures.append(f'counterfactual differs from TL in {differing}, '
                            f'not exactly {sorted(PROVENANCE_FIELDS)}')
        if not a.label.equals(b.label):
            failures.append('counterfactual labels differ from TL')
        if set(b.tp_src.unique()) != {-1}:
            failures.append('counterfactual source port is not the constant sentinel')
        check('matched test rows', f'{len(a):,}')
        check('protocol disagreement rate',
              f'{100 * (a.ip_proto != b.ip_proto).mean():.2f}\\%')
        prev = (a.label != 1).mean()
        if abs(prev - (b.label != 1).mean()) > 1e-12:
            failures.append('counterfactual attack prevalence differs from TL')
else:
    print('NOTE: prepared TL/CF CSVs absent; skipping row-matching '
          '(EXTERNAL DATA REQUIRED)')

# --- source-domain summary table -------------------------------------------
RESULTS = table_block('tab:results')
for label, key in [('Central', 'centralized'), ('Local-only', 'local_mean'),
                   ('FedAvg IID', 'fedavg_iid'), ('FedAvg non-IID', 'fedavg_noniid')]:
    row = table_row(label, RESULTS)
    for metric in ['f1', 'roc_auc', 'balanced_accuracy', 'mcc']:
        blk = SUM['summary'][key][metric]
        check(f'{label}/{metric}', f"{q(blk['mean'])}$\\pm${q(blk['std'])}", where=row)

# --- cross-dataset table (two panels sharing row labels) --------------------
CROSS = table_block('tab:cross')
FED_PANEL, CEN_PANEL = split_at(CROSS, 'Centralized model')
ROW_LABEL = {'trafficlabelling': r'\textsc{tl} (original)',
             'tl_counterfactual': r'\textsc{cf} (matched)',
             'mlcve': r'\textsc{mlcve} (real)'}
for prefix, panel in [('cross_', FED_PANEL), ('central_cross_', CEN_PANEL)]:
    for view, _ in VIEWS:
        row = table_row(ROW_LABEL[view], panel)
        # F1 was dropped from Table III to widen it; it stays in the artifact.
        for metric in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc']:
            blk = SUM['summary'][f'{prefix}{view}'][metric]
            check(f'{prefix}{view}/{metric}', f"{q(blk['mean'])}$\\pm${q(blk['std'])}", where=row)

# --- retraining ablation table ---------------------------------------------
ABL = [('All nine (reconstructed)', None), (r'\quad without ports', 'without_ports'),
       (r'\quad without protocol', 'without_protocol'),
       (r'\quad one-hot protocol', 'onehot_protocol'),
       ('Intersection, 7 fields', 'intersection')]
# Panel (b) is a within-domain LR control, not an information-theoretic bound.
ABLT = table_block('tab:ablation')
# Panel (a) reports source means without dispersion; the per-seed spread stays in
# five_seed_metrics.csv. Panel (b) is the target-trained oracle and has no source column.
TRANSFER, ORACLE = split_at(ABLT, '(b) Target-trained within-domain control')
for label, name in ABL:
    row = table_row(label, TRANSFER)
    src = SUM['summary']['fedavg_iid'] if name is None else SUM[f'ablation_{name}']
    for metric in ['f1', 'roc_auc']:
        check(f'{label}/src {metric}', q(src[metric]['mean']), where=row)
    for view, _ in VIEWS:
        blk = (SUM['summary'][f'cross_{view}'] if name is None
               else SUM[f'ablation_{name}_cross_{view}'])['roc_auc']
        check(f'{label}/{view} auc', q(blk['mean']), where=row)

for label, name in [('All nine', None), ('Intersection, 7 fields', 'intersection')]:
    row = table_row(label, ORACLE)
    for view, _ in VIEWS:
        blk = (SUM['summary'][f'oracle_{view}'] if name is None
               else SUM[f'ablation_{name}_oracle_{view}'])['roc_auc']
        check(f'oracle {label}/{view} auc', q(blk['mean']), where=row)

# --- feature audit table ----------------------------------------------------
g = AUDIT.groupby(['mirror', 'feature'])[
    ['ks', 'source_mask_f1', 'mirror_auc_delta', 'mirror_mask_mcc']].mean()
# Table IV was retired for space; the audit is now reported in prose, so the
# values the argument rests on are checked there instead of cell by cell.

# --- label-free provenance screen ------------------------------------------
sg = SCR.groupby(['mirror', 'feature']).agg(
    ratio=('std_ratio', 'mean'), oor=('out_of_range', 'mean'), tv=('tv_dist', 'mean'),
    flagged=('flagged', 'all'), any_flag=('flagged', 'any'),
    collapse=('collapsed_category', 'all'))
# Table VI was retired for space; the screen is now reported in prose, so the
# flagged cells are checked there instead of cell by cell.
for view in ('tl_counterfactual', 'mlcve'):
    assert sg.loc[(view, 'tp_src'), 'ratio'] == 0.0, view
    assert sg.loc[(view, 'tp_src'), 'oor'] == 1.0, view

# The paper claims: exactly the two substituted ports are flagged, on every seed;
# protocol is never flagged; and the rejected rule fires on every view.
EXPECTED_FLAGS = [('mlcve', 'tp_src'), ('tl_counterfactual', 'tp_src')]
flagged = sorted(map(tuple, sg[sg.any_flag].index))
if flagged != sorted(EXPECTED_FLAGS):
    failures.append(f'check flags {flagged}, but the paper claims {sorted(EXPECTED_FLAGS)}')
for pair in EXPECTED_FLAGS:
    if not bool(sg.loc[pair, 'flagged']):
        failures.append(f'{pair} is not flagged on all five seeds')
for view, _ in VIEWS:
    if bool(sg.loc[(view, 'ip_proto'), 'any_flag']):
        failures.append(f'protocol is flagged on {view}, contradicting the paper')
if len(sg) != 27:
    failures.append(f'paper claims 27 column--view pairs, artifact has {len(sg)}')
collapse = sorted(m for m, f in sg[sg.collapse].index if f == 'ip_proto')
if collapse != sorted(v for v, _ in VIEWS):
    failures.append(f'mass-collapse rule fires on {collapse}, not on all three views')

unflagged = sg[~sg.any_flag]
unflagged_obs = SCR[~SCR.flagged]
for label, value in [
    ('screen flagged ratio', f"{sg.loc[('tl_counterfactual', 'tp_src'), 'ratio']:.4f}"),
    ('screen flagged OOR', f"{sg.loc[('tl_counterfactual', 'tp_src'), 'oor']:.4f}"),
    ('screen min unflagged mean ratio', f"{unflagged.ratio.min():.3f}"),
    ('screen max unflagged mean OOR', f"{unflagged.oor.max():.4f}"),
    ('screen min unflagged observation ratio', f"{unflagged_obs.std_ratio.min():.4f}"),
    ('screen max unflagged observation OOR', f"{unflagged_obs.out_of_range.max():.4f}"),
    ('screen TL protocol TV', f"{sg.loc[('trafficlabelling', 'ip_proto'), 'tv']:.4f}"),
    ('screen CF protocol TV', f"{sg.loc[('tl_counterfactual', 'ip_proto'), 'tv']:.4f}"),
    ('screen CF protocol ratio', f"{sg.loc[('tl_counterfactual', 'ip_proto'), 'ratio']:.4f}"),
]:
    check(label, value)

# --- prose scalars ----------------------------------------------------------
s = SUM['summary']
for label, value in [
    ('abstract TL AUC', f"{s['cross_trafficlabelling']['roc_auc']['mean']:.4f}"),
    ('abstract TL MCC', f"{s['cross_trafficlabelling']['mcc']['mean']:.4f}"),
    ('abstract CF AUC', f"{s['cross_tl_counterfactual']['roc_auc']['mean']:.4f}"),
    ('abstract CF MCC', f"{abs(s['cross_tl_counterfactual']['mcc']['mean']):.4f}"),
    ('abstract ML AUC', f"{s['cross_mlcve']['roc_auc']['mean']:.4f}"),
    ('abstract ML MCC', f"{abs(s['cross_mlcve']['mcc']['mean']):.4f}"),
    ('central TL AUC', f"{s['central_cross_trafficlabelling']['roc_auc']['mean']:.4f}"),
    ('central CF AUC', f"{s['central_cross_tl_counterfactual']['roc_auc']['mean']:.4f}"),
    ('central ML AUC', f"{s['central_cross_mlcve']['roc_auc']['mean']:.4f}"),
    # Source-domain means appear in Table II rather than in prose; only the IID F1
    # is quoted inline, as the baseline of the without-protocol ablation.
    ('iid F1 in prose', f"{s['fedavg_iid']['f1']['mean']:.4f}"),
    ('source ROC-AUC in prose', f"{s['fedavg_iid']['roc_auc']['mean']:.4f}"),
    ('TL PR-AUC in prose', f"{s['cross_trafficlabelling']['pr_auc']['mean']:.4f}"),
]:
    check(label, value)

# Masking the constant sentinel is AUC-neutral but not threshold-neutral, so the
# paper quotes the operating-point shift as well.
SRC_PORT_CF_DMCC = (g.loc[('tl_counterfactual', 'tp_src'), 'mirror_mask_mcc']
                    - SUM['summary']['cross_tl_counterfactual']['mcc']['mean'])

for label, value in [
    ('source-port CF dMCC', f"{SRC_PORT_CF_DMCC:+.4f}"),
    ('protocol TL KS', f"{g.loc[('trafficlabelling', 'ip_proto'), 'ks']:.4f}"),
    ('protocol CF KS', f"{g.loc[('tl_counterfactual', 'ip_proto'), 'ks']:.4f}"),
    ('source-port TL dAUC', f"{g.loc[('trafficlabelling', 'tp_src'), 'mirror_auc_delta']:.4f}"),
    ('dest-port TL dAUC', f"{abs(g.loc[('trafficlabelling', 'tp_dst'), 'mirror_auc_delta']):.4f}"),
    ('protocol TL dAUC', f"{abs(g.loc[('trafficlabelling', 'ip_proto'), 'mirror_auc_delta']):.4f}"),
    ('protocol CF dAUC', f"{g.loc[('tl_counterfactual', 'ip_proto'), 'mirror_auc_delta']:.4f}"),
    ('protocol ML dAUC', f"{g.loc[('mlcve', 'ip_proto'), 'mirror_auc_delta']:.4f}"),
    ('byte-count TL dAUC', f"{g.loc[('trafficlabelling', 'byte_count'), 'mirror_auc_delta']:+.4f}"),
    ('byte-count CF dAUC', f"{g.loc[('tl_counterfactual', 'byte_count'), 'mirror_auc_delta']:.4f}"),
    ('byte-rate CF dAUC', f"{g.loc[('tl_counterfactual', 'byte_count_per_sec'), 'mirror_auc_delta']:.4f}"),
    ('protocol TL KS', f"{g.loc[('trafficlabelling', 'ip_proto'), 'ks']:.4f}"),
    ('protocol CF KS', f"{g.loc[('tl_counterfactual', 'ip_proto'), 'ks']:.4f}"),
    ('no-protocol source F1', f"{SUM['ablation_without_protocol']['f1']['mean']:.4f}"),
    ('no-protocol TL AUC', f"{SUM['ablation_without_protocol_cross_trafficlabelling']['roc_auc']['mean']:.4f}"),
    ('no-protocol CF AUC', f"{SUM['ablation_without_protocol_cross_tl_counterfactual']['roc_auc']['mean']:.4f}"),
    ('no-protocol ML AUC', f"{SUM['ablation_without_protocol_cross_mlcve']['roc_auc']['mean']:.4f}"),
    ('onehot TL AUC', f"{SUM['ablation_onehot_protocol_cross_trafficlabelling']['roc_auc']['mean']:.4f}"),
    ('onehot CF AUC', f"{SUM['ablation_onehot_protocol_cross_tl_counterfactual']['roc_auc']['mean']:.4f}"),
    ('no-ports TL AUC', f"{SUM['ablation_without_ports_cross_trafficlabelling']['roc_auc']['mean']:.4f}"),
]:
    check(label, value)

# --- counts, prevalence, confusion matrix, and view metadata ---------------
first = SUM['per_seed'][0]
c = first['counts']
check('sampled benign count', f"{c['insdn_benign']:,}")
check('sampled attack count', f"{c['insdn_attack']:,}")
for view, tag in VIEWS:
    b, a_ = c[f'{view}_benign'], c[f'{view}_attack']
    check(f'{tag} sampled benign/attack', f'{b:,} / {a_:,}')
    check(f'{tag} prevalence', f'{a_ / (a_ + b):.4f}')
# Class-wise recall is reported per configuration. Benign recall is not stored
# directly; balanced accuracy is the mean of the two recalls, so it recovers it.
SRC_CFG = [('centralized', 'centralized'), ('local_mean', 'local-only'),
           ('fedavg_iid', 'IID'), ('fedavg_noniid', 'non-IID')]
att = [SUM['summary'][k]['recall']['mean'] for k, _ in SRC_CFG]
ben = [2 * SUM['summary'][k]['balanced_accuracy']['mean'] - a
       for (k, _), a in zip(SRC_CFG, att)]
for (_, tag), a_, b_ in zip(SRC_CFG, att, ben):
    check(f'{tag} attack/benign recall', f'${a_:.4f}$/${b_:.4f}$')
keys = [k for k, _ in SRC_CFG]
assert att.index(max(att)) == keys.index('fedavg_noniid'), 'non-IID should top attack recall'
assert ben.index(min(ben)) == keys.index('fedavg_noniid'), 'non-IID should bottom benign recall'

# The oracle prose quotes marginal dispersion and then the paired difference.
# Pairing matters: the marginal SDs are dominated by a seed effect that cancels,
# so the TL-CF oracle gap is small but consistent rather than indistinguishable.
for view, tag in [('trafficlabelling', 'TL'), ('tl_counterfactual', 'CF')]:
    blk = SUM['summary'][f'oracle_{view}']['roc_auc']
    check(f'oracle {tag} mean+-sd prose',
          f"${blk['mean']:.4f}\\pm{blk['std']:.4f}$")


def paired(a_key, b_key):
    """Per-seed differences a-b, both scored on the same seeded row sample."""
    return [r[a_key]['roc_auc'] - r[b_key]['roc_auc'] for r in SUM['per_seed']]


od = paired('oracle_trafficlabelling', 'oracle_tl_counterfactual')
td = paired('cross_trafficlabelling', 'cross_tl_counterfactual')
n = len(od)
om = statistics.mean(od)
ose = statistics.stdev(od) / n ** 0.5
T_CRIT_DF4 = 2.776  # two-sided 0.05, df = 5 - 1
half = T_CRIT_DF4 * ose
check('oracle paired gap', f'${om:.4f}$')
check('oracle paired 95% CI', f'$[{om - half:.4f},{om + half:.4f}]$')
check('transfer paired gap',
      f'${statistics.mean(td):.4f}\\pm{statistics.stdev(td):.4f}$')
assert all(x > 0 for x in od), 'TL oracle should exceed CF on every seed'
assert om - half > 0, 'paired oracle CI should exclude zero'
if 'true IANA' in TEX:
    failures.append('manuscript still claims true IANA protocol numbers')
if 'every DDoS and PortScan row has code 6' in TEX:
    failures.append('manuscript still claims every PortScan row has code 6')
if re.search(r'masking it costs \$0\.0165\$', TEX):
    failures.append('source-port masking sign is still inverted (costs instead of gains)')
check('source-port TL masking gains', r'\emph{gains} $0.0165$')

# Protocol-code associations are computed from the prepared CSVs, not hardcoded.
try:
    sys.path.insert(0, str(ROOT / 'evaluation'))
    from protocol_label_audit import audit as protocol_audit, PATHS as PROTO_PATHS
    if all(p.exists() for p in PROTO_PATHS.values()):
        pre = protocol_audit()
    else:
        print('NOTE: prepared CSVs absent; skipping protocol-label audit '
              '(EXTERNAL DATA REQUIRED)')
        pre = None
    if pre is not None:
        z = pre['insdn_code0']
        check('insdn proto0 share', f"{100 * z['share']:.1f}\\%")
        check('insdn proto0 attack', f"{100 * z['attack_rate']:.1f}\\%")
        check('tl udp count', f"{pre['transition']['tl_udp_n']:,}")
        if pre['transition']['changed_pairs'] != [(17, 0)]:
            failures.append(f"protocol transitions {pre['transition']['changed_pairs']} are not solely 17->0")
        if pre['transition']['tl_udp_attack_rate'] != 0.0:
            failures.append('TL code-17 attack rate is not zero on the prepared split')
        ac = pre['tl_attack_codes']
        if not ac['ddos_all_code6']:
            failures.append('not every TL DDoS row has protocol code 6')
        if ac['n_portscan_code0'] == 0:
            failures.append('TL PortScan has no code-0 rows; prose/table reconciliation is stale')
        if not ac['code0_attacks_are_portscan']:
            failures.append('TL code-0 attacks are not exactly the code-0 PortScan rows')
        check('tl code0 attack pct', f"{100 * ac['code0_attack_rate']:.1f}\\%")
        check('cf code0 attack not exact zero', r'$<$0.1\%')
        _num = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
        w = _num.get(ac['n_portscan_code0'])
        if w is None:
            failures.append(f"unexpected TL PortScan code-0 count {ac['n_portscan_code0']}")
        else:
            check('portscan code0 wording', f'{w} PortScan')
        if not z['src_port_all_zero'] or not z['dst_port_all_zero']:
            failures.append('InSDN protocol 0 is not uniformly portless')
except Exception as exc:
    failures.append(f'protocol audit failed: {exc}')

for view, _ in VIEWS:
    meta = json.loads((ROOT / 'data' / MIRROR_DIR[view] / 'dataset_summary.json').read_text(encoding='utf8'))
    check(f'{view} cleaned flows', f"{meta['rows_after_clean']:,}")
    check(f'{view} test rows', f"{meta['rows_test']:,}")
cf_meta = json.loads((ROOT / 'data' / 'cicids2017_mirror_counterfactual' / 'dataset_summary.json').read_text(encoding='utf8'))
if cf_meta.get('protocol_values') != [0, 6]:
    failures.append(f"CF metadata protocol_values={cf_meta.get('protocol_values')} (expected [0, 6] from CF rows)")
if cf_meta.get('source_port_sentinel_rows') in (None, 0):
    failures.append('CF metadata source_port_sentinel_rows still describes the pre-overwrite table')

# --- Phase B robustness numbers quoted in the camera-ready -----------------
# These are generated artifacts, not hardcoded oracles. The paper may quote a
# number only if the corresponding JSON still contains that rounded mean.
PHASE_B = ROOT / 'results' / 'phase_b'


def load_phase_b(rel: str):
    path = PHASE_B / rel
    if not path.exists():
        failures.append(f'Phase B artifact absent: results/phase_b/{rel}')
        return None
    return json.loads(path.read_text(encoding='utf8'))


b1 = load_phase_b('aggregate_noniid_cross.json')
paired_b = load_phase_b('paired_effects.json')
fact = load_phase_b('mechanism/factorial_summary.json')
hgb = load_phase_b('robustness/nonlinear_summary.json')
miss = load_phase_b('missingness/policy_comparison_summary.json')

if b1 is not None:
    check('noniid TL AUC',
          f"{b1['views']['trafficlabelling']['roc_auc']['mean']:.4f}")
    check('noniid CF AUC',
          f"{b1['views']['tl_counterfactual']['roc_auc']['mean']:.4f}")
if paired_b is not None:
    check('noniid paired dAUC',
          f"{paired_b['fedavg_noniid']['tl_minus_cf']['roc_auc']['mean']:.4f}")
if fact is not None:
    iid_fact = fact['models']['fedavg_iid']['conditions_roc_auc']
    check('factorial A original TL', f"{iid_fact['A']['mean']:.4f}")
    check('factorial B protocol-only', f"{iid_fact['B']['mean']:.4f}")
    check('factorial C port-only', f"{iid_fact['C']['mean']:.4f}")
    check('factorial D combined CF', f"{iid_fact['D']['mean']:.4f}")
    locked_tl = s['cross_trafficlabelling']['roc_auc']['mean']
    locked_cf = s['cross_tl_counterfactual']['roc_auc']['mean']
    if abs(iid_fact['A']['mean'] - locked_tl) > 5e-5:
        failures.append('factorial A does not match locked IID TL ROC-AUC')
    if abs(iid_fact['D']['mean'] - locked_cf) > 5e-5:
        failures.append('factorial D does not match locked IID CF ROC-AUC')
    if iid_fact['B']['mean'] >= 0.5:
        failures.append('protocol-only factorial B is not below chance')
    if iid_fact['C']['mean'] <= iid_fact['A']['mean']:
        failures.append('port-only factorial C does not exceed original TL')
if hgb is not None:
    if hgb.get('federated'):
        failures.append('HGB summary marks federated=true')
    zs = hgb['zero_shot']
    check('HGB nine-field TL AUC',
          f"{zs['trafficlabelling']['roc_auc']['mean']:.4f}")
    check('HGB nine-field CF AUC',
          f"{zs['tl_counterfactual']['roc_auc']['mean']:.4f}")
    check('HGB nine-field unmatched MLCVE',
          f"{zs['mlcve']['roc_auc']['mean']:.4f}")
    check('HGB source AUC',
          f"{hgb['source']['roc_auc']['mean']:.4f}")
    fac = hgb['factorial']['conditions_roc_auc']
    check('HGB factorial protocol-only B', f"{fac['B']['mean']:.4f}")
    check('HGB factorial port-only C', f"{fac['C']['mean']:.4f}")
    if zs['trafficlabelling']['roc_auc']['mean'] <= 0.5:
        failures.append('HGB TL is not above chance')
    if zs['tl_counterfactual']['roc_auc']['mean'] <= 0.5:
        failures.append('HGB CF is not above chance; paper must not claim a reversal')
    if 'not a federated model' not in TEX:
        failures.append('manuscript does not state that the HGB control is not federated')
    if 'nine-field' not in TEX:
        failures.append('manuscript does not mark the 0.9602/0.9080 HGB as nine-field')
if miss is not None:
    if miss.get('federated_hgb'):
        failures.append('missingness summary marks federated_hgb=true')
    hgb_p2 = miss['by_model_policy']['hgb_categorical']['P2']
    check('HGB seven-field matched AUC',
          f"{hgb_p2['trafficlabelling']['roc_auc']['mean']:.4f}")
    check('HGB seven-field unmatched MLCVE',
          f"{hgb_p2['mlcve']['roc_auc']['mean']:.4f}")
    if abs(hgb_p2['trafficlabelling']['roc_auc']['mean']
           - hgb_p2['tl_counterfactual']['roc_auc']['mean']) > 1e-12:
        failures.append('HGB seven-field TL and CF should coincide')
    lr_p2 = miss['by_model_policy']['lr_fedavg_iid']['P2']['trafficlabelling']['roc_auc']['mean']
    locked_isec = SUM['ablation_intersection_cross_trafficlabelling']['roc_auc']['mean']
    if abs(lr_p2 - locked_isec) > 5e-5:
        failures.append('missingness LR P2 does not match locked seven-field intersection')
    if 'source-retrained seven-field' not in TEX:
        failures.append('0.9603 is not identified as a source-retrained seven-field HGB')
    leak = miss.get('leakage') or {}
    if 'mean_test_overlap_frac' in leak:
        check('source-test overlap pct',
              f"{100 * leak['mean_test_overlap_frac']:.2f}\\%")

for label, snippet in [
    ('non-IID still described as unscored',
     'the non-IID model is not scored on the target views'),
    ('non-IID still described as unevaluated',
     'The non-IID model is not evaluated on the target views'),
    ('trees still described as open',
     'Whether the reversal appears for trees'),
    ('imputations still described as untested',
     'Other imputations (mode, missing indicator, unknown category) are untested'),
    ('HGB still described as the same configuration',
     'that HGB configuration'),
    ('linearity isolation overstated (intro)',
     'specific to the linear setting'),
    ('linearity isolation overstated (method)',
     'specific to this linear setting'),
    ('universality delimitation overstated',
     'delimiting the finding to the tested linear settings'),
    ('Phase B claimed already committed on GitHub',
     'committed outputs, including the Phase'),
    ('AI disclosure denies pipeline execution',
     'did not execute the intrusion-detection experiments'),
    ('artifact section still time-dependent unpublished-on-main',
     'not yet published on that branch'),
    ('artifact section still says working tree',
     'this working tree'),
    ('protocol uniqueness overstated',
     'the only field whose'),
    ('byte-rate dAUC claimed exactly zero',
     r'$\Delta$AUC $0.0000$'),
    ('manuscript claims preregistration',
     'pre-registered'),
    ('unseen-row HGB AUC claimed',
     'near-perfect on unseen'),
]:
    if snippet in TEX:
        failures.append(label)

check('camera-ready tag url',
      'https://github.com/thtcsec/rivf2026-federated-edge/tree/rivf2026-camera-ready')
check('lanvin DOI', r'10.1007/978-3-031-31108-6\textunderscore 2')
check('AI disclosure orchestration', 'experiment orchestration')
check('pool vs sample abstract', 'Each seed scores a stratified 30,000-row sample')
check('pool vs sample conclusion', 'each seed scoring a stratified 30,000-row sample')
check('InSDN Hugging Face derivative', 'Sharukesh/INSDN')
check('screen means not observations', r'feature/view \emph{means}')
check('no unseen-row AUC in threats', 'does not compute an unseen-row source AUC')
check('no timestamped preregistration', 'not a timestamped preregistration')

# --- checksums published in data/README.md ---------------------------------
README = (ROOT / 'data' / 'README.md').read_text(encoding='utf8')
_missing_prepared = []
for rel in ['insdn/flow_stats.csv'] + [f'{MIRROR_DIR[v]}/test.csv' for v, _ in VIEWS]:
    p = ROOT / 'data' / rel
    if not p.exists():
        _missing_prepared.append(rel)
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h not in README:
        failures.append(f'{rel}: SHA-256 {h} is not published in data/README.md')
if _missing_prepared:
    print('NOTE: prepared file(s) absent; skipping checksum verification for '
          + ', '.join(_missing_prepared) + ' (EXTERNAL DATA REQUIRED)')

if failures:
    print(f'FAILED ({len(failures)} mismatch(es)):')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('OK: paper numbers, structural claims, counts and checksums all match results/verified/ and results/phase_b/')
