"""Assert that every number quoted in rivf2026.tex matches results/verified/.

Run after run_verified_federated_baseline.py. Exits non-zero on the first
mismatch so that a stale table cannot survive a rerun unnoticed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / 'rivf2026.tex').read_text(encoding='utf8')
SUM = json.loads((ROOT / 'results' / 'verified' / 'summary.json').read_text(encoding='utf8'))
AUDIT = pd.read_csv(ROOT / 'results' / 'verified' / 'feature_shift_audit.csv')
failures: list[str] = []


def q(x: float) -> str:
    """Render a value in the paper's leading-dot four-decimal convention."""
    return f'{x:.4f}'.replace('0.', '.')


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


def table_row(label: str, block: list[str]) -> str:
    """Return the body of the unique row in `block` beginning with `label&`."""
    pat = re.compile(r'^\s*' + re.escape(label) + r'\s*&(.*?)\\\\')
    hits = [m.group(1) for m in map(pat.match, block) if m]
    if len(hits) != 1:
        failures.append(f'{label}: expected exactly one table row, found {len(hits)}')
        return ''
    return hits[0]


# --- source-domain summary table -------------------------------------------
RESULTS = table_block('tab:results')
for label, key in [('Central', 'centralized'), ('Local-only', 'local_mean'),
                   ('FedAvg IID', 'fedavg_iid'), ('FedAvg non-IID', 'fedavg_noniid')]:
    row = table_row(label, RESULTS)
    for metric in ['f1', 'roc_auc', 'balanced_accuracy', 'mcc']:
        blk = SUM['summary'][key][metric]
        cell = f"{q(blk['mean'])}$\\pm${q(blk['std'])}"
        check(f'{label}/{metric}', cell, where=row)

# --- cross-dataset table ----------------------------------------------------
CROSS = table_block('tab:cross')
for label, mirror in [(r'\textsc{tl} (original fields)', 'trafficlabelling'),
                      (r'\textsc{ml} (inferred fields)', 'mlcve')]:
    row = table_row(label, CROSS)
    for metric in ['f1', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc']:
        blk = SUM['summary'][f'cross_{mirror}'][metric]
        cell = f"{q(blk['mean'])}$\\pm${q(blk['std'])}"
        check(f'cross_{mirror}/{metric}', cell, where=row)

# --- retraining ablation table ---------------------------------------------
ABL = [('All nine', None), ('Without ports', 'without_ports'),
       ('Without protocol', 'without_protocol'), ('One-hot protocol', 'onehot_protocol')]
ABLT = table_block('tab:ablation')
for label, name in ABL:
    row = table_row(label, ABLT)
    src = SUM['summary']['fedavg_iid'] if name is None else SUM[f'ablation_{name}']
    for metric in ['f1', 'roc_auc']:
        check(f'{label}/src {metric}', f"{q(src[metric]['mean'])}$\\pm${q(src[metric]['std'])}", where=row)
    for mirror in ['trafficlabelling', 'mlcve']:
        blk = (SUM['summary'][f'cross_{mirror}'] if name is None
               else SUM[f'ablation_{name}_cross_{mirror}'])['roc_auc']
        check(f'{label}/{mirror} auc', f"{q(blk['mean'])}$\\pm${q(blk['std'])}", where=row)

# --- feature audit table ----------------------------------------------------
g = AUDIT.groupby(['mirror', 'feature'])[['ks', 'source_mask_f1', 'mirror_auc_delta']].mean()
AUD = table_block('tab:audit')
for label, feat in [('Protocol', 'ip_proto'), ('Destination port', 'tp_dst'),
                    ('Mean packet size', 'packet_size_avg'), ('Source port', 'tp_src')]:
    row = table_row(label, AUD)
    check(f'{label}/source F1', q(g.loc[('trafficlabelling', feat), 'source_mask_f1']), where=row)
    for mirror in ['trafficlabelling', 'mlcve']:
        check(f'{label}/{mirror} KS', q(g.loc[(mirror, feat), 'ks']), where=row)
        d = g.loc[(mirror, feat), 'mirror_auc_delta']
        # The paper writes an explicit sign for non-negative deltas except exact zero.
        rendered = q(d) if d < 0 else ('.0000' if abs(d) < 5e-5 else '+' + q(d))
        check(f'{label}/{mirror} dAUC', rendered, where=row)

# --- label-free provenance screen ------------------------------------------
SCR = pd.read_csv(ROOT / 'results' / 'verified' / 'provenance_screen.csv')
sg = SCR.groupby(['mirror', 'feature']).agg(ratio=('std_ratio', 'mean'), oor=('out_of_range', 'mean'),
                                            tv=('tv_dist', 'mean'), flagged=('flagged', 'all'),
                                            any_flag=('flagged', 'any'),
                                            collapse=('collapsed_category', 'all'))
SCRT = table_block('tab:screen')
for label, feat in [('Protocol', 'ip_proto'), ('Source port', 'tp_src'),
                    ('Destination port', 'tp_dst'), ('Byte rate', 'byte_count_per_sec')]:
    row = table_row(label, SCRT)
    for mirror in ['trafficlabelling', 'mlcve']:
        check(f'screen {label}/{mirror} ratio', f"{sg.loc[(mirror, feat), 'ratio']:.4f}", where=row)
        check(f'screen {label}/{mirror} OOR', q(sg.loc[(mirror, feat), 'oor']), where=row)

# the screen must flag exactly one column--mirror pair, on every seed
flagged = sorted(map(tuple, sg[sg.any_flag].index))
if flagged != [('mlcve', 'tp_src')]:
    failures.append(f'screen flags {flagged}, but the paper claims only the ML source port')
if not bool(sg.loc[('mlcve', 'tp_src'), 'flagged']):
    failures.append('screen does not flag the ML source port on all five seeds')
if len(sg) != 18:
    failures.append(f'paper claims 18 column--mirror pairs, artifact has {len(sg)}')
# the rejected categorical rule must fire on both mirrors, i.e. fail to discriminate
collapse = sorted(m for m, f in sg[sg.collapse].index if f == 'ip_proto')
if collapse != ['mlcve', 'trafficlabelling']:
    failures.append(f'mass-collapse rule fires on {collapse}, not on both mirrors')

unflagged = sg[~sg.any_flag]
for label, value in [
    ('screen min unflagged ratio', f"{unflagged.ratio.min():.3f}"),
    ('screen max unflagged OOR', f"{unflagged.oor.max():.4f}"),
    ('screen TL protocol TV', f"{sg.loc[('trafficlabelling', 'ip_proto'), 'tv']:.4f}"),
    ('screen ML protocol TV', f"{sg.loc[('mlcve', 'ip_proto'), 'tv']:.4f}"),
    ('screen ML protocol ratio', f"{sg.loc[('mlcve', 'ip_proto'), 'ratio']:.4f}"),
]:
    check(label, value)

# --- seed-level table -------------------------------------------------------
SEED = table_block('tab:seed')
for r in SUM['per_seed']:
    row = table_row(str(r['seed']), SEED)
    for blk in ['fedavg_iid', 'fedavg_noniid']:
        for metric in ['f1', 'mcc']:
            check(f"seed {r['seed']}/{blk}/{metric}", q(r[blk][metric]), where=row)

# --- prose scalars ----------------------------------------------------------
s = SUM['summary']
for label, value in [
    ('abstract TL AUC', f"{s['cross_trafficlabelling']['roc_auc']['mean']:.4f}"),
    ('abstract TL MCC', f"{s['cross_trafficlabelling']['mcc']['mean']:.4f}"),
    ('abstract ML AUC', f"{s['cross_mlcve']['roc_auc']['mean']:.4f}"),
    ('abstract ML MCC', f"{abs(s['cross_mlcve']['mcc']['mean']):.4f}"),
    ('abstract central F1', f"{s['centralized']['f1']['mean']:.4f}"),
    ('abstract local F1', f"{s['local_mean']['f1']['mean']:.4f}"),
    ('abstract iid F1', f"{s['fedavg_iid']['f1']['mean']:.4f}"),
    ('abstract noniid F1', f"{s['fedavg_noniid']['f1']['mean']:.4f}"),
    ('source ROC-AUC in prose', f"{s['fedavg_iid']['roc_auc']['mean']:.4f}"),
    ('TL PR-AUC in prose', f"{s['cross_trafficlabelling']['pr_auc']['mean']:.4f}"),
]:
    check(label, value)

for label, value in [
    ('protocol TL dAUC', f"{abs(g.loc[('trafficlabelling', 'ip_proto'), 'mirror_auc_delta']):.4f}"),
    ('protocol ML dAUC', f"{g.loc[('mlcve', 'ip_proto'), 'mirror_auc_delta']:.4f}"),
    ('protocol TL KS', f"{g.loc[('trafficlabelling', 'ip_proto'), 'ks']:.4f}"),
    ('protocol ML KS', f"{g.loc[('mlcve', 'ip_proto'), 'ks']:.4f}"),
    ('no-protocol source F1', f"{SUM['ablation_without_protocol']['f1']['mean']:.4f}"),
    ('no-protocol TL AUC', f"{SUM['ablation_without_protocol_cross_trafficlabelling']['roc_auc']['mean']:.4f}"),
    ('no-protocol ML AUC', f"{SUM['ablation_without_protocol_cross_mlcve']['roc_auc']['mean']:.4f}"),
    ('onehot TL AUC', f"{SUM['ablation_onehot_protocol_cross_trafficlabelling']['roc_auc']['mean']:.4f}"),
    ('onehot ML AUC', f"{SUM['ablation_onehot_protocol_cross_mlcve']['roc_auc']['mean']:.4f}"),
    ('no-ports TL AUC', f"{SUM['ablation_without_ports_cross_trafficlabelling']['roc_auc']['mean']:.4f}"),
]:
    check(label, value)

# --- counts, prevalence, confusion matrix, and mirror metadata -------------
first = SUM['per_seed'][0]
c = first['counts']
check('seed-11 client benign 1', f"{c['insdn_benign']:,}")
check('sampled attack count', f"{c['insdn_attack']:,}")
for mirror, tag in [('trafficlabelling', 'TL'), ('mlcve', 'ML')]:
    b, a = c[f'{mirror}_benign'], c[f'{mirror}_attack']
    check(f'{tag} sampled benign/attack', f'{b:,} / {a:,}')
    check(f'{tag} prevalence', f'{a / (a + b):.4f}')
for cell in ['tn', 'fp', 'fn', 'tp']:
    check(f'seed-11 confusion {cell}', f"{first['fedavg_iid'][cell]:,}")

for mirror, path in [('trafficlabelling', ROOT / 'data' / 'cicids2017_mirror'),
                     ('mlcve', ROOT / 'data' / 'cicids2017_mirror_mlcve')]:
    meta = json.loads((path / 'dataset_summary.json').read_text(encoding='utf8'))
    check(f'{mirror} cleaned flows', f"{meta['rows_after_clean']:,}")
    check(f'{mirror} test rows', f"{meta['rows_test']:,}")

# --- checksums published in data/README.md ---------------------------------
README = (ROOT / 'data' / 'README.md').read_text(encoding='utf8')
for rel in ['insdn/flow_stats.csv', 'cicids2017_mirror/test.csv',
            'cicids2017_mirror_mlcve/test.csv']:
    p = ROOT / 'data' / rel
    if not p.exists():
        failures.append(f'{rel}: prepared file absent, cannot verify checksum')
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h not in README:
        failures.append(f'{rel}: SHA-256 {h} is not published in data/README.md')

if failures:
    print(f'FAILED ({len(failures)} mismatch(es)):')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('OK: paper numbers, counts, and checksums all match results/verified/')
