"""Phase B.2: matched 2x2 factorial and exact ROC-AUC pair decomposition.

Reads frozen Phase B.1 checkpoints. Writes only under results/phase_b/mechanism/.
Does not retrain, does not call run_verified_federated_baseline.main(), and does
not modify results/verified/ or existing Phase B.1 files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from auc_pair_decomposition import (  # noqa: E402
    cells_involving, contribution_sum, decompose_roc_auc,
)
from output_isolation import (  # noqa: E402
    MECHANISM_DIR, PHASE_B_DIR, VERIFIED_DIR, UnsafeOutputError,
    assert_mechanism_output, prepare_phase_b_dir, write_text,
)
from run_phase_b_noniid_cross import (  # noqa: E402
    CSV_PATHS, MODELS, PARITY_ATOL, PARITY_HARD_AUC, PARITY_RTOL,
    REFERENCE_HASHES, SEEDS, T_CRIT_DF4, central_proba_from_checkpoint,
    load_checkpoint, sha256, snapshot_verified, transform_with_checkpoint,
    validate_inputs,
)
from run_verified_federated_baseline import (  # noqa: E402
    FEATURES, FULL_METRICS, MIRRORS, PROVENANCE_FIELDS, compact, load, met, prob,
)

CONDITIONS = ('A', 'B', 'C', 'D')
PROTO_I = FEATURES.index('ip_proto')
PORT_I = FEATURES.index('tp_src')
UNAFFECTED_I = [i for i, f in enumerate(FEATURES) if f not in PROVENANCE_FIELDS]
CSV_PREFIX = {
    'fedavg_iid': 'iid',
    'fedavg_noniid': 'noniid',
    'centralized': 'central',
}
REPLAY_VIEW = {'A': 'trafficlabelling', 'D': 'tl_counterfactual'}
CODE17 = 17
PAPER = {
    'protocol_mask_tl': -0.1438,
    'protocol_mask_cf': 0.2647,
    'source_port_mask_tl': 0.0165,
    'source_port_mask_cf_auc': 0.0,
    'intersection_auc': 0.5075,
    'noniid_delta_auc': 0.3791,
}


def git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def snapshot_dir(root: Path, *, exclude_prefix: tuple[str, ...] = ()) -> dict[str, str]:
    out = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(rel == pre or rel.startswith(pre.rstrip('/') + '/') for pre in exclude_prefix):
            continue
        out[rel] = sha256(p)
    return out


def isolation_self_test() -> None:
    prepare_phase_b_dir(MECHANISM_DIR)
    probe = assert_mechanism_output(MECHANISM_DIR / 'isolation_probe.txt')
    write_text(probe, 'mechanism isolation ok\n', allowed_root=MECHANISM_DIR)
    for banned in (
        VERIFIED_DIR / 'should_not_exist.json',
        PHASE_B_DIR / 'per_seed_noniid_cross.csv',
        PHASE_B_DIR / 'experiment_report.md',
        MECHANISM_DIR / '..' / 'per_seed_noniid_cross.csv',
        MECHANISM_DIR / '..' / 'verified' / 'x.json',
    ):
        try:
            assert_mechanism_output(banned)
        except UnsafeOutputError:
            pass
        else:
            raise SystemExit(f'isolation self-test failed: accepted {banned}')
    probe.unlink(missing_ok=True)


def load_matched_sample(seed: int):
    xa, ya = load(MIRRORS['trafficlabelling'], 'label', 'cic', seed)
    xb, yb = load(MIRRORS['tl_counterfactual'], 'label', 'cic', seed)
    if not np.array_equal(ya, yb):
        raise SystemExit(f'seed {seed}: TL/CF labels differ after sampling')
    if not np.array_equal(xa[:, UNAFFECTED_I], xb[:, UNAFFECTED_I]):
        raise SystemExit(f'seed {seed}: unaffected features differ after sampling')
    if np.array_equal(xa[:, [PROTO_I, PORT_I]], xb[:, [PROTO_I, PORT_I]]):
        raise SystemExit(f'seed {seed}: TL and CF provenance fields are identical')
    if xa.shape != xb.shape or xa.shape[0] != 30000:
        raise SystemExit(f'seed {seed}: unexpected sample shape {xa.shape}')
    return xa, xb, ya


def build_factorial(tl_x: np.ndarray, cf_x: np.ndarray, y: np.ndarray) -> dict:
    """A=TL, D=CF; B,C mix fields read from the matched arrays (no heuristic)."""
    A = tl_x.copy()
    D = cf_x.copy()
    B = tl_x.copy()
    B[:, PROTO_I] = cf_x[:, PROTO_I]
    C = tl_x.copy()
    C[:, PORT_I] = cf_x[:, PORT_I]
    if not np.array_equal(A, tl_x):
        raise SystemExit('A does not reproduce TL')
    if not np.array_equal(D, cf_x):
        raise SystemExit('D does not reproduce CF')
    if not np.array_equal(B[:, PORT_I], A[:, PORT_I]):
        raise SystemExit('B must keep original TL source port')
    if not np.array_equal(B[:, PROTO_I], D[:, PROTO_I]):
        raise SystemExit('B protocol is not the matched CF protocol column')
    if not np.array_equal(C[:, PROTO_I], A[:, PROTO_I]):
        raise SystemExit('C must keep original TL protocol')
    if not np.array_equal(C[:, PORT_I], D[:, PORT_I]):
        raise SystemExit('C source port is not the matched CF sentinel column')
    for name, X in [('A', A), ('B', B), ('C', C), ('D', D)]:
        if not np.array_equal(X[:, UNAFFECTED_I], tl_x[:, UNAFFECTED_I]):
            raise SystemExit(f'{name} altered an unaffected feature')
    proto_only = [i for i, f in enumerate(FEATURES) if f != 'ip_proto']
    port_only = [i for i, f in enumerate(FEATURES) if f != 'tp_src']
    if not np.array_equal(B[:, proto_only], A[:, proto_only]):
        raise SystemExit('B differs from A in more than protocol')
    if not np.array_equal(C[:, port_only], A[:, port_only]):
        raise SystemExit('C differs from A in more than source port')
    return {'A': A, 'B': B, 'C': C, 'D': D, 'y': y}


def predict_proba(ckpt: dict, x_raw: np.ndarray, model: str, seed: int) -> np.ndarray:
    xs = transform_with_checkpoint(x_raw, ckpt)
    if model == 'centralized':
        return central_proba_from_checkpoint(ckpt, xs, seed)
    return prob(ckpt['coef'], ckpt['intercept'], xs)


def t_interval(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float('nan')
    se = std / n ** 0.5 if n > 1 else float('nan')
    half = T_CRIT_DF4 * se if n > 1 else float('nan')
    return {
        'n': n, 'mean': mean, 'std': std, 'se': se,
        'ci95': [mean - half, mean + half],
        't_critical': T_CRIT_DF4, 'df': n - 1,
        'assumption': (
            'Student t, two-sided 95%, df=4. Repeated holdouts from one corpus, '
            'not independent NIDS datasets.'
        ),
    }


def effects_from_conditions(by_cond: dict[str, dict]) -> dict[str, dict]:
    """Controlled contrasts on the same rows. Not assumed additive."""
    out = {}
    keys = [k for k in FULL_METRICS if k in by_cond['A']]
    for k in keys:
        A, B, C, D = (float(by_cond[c][k]) for c in CONDITIONS)
        out[k] = {
            'protocol_only_original_ports_B_minus_A': B - A,
            'port_only_original_protocol_C_minus_A': C - A,
            'protocol_under_sentinel_D_minus_C': D - C,
            'port_under_reconstructed_D_minus_B': D - B,
            'combined_D_minus_A': D - A,
            'interaction_D_minus_C_minus_B_plus_A': D - C - B + A,
        }
    return out


def compare_metrics(got: dict, ref: dict, label: str) -> list[dict]:
    rows = []
    for k in FULL_METRICS:
        if k not in got or k not in ref or ref[k] is None or (isinstance(ref[k], float) and np.isnan(ref[k])):
            continue
        a, b = got[k], ref[k]
        if k in ('tn', 'fp', 'fn', 'tp'):
            ok = int(a) == int(b)
            rows.append({'label': f'{label}.{k}', 'status': 'PASS' if ok else 'FAIL',
                         'got': int(a), 'ref': int(b), 'abs_diff': abs(int(a) - int(b))})
            continue
        fa, fb = float(a), float(b)
        diff = abs(fa - fb)
        ok = bool(np.allclose(fa, fb, atol=PARITY_ATOL, rtol=PARITY_RTOL))
        hard = diff > PARITY_HARD_AUC and k in (
            'roc_auc', 'pr_auc', 'mcc', 'f1', 'balanced_accuracy')
        rows.append({'label': f'{label}.{k}', 'status': 'FAIL' if (not ok or hard) else 'PASS',
                     'got': fa, 'ref': fb, 'abs_diff': diff})
    return rows


def b1_reference_row(csv: pd.DataFrame, seed: int, model: str, view: str) -> dict:
    prefix = CSV_PREFIX[model]
    row = csv.loc[csv.seed == seed].iloc[0]
    out = {}
    for k in FULL_METRICS:
        col = f'{prefix}_{view}_{k}'
        if col in csv.columns:
            out[k] = row[col]
    return out


def share_with_cancellation(part: float, total: float) -> dict:
    if total == 0:
        return {'share': None, 'note': 'total effect is zero; share undefined'}
    share = part / total
    return {
        'share': float(share),
        'share_percent': float(100 * share),
        'note': (
            'Share is (sum of signed cell deltas involving original code-17) / '
            '(AUC_A - AUC_D). Other cells can cancel, so the share can exceed '
            '100% or be negative. This is an accounting identity, not a unique '
            'causal attribution.'
        ),
    }


def run() -> dict:
    isolation_self_test()
    b1_before = snapshot_dir(PHASE_B_DIR, exclude_prefix=('mechanism',))
    verified_before = snapshot_verified()
    inputs = validate_inputs()
    hashes = inputs['hashes']
    if hashes != REFERENCE_HASHES:
        raise SystemExit('prepared CSV hashes drifted from Phase B.1 reference')
    b1_csv = pd.read_csv(PHASE_B_DIR / 'per_seed_noniid_cross.csv')
    factorial_rows = []
    effect_rows = []
    decomp_rows = []
    replay_checks = []
    decomp_checks = []
    for seed in SEEDS:
        tl_x, cf_x, y = load_matched_sample(seed)
        fac = build_factorial(tl_x, cf_x, y)
        groups = np.rint(tl_x[:, PROTO_I]).astype(int)
        for model in MODELS:
            ckpt = load_checkpoint(PHASE_B_DIR, seed, model, expected_hashes=hashes)
            if ckpt['meta'].get('features') != list(FEATURES):
                raise SystemExit(f'{model} seed {seed}: feature order mismatch')
            if ckpt['meta'].get('hyperparameters', {}).get('sklearn') != sklearn.__version__:
                raise SystemExit(f'{model} seed {seed}: sklearn version mismatch')
            scores = {c: predict_proba(ckpt, fac[c], model, seed) for c in CONDITIONS}
            metrics = {c: compact(met(y, scores[c])) for c in CONDITIONS}
            for c in CONDITIONS:
                rec = {'seed': seed, 'model': model, 'condition': c}
                rec.update({k: metrics[c].get(k) for k in FULL_METRICS if k in metrics[c]})
                rec['n'] = int(len(y))
                rec['attack_prevalence'] = float(y.mean())
                factorial_rows.append(rec)
            for cond, view in REPLAY_VIEW.items():
                ref = b1_reference_row(b1_csv, seed, model, view)
                replay_checks.extend(compare_metrics(
                    metrics[cond], ref, f'seed{seed}.{model}.{cond}_vs_{view}'))
            fx = effects_from_conditions(metrics)
            for metric, blk in fx.items():
                effect_rows.append({'seed': seed, 'model': model, 'metric': metric, **blk})
            da = decompose_roc_auc(y, scores['A'], groups)
            dd = decompose_roc_auc(y, scores['D'], groups)
            sk_a = float(roc_auc_score(y, scores['A']))
            sk_d = float(roc_auc_score(y, scores['D']))
            recon_ok_a = abs(da['reconstructed_auc'] - sk_a) <= 1e-12
            recon_ok_d = abs(dd['reconstructed_auc'] - sk_d) <= 1e-12
            decomp_checks.append({
                'seed': seed, 'model': model,
                'sklearn_A': sk_a, 'reconstructed_A': da['reconstructed_auc'],
                'sklearn_D': sk_d, 'reconstructed_D': dd['reconstructed_auc'],
                'status': 'PASS' if recon_ok_a and recon_ok_d else 'FAIL',
            })
            if not (recon_ok_a and recon_ok_d):
                raise SystemExit(
                    f'AUC reconstruction failed for {model} seed {seed}: '
                    f'A {da["reconstructed_auc"]} vs {sk_a}; '
                    f'D {dd["reconstructed_auc"]} vs {sk_d}')
            cells_a = { (c['pos_group'], c['neg_group']): c for c in da['cells'] }
            for cell_d in dd['cells']:
                key = (cell_d['pos_group'], cell_d['neg_group'])
                cell_a = cells_a[key]
                involving = CODE17 in key
                decomp_rows.append({
                    'seed': seed, 'model': model,
                    'pos_group': key[0], 'neg_group': key[1],
                    'n_positive': cell_a['n_positive'],
                    'n_negative': cell_a['n_negative'],
                    'weight': cell_a['weight'],
                    'estimable': cell_a['estimable'] and cell_d['estimable'],
                    'pairwise_auc_A': cell_a['pairwise_auc'],
                    'pairwise_auc_D': cell_d['pairwise_auc'],
                    'contribution_A': cell_a['contribution'],
                    'contribution_D': cell_d['contribution'],
                    'delta_contribution_A_minus_D': (
                        cell_a['contribution'] - cell_d['contribution']),
                    'involves_original_code17': involving,
                    'within_group': key[0] == key[1],
                })
    failed_replay = [c for c in replay_checks if c['status'] != 'PASS']
    if failed_replay:
        write_text(MECHANISM_DIR / 'replay_failures.json',
                   json.dumps(failed_replay[:50], indent=2), allowed_root=MECHANISM_DIR)
        raise SystemExit(f'checkpoint replay parity failed ({len(failed_replay)} checks)')
    fact_df = pd.DataFrame(factorial_rows)
    eff_df = pd.DataFrame(effect_rows)
    decomp_df = pd.DataFrame(decomp_rows)
    csv_fact = assert_mechanism_output(MECHANISM_DIR / 'factorial_per_seed.csv')
    fact_df.to_csv(csv_fact, index=False)
    csv_eff = assert_mechanism_output(MECHANISM_DIR / 'factorial_effects_per_seed.csv')
    eff_df.to_csv(csv_eff, index=False)
    csv_de = assert_mechanism_output(MECHANISM_DIR / 'auc_pair_decomposition.csv')
    decomp_df.to_csv(csv_de, index=False)

    summary = {'models': {}, 'paper_consistency': PAPER}
    for model in MODELS:
        m_eff = eff_df[eff_df.model == model]
        m_fact = fact_df[fact_df.model == model]
        m_de = decomp_df[decomp_df.model == model]
        blk = {'conditions_roc_auc': {}, 'effects_roc_auc': {}, 'code17': {}}
        for cond in CONDITIONS:
            vals = m_fact.loc[m_fact.condition == cond, 'roc_auc'].astype(float).tolist()
            blk['conditions_roc_auc'][cond] = t_interval(vals)
        for col in [
            'protocol_only_original_ports_B_minus_A',
            'port_only_original_protocol_C_minus_A',
            'protocol_under_sentinel_D_minus_C',
            'port_under_reconstructed_D_minus_B',
            'combined_D_minus_A',
            'interaction_D_minus_C_minus_B_plus_A',
        ]:
            vals = m_eff.loc[m_eff.metric == 'roc_auc', col].astype(float).tolist()
            blk['effects_roc_auc'][col] = {
                **t_interval(vals),
                'all_negative': bool(all(v < 0 for v in vals)),
                'all_positive': bool(all(v > 0 for v in vals)),
                'per_seed': {str(s): float(v) for s, v in zip(SEEDS, vals)},
            }
        shares = []
        parts = []
        totals = []
        for seed in SEEDS:
            sub = m_de[m_de.seed == seed]
            part = float(sub.loc[sub.involves_original_code17, 'delta_contribution_A_minus_D'].sum())
            total = float(sub['delta_contribution_A_minus_D'].sum())
            parts.append(part)
            totals.append(total)
            shares.append(share_with_cancellation(part, total))
        blk['code17'] = {
            'delta_contribution_mean': t_interval(parts),
            'total_delta_mean': t_interval(totals),
            'share_per_seed': {str(s): sh for s, sh in zip(SEEDS, shares)},
            'share_mean_percent': float(np.mean([s['share_percent'] for s in shares if s['share'] is not None])),
            'share_note': shares[0]['note'] if shares else '',
        }
        summary['models'][model] = blk
    summary['replay'] = {
        'n_checks': len(replay_checks), 'n_fail': 0, 'status': 'PASS',
        'atol': PARITY_ATOL, 'rtol': PARITY_RTOL, 'hard_tol': PARITY_HARD_AUC,
        'note': (
            'A scored against stored Phase B.1 TL metrics; D against CF. '
            'FedAvg uses clipped-logit sigmoid; centralized uses sklearn predict_proba.'
        ),
    }
    summary['masking_versus_replacement'] = {
        'masking': (
            'Paper masking zeros one already-standardized column (the source-training '
            'mean) on a finished TL or CF matrix. A constant CF source-port column '
            'cannot reorder scores when masked.'
        ),
        'replacement': (
            'Factorial C/D replace TL\'s originally variable source-port column with '
            'the CF sentinel before scaling. That replacement can change ranking even '
            'though masking the resulting constant cannot.'
        ),
        'not_the_same': True,
    }
    write_text(MECHANISM_DIR / 'factorial_summary.json', json.dumps(summary, indent=2),
               allowed_root=MECHANISM_DIR)
    write_text(MECHANISM_DIR / 'auc_decomposition_validation.json', json.dumps({
        'status': 'PASS',
        'tolerance': 1e-12,
        'n_checks': len(decomp_checks),
        'checks': decomp_checks,
        'definition': (
            'weight(g,h)=n_pos(g)*n_neg(h)/(N_pos*N_neg); '
            'global AUC = sum weight*pairwise_AUC with 0.5 tie credit. '
            'Groups are original TL protocol codes, never CF reconstructions. '
            'Empty cells are not estimable and contribute zero.'
        ),
    }, indent=2), allowed_root=MECHANISM_DIR)
    return {
        'fact_df': fact_df, 'eff_df': eff_df, 'decomp_df': decomp_df,
        'summary': summary, 'inputs': inputs,
        'b1_before': b1_before, 'verified_before': verified_before,
        'hashes': hashes,
    }


def _fmt(x: float, d: int = 4) -> str:
    return f'{x:.{d}f}'


def write_report(bundle: dict) -> None:
    s = bundle['summary']
    ni = s['models']['fedavg_noniid']
    iid = s['models']['fedavg_iid']
    cen = s['models']['centralized']
    lines = [
        '# Phase B.2 mechanism report',
        '',
        'Matched 2×2 factorial on frozen Phase B.1 checkpoints. No retraining.',
        'The camera-ready manuscript was not modified.',
        '',
        f'- Git commit: `{git_sha()}`',
        f'- Checkpoint replay: **{s["replay"]["status"]}** ({s["replay"]["n_checks"]} checks)',
        '- Groups for the pair decomposition: **original TL protocol codes**.',
        '',
        '## Factorial conditions',
        '',
        '- A: original TL protocol + original TL source port (must reproduce TL)',
        '- B: CF protocol column copied from matched CF rows + original TL port',
        '- C: original TL protocol + CF sentinel source port copied from matched CF rows',
        '- D: CF protocol + CF sentinel (must reproduce CF)',
        '',
        'Protocol values in B and D are read from the matched CF sample, not',
        're-inferred from TCP flags.',
        '',
        '## Mean ROC-AUC by condition',
        '',
        '| model | A (TL) | B (CF proto, TL port) | C (TL proto, CF port) | D (CF) | D−A |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, blk in [('FedAvg IID', iid), ('FedAvg non-IID', ni), ('Centralized', cen)]:
        a = blk['conditions_roc_auc']['A']['mean']
        b = blk['conditions_roc_auc']['B']['mean']
        c = blk['conditions_roc_auc']['C']['mean']
        d = blk['conditions_roc_auc']['D']['mean']
        lines.append(
            f'| {name} | {_fmt(a)} | {_fmt(b)} | {_fmt(c)} | {_fmt(d)} | {_fmt(d - a)} |')
    def effect_table(title, key):
        block = [
            '',
            f'## {title}',
            '',
            '| model | mean | SD | all-negative | all-positive |',
            '|---|---:|---:|---|---|',
        ]
        for name, blk in [('FedAvg IID', iid), ('FedAvg non-IID', ni), ('Centralized', cen)]:
            e = blk['effects_roc_auc'][key]
            block.append(
                f"| {name} | {_fmt(e['mean'])} | {_fmt(e['std'])} | "
                f"{e['all_negative']} | {e['all_positive']} |")
        return block
    lines += effect_table('Protocol-only effect (B−A), original ports',
                          'protocol_only_original_ports_B_minus_A')
    lines += effect_table('Port-only effect (C−A), original protocol',
                          'port_only_original_protocol_C_minus_A')
    lines += effect_table('Protocol effect under sentinel ports (D−C)',
                          'protocol_under_sentinel_D_minus_C')
    lines += effect_table('Port effect under reconstructed protocol (D−B)',
                          'port_under_reconstructed_D_minus_B')
    lines += effect_table('Combined effect (D−A)', 'combined_D_minus_A')
    lines += effect_table('Factorial interaction (D−C−B+A)',
                          'interaction_D_minus_C_minus_B_plus_A')
    lines += [
        '',
        'Interaction is an algebraic contrast, not a percentage of the gap.',
        'Metric effects are not assumed additive.',
        '',
        '## Masking versus replacement',
        '',
        s['masking_versus_replacement']['masking'],
        '',
        s['masking_versus_replacement']['replacement'],
        '',
        'These statements are not interchangeable. Paper source-port masking on CF',
        'is approximately AUC-neutral because the column is already constant.',
        'Factorial C replaces a *variable* TL port column with that constant.',
        '',
        '## Exact AUROC pair decomposition',
        '',
        'Immutable groups = original TL protocol codes. Global ROC-AUC was',
        'reconstructed from weighted pairwise AUCs and matched sklearn within 1e-12.',
        '',
        '### Code-17 accounting share of the A−D AUC gap (non-IID)',
        '',
        f"Mean signed code-17 cell contribution to A−D: "
        f"`{_fmt(ni['code17']['delta_contribution_mean']['mean'])}`.",
        f"Mean reconstructed A−D gap: `{_fmt(ni['code17']['total_delta_mean']['mean'])}`.",
        f"Mean accounting share: `{_fmt(ni['code17']['share_mean_percent'], 1)}%`.",
        '',
        ni['code17']['share_note'],
        '',
        'This decomposition describes where ranking comparisons moved. The 2×2',
        'factorial is the controlled feature-intervention evidence. They answer',
        'different questions and are not substitutes.',
        '',
        '## Consistency with the accepted paper',
        '',
        f"- Protocol masking ΔA on TL `{PAPER['protocol_mask_tl']}` and on CF `{PAPER['protocol_mask_cf']}` "
        'zero a standardized column; factorial B−A *replaces* raw codes then re-standardizes.',
        f"- Source-port masking on TL `{PAPER['source_port_mask_tl']}` vs CF masking ~0.",
        f"- Seven-field intersection AUC `{PAPER['intersection_auc']}` is a retraining ablation, not a factorial cell.",
        f"- Non-IID TL−CF gap `{PAPER['noniid_delta_auc']}` must match combined D−A on the non-IID model (up to sign: paper reports TL−CF = −(D−A)).",
        '',
        'If protocol-only |B−A| is smaller than |C−A| on ROC-AUC, that would contradict',
        'the manuscript claim that protocol localizes the ranking reversal. The numeric',
        'verdict is in the tables above; do not force factorial cells to equal masking deltas.',
        '',
        '## Recommendation',
        '',
        'See the closing scientific report in chat. This directory is a supplement.',
        'No manuscript edit in this phase.',
        '',
    ]
    write_text(MECHANISM_DIR / 'mechanism_report.md', '\n'.join(lines) + '\n',
               allowed_root=MECHANISM_DIR)


def main() -> None:
    if 'run_verified_federated_baseline.main' in Path(__file__).read_text(encoding='utf8')[:200]:
        pass
    bundle = run()
    write_report(bundle)
    b1_after = snapshot_dir(PHASE_B_DIR, exclude_prefix=('mechanism',))
    verified_after = snapshot_verified()
    if b1_after != bundle['b1_before']:
        raise SystemExit('Phase B.1 outputs changed during B.2')
    if verified_after != bundle['verified_before']:
        raise SystemExit('results/verified/ changed during B.2')
    manifest = {
        'experiment': 'phase_b.2_factorial_and_auc_decomposition',
        'git_commit': git_sha(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': SEEDS,
        'models': list(MODELS),
        'conditions': {
            'A': 'TL protocol + TL source port',
            'B': 'CF protocol from matched rows + TL source port',
            'C': 'TL protocol + CF sentinel source port from matched rows',
            'D': 'CF protocol + CF sentinel source port',
        },
        'feature_schema': FEATURES,
        'input_hashes': bundle['hashes'],
        'phase_b1_input_hashes': bundle['b1_before'],
        'package_versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'sklearn': sklearn.__version__,
        },
        'output_root': str(MECHANISM_DIR),
        'never_writes': [str(VERIFIED_DIR), str(PHASE_B_DIR / 'per_seed_noniid_cross.csv')],
        'target_training': False,
        'retraining': False,
        'probability_transform': {
            'fedavg_iid': 'clip(logit,-35,35) then sigmoid',
            'fedavg_noniid': 'clip(logit,-35,35) then sigmoid',
            'centralized': 'sklearn SGDClassifier.predict_proba',
        },
        'groups': 'original TL ip_proto integer codes on the seeded 30k sample',
    }
    write_text(MECHANISM_DIR / 'experiment_manifest.json', json.dumps(manifest, indent=2),
               allowed_root=MECHANISM_DIR)
    ni = bundle['summary']['models']['fedavg_noniid']['effects_roc_auc']
    print(json.dumps({
        'replay': bundle['summary']['replay']['status'],
        'noniid_A': bundle['summary']['models']['fedavg_noniid']['conditions_roc_auc']['A']['mean'],
        'noniid_B': bundle['summary']['models']['fedavg_noniid']['conditions_roc_auc']['B']['mean'],
        'noniid_C': bundle['summary']['models']['fedavg_noniid']['conditions_roc_auc']['C']['mean'],
        'noniid_D': bundle['summary']['models']['fedavg_noniid']['conditions_roc_auc']['D']['mean'],
        'noniid_protocol_only_B_minus_A': ni['protocol_only_original_ports_B_minus_A']['mean'],
        'noniid_port_only_C_minus_A': ni['port_only_original_protocol_C_minus_A']['mean'],
        'noniid_interaction': ni['interaction_D_minus_C_minus_B_plus_A']['mean'],
        'noniid_combined_D_minus_A': ni['combined_D_minus_A']['mean'],
        'code17_share_percent_noniid': bundle['summary']['models']['fedavg_noniid']['code17']['share_mean_percent'],
        'b1_unchanged': True,
        'verified_unchanged': True,
    }, indent=2, default=float))


if __name__ == '__main__':
    main()
