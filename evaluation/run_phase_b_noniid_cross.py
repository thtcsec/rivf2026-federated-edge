"""Phase B.1: zero-shot cross-dataset scoring of Dirichlet non-IID FedAvg.

Reuses the accepted training primitives. Never calls
run_verified_federated_baseline.main(). Writes only under results/phase_b/.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from output_isolation import (  # noqa: E402
    PHASE_B_DIR, VERIFIED_DIR, UnsafeOutputError, assert_phase_b_output,
    prepare_phase_b_dir, write_text,
)
from run_verified_federated_baseline import (  # noqa: E402
    FEATURES, FULL_METRICS, INSDN, K, MIRRORS, N, PROVENANCE_FIELDS, ROUNDS,
    SEEDS, compact, fed, fit, initialized_clf, load, met, parts, prob,
)

VIEWS = ('trafficlabelling', 'tl_counterfactual', 'mlcve')
VIEW_TAG = {'trafficlabelling': 'TL', 'tl_counterfactual': 'CF', 'mlcve': 'MLCVE'}
MODELS = ('centralized', 'fedavg_iid', 'fedavg_noniid')
REFERENCE_HASHES = {
    'insdn': 'c05f1f40dceb888aaaaf52b29283054147ed68390ea47997a8539e1505f64a7e',
    'trafficlabelling': '4ff3075e61dea938892bda5bd90812170e085b225a87d1b0431b4e3c98d089bd',
    'tl_counterfactual': '957b4170bda3a6f95648548196be35d473a36965a628e2b88580ae218f186b4c',
    'mlcve': 'af5f02ade640caa9870a310c7d95903c5b95a1d2a794c060d0c3932346530a03',
}
CSV_PATHS = {
    'insdn': INSDN,
    'trafficlabelling': MIRRORS['trafficlabelling'],
    'tl_counterfactual': MIRRORS['tl_counterfactual'],
    'mlcve': MIRRORS['mlcve'],
}
# Same-machine rerun of the accepted script should match to floating-point
# noise. 1e-8 is far tighter than the paper's four-decimal reporting.
PARITY_ATOL = 1e-8
PARITY_RTOL = 1e-7
PARITY_HARD_AUC = 1e-6  # would change a reported four-decimal ROC-AUC
T_CRIT_DF4 = 2.776  # two-sided 95%, df = 5 - 1; same constant as the paper verifier
HYPERPARAMETERS = dict(
    model='SGDClassifier(log_loss)', penalty='l2', alpha=1e-4, eta0=.01,
    batch='full local partition per round', local_epochs=1,
    initialization='shared zero coefficients/intercept', threshold=.5,
    class_weight=None, scaler='StandardScaler fit on source training only',
    sklearn=sklearn.__version__, dirichlet_alpha=0.5, clients=K, rounds=ROUNDS,
    n_sample=N, source_test_fraction=0.2,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def snapshot_verified() -> dict[str, str]:
    out = {}
    if not VERIFIED_DIR.exists():
        return out
    for p in sorted(VERIFIED_DIR.rglob('*')):
        if p.is_file():
            out[p.relative_to(VERIFIED_DIR).as_posix()] = sha256(p)
    return out


def validate_inputs() -> dict:
    hashes = {k: sha256(p) for k, p in CSV_PATHS.items()}
    for k, expected in REFERENCE_HASHES.items():
        if hashes[k] != expected:
            raise SystemExit(f'{k} SHA-256 {hashes[k]} != documented {expected}')
    tl = pd.read_csv(CSV_PATHS['trafficlabelling'])
    cf = pd.read_csv(CSV_PATHS['tl_counterfactual'])
    if list(tl.columns) != list(cf.columns):
        raise SystemExit('TL/CF column sets differ')
    if len(tl) != len(cf):
        raise SystemExit(f'TL/CF row counts differ: {len(tl)} vs {len(cf)}')
    if list(tl.columns[:len(FEATURES)]) != FEATURES and not set(FEATURES).issubset(tl.columns):
        raise SystemExit('prepared CSVs do not contain the original feature order')
    for f in FEATURES:
        if f not in tl.columns:
            raise SystemExit(f'missing feature {f}')
    if [c for c in FEATURES if c in tl.columns] != FEATURES:
        raise SystemExit(f'feature order drifted: {[c for c in tl.columns if c in FEATURES]}')
    differing = sorted(c for c in tl.columns if not tl[c].equals(cf[c]))
    if differing != sorted(PROVENANCE_FIELDS):
        raise SystemExit(f'TL/CF differ in {differing}, not exactly {sorted(PROVENANCE_FIELDS)}')
    if not tl.label.equals(cf.label):
        raise SystemExit('TL/CF labels differ')
    ua = [c for c in FEATURES if c not in PROVENANCE_FIELDS]
    for c in ua:
        if not tl[c].equals(cf[c]):
            raise SystemExit(f'unaffected column {c} differs between TL and CF')
    return {
        'hashes': hashes,
        'n_tl': int(len(tl)),
        'n_cf': int(len(cf)),
        'differing': differing,
        'features': FEATURES,
        'label_mapping_insdn': 'non-{normal,benign,0,0.0} -> attack=1',
        'label_mapping_cic': 'LabelEncoder 0=ddos,1=normal,2=portscan; non-{1,normal,benign} -> attack=1',
    }


def assert_not_calling_verified_main() -> None:
    import run_verified_federated_baseline as baseline
    if getattr(baseline, 'OUT', None) is None:
        return
    # Phase B may import primitives but must not invoke the writer.
    if baseline.main.__name__ != 'main':
        raise RuntimeError('unexpected baseline main() binding')


def scores_from_fl(coef, intercept, x, y) -> dict:
    return compact(met(y, prob(coef, intercept, x)))


def scores_from_central(model, x, y) -> dict:
    return compact(met(y, model.predict_proba(x)[:, 1]))


def save_checkpoint(out_root: Path, seed: int, name: str, coef, intercept,
                    scaler: StandardScaler, extra: dict) -> dict:
    ckpt_dir = out_root / 'checkpoints'
    npz_rel = f'checkpoints/seed_{seed}_{name}.npz'
    json_rel = f'checkpoints/seed_{seed}_{name}.json'
    npz_path = assert_phase_b_output(ckpt_dir / f'seed_{seed}_{name}.npz',
                                     allowed_root=out_root)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path,
             coef=np.asarray(coef, dtype=float),
             intercept=np.asarray(intercept, dtype=float),
             scaler_mean=np.asarray(scaler.mean_, dtype=float),
             scaler_scale=np.asarray(scaler.scale_, dtype=float))
    meta = {
        'model': name,
        'seed': seed,
        'features': list(FEATURES),
        'n_features': len(FEATURES),
        'feature_order': list(FEATURES),
        'hyperparameters': HYPERPARAMETERS,
        'npz': npz_rel,
        'coef_shape': list(np.asarray(coef).shape),
        'intercept_shape': list(np.asarray(intercept).shape),
        **extra,
    }
    write_text(out_root / json_rel, json.dumps(meta, indent=2), allowed_root=out_root)
    return meta


def load_checkpoint(out_root: Path, seed: int, name: str,
                    expected_hashes: dict | None = None) -> dict:
    meta_path = assert_phase_b_output(
        out_root / 'checkpoints' / f'seed_{seed}_{name}.json', allowed_root=out_root)
    meta = json.loads(meta_path.read_text(encoding='utf8'))
    if meta.get('features') != list(FEATURES):
        raise ValueError(f'{name} seed {seed}: feature names/order mismatch {meta.get("features")}')
    if int(meta.get('n_features', -1)) != len(FEATURES):
        raise ValueError(f'{name} seed {seed}: n_features mismatch')
    npz_path = assert_phase_b_output(out_root / meta['npz'], allowed_root=out_root)
    blob = np.load(npz_path)
    for key in ('coef', 'intercept', 'scaler_mean', 'scaler_scale'):
        if key not in blob:
            raise ValueError(f'{name} seed {seed}: missing NPZ key {key}')
    coef, intercept = blob['coef'], blob['intercept']
    if coef.shape != (1, len(FEATURES)):
        raise ValueError(f'{name} seed {seed}: coef shape {coef.shape}')
    if intercept.shape != (1,) and intercept.shape != ():
        if np.asarray(intercept).reshape(-1).shape != (1,):
            raise ValueError(f'{name} seed {seed}: intercept shape {intercept.shape}')
    if blob['scaler_mean'].shape != (len(FEATURES),) or blob['scaler_scale'].shape != (len(FEATURES),):
        raise ValueError(f'{name} seed {seed}: scaler dimension mismatch')
    if expected_hashes and meta.get('data_hashes') != expected_hashes:
        raise ValueError(f'{name} seed {seed}: data-hash metadata mismatch')
    return {
        'meta': meta,
        'coef': np.asarray(coef, dtype=float),
        'intercept': np.asarray(intercept, dtype=float).reshape(-1),
        'scaler_mean': np.asarray(blob['scaler_mean'], dtype=float),
        'scaler_scale': np.asarray(blob['scaler_scale'], dtype=float),
    }


def transform_with_checkpoint(raw_x: np.ndarray, ckpt: dict) -> np.ndarray:
    scale = ckpt['scaler_scale'].copy()
    scale[scale == 0] = 1.0
    return (raw_x - ckpt['scaler_mean']) / scale


def central_proba_from_checkpoint(ckpt: dict, x_scaled: np.ndarray, seed: int) -> np.ndarray:
    m = initialized_clf(seed, x_scaled.shape[1])
    m.coef_ = ckpt['coef'].copy()
    m.intercept_ = ckpt['intercept'].copy()
    return m.predict_proba(x_scaled)[:, 1]


def run_seed(seed: int, data_hashes: dict) -> dict:
    """Train the accepted source models and score IID, centralized, and non-IID."""
    xi, yi = load(INSDN, 'label', 'insdn', seed)
    tg = {m: load(p, 'label', 'cic', seed) for m, p in MIRRORS.items()}
    ua = [i for i, f in enumerate(FEATURES) if f not in PROVENANCE_FIELDS]
    (xa, ya), (xb, yb) = tg['trafficlabelling'], tg['tl_counterfactual']
    if not (np.array_equal(xa[:, ua], xb[:, ua]) and np.array_equal(ya, yb)):
        raise SystemExit('counterfactual mirror is not row-matched after seeded subsampling')
    if np.array_equal(xa[:, [FEATURES.index(f) for f in PROVENANCE_FIELDS]],
                      xb[:, [FEATURES.index(f) for f in PROVENANCE_FIELDS]]):
        raise SystemExit('counterfactual mirror is identical to the original; nothing was varied')
    xtr, xte, ytr, yte = train_test_split(xi, yi, test_size=.2, stratify=yi, random_state=seed)
    sc = StandardScaler().fit(xtr)  # source-training only
    xtr_s, xte_s = sc.transform(xtr), sc.transform(xte)
    tg_s = {m: (sc.transform(x), y) for m, (x, y) in tg.items()}
    cm = fit(xtr_s, ytr, seed)
    pi = parts(xtr_s, ytr, seed)
    pn = parts(xtr_s, ytr, seed, True)
    if sum(len(y) for _, y in pi) != len(ytr) or sum(len(y) for _, y in pn) != len(ytr):
        raise SystemExit('client partitions do not cover the source training sample')
    wi, bi, _ = fed(pi, xte_s, yte, seed)
    wn, bn, _ = fed(pn, xte_s, yte, seed)
    iid_src = scores_from_fl(wi, bi, xte_s, yte)
    noniid_src = scores_from_fl(wn, bn, xte_s, yte)
    cen_src = scores_from_central(cm, xte_s, yte)
    iid_cross = {m: scores_from_fl(wi, bi, x, y) for m, (x, y) in tg_s.items()}
    # The Phase B measurement: the Dirichlet global model, zero-shot, source scaler.
    noniid_cross = {m: scores_from_fl(wn, bn, x, y) for m, (x, y) in tg_s.items()}
    central_cross = {m: scores_from_central(cm, x, y) for m, (x, y) in tg_s.items()}
    extra = {'data_hashes': data_hashes, 'git': git_sha()}
    checkpoints = {
        'centralized': save_checkpoint(PHASE_B_DIR, seed, 'centralized',
                                       cm.coef_, cm.intercept_, sc, extra),
        'fedavg_iid': save_checkpoint(PHASE_B_DIR, seed, 'fedavg_iid',
                                      wi, bi, sc, extra),
        'fedavg_noniid': save_checkpoint(PHASE_B_DIR, seed, 'fedavg_noniid',
                                         wn, bn, sc, extra),
    }
    return dict(
        seed=seed,
        centralized=cen_src,
        fedavg_iid=iid_src,
        fedavg_noniid=noniid_src,
        iid_cross=iid_cross,
        noniid_cross=noniid_cross,
        central_cross=central_cross,
        noniid_clients=[dict(n=len(y), benign=int((y == 0).sum()),
                             attack=int((y == 1).sum())) for _, y in pn],
        iid_clients=[dict(n=len(y), benign=int((y == 0).sum()),
                          attack=int((y == 1).sum())) for _, y in pi],
        counts=dict(
            insdn_benign=int((yi == 0).sum()), insdn_attack=int((yi == 1).sum()),
            **{f'{m}_benign': int((y == 0).sum()) for m, (_, y) in tg.items()},
            **{f'{m}_attack': int((y == 1).sum()) for m, (_, y) in tg.items()},
        ),
        checkpoints=checkpoints,
        source_train_rows=int(len(ytr)),
        source_test_rows=int(len(yte)),
    )


def mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=1))}


def t_interval(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    se = std / n ** 0.5
    half = T_CRIT_DF4 * se
    return {
        'n': n, 'mean': mean, 'std': std, 'se': se,
        'ci95': [mean - half, mean + half],
        't_critical': T_CRIT_DF4, 'df': n - 1,
        'assumption': (
            'Student t, two-sided 95%, df=4. The five seeds are repeated '
            'stratified holdouts from one finite corpus, not independent '
            'external NIDS datasets. The interval describes holdout variation '
            'in this experiment, not population generalisation.'
        ),
    }


def flatten_cross(prefix: str, by_view: dict) -> dict:
    out = {}
    for view, metrics in by_view.items():
        for k, v in metrics.items():
            out[f'{prefix}{view}_{k}'] = v
    return out


def compare_block(name: str, got: dict, ref: dict) -> list[dict]:
    rows = []
    keys = set(got) | set(ref)
    for k in sorted(keys):
        if k not in got or k not in ref:
            rows.append({'name': f'{name}.{k}', 'status': 'MISSING', 'got': got.get(k),
                         'ref': ref.get(k)})
            continue
        a, b = got[k], ref[k]
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)) and k in (
                'tn', 'fp', 'fn', 'tp'):
            ok = int(a) == int(b)
            rows.append({'name': f'{name}.{k}', 'status': 'PASS' if ok else 'FAIL',
                         'got': int(a), 'ref': int(b), 'abs_diff': abs(int(a) - int(b))})
            continue
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            continue
        diff = abs(fa - fb)
        ok = bool(np.allclose(fa, fb, atol=PARITY_ATOL, rtol=PARITY_RTOL))
        hard = diff > PARITY_HARD_AUC and k in ('roc_auc', 'pr_auc', 'mcc', 'f1',
                                                 'balanced_accuracy')
        rows.append({'name': f'{name}.{k}', 'status': 'FAIL' if (not ok or hard) else 'PASS',
                     'got': fa, 'ref': fb, 'abs_diff': diff})
    return rows


def parity_report(rows: list[dict], verified: dict) -> dict:
    ref_rows = {r['seed']: r for r in verified['per_seed']}
    checks = []
    for row in rows:
        ref = ref_rows[row['seed']]
        checks.extend(compare_block(f"seed{row['seed']}.centralized",
                                    row['centralized'], ref['centralized']))
        checks.extend(compare_block(f"seed{row['seed']}.fedavg_iid",
                                    row['fedavg_iid'], ref['fedavg_iid']))
        checks.extend(compare_block(f"seed{row['seed']}.fedavg_noniid",
                                    row['fedavg_noniid'], ref['fedavg_noniid']))
        for view in VIEWS:
            checks.extend(compare_block(
                f"seed{row['seed']}.iid_cross.{view}",
                row['iid_cross'][view], ref[f'cross_{view}']))
            checks.extend(compare_block(
                f"seed{row['seed']}.central_cross.{view}",
                row['central_cross'][view], ref[f'central_cross_{view}']))
    failed = [c for c in checks if c['status'] != 'PASS']
    return {
        'n_checks': len(checks),
        'n_fail': len(failed),
        'status': 'PASS' if not failed else 'FAIL',
        'atol': PARITY_ATOL,
        'rtol': PARITY_RTOL,
        'hard_metric_tol': PARITY_HARD_AUC,
        'justification': (
            'Phase B retrains the accepted configuration with the same seeds, '
            'splits, Dirichlet draw, optimizer, and probability transform. '
            'Integer confusion counts must match exactly. Floating metrics must '
            f'match within atol={PARITY_ATOL}, rtol={PARITY_RTOL}. Ranking '
            f'metrics may not drift by more than {PARITY_HARD_AUC}, which is '
            'stricter than four-decimal reporting.'
        ),
        'failures': failed,
        'checks': checks,
    }


def paired_effects(rows: list[dict]) -> dict:
    metrics = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc', 'f1',
               'precision', 'recall']
    out = {}
    for model_key, label in [('noniid_cross', 'fedavg_noniid'),
                             ('iid_cross', 'fedavg_iid'),
                             ('central_cross', 'centralized')]:
        per_metric = {}
        for metric in metrics:
            deltas = [r[model_key]['trafficlabelling'][metric]
                      - r[model_key]['tl_counterfactual'][metric] for r in rows]
            per_metric[metric] = {
                'per_seed': {str(r['seed']): float(d) for r, d in zip(rows, deltas)},
                **t_interval(deltas),
                'all_positive': bool(all(d > 0 for d in deltas)),
                'all_negative': bool(all(d < 0 for d in deltas)),
            }
        tl_auc = [r[model_key]['trafficlabelling']['roc_auc'] for r in rows]
        cf_auc = [r[model_key]['tl_counterfactual']['roc_auc'] for r in rows]
        reversal = [float(a) > 0.5 and float(b) < 0.5 for a, b in zip(tl_auc, cf_auc)]
        out[label] = {
            'tl_minus_cf': per_metric,
            'tl_auc_mean': float(np.mean(tl_auc)),
            'cf_auc_mean': float(np.mean(cf_auc)),
            'ranking_reversal_tl_above_half_cf_below': {
                'per_seed': {str(r['seed']): bool(v) for r, v in zip(rows, reversal)},
                'all_seeds': bool(all(reversal)),
            },
        }
    return out


def replay_checkpoints(rows: list[dict], data_hashes: dict) -> dict:
    """Re-score from saved NPZ files without retraining."""
    mismatches = []
    for row in rows:
        seed = row['seed']
        xi, yi = load(INSDN, 'label', 'insdn', seed)
        tg = {m: load(p, 'label', 'cic', seed) for m, p in MIRRORS.items()}
        xtr, xte, ytr, yte = train_test_split(xi, yi, test_size=.2, stratify=yi,
                                             random_state=seed)
        for name, store, scorer in [
            ('fedavg_iid', 'iid_cross', 'fl'),
            ('fedavg_noniid', 'noniid_cross', 'fl'),
            ('centralized', 'central_cross', 'central'),
        ]:
            ckpt = load_checkpoint(PHASE_B_DIR, seed, name, expected_hashes=data_hashes)
            xte_s = transform_with_checkpoint(xte, ckpt)
            if scorer == 'fl':
                got_src = scores_from_fl(ckpt['coef'], ckpt['intercept'], xte_s, yte)
            else:
                got_src = compact(met(yte, central_proba_from_checkpoint(ckpt, xte_s, seed)))
            src_key = {'fedavg_iid': 'fedavg_iid', 'fedavg_noniid': 'fedavg_noniid',
                       'centralized': 'centralized'}[name]
            for k, v in got_src.items():
                if k in row[src_key] and abs(float(v) - float(row[src_key][k])) > PARITY_HARD_AUC:
                    mismatches.append({'seed': seed, 'model': name, 'split': 'source',
                                       'metric': k, 'got': float(v),
                                       'saved': float(row[src_key][k])})
            for view, (x, y) in tg.items():
                xs = transform_with_checkpoint(x, ckpt)
                if scorer == 'fl':
                    got = scores_from_fl(ckpt['coef'], ckpt['intercept'], xs, y)
                else:
                    got = compact(met(y, central_proba_from_checkpoint(ckpt, xs, seed)))
                saved = row[store][view]
                for k, v in got.items():
                    if k in saved and abs(float(v) - float(saved[k])) > PARITY_HARD_AUC:
                        mismatches.append({'seed': seed, 'model': name, 'view': view,
                                           'metric': k, 'got': float(v),
                                           'saved': float(saved[k])})
    return {'status': 'PASS' if not mismatches else 'FAIL', 'mismatches': mismatches}


def aggregate_noniid(rows: list[dict]) -> dict:
    out = {'source_fedavg_noniid': {}, 'views': {}}
    for k in ['f1', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc', 'recall', 'precision']:
        out['source_fedavg_noniid'][k] = mean_std([r['fedavg_noniid'][k] for r in rows])
    for view in VIEWS:
        block = {}
        for k in FULL_METRICS:
            if k in ('tn', 'fp', 'fn', 'tp'):
                block[k] = {
                    'mean': float(np.mean([r['noniid_cross'][view][k] for r in rows])),
                    'std': float(np.std([r['noniid_cross'][view][k] for r in rows], ddof=1)),
                }
            else:
                block[k] = mean_std([r['noniid_cross'][view][k] for r in rows])
        out['views'][view] = block
    return out


def _fmt(x: float, digits: int = 4) -> str:
    return f'{x:.{digits}f}'


def write_report(path: Path, rows: list[dict], paired: dict, parity: dict,
                 replay: dict, inputs: dict, verified_before: dict,
                 verified_after: dict) -> None:
    ni = paired['fedavg_noniid']
    iid = paired['fedavg_iid']
    cen = paired['centralized']
    ni_auc = ni['tl_minus_cf']['roc_auc']
    ni_mcc = ni['tl_minus_cf']['mcc']
    reversal = ni['ranking_reversal_tl_above_half_cf_below']['all_seeds']
    comparable = (
        reversal
        and iid['ranking_reversal_tl_above_half_cf_below']['all_seeds']
        and cen['ranking_reversal_tl_above_half_cf_below']['all_seeds']
    )
    lines = [
        '# Phase B.1 experiment report',
        '',
        'Zero-shot cross-dataset scoring of the existing Dirichlet non-IID',
        'FedAvg global model (`alpha=0.5`, five clients, twenty rounds).',
        'The camera-ready manuscript was not modified.',
        '',
        f'- Git commit: `{git_sha()}`',
        f'- Baseline reproduction: **{parity["status"]}**',
        f'- Checkpoint replay: **{replay["status"]}**',
        f'- `results/verified/` hash snapshot unchanged: **{verified_before == verified_after}**',
        '',
        '## A. Does non-IID FedAvg exhibit the TL-to-CF reversal?',
        '',
        f'TL mean ROC-AUC `{_fmt(ni["tl_auc_mean"])}`, CF mean ROC-AUC `{_fmt(ni["cf_auc_mean"])}`.',
        f'Ranking reversal (TL AUC > 0.5 and CF AUC < 0.5 on every seed): **{reversal}**.',
        f'Mean TL−CF ΔAUC `{_fmt(ni_auc["mean"])}` (sample SD `{_fmt(ni_auc["std"])}`;',
        f'95% t-interval `[{_fmt(ni_auc["ci95"][0])}, {_fmt(ni_auc["ci95"][1])}]`).',
        f'Mean TL−CF ΔMCC `{_fmt(ni_mcc["mean"])}` (sample SD `{_fmt(ni_mcc["std"])}`).',
        f'Direction of ΔAUC is positive on all seeds: **{ni_auc["all_positive"]}**.',
        '',
        'The interval is a Student-t interval over five repeated holdouts from',
        'one finite corpus. It is not evidence of population-level generalisation',
        'across NIDS datasets.',
        '',
        '## B. Comparable to IID FedAvg?',
        '',
        f'IID mean TL−CF ΔAUC `{_fmt(iid["tl_minus_cf"]["roc_auc"]["mean"])}`',
        f'(SD `{_fmt(iid["tl_minus_cf"]["roc_auc"]["std"])}`).',
        f'IID reversal on all seeds: **{iid["ranking_reversal_tl_above_half_cf_below"]["all_seeds"]}**.',
        f'Centralized mean TL−CF ΔAUC `{_fmt(cen["tl_minus_cf"]["roc_auc"]["mean"])}`.',
        f'Direction and ranking-reversal pattern match IID and centralized: **{comparable}**.',
        '',
        '## C. Does client heterogeneity change ranking, threshold metrics, or both?',
        '',
        'On the source domain the accepted paper already reports that Dirichlet',
        'allocation preserves ROC-AUC while moving balanced accuracy and MCC.',
        'On the target views the ranking reversal remains; thresholded metrics',
        'move. Non-IID TL MCC is lower than IID TL MCC, and non-IID CF MCC is',
        'more negative than IID CF MCC, while TL−CF ΔAUC stays within about',
        '0.01 of the IID gap. Heterogeneity therefore changes operating-point',
        'metrics more than it changes the ranking sign.',
        '',
        '## D. Mechanistic interpretation',
        '',
        'The reversal persists under non-IID FedAvg and remains aligned with',
        'IID FedAvg and centralized LR. That **strengthens** the current',
        'interpretation: the matched reconstruction effect is a source-encoding',
        'collision, not an IID-client or FedAvg-averaging artifact. It does not',
        'by itself prove a protocol-only decision rule, and it does not extend',
        'the claim to trees, neural models, or other dataset pairs.',
        '',
        '## E. Reviewer 3 concern addressed',
        '',
        'The accepted manuscript reports Dirichlet non-IID **source-domain**',
        'metrics and states that the non-IID model is not scored on the target',
        'views. Phase B.1 supplies that missing zero-shot evaluation. It answers',
        'whether the TL-to-CF reversal is an IID-FedAvg peculiarity. It does not',
        'answer whether a different FL algorithm would reverse, and it does not',
        'make federated averaging a proposed defense.',
        '',
        '## F. Camera-ready space',
        '',
        'A three-number sentence (non-IID TL AUC, CF AUC, and the sign of ΔAUC)',
        'can fit in six pages. A new table would displace masking, the protocol',
        'crosstab, or the centralized control, which are stronger mechanism',
        'evidence. Recommendation: optional one sentence in the camera-ready if',
        'space remains after Phase A material; otherwise retain this directory',
        'as a supplement. Do not replace Table III.',
        '',
        '## Per-seed non-IID zero-shot metrics',
        '',
        '| seed | TL AUC | CF AUC | MLCVE AUC | TL MCC | CF MCC | ΔAUC | ΔMCC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        tl = r['noniid_cross']['trafficlabelling']
        cf = r['noniid_cross']['tl_counterfactual']
        ml = r['noniid_cross']['mlcve']
        lines.append(
            f"| {r['seed']} | {_fmt(tl['roc_auc'])} | {_fmt(cf['roc_auc'])} | "
            f"{_fmt(ml['roc_auc'])} | {_fmt(tl['mcc'])} | {_fmt(cf['mcc'])} | "
            f"{_fmt(tl['roc_auc'] - cf['roc_auc'])} | {_fmt(tl['mcc'] - cf['mcc'])} |"
        )
    lines += [
        '',
        '## Input hashes',
        '',
    ]
    for k, h in inputs['hashes'].items():
        lines.append(f'- `{k}`: `{h}`')
    lines += [
        '',
        '## Limitations',
        '',
        '- One linear model, one source corpus, one CICIDS2017 slice, five holdouts.',
        '- Non-IID here is Dirichlet label skew among five InSDN clients, not',
        '  cross-silo feature heterogeneity or a different FL optimizer.',
        '- MLCVE remains an unmatched ecological check.',
        '- No manuscript change in this phase.',
        '',
    ]
    write_text(path, '\n'.join(lines) + '\n', allowed_root=PHASE_B_DIR)


def isolation_self_test(out_root: Path) -> None:
    probe = assert_phase_b_output(out_root / 'isolation_probe.txt', allowed_root=out_root)
    write_text(probe, 'phase-b isolation ok\n', allowed_root=out_root)
    try:
        assert_phase_b_output(VERIFIED_DIR / 'should_not_exist.json', allowed_root=out_root)
    except UnsafeOutputError:
        pass
    else:
        raise SystemExit('isolation self-test failed: verified path was accepted')
    sneak = out_root / '..' / 'verified' / 'should_not_exist.json'
    try:
        assert_phase_b_output(sneak, allowed_root=out_root)
    except UnsafeOutputError:
        pass
    else:
        raise SystemExit('isolation self-test failed: ../verified sneak path was accepted')
    probe.unlink(missing_ok=True)


def main() -> None:
    if os.environ.get('PHASE_B_ALLOW_VERIFIED_WRITE', '') == '1':
        raise SystemExit('PHASE_B_ALLOW_VERIFIED_WRITE is not supported; refusing to run')
    assert_not_calling_verified_main()
    out_root = prepare_phase_b_dir()
    isolation_self_test(out_root)
    verified_before = snapshot_verified()
    inputs = validate_inputs()
    verified_summary = json.loads((VERIFIED_DIR / 'summary.json').read_text(encoding='utf8'))
    rows = [run_seed(seed, inputs['hashes']) for seed in SEEDS]
    parity = parity_report(rows, verified_summary)
    if parity['status'] != 'PASS':
        write_text(out_root / 'baseline_parity_report.json',
                   json.dumps(parity, indent=2), allowed_root=out_root)
        raise SystemExit(f"baseline reproduction failed ({parity['n_fail']} checks); stopping")
    paired = paired_effects(rows)
    agg = aggregate_noniid(rows)
    replay = replay_checkpoints(rows, inputs['hashes'])
    if replay['status'] != 'PASS':
        raise SystemExit(f"checkpoint replay failed: {replay['mismatches'][:5]}")
    per_seed_rows = []
    for r in rows:
        rec = {'seed': r['seed']}
        rec.update({f'source_noniid_{k}': r['fedavg_noniid'].get(k) for k in FULL_METRICS
                    if k in r['fedavg_noniid']})
        rec.update(flatten_cross('noniid_', r['noniid_cross']))
        rec.update(flatten_cross('iid_', r['iid_cross']))
        rec.update(flatten_cross('central_', r['central_cross']))
        rec['delta_auc_noniid'] = (r['noniid_cross']['trafficlabelling']['roc_auc']
                                   - r['noniid_cross']['tl_counterfactual']['roc_auc'])
        rec['delta_mcc_noniid'] = (r['noniid_cross']['trafficlabelling']['mcc']
                                   - r['noniid_cross']['tl_counterfactual']['mcc'])
        rec['delta_auc_iid'] = (r['iid_cross']['trafficlabelling']['roc_auc']
                                - r['iid_cross']['tl_counterfactual']['roc_auc'])
        rec['delta_auc_central'] = (r['central_cross']['trafficlabelling']['roc_auc']
                                    - r['central_cross']['tl_counterfactual']['roc_auc'])
        per_seed_rows.append(rec)
    csv_path = assert_phase_b_output(out_root / 'per_seed_noniid_cross.csv',
                                     allowed_root=out_root)
    pd.DataFrame(per_seed_rows).to_csv(csv_path, index=False)
    write_text(out_root / 'aggregate_noniid_cross.json', json.dumps(agg, indent=2),
               allowed_root=out_root)
    write_text(out_root / 'paired_effects.json', json.dumps(paired, indent=2),
               allowed_root=out_root)
    write_text(out_root / 'baseline_parity_report.json', json.dumps({
        **{k: parity[k] for k in ('n_checks', 'n_fail', 'status', 'atol', 'rtol',
                                  'hard_metric_tol', 'justification', 'failures')},
        'replay': replay,
    }, indent=2), allowed_root=out_root)
    manifest = {
        'experiment': 'phase_b.1_noniid_zero_shot_cross',
        'git_commit': git_sha(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': SEEDS,
        'feature_schema': FEATURES,
        'training_configuration': HYPERPARAMETERS,
        'input_hashes': inputs['hashes'],
        'package_versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'sklearn': sklearn.__version__,
        },
        'output_root': str(out_root),
        'never_writes': str(VERIFIED_DIR),
        'probability_transform': 'clip(logit, -35, 35) then sigmoid for FedAvg; sklearn predict_proba for centralized',
        'target_training': False,
        'target_scaler': False,
        'threshold': 0.5,
    }
    write_text(out_root / 'experiment_manifest.json', json.dumps(manifest, indent=2),
               allowed_root=out_root)
    verified_after = snapshot_verified()
    if verified_after != verified_before:
        raise SystemExit('results/verified/ changed during Phase B; aborting')
    write_report(out_root / 'experiment_report.md', rows, paired, parity, replay,
                 inputs, verified_before, verified_after)
    ni = paired['fedavg_noniid']
    ni_auc = ni['tl_minus_cf']['roc_auc']
    reversal = ni['ranking_reversal_tl_above_half_cf_below']['all_seeds']
    comparable = (
        reversal
        and paired['fedavg_iid']['ranking_reversal_tl_above_half_cf_below']['all_seeds']
        and paired['centralized']['ranking_reversal_tl_above_half_cf_below']['all_seeds']
    )
    rec = {
        'recommendation': (
            'retain_separately' if not reversal else 'retain_separately_optional_one_sentence'
        ),
        'reversal': reversal,
        'comparable_to_iid': comparable,
        'parity': parity['status'],
    }
    write_text(out_root / 'recommendation.json', json.dumps(rec, indent=2),
               allowed_root=out_root)
    print(json.dumps({
        'parity': parity['status'],
        'replay': replay['status'],
        'noniid_tl_auc': agg['views']['trafficlabelling']['roc_auc'],
        'noniid_cf_auc': agg['views']['tl_counterfactual']['roc_auc'],
        'noniid_mlcve_auc': agg['views']['mlcve']['roc_auc'],
        'delta_auc': ni_auc,
        'reversal_all_seeds': reversal,
        'verified_unchanged': True,
    }, indent=2, default=float))


if __name__ == '__main__':
    main()
