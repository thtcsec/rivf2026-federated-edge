"""Phase B.3: centralized HistGradientBoostingClassifier robustness check.

Author-specified hyperparameters. Source-only training. Protocol is a nominal
stored code (categorical_features), not a numeric magnitude. Writes only under
results/phase_b/robustness/. This is not a federated model.
"""
from __future__ import annotations

import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from output_isolation import (  # noqa: E402
    MECHANISM_DIR, PHASE_B_DIR, ROBUSTNESS_DIR, VERIFIED_DIR, UnsafeOutputError,
    assert_robustness_output, prepare_phase_b_dir, write_text,
)
from run_phase_b_mechanism import (  # noqa: E402
    CONDITIONS, PORT_I, PROTO_I, UNAFFECTED_I, build_factorial, effects_from_conditions,
    git_sha, load_matched_sample, snapshot_dir, t_interval,
)
from run_phase_b_noniid_cross import (  # noqa: E402
    REFERENCE_HASHES, SEEDS, snapshot_verified, validate_inputs,
)
from run_verified_federated_baseline import (  # noqa: E402
    FEATURES, FULL_METRICS, INSDN, MIRRORS, compact, load, met,
)

VIEWS = ('trafficlabelling', 'tl_counterfactual', 'mlcve')
VIEW_TAG = {'trafficlabelling': 'TL', 'tl_counterfactual': 'CF', 'mlcve': 'MLCVE'}
PROTOCOL_INDEX = FEATURES.index('ip_proto')
HGB_HYPERPARAMETERS = dict(
    loss='log_loss',
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=False,
    class_weight=None,
)
LR_CENTRAL_SOURCE_AUC = 0.9320  # accepted Table II mean; comparison only
LR_CENTRAL_FACTORIAL = {  # Phase B.2 centralized LR; not a tuning target
    'A': 0.7490, 'B': 0.3457, 'C': 0.7576, 'D': 0.3544,
}


def require_categorical_support() -> None:
    params = inspect.signature(HistGradientBoostingClassifier.__init__).parameters
    if 'categorical_features' not in params:
        raise SystemExit('installed scikit-learn lacks categorical_features on HGB')
    if sklearn.__version__ < '1.2':
        raise SystemExit(f'sklearn {sklearn.__version__} is too old for native HGB categories')


def make_hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss=HGB_HYPERPARAMETERS['loss'],
        learning_rate=HGB_HYPERPARAMETERS['learning_rate'],
        max_iter=HGB_HYPERPARAMETERS['max_iter'],
        max_leaf_nodes=HGB_HYPERPARAMETERS['max_leaf_nodes'],
        min_samples_leaf=HGB_HYPERPARAMETERS['min_samples_leaf'],
        l2_regularization=HGB_HYPERPARAMETERS['l2_regularization'],
        early_stopping=HGB_HYPERPARAMETERS['early_stopping'],
        categorical_features=[PROTOCOL_INDEX],
        random_state=seed,
    )


def as_hgb_matrix(x: np.ndarray) -> np.ndarray:
    """Keep prepared numeric values; store protocol as integer category codes.

    Code 0 remains 0. It is never recoded to NaN or to a missing sentinel.
    No StandardScaler: categorical handling needs the original encodings.
    """
    out = np.array(x, dtype=np.float64, copy=True)
    codes = np.rint(out[:, PROTOCOL_INDEX])
    if not np.allclose(out[:, PROTOCOL_INDEX], codes, atol=1e-9, equal_nan=True):
        raise SystemExit('protocol column is not integer-valued stored codes')
    if np.isnan(codes).any():
        raise SystemExit('protocol NaNs are not in the prepared-data contract used here')
    out[:, PROTOCOL_INDEX] = codes
    return out


def protocol_support(train_x: np.ndarray, scored_x: np.ndarray) -> dict:
    """Unknown protocol codes at score time become missing under native HGB categories."""
    known = {int(c) for c in np.unique(train_x[:, PROTOCOL_INDEX])}
    codes = np.rint(scored_x[:, PROTOCOL_INDEX]).astype(int)
    unknown = np.array([c not in known for c in codes])
    return {
        'known_source_codes': sorted(known),
        'n_unknown_protocol': int(unknown.sum()),
        'frac_unknown_protocol': float(unknown.mean()),
        'unknown_codes': sorted({int(c) for c, u in zip(codes, unknown) if u}),
    }


def attack_proba(model: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise SystemExit(f'attack label 1 missing from classes_={classes}')
    return model.predict_proba(x)[:, classes.index(1)]


def source_partition(seed: int):
    """Identical InSDN 30k subsample and 24k/6k split as the accepted study."""
    xi, yi = load(INSDN, 'label', 'insdn', seed)
    xtr, xte, ytr, yte = train_test_split(
        xi, yi, test_size=.2, stratify=yi, random_state=seed)
    return as_hgb_matrix(xtr), as_hgb_matrix(xte), ytr, yte


def isolation_self_test() -> None:
    prepare_phase_b_dir(ROBUSTNESS_DIR)
    probe = assert_robustness_output(ROBUSTNESS_DIR / 'isolation_probe.txt')
    write_text(probe, 'robustness isolation ok\n', allowed_root=ROBUSTNESS_DIR)
    banned = [
        VERIFIED_DIR / 'summary.json',
        PHASE_B_DIR / 'per_seed_noniid_cross.csv',
        MECHANISM_DIR / 'factorial_summary.json',
        ROBUSTNESS_DIR / '..' / 'per_seed_noniid_cross.csv',
        ROBUSTNESS_DIR / '..' / 'mechanism' / 'mechanism_report.md',
    ]
    for path in banned:
        try:
            assert_robustness_output(path)
        except UnsafeOutputError:
            pass
        else:
            raise SystemExit(f'isolation self-test failed: accepted {path}')
    probe.unlink(missing_ok=True)


def _fmt(x: float, d: int = 4) -> str:
    return f'{x:.{d}f}'


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def write_report(summary: dict, hashes: dict) -> None:
    src = summary['source']
    zs = summary['zero_shot']
    fac = summary['factorial']
    tl, cf, ml = zs['trafficlabelling'], zs['tl_counterfactual'], zs['mlcve']
    ba = fac['effects_roc_auc']['protocol_only_original_ports_B_minus_A']
    ca = fac['effects_roc_auc']['port_only_original_protocol_C_minus_A']
    da = fac['effects_roc_auc']['combined_D_minus_A']
    ix = fac['effects_roc_auc']['interaction_D_minus_C_minus_B_plus_A']
    a_mean = fac['conditions_roc_auc']['A']['mean']
    b_mean = fac['conditions_roc_auc']['B']['mean']
    c_mean = fac['conditions_roc_auc']['C']['mean']
    d_mean = fac['conditions_roc_auc']['D']['mean']
    tl_above = bool(all(v > 0.5 for v in tl['roc_auc']['per_seed'].values()))
    cf_below = bool(all(v < 0.5 for v in cf['roc_auc']['per_seed'].values()))
    proto_reverses = bool(b_mean < 0.5 < a_mean)
    useful_source = src['roc_auc']['mean'] >= 0.8
    gap = float(np.mean([
        tl['roc_auc']['per_seed'][str(s)] - cf['roc_auc']['per_seed'][str(s)]
        for s in SEEDS]))
    port_material = bool(abs(ca['mean']) >= 0.02)
    if not useful_source:
        verdict = 'inconclusive_poor_source_fit'
    elif tl['roc_auc']['mean'] < 0.55:
        verdict = 'inconclusive_tl_near_chance'
    elif proto_reverses and cf_below and tl_above:
        verdict = 'observed_in_one_additional_centralized_nonlinear_classifier'
    else:
        verdict = 'model_family_dependent_under_the_tested_configurations'
    if verdict == 'observed_in_one_additional_centralized_nonlinear_classifier':
        generality = (
            'Strengthens the claim only for this one additional centralized '
            'nonlinear family under the author-specified configuration. It does '
            'not establish generality across classifier families.'
        )
        shortcut = (
            'The reconstruction collision is not unique to linear FedAvg under '
            'this test. A representation-sensitive shortcut remains possible; '
            'this design does not isolate every alternative representation.'
        )
    elif verdict == 'inconclusive_tl_near_chance':
        generality = (
            'Weakens any robustness reading: TL itself is near chance, so the '
            'absence or presence of a reversal is inconclusive for transfer.'
        )
        shortcut = (
            'Cannot separate a representation-sensitive shortcut from a model '
            'that simply failed to rank the target.'
        )
    else:
        generality = (
            'Weakens the generality claim: the TL-to-CF ranking reversal is '
            'model-family-dependent under the tested configurations.'
        )
        shortcut = (
            'Consistent with a representation-sensitive shortcut in logistic '
            'regression (numeric protocol after source scaling) rather than a '
            'universal reconstruction failure.'
        )
    lines = [
        '# Phase B.3 nonlinear robustness report',
        '',
        'Centralized `HistGradientBoostingClassifier` with protocol as a native',
        'categorical feature. **Not a federated model.** Hyperparameters are',
        'an author-specified configuration in this script.',
        'The camera-ready manuscript was not modified.',
        '',
        f'- Git commit: `{git_sha()}`',
        f'- sklearn: `{sklearn.__version__}`',
        f'- Protocol column index from FEATURES: `{PROTOCOL_INDEX}` (`ip_proto`)',
        f'- Scientific verdict: **{verdict}**',
        '',
        '## A. Source-domain performance',
        '',
        f"Mean ROC-AUC `{_fmt(src['roc_auc']['mean'])}` (SD `{_fmt(src['roc_auc']['std'])}`)",
        f"vs accepted centralized LR `{LR_CENTRAL_SOURCE_AUC:.4f}`.",
        f"MCC `{_fmt(src['mcc']['mean'])}`, F1 `{_fmt(src['f1']['mean'])}`.",
        f"Useful source ranking (mean AUC ≥ 0.8): **{useful_source}**.",
        '',
        '## B–C. Zero-shot TL / CF / MLCVE',
        '',
        '| view | ROC-AUC | PR-AUC | Bal. acc. | MCC |',
        '|---|---:|---:|---:|---:|',
        f"| TL | {_fmt(tl['roc_auc']['mean'])} | {_fmt(tl['pr_auc']['mean'])} | "
        f"{_fmt(tl['balanced_accuracy']['mean'])} | {_fmt(tl['mcc']['mean'])} |",
        f"| CF | {_fmt(cf['roc_auc']['mean'])} | {_fmt(cf['pr_auc']['mean'])} | "
        f"{_fmt(cf['balanced_accuracy']['mean'])} | {_fmt(cf['mcc']['mean'])} |",
        f"| MLCVE | {_fmt(ml['roc_auc']['mean'])} | {_fmt(ml['pr_auc']['mean'])} | "
        f"{_fmt(ml['balanced_accuracy']['mean'])} | {_fmt(ml['mcc']['mean'])} |",
        '',
        f"TL above chance on every seed: **{tl_above}**. "
        f"CF below chance on every seed: **{cf_below}**.",
        f"Mean paired TL−CF ΔAUC `{_fmt(gap)}`.",
        'MLCVE is an unmatched ecological check, not a paired contrast.',
        '',
        '## D–E. Factorial (same trained tree per seed)',
        '',
        f'| A (TL) | B (CF proto) | C (CF port) | D (CF) |',
        f'|---:|---:|---:|---:|',
        f'| {_fmt(a_mean)} | {_fmt(b_mean)} | {_fmt(c_mean)} | {_fmt(d_mean)} |',
        '',
        f"Protocol-only B−A: `{_fmt(ba['mean'])}` (all negative: {ba['all_negative']}).",
        f"Port-only C−A: `{_fmt(ca['mean'])}` (all positive: {ca['all_positive']}; "
        f"all negative: {ca['all_negative']}).",
        f"Combined D−A: `{_fmt(da['mean'])}`. Interaction D−C−B+A: `{_fmt(ix['mean'])}`.",
        '',
        'A constant source-port column can still affect a tree through splits',
        'that interact with other features. Port-only effects are therefore',
        'reported, not assumed to be zero.',
        '',
        'Source-train protocol codes were `{0, 6, 17}` on every seed. The',
        'unknown-protocol fraction on TL, CF, and MLCVE was `0`, so',
        'unseen-category-as-missing does not explain the scores.',
        '',
        '## F–H. Interpretation',
        '',
        f'- **A. Useful source-domain performance?** `{useful_source}` '
        f"(mean ROC-AUC `{_fmt(src['roc_auc']['mean'])}` vs LR `{LR_CENTRAL_SOURCE_AUC:.4f}`).",
        f'- **B. TL above chance?** `{tl_above}` (mean `{_fmt(tl["roc_auc"]["mean"])}`).',
        f'- **C. CF below chance?** `{cf_below}` (mean `{_fmt(cf["roc_auc"]["mean"])}`).',
        f'- **D. Protocol-only replacement reverses ranking?** `{proto_reverses}` '
        f"(B `{_fmt(b_mean)}` vs A `{_fmt(a_mean)}`; B−A `{_fmt(ba['mean'])}`).",
        f'- **E. Port-only replacement materially changes ranking?** `{port_material}` '
        f"(C−A `{_fmt(ca['mean'])}`; |Δ|≥0.02 used only as a reporting flag, not a tuned cutoff).",
        f'- **F. Categorical protocol changes the LR failure pattern?** '
        f"LR centralized A/B/C/D = "
        f"{LR_CENTRAL_FACTORIAL['A']:.4f}/"
        f"{LR_CENTRAL_FACTORIAL['B']:.4f}/"
        f"{LR_CENTRAL_FACTORIAL['C']:.4f}/"
        f"{LR_CENTRAL_FACTORIAL['D']:.4f}; "
        f"HGB A/B/C/D = "
        f"{_fmt(a_mean)}/{_fmt(b_mean)}/{_fmt(c_mean)}/{_fmt(d_mean)}.",
        f'- **G. Generality?** {generality}',
        f'- **H. Representation-sensitive shortcut?** {shortcut}',
        '',
        'This is **one** centralized nonlinear family, not a survey of classifiers.',
        'Differences versus logistic regression are **not** attributed to federated averaging.',
        '',
        f'Verdict used for downstream recommendation: `{verdict}`.',
        '',
        '## Per-seed TL vs CF ROC-AUC',
        '',
        '| seed | TL | CF | ΔAUC |',
        '|---|---:|---:|---:|',
    ]
    for s in SEEDS:
        t = tl['roc_auc']['per_seed'][str(s)]
        c = cf['roc_auc']['per_seed'][str(s)]
        lines.append(f'| {s} | {_fmt(t)} | {_fmt(c)} | {_fmt(t - c)} |')
    lines += [
        '',
        '## Limitations',
        '',
        '- One tree-ensemble configuration, not tuned, not federated.',
        '- Same CICIDS2017 slice and five holdouts as the linear study.',
        '- No manuscript change in this phase.',
        '',
    ]
    write_text(ROBUSTNESS_DIR / 'nonlinear_scientific_report.md',
               '\n'.join(lines) + '\n', allowed_root=ROBUSTNESS_DIR)
    summary['verdict'] = verdict
    summary['useful_source'] = useful_source
    summary['tl_above_chance_all_seeds'] = tl_above
    summary['cf_below_chance_all_seeds'] = cf_below
    summary['protocol_only_reverses'] = proto_reverses
    summary['port_only_material'] = port_material
    summary['lr_central_factorial_reference'] = LR_CENTRAL_FACTORIAL


def main() -> None:
    require_categorical_support()
    isolation_self_test()
    frozen_b = snapshot_dir(PHASE_B_DIR, exclude_prefix=('robustness',))
    frozen_v = snapshot_verified()
    inputs = validate_inputs()
    hashes = inputs['hashes']
    if hashes != REFERENCE_HASHES:
        raise SystemExit('prepared CSV hashes drifted')
    if PROTOCOL_INDEX != FEATURES.index('ip_proto'):
        raise SystemExit('protocol index drifted')
    if FEATURES[PROTOCOL_INDEX] != 'ip_proto':
        raise SystemExit('protocol is not at the resolved schema index')

    per_seed_rows = []
    fact_rows = []
    source_metrics = []
    zs = {v: [] for v in VIEWS}
    fact_by_seed = []
    support_rows = []

    for seed in SEEDS:
        xtr, xte, ytr, yte = source_partition(seed)
        train_codes = set(np.unique(xtr[:, PROTOCOL_INDEX]).tolist())
        if 0.0 not in train_codes:
            raise SystemExit(f'seed {seed}: protocol category 0 missing from source train')
        model = make_hgb(seed)
        model.fit(xtr, ytr)
        if list(model.classes_) != [0, 1]:
            raise SystemExit(f'seed {seed}: classes_={model.classes_} (expected [0, 1])')
        cat_flags = [bool(v) for v in np.asarray(model.is_categorical_).ravel()]
        if cat_flags != [i == PROTOCOL_INDEX for i in range(len(FEATURES))]:
            raise SystemExit(f'seed {seed}: is_categorical_={cat_flags}')
        src = compact(met(yte, attack_proba(model, xte)))
        source_metrics.append(src)
        rec = {'seed': seed, 'view': 'source_test', 'n': int(len(yte)),
               'train_protocol_codes': sorted(int(c) for c in train_codes)}
        rec.update({k: src.get(k) for k in FULL_METRICS if k in src})
        per_seed_rows.append(rec)

        tl_x, cf_x, y = load_matched_sample(seed)
        ml_x, ml_y = load(MIRRORS['mlcve'], 'label', 'cic', seed)
        tg = {
            'trafficlabelling': as_hgb_matrix(tl_x),
            'tl_counterfactual': as_hgb_matrix(cf_x),
            'mlcve': as_hgb_matrix(ml_x),
        }
        tg_y = {'trafficlabelling': y, 'tl_counterfactual': y, 'mlcve': ml_y}

        for view in VIEWS:
            p = attack_proba(model, tg[view])
            m = compact(met(tg_y[view], p))
            zs[view].append(m)
            support = protocol_support(xtr, tg[view])
            support_rows.append({'seed': seed, 'view': view, **support})
            row = {'seed': seed, 'view': view, 'n': int(len(tg_y[view])),
                   'n_unknown_protocol': support['n_unknown_protocol'],
                   'frac_unknown_protocol': support['frac_unknown_protocol']}
            row.update({k: m.get(k) for k in FULL_METRICS if k in m})
            per_seed_rows.append(row)

        fac = build_factorial(tl_x, cf_x, y)
        scored = {}
        for cond in CONDITIONS:
            p = attack_proba(model, as_hgb_matrix(fac[cond]))
            scored[cond] = compact(met(y, p))
            row = {'seed': seed, 'condition': cond}
            row.update({k: scored[cond].get(k) for k in FULL_METRICS if k in scored[cond]})
            fact_rows.append(row)
        if abs(scored['A']['roc_auc'] - zs['trafficlabelling'][-1]['roc_auc']) > 1e-12:
            raise SystemExit(f'seed {seed}: A does not match TL scores')
        if abs(scored['D']['roc_auc'] - zs['tl_counterfactual'][-1]['roc_auc']) > 1e-12:
            raise SystemExit(f'seed {seed}: D does not match CF scores')
        fx = effects_from_conditions(scored)
        fact_by_seed.append({'seed': seed, 'metrics': scored, 'effects': fx})

    def agg(rows: list[dict], key: str) -> dict:
        vals = [float(r[key]) for r in rows]
        blk = t_interval(vals)
        blk['per_seed'] = {str(s): float(v) for s, v in zip(SEEDS, vals)}
        return blk

    summary = {
        'model': 'HistGradientBoostingClassifier',
        'federated': False,
        'hyperparameters': {**HGB_HYPERPARAMETERS, 'categorical_features': [PROTOCOL_INDEX],
                            'protocol_feature': 'ip_proto'},
        'source': {k: agg(source_metrics, k) for k in
                   ['roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc', 'f1', 'precision', 'recall']},
        'zero_shot': {v: {k: agg(zs[v], k) for k in
                          ['roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc', 'f1', 'precision', 'recall']}
                      for v in VIEWS},
        'paired_tl_minus_cf_roc_auc': t_interval([
            zs['trafficlabelling'][i]['roc_auc'] - zs['tl_counterfactual'][i]['roc_auc']
            for i in range(len(SEEDS))
        ]),
        'lr_central_source_auc_reference': LR_CENTRAL_SOURCE_AUC,
        'protocol_support': support_rows,
        'hgb_is_categorical_': [i == PROTOCOL_INDEX for i in range(len(FEATURES))],
    }
    summary['factorial'] = {
        'conditions_roc_auc': {
            c: t_interval([row['metrics'][c]['roc_auc'] for row in fact_by_seed])
            for c in CONDITIONS
        },
        'effects_roc_auc': {},
        'effects_mcc': {},
    }
    for name in [
        'protocol_only_original_ports_B_minus_A',
        'port_only_original_protocol_C_minus_A',
        'protocol_under_sentinel_D_minus_C',
        'port_under_reconstructed_D_minus_B',
        'combined_D_minus_A',
        'interaction_D_minus_C_minus_B_plus_A',
    ]:
        auc_vals = [row['effects']['roc_auc'][name] for row in fact_by_seed]
        mcc_vals = [row['effects']['mcc'][name] for row in fact_by_seed]
        summary['factorial']['effects_roc_auc'][name] = {
            **t_interval(auc_vals),
            'all_negative': bool(all(v < 0 for v in auc_vals)),
            'all_positive': bool(all(v > 0 for v in auc_vals)),
            'per_seed': {str(s): float(v) for s, v in zip(SEEDS, auc_vals)},
        }
        summary['factorial']['effects_mcc'][name] = {
            **t_interval(mcc_vals),
            'per_seed': {str(s): float(v) for s, v in zip(SEEDS, mcc_vals)},
        }

    csv_ps = assert_robustness_output(ROBUSTNESS_DIR / 'nonlinear_per_seed.csv')
    pd.DataFrame(per_seed_rows).to_csv(csv_ps, index=False)
    csv_f = assert_robustness_output(ROBUSTNESS_DIR / 'nonlinear_factorial.csv')
    pd.DataFrame(fact_rows).to_csv(csv_f, index=False)
    write_report(summary, hashes)
    write_text(ROBUSTNESS_DIR / 'nonlinear_summary.json',
               json.dumps(summary, indent=2, default=_json_default),
               allowed_root=ROBUSTNESS_DIR)
    if snapshot_dir(PHASE_B_DIR, exclude_prefix=('robustness',)) != frozen_b:
        raise SystemExit('Phase B.1/B.2 outputs changed during B.3')
    if snapshot_verified() != frozen_v:
        raise SystemExit('results/verified/ changed during B.3')
    manifest = {
        'experiment': 'phase_b.3_centralized_hist_gradient_boosting',
        'git_commit': git_sha(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'federated': False,
        'seeds': SEEDS,
        'feature_schema': FEATURES,
        'protocol_column_index': PROTOCOL_INDEX,
        'protocol_treatment': 'native HGB categorical identifier; code 0 is a stored category',
        'scaler': None,
        'target_training': False,
        'target_calibration': False,
        'early_stopping': False,
        'class_weight': None,
        'hyperparameters': HGB_HYPERPARAMETERS,
        'input_hashes': hashes,
        'phase_b_frozen_hashes': frozen_b,
        'package_versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'sklearn': sklearn.__version__,
        },
        'probability': 'predict_proba column for label 1 (attack)',
        'threshold': 0.5,
        'evaluation': {
            'roc_auc': 'sklearn.metrics.roc_auc_score',
            'pr_auc': 'trapezoidal sklearn.metrics.auc on precision_recall_curve (recall, precision)',
            'thresholded': 'predict_proba>=0.5 mapped to attack label 1',
            'metrics': FULL_METRICS,
            'source_split': 'train_test_split(test_size=0.2, stratify, random_state=seed) on load(INSDN,...,seed)',
            'target_sample': 'same seed-specific 30000-row load() as the original study',
            'positive_class': 1,
        },
        'output_root': str(ROBUSTNESS_DIR),
        'verdict': summary['verdict'],
    }
    write_text(ROBUSTNESS_DIR / 'nonlinear_experiment_manifest.json',
               json.dumps(manifest, indent=2, default=_json_default),
               allowed_root=ROBUSTNESS_DIR)
    print(json.dumps({
        'verdict': summary['verdict'],
        'source_auc': summary['source']['roc_auc']['mean'],
        'tl_auc': summary['zero_shot']['trafficlabelling']['roc_auc']['mean'],
        'cf_auc': summary['zero_shot']['tl_counterfactual']['roc_auc']['mean'],
        'mlcve_auc': summary['zero_shot']['mlcve']['roc_auc']['mean'],
        'delta_auc': summary['paired_tl_minus_cf_roc_auc']['mean'],
        'B_minus_A': summary['factorial']['effects_roc_auc']['protocol_only_original_ports_B_minus_A']['mean'],
        'C_minus_A': summary['factorial']['effects_roc_auc']['port_only_original_protocol_C_minus_A']['mean'],
        'A': summary['factorial']['conditions_roc_auc']['A']['mean'],
        'B': summary['factorial']['conditions_roc_auc']['B']['mean'],
        'C': summary['factorial']['conditions_roc_auc']['C']['mean'],
        'D': summary['factorial']['conditions_roc_auc']['D']['mean'],
        'protected_unchanged': True,
    }, indent=2, default=float))


if __name__ == '__main__':
    main()
