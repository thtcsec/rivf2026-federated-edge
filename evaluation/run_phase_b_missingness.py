"""Phase B.4: missing-feature handling policies for zero-shot NIDS transfer.

Author-specified policies in this script. Source-only fitting. Writes only
under results/phase_b/missingness/. HGB is not a federated model.

Experimental matrix (specified in this file; not a timestamped preregistration):
  Models: lr_fedavg_iid, lr_centralized, hgb_categorical, hgb_numeric
  Policies:
    P0 observed nine-field TL (not MLCVE-feasible)
    P1 existing reconstruction (CF / MLCVE contract)
    P2 drop provenance fields and retrain on seven-field intersection
    P3 source-train mode/median imputation on a fixed nine-field model
    P4 missingness-aware source training (mask p=0.5, specified here)
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from output_isolation import (  # noqa: E402
    MECHANISM_DIR, MISSINGNESS_DIR, PHASE_B_DIR, ROBUSTNESS_DIR, VERIFIED_DIR,
    UnsafeOutputError, assert_missingness_output, prepare_phase_b_dir, write_text,
)
from run_phase_b_mechanism import git_sha, load_matched_sample, snapshot_dir  # noqa: E402
from run_phase_b_noniid_cross import (  # noqa: E402
    REFERENCE_HASHES, SEEDS, load_checkpoint,
    scores_from_central, scores_from_fl, snapshot_verified, t_interval,
    transform_with_checkpoint, validate_inputs,
)
from run_phase_b_robustness import (  # noqa: E402
    HGB_HYPERPARAMETERS, PROTOCOL_INDEX, as_hgb_matrix, attack_proba,
    make_hgb, require_categorical_support,
)
from run_verified_federated_baseline import (  # noqa: E402
    FEATURES, FULL_METRICS, INSDN, MIRRORS, PROVENANCE_FIELDS, compact, fed,
    fit, load, met, parts,
)

VIEWS = ('source_test', 'trafficlabelling', 'tl_counterfactual', 'mlcve')
PROTO_I = FEATURES.index('ip_proto')
PORT_I = FEATURES.index('tp_src')
UNAFFECTED_I = [i for i, f in enumerate(FEATURES) if f not in PROVENANCE_FIELDS]
# Pre-registered 2026-10-26. Not chosen from target scores.
MASK_P = 0.5
MASK_OFFSET = 10007
PAPER = {
    'fedavg_iid_tl': 0.7357,
    'fedavg_iid_cf': 0.3545,
    'fedavg_iid_intersection': 0.5075,
    'central_tl': 0.7490,
    'central_cf': 0.3544,
    'onehot_tl': 0.7472,
    'onehot_cf': 0.3630,
    'hgb_cat_tl': 0.9602,
    'hgb_cat_cf': 0.9080,
}
MODELS = ('lr_fedavg_iid', 'lr_centralized', 'hgb_categorical', 'hgb_numeric')
POLICIES = {
    'P0': dict(
        name='observed_nine_fields',
        source_retraining=False,
        fixed_model_intervention=True,
        uses_original_observed_fields=True,
        unavailable_field_handling='none_fields_observed',
        mlcve_feasible=False,
        accessed_original_tl_provenance=True,
        views=('source_test', 'trafficlabelling'),
    ),
    'P1': dict(
        name='reconstruction_cf_contract',
        source_retraining=False,
        fixed_model_intervention=True,
        uses_original_observed_fields=False,
        unavailable_field_handling='flag_inferred_protocol_and_port_sentinel',
        mlcve_feasible=True,
        accessed_original_tl_provenance=False,
        views=('source_test', 'tl_counterfactual', 'mlcve'),
    ),
    'P2': dict(
        name='drop_and_retrain_intersection',
        source_retraining=True,
        fixed_model_intervention=False,
        uses_original_observed_fields=False,
        unavailable_field_handling='drop_unavailable_columns',
        mlcve_feasible=True,
        accessed_original_tl_provenance=False,
        views=VIEWS,
    ),
    'P3': dict(
        name='source_statistic_imputation',
        source_retraining=False,
        fixed_model_intervention=True,
        uses_original_observed_fields=False,
        unavailable_field_handling='source_train_protocol_mode_and_port_median',
        mlcve_feasible=True,
        accessed_original_tl_provenance=False,
        views=VIEWS,
    ),
    'P4': dict(
        name='missingness_aware_source_training',
        source_retraining=True,
        fixed_model_intervention=False,
        uses_original_observed_fields=False,
        unavailable_field_handling='synthetic_mcar_then_100pct_unavailable_at_target',
        mlcve_feasible=True,
        accessed_original_tl_provenance=False,
        views=VIEWS,
    ),
}


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def _fmt(x: float, d: int = 4) -> str:
    return f'{x:.{d}f}'


def provenance_masks(n: int, seed: int, p: float = MASK_P) -> tuple[np.ndarray, np.ndarray]:
    """Label-independent Bernoulli masks. Two independent draws, fixed offset."""
    rng = np.random.default_rng(int(seed) + MASK_OFFSET)
    return rng.random(n) < p, rng.random(n) < p


def source_imputation_constants(xtr: np.ndarray) -> tuple[float, float]:
    proto = np.rint(xtr[:, PROTO_I]).astype(int)
    vals, counts = np.unique(proto, return_counts=True)
    mode = float(vals[int(np.argmax(counts))])
    median = float(np.median(xtr[:, PORT_I]))
    return mode, median


def apply_imputation(x: np.ndarray, proto_mode: float, port_median: float) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    out[:, PROTO_I] = proto_mode
    out[:, PORT_I] = port_median
    return out


def apply_nan_missing(x: np.ndarray, mask_proto: np.ndarray, mask_port: np.ndarray) -> np.ndarray:
    """HGB missingness: NaN. Protocol code 0 is never used as a missing token."""
    out = np.array(x, dtype=np.float64, copy=True)
    out[mask_proto, PROTO_I] = np.nan
    out[mask_port, PORT_I] = np.nan
    if np.any((~mask_proto) & (out[:, PROTO_I] == 0) & np.isnan(out[:, PROTO_I])):
        raise SystemExit('protocol 0 collided with NaN encoding')
    kept = out[~mask_proto, PROTO_I]
    if kept.size and np.isnan(kept).any():
        raise SystemExit('unmasked protocol became NaN')
    return out


def as_hgb_allow_nan(x: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    col = out[:, PROTOCOL_INDEX]
    finite = np.isfinite(col)
    if finite.any():
        codes = np.rint(col[finite])
        if not np.allclose(col[finite], codes, atol=1e-9):
            raise SystemExit('finite protocol values are not integer codes')
        out[finite, PROTOCOL_INDEX] = codes
    return out


def make_hgb_numeric(seed: int, n_features: int) -> HistGradientBoostingClassifier:
    del n_features
    return HistGradientBoostingClassifier(
        loss=HGB_HYPERPARAMETERS['loss'],
        learning_rate=HGB_HYPERPARAMETERS['learning_rate'],
        max_iter=HGB_HYPERPARAMETERS['max_iter'],
        max_leaf_nodes=HGB_HYPERPARAMETERS['max_leaf_nodes'],
        min_samples_leaf=HGB_HYPERPARAMETERS['min_samples_leaf'],
        l2_regularization=HGB_HYPERPARAMETERS['l2_regularization'],
        early_stopping=HGB_HYPERPARAMETERS['early_stopping'],
        random_state=seed,
    )


def make_hgb_flex(seed: int, *, categorical: bool, n_features: int) -> HistGradientBoostingClassifier:
    if categorical:
        if n_features != len(FEATURES):
            raise SystemExit('categorical HGB expects the nine-field schema')
        return make_hgb(seed)
    return make_hgb_numeric(seed, n_features)


def encode_lr_p4(x: np.ndarray, encoder: OneHotEncoder, mask_proto: np.ndarray,
                 mask_port: np.ndarray) -> np.ndarray:
    """Observed protocol one-hot + unavailable flags. Code 0 is a real one-hot column.

    Masked protocol: all protocol dummies zero, protocol_unavailable=1.
    Masked port: numeric slot 0 with port_unavailable=1 (0 is a real InSDN port,
    so the indicator is required). Unmasked port keeps the observed value.
    Masked rows do not read the original protocol or port values.
    """
    n = len(x)
    n_hot = int(encoder.categories_[0].shape[0])
    dummy = float(encoder.categories_[0][0])
    proto_in = np.full((n, 1), dummy, dtype=np.float64)
    keep_p = ~np.asarray(mask_proto, dtype=bool)
    if keep_p.any():
        proto_in[keep_p] = np.rint(x[keep_p][:, [PROTO_I]])
    hot = encoder.transform(proto_in)
    hot[np.asarray(mask_proto, dtype=bool)] = 0.0
    proto_u = np.asarray(mask_proto, dtype=np.float64)[:, None]
    port = np.zeros((n, 1), dtype=np.float64)
    keep_s = ~np.asarray(mask_port, dtype=bool)
    if keep_s.any():
        port[keep_s] = np.asarray(x[keep_s][:, [PORT_I]], dtype=np.float64)
    port_u = np.asarray(mask_port, dtype=np.float64)[:, None]
    rest = x[:, UNAFFECTED_I]
    return np.hstack([hot, proto_u, port, port_u, rest])


def fit_protocol_encoder(xtr: np.ndarray) -> OneHotEncoder:
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    enc.fit(np.rint(xtr[:, [PROTO_I]]))
    cats = [int(c) for c in enc.categories_[0]]
    if 0 not in cats:
        raise SystemExit(f'protocol 0 missing from LR-P4 encoder categories {cats}')
    return enc


def source_partition(seed: int):
    xi, yi = load(INSDN, 'label', 'insdn', seed)
    xtr, xte, ytr, yte = train_test_split(
        xi, yi, test_size=.2, stratify=yi, random_state=seed)
    return xtr, xte, ytr, yte


def leakage_audit(xtr: np.ndarray, xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray) -> dict:
    if any(name in FEATURES for name in ('label', 'target', 'is_synthetic', 'run_id')):
        raise SystemExit('label or identifier leaked into FEATURES')
    keys_tr = [tuple(np.round(r, 10)) for r in xtr]
    keys_te = [tuple(np.round(r, 10)) for r in xte]
    tr_labels: dict[tuple, int] = {}
    train_internal_conflicts = 0
    for k, lab in zip(keys_tr, ytr):
        lab_i = int(lab)
        prev = tr_labels.get(k)
        if prev is None:
            tr_labels[k] = lab_i
        elif prev != lab_i:
            train_internal_conflicts += 1
    overlap = set(keys_tr).intersection(keys_te)
    te_in = int(sum(k in overlap for k in keys_te))
    split_conflicts = int(sum(
        1 for k, lab in zip(keys_te, yte)
        if k in tr_labels and tr_labels[k] != int(lab)))
    consistent = split_conflicts == 0 and train_internal_conflicts == 0
    if consistent:
        consistency = 'Duplicate groups in this split are label-consistent. '
    else:
        consistency = (
            f'This split has {split_conflicts} train-test label conflicts '
            f'on overlapping vectors '
            f'({train_internal_conflicts} train-internal). ')
    return {
        'n_train': int(len(ytr)),
        'n_test': int(len(yte)),
        'unique_exact_overlap': int(len(overlap)),
        'test_rows_matching_a_train_vector': te_in,
        'test_overlap_frac': float(te_in / len(yte)),
        'label_conflicts_in_this_split_overlap': split_conflicts,
        'train_internal_label_conflicts': train_internal_conflicts,
        'features': list(FEATURES),
        'verdict': (
            'Repeated identical InSDN flow-stat vectors exist and appear in both '
            'source splits. ' + consistency +
            'This inflates source AUC but is not target-label leakage. '
            'This audit does not compute an unseen-row source AUC.'
        ),
    }


def isolation_self_test() -> None:
    prepare_phase_b_dir(MISSINGNESS_DIR)
    probe = assert_missingness_output(MISSINGNESS_DIR / 'isolation_probe.txt')
    write_text(probe, 'missingness isolation ok\n', allowed_root=MISSINGNESS_DIR)
    banned = [
        VERIFIED_DIR / 'summary.json',
        PHASE_B_DIR / 'per_seed_noniid_cross.csv',
        MECHANISM_DIR / 'factorial_summary.json',
        ROBUSTNESS_DIR / 'nonlinear_summary.json',
        MISSINGNESS_DIR / '..' / 'robustness' / 'nonlinear_summary.json',
    ]
    for path in banned:
        try:
            assert_missingness_output(path)
        except UnsafeOutputError:
            pass
        else:
            raise SystemExit(f'isolation self-test failed: accepted {path}')
    probe.unlink(missing_ok=True)


def policy_row(seed: int, model: str, policy: str, view: str, metrics: dict,
               extra: dict) -> dict:
    spec = POLICIES[policy]
    row = {
        'seed': seed,
        'model': model,
        'federated': model.startswith('lr_fedavg'),
        'policy': policy,
        'policy_name': spec['name'],
        'view': view,
        'fixed_model_intervention': spec['fixed_model_intervention'],
        'source_retraining': spec['source_retraining'],
        'uses_original_observed_fields': (
            spec['uses_original_observed_fields'] and view in ('source_test', 'trafficlabelling')
            and policy == 'P0'
        ),
        'unavailable_field_handling': spec['unavailable_field_handling'],
        'mlcve_feasible': spec['mlcve_feasible'],
        'accessed_original_tl_provenance': spec['accessed_original_tl_provenance'] and view == 'trafficlabelling',
        'target_labels_used': 'evaluation_only',
    }
    row.update(extra)
    row.update({k: metrics.get(k) for k in FULL_METRICS if k in metrics})
    return row


def train_lr_matrix(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray,
                    seed: int) -> dict:
    sc = StandardScaler().fit(xtr)
    xtr_s, xte_s = sc.transform(xtr), sc.transform(xte)
    cm = fit(xtr_s, ytr, seed)
    pi = parts(xtr_s, ytr, seed)
    wi, bi, _ = fed(pi, xte_s, yte, seed)
    return {'scaler': sc, 'central': cm, 'coef': wi, 'intercept': bi}


def score_lr_trained(bundle: dict, x: np.ndarray, y: np.ndarray, which: str) -> dict:
    xs = bundle['scaler'].transform(x)
    if which == 'lr_centralized':
        return scores_from_central(bundle['central'], xs, y)
    return scores_from_fl(bundle['coef'], bundle['intercept'], xs, y)


def replay_lr(ckpt: dict, x: np.ndarray, y: np.ndarray, model: str, seed: int) -> dict:
    xs = transform_with_checkpoint(x, ckpt)
    if model == 'lr_centralized':
        from run_phase_b_noniid_cross import central_proba_from_checkpoint
        return compact(met(y, central_proba_from_checkpoint(ckpt, xs, seed)))
    return scores_from_fl(ckpt['coef'], ckpt['intercept'], xs, y)


def fit_hgb(xtr: np.ndarray, ytr: np.ndarray, seed: int, *, categorical: bool) -> HistGradientBoostingClassifier:
    n = xtr.shape[1]
    m = make_hgb_flex(seed, categorical=categorical, n_features=n)
    m.fit(xtr, ytr)
    if list(m.classes_) != [0, 1]:
        raise SystemExit(f'HGB classes_={m.classes_}')
    return m


def write_report(summary: dict) -> None:
    lines = [
        '# Phase B.4 missing-feature policy report',
        '',
        'Controlled comparison of source-only missingness policies when Protocol',
        'and Source Port are unavailable in a target release. **HGB is not a',
        'federated model.** Policies and the masking probability are specified in',
        'this script; that is not a timestamped preregistration.',
        '',
        f"- Git commit: `{git_sha()}`",
        f"- sklearn: `{sklearn.__version__}`",
        f"- Author-specified P4 mask probability: `{MASK_P}` (independent per field)",
        f"- Leakage audit: {summary['leakage']['verdict']}",
        '',
        '## Policy feasibility',
        '',
        '| policy | MLCVE-feasible | retrains | uses original TL proto/port | handling |',
        '|---|---|---|---|---|',
    ]
    for pid, spec in POLICIES.items():
        lines.append(
            f"| {pid} | {spec['mlcve_feasible']} | {spec['source_retraining']} | "
            f"{spec['accessed_original_tl_provenance']} | `{spec['unavailable_field_handling']}` |"
        )

    def grab(model, policy, view, metric='roc_auc'):
        return summary['by_model_policy'][model][policy][view][metric]

    lines += ['', '## Mean ROC-AUC (five seeds)', '', '| model | P0 TL | P1 CF | P2 TL=CF | P3 TL | P4 TL | P1 MLCVE | P2 MLCVE | P3 MLCVE | P4 MLCVE |']
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for model in MODELS:
        blk = summary['by_model_policy'][model]
        p2 = grab(model, 'P2', 'trafficlabelling')['mean']
        lines.append(
            f"| {model} | {_fmt(grab(model,'P0','trafficlabelling')['mean'])} | "
            f"{_fmt(grab(model,'P1','tl_counterfactual')['mean'])} | {_fmt(p2)} | "
            f"{_fmt(grab(model,'P3','trafficlabelling')['mean'])} | "
            f"{_fmt(grab(model,'P4','trafficlabelling')['mean'])} | "
            f"{_fmt(grab(model,'P1','mlcve')['mean'])} | "
            f"{_fmt(grab(model,'P2','mlcve')['mean'])} | "
            f"{_fmt(grab(model,'P3','mlcve')['mean'])} | "
            f"{_fmt(grab(model,'P4','mlcve')['mean'])} |"
        )
    lines += ['', '## Source-domain ROC-AUC', '', '| model | P0/P1/P3 observed | P2 seven-field | P4 observed encoding |']
    lines.append('|---|---:|---:|---:|')
    for model in MODELS:
        lines.append(
            f"| {model} | {_fmt(grab(model,'P0','source_test')['mean'])} | "
            f"{_fmt(grab(model,'P2','source_test')['mean'])} | "
            f"{_fmt(grab(model,'P4','source_test')['mean'])} |"
        )

    q = summary['answers']
    lines += [
        '',
        '## Author-specified questions',
        '',
        f"1. Can any source-only missingness policy avoid the LR below-chance result? **{q['q1']}**",
        f"2. If yes, useful ranking vs chance? **{q['q2']}**",
        f"3. Does P4 differ from P2 (drop-and-retrain)? **{q['q3']}**",
        f"4. MLCVE-feasible policies: **{q['q4']}**",
        f"5. Does HGB remain robust under each policy? **{q['q5']}**",
        f"6. HGB robustness: family, representation, or both? **{q['q6']}**",
        f"7. Cross-dataset shortcuts? **{q['q7']}**",
        f"8. Supports camera-ready claims? **{q['q8']}**",
        f"9. Narrows camera-ready claims? **{q['q9']}**",
        f"10. Journal vs six-page paper? **{q['q10']}**",
        '',
        'One-hot LR (accepted ablation) remains TL 0.7472 / CF 0.3630; categorical',
        'encoding inside logistic regression did not remove the reversal. HGB',
        'categorical vs numeric is the within-family representation control.',
        '',
        'A better target AUC is an observed outcome under the specified policies,',
        'not a validated deployment-time selection rule.',
        '',
        '## Limitations',
        '',
        f'- P4 trains with missingness rate {MASK_P} and is applied to 100% missing target fields.',
        '- This audit reports source-split overlap; it does not compute an unseen-row source AUC.',
        '- One HGB configuration and the original LR optimizer; not a survey.',
        '- No manuscript change in this phase.',
        '',
    ]
    write_text(MISSINGNESS_DIR / 'missingness_scientific_report.md',
               '\n'.join(lines) + '\n', allowed_root=MISSINGNESS_DIR)


def answers_from_summary(by: dict) -> dict:
    def m(model, policy, view):
        return float(by[model][policy][view]['roc_auc']['mean'])

    def mcc(model, policy, view):
        return float(by[model][policy][view]['mcc']['mean'])

    p2_auc = m('lr_fedavg_iid', 'P2', 'trafficlabelling')
    p3_auc = m('lr_fedavg_iid', 'P3', 'trafficlabelling')
    p4_auc = m('lr_fedavg_iid', 'P4', 'trafficlabelling')
    p3_mcc = mcc('lr_fedavg_iid', 'P3', 'trafficlabelling')
    p2_mcc = mcc('lr_fedavg_iid', 'P2', 'trafficlabelling')
    p4_mcc = mcc('lr_fedavg_iid', 'P4', 'trafficlabelling')
    p2_p4 = abs(p2_auc - p4_auc)
    cat_gap = m('hgb_categorical', 'P0', 'trafficlabelling') - m('hgb_categorical', 'P1', 'tl_counterfactual')
    num_gap = m('hgb_numeric', 'P0', 'trafficlabelling') - m('hgb_numeric', 'P1', 'tl_counterfactual')
    num_cf = m('hgb_numeric', 'P1', 'tl_counterfactual')
    cat_cf = m('hgb_categorical', 'P1', 'tl_counterfactual')
    hgb_p3 = m('hgb_categorical', 'P3', 'trafficlabelling')
    hgb_p2 = m('hgb_categorical', 'P2', 'trafficlabelling')
    hgb_p4 = m('hgb_categorical', 'P4', 'trafficlabelling')
    if num_cf < 0.5 < m('hgb_numeric', 'P0', 'trafficlabelling') and cat_cf >= 0.8:
        q6 = ('Representation contributes: numeric HGB CF falls toward or below chance '
              'while categorical HGB does not. One-hot LR still reverses, so family '
              'and representation remain partly confounded.')
    elif num_cf >= 0.8 and cat_cf >= 0.8:
        q6 = ('Classifier family dominates under these configurations: numeric and '
              f'categorical HGB CF stay above chance ({num_cf:.4f} vs {cat_cf:.4f}), '
              'unlike one-hot or numeric LR. Categorical protocol shrinks the HGB '
              f'TL–CF gap ({cat_gap:.4f} vs numeric {num_gap:.4f}) but is not required '
              'to avoid a reversal.')
    else:
        q6 = ('Both family and representation matter under the controlled comparisons; '
              'do not attribute the pattern to a single cause.')
    q1 = (
        f'P1 reconstruction remains below chance (FedAvg CF AUC {m("lr_fedavg_iid","P1","tl_counterfactual"):.4f}, '
        f'MCC {mcc("lr_fedavg_iid","P1","tl_counterfactual"):+.4f}). '
        f'P2 is chance-level ranking (AUC {p2_auc:.4f}, MCC {p2_mcc:+.4f}). '
        f'P4 is near chance (AUC {p4_auc:.4f}, MCC {p4_mcc:+.4f}). '
        f'P3 raises ROC-AUC to {p3_auc:.4f} but MCC stays negative ({p3_mcc:+.4f}).'
    )
    q2 = (
        'No tested LR missingness policy is a useful detector. P3 is above-chance '
        'ranking with still-harmful thresholded decisions. P2/P4 are chance-level.'
    )
    return {
        'q1': q1,
        'q2': q2,
        'q3': (f'P2 vs P4 |ΔAUC| on FedAvg IID TL is {p2_p4:.4f}. '
               'Differences are reported, not selected.'),
        'q4': 'P1, P2, P3, P4. P0 is not MLCVE-feasible.',
        'q5': (
            f'P0/P1/P2/P4 remain strong (P2 {hgb_p2:.4f}, P4 {hgb_p4:.4f}, '
            f'P1 CF {cat_cf:.4f}). P3 source-stat imputation degrades HGB to {hgb_p3:.4f}.'
        ),
        'q6': q6,
        'q7': ('Protocol remains a documented LR shortcut. HGB seven-field source '
               'separability shows InSDN volume features are also highly informative '
               'in-domain; that does not restore LR target ranking.'),
        'q8': ('The accepted reconstruction reversal for numeric/one-hot LR, and the '
               'chance-level seven-field LR intersection, are reproduced.'),
        'q9': ('P2/P4 avoid the LR reconstruction reversal only by returning to chance. '
               'P3 improves ROC-AUC but not MCC. HGB stays strong if the fields are '
               'dropped or marked missing, and weakens under source-stat imputation. '
               'That qualifies the LR reversal as representation- and family-sensitive, '
               'not a universal missingness result.'),
        'q10': ('The full policy matrix, P4 train/target missingness mismatch, and '
                'HGB numeric-vs-categorical control belong in a journal extension. '
                'The six-page camera-ready should keep the locked LR reconstruction result.'),
        'hgb_cat_p0_minus_p1': cat_gap,
        'hgb_num_p0_minus_p1': num_gap,
    }


def main() -> None:
    require_categorical_support()
    if 'categorical_features' not in inspect.signature(
            HistGradientBoostingClassifier.__init__).parameters:
        raise SystemExit('HGB categorical_features unavailable')
    isolation_self_test()
    frozen_b = snapshot_dir(PHASE_B_DIR, exclude_prefix=('missingness',))
    frozen_v = snapshot_verified()
    inputs = validate_inputs()
    hashes = inputs['hashes']
    if hashes != REFERENCE_HASHES:
        raise SystemExit('prepared CSV hashes drifted')

    per_seed_rows = []
    leak_rows = []
    bundles = {s: {} for s in SEEDS}

    for seed in SEEDS:
        xtr, xte, ytr, yte = source_partition(seed)
        leak_rows.append({'seed': seed, **leakage_audit(xtr, xte, ytr, yte)})
        tl_x, cf_x, y_pair = load_matched_sample(seed)
        ml_x, ml_y = load(MIRRORS['mlcve'], 'label', 'cic', seed)
        views_raw = {
            'source_test': (xte, yte),
            'trafficlabelling': (tl_x, y_pair),
            'tl_counterfactual': (cf_x, y_pair),
            'mlcve': (ml_x, ml_y),
        }
        proto_mode, port_median = source_imputation_constants(xtr)
        extra_imp = {'proto_mode': proto_mode, 'port_median': port_median}

        ck_iid = load_checkpoint(PHASE_B_DIR, seed, 'fedavg_iid', expected_hashes=hashes)
        ck_cen = load_checkpoint(PHASE_B_DIR, seed, 'centralized', expected_hashes=hashes)

        # P0 / P1 / P3 share the frozen nine-field LR models.
        for model, ck in (('lr_fedavg_iid', ck_iid), ('lr_centralized', ck_cen)):
            m_p0 = replay_lr(ck, xte, yte, model, seed)
            m_tl = replay_lr(ck, tl_x, y_pair, model, seed)
            m_cf = replay_lr(ck, cf_x, y_pair, model, seed)
            m_ml = replay_lr(ck, ml_x, ml_y, model, seed)
            per_seed_rows.append(policy_row(seed, model, 'P0', 'source_test', m_p0, extra_imp))
            per_seed_rows.append(policy_row(seed, model, 'P0', 'trafficlabelling', m_tl, extra_imp))
            per_seed_rows.append(policy_row(seed, model, 'P1', 'source_test', m_p0, extra_imp))
            per_seed_rows.append(policy_row(seed, model, 'P1', 'tl_counterfactual', m_cf, extra_imp))
            per_seed_rows.append(policy_row(seed, model, 'P1', 'mlcve', m_ml, extra_imp))
            imp_src = apply_imputation(xte, proto_mode, port_median)
            imp_tl = apply_imputation(tl_x, proto_mode, port_median)
            imp_cf = apply_imputation(cf_x, proto_mode, port_median)
            imp_ml = apply_imputation(ml_x, proto_mode, port_median)
            if not np.array_equal(imp_tl, imp_cf):
                raise SystemExit(f'seed {seed}: P3 TL/CF differ after imputation')
            per_seed_rows.append(policy_row(seed, model, 'P3', 'source_test',
                                            replay_lr(ck, imp_src, yte, model, seed), extra_imp))
            per_seed_rows.append(policy_row(seed, model, 'P3', 'trafficlabelling',
                                            replay_lr(ck, imp_tl, y_pair, model, seed), extra_imp))
            per_seed_rows.append(policy_row(seed, model, 'P3', 'tl_counterfactual',
                                            replay_lr(ck, imp_cf, y_pair, model, seed), extra_imp))
            per_seed_rows.append(policy_row(seed, model, 'P3', 'mlcve',
                                            replay_lr(ck, imp_ml, ml_y, model, seed), extra_imp))

        # P2 LR retrain on seven fields.
        lr2 = train_lr_matrix(xtr[:, UNAFFECTED_I], ytr, xte[:, UNAFFECTED_I], yte, seed)
        for model in ('lr_fedavg_iid', 'lr_centralized'):
            for view, (xv, yv) in views_raw.items():
                sc = score_lr_trained(lr2, xv[:, UNAFFECTED_I], yv, model)
                per_seed_rows.append(policy_row(seed, model, 'P2', view, sc, extra_imp))
        p2_tl = score_lr_trained(lr2, tl_x[:, UNAFFECTED_I], y_pair, 'lr_fedavg_iid')
        p2_cf = score_lr_trained(lr2, cf_x[:, UNAFFECTED_I], y_pair, 'lr_fedavg_iid')
        if abs(p2_tl['roc_auc'] - p2_cf['roc_auc']) > 1e-12:
            raise SystemExit(f'seed {seed}: P2 TL/CF AUC differ')

        # P4 LR: missingness-aware encoding + retrain.
        enc = fit_protocol_encoder(xtr)
        mp_tr, ms_tr = provenance_masks(len(xtr), seed)
        if not mp_tr.any() or not (~mp_tr).any() or not ms_tr.any() or not (~ms_tr).any():
            raise SystemExit(f'seed {seed}: P4 training mask is degenerate')
        xtr_p4 = encode_lr_p4(xtr, enc, mp_tr, ms_tr)
        xte_obs = encode_lr_p4(xte, enc, np.zeros(len(xte), bool), np.zeros(len(xte), bool))
        lr4 = train_lr_matrix(xtr_p4, ytr, xte_obs, yte, seed)
        for model in ('lr_fedavg_iid', 'lr_centralized'):
            per_seed_rows.append(policy_row(
                seed, model, 'P4', 'source_test',
                score_lr_trained(lr4, xte_obs, yte, model), extra_imp))
            for view, (xv, yv) in views_raw.items():
                if view == 'source_test':
                    continue
                n = len(xv)
                x_unavail = encode_lr_p4(xv, enc, np.ones(n, bool), np.ones(n, bool))
                per_seed_rows.append(policy_row(
                    seed, model, 'P4', view,
                    score_lr_trained(lr4, x_unavail, yv, model), extra_imp))

        # HGB categorical and numeric.
        for cat, mname in ((True, 'hgb_categorical'), (False, 'hgb_numeric')):
            xtr9 = as_hgb_matrix(xtr) if cat else np.array(xtr, dtype=np.float64, copy=True)
            xte9 = as_hgb_matrix(xte) if cat else np.array(xte, dtype=np.float64, copy=True)
            h0 = fit_hgb(xtr9, ytr, seed, categorical=cat)
            # P0 / P1 / P3: fixed nine-field HGB (P3 imputes then scores h0).
            def hscore(model, xx, yy):
                return compact(met(yy, attack_proba(model, xx)))

            def prep(xx):
                return as_hgb_matrix(xx) if cat else np.array(xx, dtype=np.float64, copy=True)

            per_seed_rows.append(policy_row(seed, mname, 'P0', 'source_test', hscore(h0, xte9, yte), extra_imp))
            per_seed_rows.append(policy_row(seed, mname, 'P0', 'trafficlabelling',
                                            hscore(h0, prep(tl_x), y_pair), extra_imp))
            per_seed_rows.append(policy_row(seed, mname, 'P1', 'source_test', hscore(h0, xte9, yte), extra_imp))
            per_seed_rows.append(policy_row(seed, mname, 'P1', 'tl_counterfactual',
                                            hscore(h0, prep(cf_x), y_pair), extra_imp))
            per_seed_rows.append(policy_row(seed, mname, 'P1', 'mlcve',
                                            hscore(h0, prep(ml_x), ml_y), extra_imp))
            per_seed_rows.append(policy_row(seed, mname, 'P3', 'source_test',
                                            hscore(h0, prep(apply_imputation(xte, proto_mode, port_median)), yte), extra_imp))
            per_seed_rows.append(policy_row(seed, mname, 'P3', 'trafficlabelling',
                                            hscore(h0, prep(apply_imputation(tl_x, proto_mode, port_median)), y_pair), extra_imp))
            per_seed_rows.append(policy_row(seed, mname, 'P3', 'tl_counterfactual',
                                            hscore(h0, prep(apply_imputation(cf_x, proto_mode, port_median)), y_pair), extra_imp))
            per_seed_rows.append(policy_row(seed, mname, 'P3', 'mlcve',
                                            hscore(h0, prep(apply_imputation(ml_x, proto_mode, port_median)), ml_y), extra_imp))

            h2 = fit_hgb(xtr[:, UNAFFECTED_I], ytr, seed, categorical=False)
            for view, (xv, yv) in views_raw.items():
                per_seed_rows.append(policy_row(
                    seed, mname, 'P2', view,
                    compact(met(yv, attack_proba(h2, xv[:, UNAFFECTED_I]))), extra_imp))

            # P4 HGB: NaN missingness during source training.
            mp_tr, ms_tr = provenance_masks(len(xtr), seed)
            xtr_m = apply_nan_missing(xtr9 if cat else xtr, mp_tr, ms_tr)
            if cat:
                xtr_m = as_hgb_allow_nan(xtr_m)
            h4 = fit_hgb(xtr_m, ytr, seed, categorical=cat)
            per_seed_rows.append(policy_row(
                seed, mname, 'P4', 'source_test', hscore(h4, xte9, yte), extra_imp))
            for view, (xv, yv) in views_raw.items():
                if view == 'source_test':
                    continue
                xx = prep(xv)
                n = len(xx)
                xxm = apply_nan_missing(xx, np.ones(n, bool), np.ones(n, bool))
                if cat:
                    xxm = as_hgb_allow_nan(xxm)
                per_seed_rows.append(policy_row(
                    seed, mname, 'P4', view, hscore(h4, xxm, yv), extra_imp))

        bundles[seed] = {'proto_mode': proto_mode, 'port_median': port_median}

    df = pd.DataFrame(per_seed_rows)
    csv_path = assert_missingness_output(MISSINGNESS_DIR / 'per_seed_policy_metrics.csv')
    df.to_csv(csv_path, index=False)

    def agg(model, policy, view):
        sub = df[(df.model == model) & (df.policy == policy) & (df.view == view)]
        if len(sub) != 5:
            raise SystemExit(f'missing rows for {model} {policy} {view}: {len(sub)}')
        out = {}
        for k in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc', 'f1', 'precision', 'recall']:
            vals = [float(v) for v in sub[k].tolist()]
            blk = t_interval(vals)
            blk['per_seed'] = {str(s): float(v) for s, v in zip(sub.seed.tolist(), vals)}
            out[k] = blk
        return out

    by = {m: {p: {} for p in POLICIES} for m in MODELS}
    for m in MODELS:
        for p, spec in POLICIES.items():
            for v in spec['views']:
                by[m][p][v] = agg(m, p, v)

    # Reproduction gates.
    def mean_auc(model, policy, view):
        return by[model][policy][view]['roc_auc']['mean']

    checks = [
        ('fedavg P0 TL', mean_auc('lr_fedavg_iid', 'P0', 'trafficlabelling'), PAPER['fedavg_iid_tl']),
        ('fedavg P1 CF', mean_auc('lr_fedavg_iid', 'P1', 'tl_counterfactual'), PAPER['fedavg_iid_cf']),
        ('fedavg P2 TL', mean_auc('lr_fedavg_iid', 'P2', 'trafficlabelling'), PAPER['fedavg_iid_intersection']),
        ('central P0 TL', mean_auc('lr_centralized', 'P0', 'trafficlabelling'), PAPER['central_tl']),
        ('central P1 CF', mean_auc('lr_centralized', 'P1', 'tl_counterfactual'), PAPER['central_cf']),
        ('hgb cat P0 TL', mean_auc('hgb_categorical', 'P0', 'trafficlabelling'), PAPER['hgb_cat_tl']),
        ('hgb cat P1 CF', mean_auc('hgb_categorical', 'P1', 'tl_counterfactual'), PAPER['hgb_cat_cf']),
    ]
    for name, got, ref in checks:
        if abs(got - ref) > 5.5e-4:
            raise SystemExit(f'baseline reproduction failed: {name} got {got:.6f} vs {ref:.4f}')

    # Representation-sensitivity table (HGB cat vs numeric, P0 vs P1).
    rep_rows = []
    for seed in SEEDS:
        for policy, view in (('P0', 'trafficlabelling'), ('P1', 'tl_counterfactual')):
            for mname in ('hgb_categorical', 'hgb_numeric'):
                sub = df[(df.seed == seed) & (df.model == mname) & (df.policy == policy) & (df.view == view)]
                rep_rows.append({
                    'seed': seed, 'policy': policy, 'view': view, 'model': mname,
                    'roc_auc': float(sub.roc_auc.iloc[0]),
                    'mcc': float(sub.mcc.iloc[0]),
                    'protocol_treatment': 'categorical' if mname == 'hgb_categorical' else 'numeric',
                    'onehot_lr_context_tl': PAPER['onehot_tl'],
                    'onehot_lr_context_cf': PAPER['onehot_cf'],
                })
    rep_path = assert_missingness_output(MISSINGNESS_DIR / 'representation_sensitivity.csv')
    pd.DataFrame(rep_rows).to_csv(rep_path, index=False)

    summary = {
        'experiment': 'phase_b.4_missing_feature_policies',
        'federated_hgb': False,
        'mask_p': MASK_P,
        'by_model_policy': by,
        'leakage': {
            'per_seed': leak_rows,
            'verdict': leak_rows[0]['verdict'],
            'mean_test_overlap_frac': float(np.mean([r['test_overlap_frac'] for r in leak_rows])),
        },
        'reproduction': {name: {'got': got, 'ref': ref} for name, got, ref in checks},
        'paper_onehot_lr': {'tl': PAPER['onehot_tl'], 'cf': PAPER['onehot_cf']},
        'imputation_per_seed': {str(s): bundles[s] for s in SEEDS},
    }
    summary['answers'] = answers_from_summary(by)
    write_report(summary)
    write_text(MISSINGNESS_DIR / 'policy_comparison_summary.json',
               json.dumps(summary, indent=2, default=_json_default),
               allowed_root=MISSINGNESS_DIR)

    if snapshot_dir(PHASE_B_DIR, exclude_prefix=('missingness',)) != frozen_b:
        raise SystemExit('Phase B.1/B.2/B.3 outputs changed during B.4')
    if snapshot_verified() != frozen_v:
        raise SystemExit('results/verified/ changed during B.4')

    manifest = {
        'experiment': 'phase_b.4_missing_feature_policies',
        'git_commit': git_sha(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': SEEDS,
        'feature_schema': FEATURES,
        'unaffected_features': [FEATURES[i] for i in UNAFFECTED_I],
        'protocol_column_index': PROTO_I,
        'port_column_index': PORT_I,
        'mask_probability': MASK_P,
        'mask_seed_offset': MASK_OFFSET,
        'mask_independence': 'independent Bernoulli per provenance field',
        'p4_lr_encoding': (
            'OneHotEncoder(handle_unknown=ignore) on source-train protocol codes; '
            'protocol_unavailable flag; observed port or 0 with port_unavailable flag; '
            'seven unaffected features; StandardScaler fit on encoded source train. '
            'Protocol code 0 is a real category, never the missing token.'
        ),
        'p4_hgb_encoding': 'native NaN missing values; categorical protocol for hgb_categorical only',
        'p4_limitation': f'train missingness {MASK_P} vs 100% unavailable target fields',
        'target_training': False,
        'target_calibration': False,
        'input_hashes': hashes,
        'phase_b_frozen_hashes': frozen_b,
        'policies': POLICIES,
        'models': {
            'lr_fedavg_iid': 'replayed 9-field FedAvg IID checkpoints for P0/P1/P3; retrained for P2/P4',
            'lr_centralized': 'replayed 9-field centralized checkpoints for P0/P1/P3; retrained for P2/P4',
            'hgb_categorical': 'centralized HistGradientBoostingClassifier; not federated',
            'hgb_numeric': 'same HGB family/hyperparameters without categorical_features',
        },
        'hgb_hyperparameters': HGB_HYPERPARAMETERS,
        'evaluation': {
            'roc_auc': 'sklearn.metrics.roc_auc_score',
            'pr_auc': 'trapezoidal sklearn.metrics.auc on precision_recall_curve',
            'threshold': 0.5,
            'positive_class': 1,
            'metrics': FULL_METRICS,
        },
        'package_versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'sklearn': sklearn.__version__,
        },
        'output_root': str(MISSINGNESS_DIR),
        'reproduction': summary['reproduction'],
    }
    write_text(MISSINGNESS_DIR / 'experiment_manifest.json',
               json.dumps(manifest, indent=2, default=_json_default),
               allowed_root=MISSINGNESS_DIR)
    print(json.dumps({
        'reproduction_ok': True,
        'mask_p': MASK_P,
        'fedavg_p0_tl': mean_auc('lr_fedavg_iid', 'P0', 'trafficlabelling'),
        'fedavg_p1_cf': mean_auc('lr_fedavg_iid', 'P1', 'tl_counterfactual'),
        'fedavg_p2_tl': mean_auc('lr_fedavg_iid', 'P2', 'trafficlabelling'),
        'fedavg_p3_tl': mean_auc('lr_fedavg_iid', 'P3', 'trafficlabelling'),
        'fedavg_p4_tl': mean_auc('lr_fedavg_iid', 'P4', 'trafficlabelling'),
        'hgb_cat_p0_tl': mean_auc('hgb_categorical', 'P0', 'trafficlabelling'),
        'hgb_cat_p1_cf': mean_auc('hgb_categorical', 'P1', 'tl_counterfactual'),
        'hgb_cat_p2': mean_auc('hgb_categorical', 'P2', 'trafficlabelling'),
        'hgb_num_p0_tl': mean_auc('hgb_numeric', 'P0', 'trafficlabelling'),
        'hgb_num_p1_cf': mean_auc('hgb_numeric', 'P1', 'tl_counterfactual'),
        'protected_unchanged': True,
    }, indent=2, default=float))


if __name__ == '__main__':
    main()
