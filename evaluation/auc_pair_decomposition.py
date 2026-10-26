"""Exact ROC-AUC pair decomposition by immutable row groups.

Groups are supplied by the caller (Phase B.2 uses original TL protocol codes).
Pairwise AUC uses 0.5 credit for ties, matching sklearn's Mann-Whitney form.
Empty (g,h) cells are marked not estimable and contribute zero weight.
Undefined within-group AUC is never replaced by 0.5 or 0.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def pairwise_auc(pos_scores, neg_scores) -> tuple[float | None, int]:
    """Mann-Whitney AUC of pos vs neg with 0.5 tie credit.

    Returns (auc_or_none, n_pairs). Does not materialize the pairwise matrix.
    """
    pos = np.asarray(pos_scores, dtype=float).reshape(-1)
    neg = np.asarray(neg_scores, dtype=float).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return None, 0
    neg_sorted = np.sort(neg)
    n_less = np.searchsorted(neg_sorted, pos, side='left')
    n_le = np.searchsorted(neg_sorted, pos, side='right')
    wins = n_less + 0.5 * (n_le - n_less)
    n_pairs = int(pos.size * neg.size)
    return float(wins.sum() / n_pairs), n_pairs


def decompose_roc_auc(y, scores, groups) -> dict[str, Any]:
    """Weighted sum of cross-group pairwise AUCs equals global ROC-AUC.

    weight(g,h) = n_pos(g) * n_neg(h) / (N_pos * N_neg)
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    if y.shape[0] != scores.shape[0] or y.shape[0] != groups.shape[0]:
        raise ValueError('y, scores, and groups must have the same length')
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    denom = n_pos * n_neg
    uniq = []
    seen = set()
    for g in groups.tolist():
        key = int(g) if isinstance(g, (int, np.integer)) or (
            isinstance(g, (float, np.floating)) and float(g).is_integer()) else g
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    uniq = sorted(uniq, key=lambda x: (str(type(x)), x))
    cells = []
    reconstructed = 0.0
    for g in uniq:
        gmask = groups == g
        n_pg = int((gmask & pos).sum())
        for h in uniq:
            hmask = groups == h
            n_nh = int((hmask & neg).sum())
            weight = (n_pg * n_nh / denom) if denom else 0.0
            auc_gh, n_pairs = pairwise_auc(scores[gmask & pos], scores[hmask & neg])
            estimable = auc_gh is not None
            contrib = (weight * auc_gh) if estimable else 0.0
            if estimable:
                reconstructed += contrib
            cells.append({
                'pos_group': g,
                'neg_group': h,
                'n_positive': n_pg,
                'n_negative': n_nh,
                'n_pairs': n_pairs,
                'weight': float(weight),
                'pairwise_auc': None if not estimable else float(auc_gh),
                'contribution': float(contrib),
                'estimable': bool(estimable),
            })
    return {
        'n_positive': n_pos,
        'n_negative': n_neg,
        'n_pairs_total': int(denom),
        'groups': uniq,
        'reconstructed_auc': float(reconstructed) if denom else float('nan'),
        'cells': cells,
    }


def cells_involving(cells: list[dict], code) -> list[dict]:
    return [c for c in cells if c['pos_group'] == code or c['neg_group'] == code]


def contribution_sum(cells: list[dict]) -> float:
    return float(sum(c['contribution'] for c in cells))
