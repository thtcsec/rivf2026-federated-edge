# Phase B.3 nonlinear robustness report

Centralized `HistGradientBoostingClassifier` with protocol as a native
categorical feature. **Not a federated model.** Hyperparameters were
pre-registered and were not changed after seeing target scores.
The camera-ready manuscript was not modified.

- Git commit: `de8ffdd6fb153471127947109d78b1435080b57d`
- sklearn: `1.7.2`
- Protocol column index from FEATURES: `0` (`ip_proto`)
- Scientific verdict: **model_family_dependent_under_the_tested_configurations**

## A. Source-domain performance

Mean ROC-AUC `0.9999` (SD `0.0001`)
vs accepted centralized LR `0.9320`.
MCC `0.9925`, F1 `0.9985`.
Useful source ranking (mean AUC ≥ 0.8): **True**.

## B–C. Zero-shot TL / CF / MLCVE

| view | ROC-AUC | PR-AUC | Bal. acc. | MCC |
|---|---:|---:|---:|---:|
| TL | 0.9602 | 0.8748 | 0.9162 | 0.7613 |
| CF | 0.9080 | 0.7286 | 0.7920 | 0.5283 |
| MLCVE | 0.9073 | 0.7245 | 0.7886 | 0.5044 |

TL above chance on every seed: **True**. CF below chance on every seed: **False**.
Mean paired TL−CF ΔAUC `0.0522`.
MLCVE is an unmatched ecological check, not a paired contrast.

## D–E. Factorial (same trained tree per seed)

| A (TL) | B (CF proto) | C (CF port) | D (CF) |
|---:|---:|---:|---:|
| 0.9602 | 0.9601 | 0.9096 | 0.9080 |

Protocol-only B−A: `-0.0001` (all negative: True).
Port-only C−A: `-0.0506` (all positive: False; all negative: True).
Combined D−A: `-0.0522`. Interaction D−C−B+A: `-0.0015`.

A constant source-port column can still affect a tree through splits
that interact with other features. Port-only effects are therefore
reported, not assumed to be zero.

Source-train protocol codes were `{0, 6, 17}` on every seed. The
unknown-protocol fraction on TL, CF, and MLCVE was `0`, so
unseen-category-as-missing does not explain the scores.

## F–H. Interpretation

- **A. Useful source-domain performance?** `True` (mean ROC-AUC `0.9999` vs LR `0.9320`).
- **B. TL above chance?** `True` (mean `0.9602`).
- **C. CF below chance?** `False` (mean `0.9080`).
- **D. Protocol-only replacement reverses ranking?** `False` (B `0.9601` vs A `0.9602`; B−A `-0.0001`).
- **E. Port-only replacement materially changes ranking?** `True` (C−A `-0.0506`; |Δ|≥0.02 used only as a reporting flag, not a tuned cutoff).
- **F. Categorical protocol changes the LR failure pattern?** LR centralized A/B/C/D = 0.7490/0.3457/0.7576/0.3544; HGB A/B/C/D = 0.9602/0.9601/0.9096/0.9080.
- **G. Generality?** Weakens the generality claim: the TL-to-CF ranking reversal is model-family-dependent under the tested configurations.
- **H. Representation-sensitive shortcut?** Consistent with a representation-sensitive shortcut in logistic regression (numeric protocol after source scaling) rather than a universal reconstruction failure.

This is **one** centralized nonlinear family, not a survey of classifiers.
Differences versus logistic regression are **not** attributed to federated averaging.

Verdict used for downstream recommendation: `model_family_dependent_under_the_tested_configurations`.

## Per-seed TL vs CF ROC-AUC

| seed | TL | CF | ΔAUC |
|---|---:|---:|---:|
| 11 | 0.9563 | 0.9198 | 0.0365 |
| 23 | 0.9614 | 0.8758 | 0.0856 |
| 42 | 0.9544 | 0.8891 | 0.0653 |
| 77 | 0.9629 | 0.9080 | 0.0549 |
| 101 | 0.9661 | 0.9472 | 0.0189 |

## Limitations

- One tree-ensemble configuration, not tuned, not federated.
- Same CICIDS2017 slice and five holdouts as the linear study.
- No manuscript change in this phase.

