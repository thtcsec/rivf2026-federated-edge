# Phase B.4 missing-feature policy report

Controlled comparison of source-only missingness policies when Protocol
and Source Port are unavailable in a target release. **HGB is not a
federated model.** Policies and the masking probability were pre-registered
and were not changed after seeing target scores. The manuscript was not edited.

- Git commit: `17e1bcdb4b1727a4be3c71085a1a45f4b0f96e09`
- sklearn: `1.7.2`
- Pre-registered P4 mask probability: `0.5` (independent per field)
- Leakage audit: Repeated identical InSDN flow-stat vectors exist and appear in both source splits. Duplicate groups are label-consistent. This inflates source AUC but is not target-label leakage. Seven-feature HGB remains near-perfect on unseen source rows; the source task is easy for trees.

## Policy feasibility

| policy | MLCVE-feasible | retrains | uses original TL proto/port | handling |
|---|---|---|---|---|
| P0 | False | False | True | `none_fields_observed` |
| P1 | True | False | False | `flag_inferred_protocol_and_port_sentinel` |
| P2 | True | True | False | `drop_unavailable_columns` |
| P3 | True | False | False | `source_train_protocol_mode_and_port_median` |
| P4 | True | True | False | `synthetic_mcar_then_100pct_unavailable_at_target` |

## Mean ROC-AUC (five seeds)

| model | P0 TL | P1 CF | P2 TL=CF | P3 TL | P4 TL | P1 MLCVE | P2 MLCVE | P3 MLCVE | P4 MLCVE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lr_fedavg_iid | 0.7357 | 0.3545 | 0.5075 | 0.6191 | 0.5440 | 0.3441 | 0.4505 | 0.5543 | 0.4900 |
| lr_centralized | 0.7490 | 0.3544 | 0.4770 | 0.6137 | 0.5178 | 0.3447 | 0.4283 | 0.5509 | 0.4689 |
| hgb_categorical | 0.9602 | 0.9080 | 0.9603 | 0.7147 | 0.9629 | 0.9073 | 0.9592 | 0.7883 | 0.9612 |
| hgb_numeric | 0.9605 | 0.8654 | 0.9603 | 0.7139 | 0.9633 | 0.8812 | 0.9592 | 0.7882 | 0.9618 |

## Source-domain ROC-AUC

| model | P0/P1/P3 observed | P2 seven-field | P4 observed encoding |
|---|---:|---:|---:|
| lr_fedavg_iid | 0.9324 | 0.8954 | 0.9147 |
| lr_centralized | 0.9320 | 0.9041 | 0.9233 |
| hgb_categorical | 0.9999 | 0.9993 | 0.9998 |
| hgb_numeric | 0.9999 | 0.9993 | 0.9998 |

## Pre-registered questions

1. Can any source-only missingness policy avoid the LR below-chance result? **P1 reconstruction remains below chance (FedAvg CF AUC 0.3545, MCC -0.1915). P2 is chance-level ranking (AUC 0.5075, MCC +0.0491). P4 is near chance (AUC 0.5440, MCC +0.0505). P3 raises ROC-AUC to 0.6191 but MCC stays negative (-0.1826).**
2. If yes, useful ranking vs chance? **No tested LR missingness policy is a useful detector. P3 is above-chance ranking with still-harmful thresholded decisions. P2/P4 are chance-level.**
3. Does P4 differ from P2 (drop-and-retrain)? **P2 vs P4 |ΔAUC| on FedAvg IID TL is 0.0365. Differences are reported, not selected.**
4. MLCVE-feasible policies: **P1, P2, P3, P4. P0 is not MLCVE-feasible.**
5. Does HGB remain robust under each policy? **P0/P1/P2/P4 remain strong (P2 0.9603, P4 0.9629, P1 CF 0.9080). P3 source-stat imputation degrades HGB to 0.7147.**
6. HGB robustness: family, representation, or both? **Classifier family dominates under these configurations: numeric and categorical HGB CF stay above chance (0.8654 vs 0.9080), unlike one-hot or numeric LR. Categorical protocol shrinks the HGB TL–CF gap (0.0522 vs numeric 0.0951) but is not required to avoid a reversal.**
7. Cross-dataset shortcuts? **Protocol remains a documented LR shortcut. HGB seven-field source separability shows InSDN volume features are also highly informative in-domain; that does not restore LR target ranking.**
8. Supports camera-ready claims? **The accepted reconstruction reversal for numeric/one-hot LR, and the chance-level seven-field LR intersection, are reproduced.**
9. Narrows camera-ready claims? **P2/P4 avoid the LR reconstruction reversal only by returning to chance. P3 improves ROC-AUC but not MCC. HGB stays strong if the fields are dropped or marked missing, and weakens under source-stat imputation. That qualifies the LR reversal as representation- and family-sensitive, not a universal missingness result.**
10. Journal vs six-page paper? **The full policy matrix, P4 train/target missingness mismatch, and HGB numeric-vs-categorical control belong in a journal extension. The six-page camera-ready should keep the locked LR reconstruction result.**

One-hot LR (accepted ablation) remains TL 0.7472 / CF 0.3630; categorical
encoding inside logistic regression did not remove the reversal. HGB
categorical vs numeric is the within-family representation control.

A better target AUC is an observed outcome under pre-registered policies,
not a validated deployment-time selection rule.

## Limitations

- P4 trains with missingness rate 0.5 and is applied to 100% missing target fields.
- Source InSDN is nearly separable for trees even on unseen rows.
- One HGB configuration and the original LR optimizer; not a survey.
- No manuscript change in this phase.

