# Phase B.1 experiment report

Zero-shot cross-dataset scoring of the existing Dirichlet non-IID
FedAvg global model (`alpha=0.5`, five clients, twenty rounds).
The camera-ready manuscript was not modified.

- Git commit: `ed880b51214a3b7906898b9086b5aa341600afbe`
- Baseline reproduction: **PASS** (540 checks, 0 failures)
- Checkpoint replay: **PASS**
- `results/verified/` hash snapshot unchanged: **True**

## A. Does non-IID FedAvg exhibit the TL-to-CF reversal?

Yes. TL mean ROC-AUC `0.7297`, CF mean ROC-AUC `0.3507`.
Ranking reversal (TL AUC > 0.5 and CF AUC < 0.5 on every seed): **True**.
Mean TL−CF ΔAUC `0.3791` (sample SD `0.0143`;
95% t-interval `[0.3613, 0.3969]`).
Mean TL−CF ΔMCC `0.4498` (sample SD `0.0153`).
Direction of ΔAUC is positive on all seeds: **True**.

The interval is a Student-t interval over five repeated holdouts from
one finite corpus. It is not evidence of population-level generalisation
across NIDS datasets.

## B. Comparable to IID FedAvg?

Yes in direction and ranking magnitude. IID mean TL−CF ΔAUC `0.3813`
(SD `0.0052`). IID reversal on all seeds: **True**.
Centralized mean TL−CF ΔAUC `0.3946`.
Non-IID ΔAUC is within `0.003` of IID and within `0.016` of centralized.
Direction and ranking-reversal pattern match IID and centralized: **True**.

Mean ROC-AUC comparison:

| model | TL | CF | MLCVE | TL−CF ΔAUC |
|---|---:|---:|---:|---:|
| FedAvg IID (accepted) | 0.7357 | 0.3545 | 0.3441 | 0.3813 |
| FedAvg non-IID (this run) | 0.7297 | 0.3507 | 0.3384 | 0.3791 |
| Centralized LR (accepted) | 0.7490 | 0.3544 | 0.3447 | 0.3946 |

## C. Does client heterogeneity change ranking, threshold metrics, or both?

It changes thresholded metrics more than ranking. On the source domain the
accepted paper already reports that Dirichlet allocation preserves ROC-AUC
while moving balanced accuracy and MCC. On the target views the ranking
reversal remains. Non-IID TL MCC is `0.2147` against IID `0.2511`; non-IID
CF MCC is `-0.2351` against IID `-0.1915`; non-IID TL balanced accuracy is
`0.6196` against IID `0.6403`. TL−CF ΔAUC stays within about `0.01` of the
IID gap. Heterogeneity therefore moves the 0.5-threshold operating point
without removing the ranking sign change.

## D. Mechanistic interpretation

The reversal persists under non-IID FedAvg and remains aligned with IID
FedAvg and centralized LR. That **strengthens** the current interpretation:
the matched reconstruction effect is a source-encoding collision, not an
IID-client or FedAvg-averaging artifact. It does not by itself prove a
protocol-only decision rule, and it does not extend the claim to trees,
neural models, or other dataset pairs.

## E. Reviewer 3 concern addressed

The accepted manuscript reports Dirichlet non-IID **source-domain**
metrics and states that the non-IID model is not scored on the target
views. Phase B.1 supplies that missing zero-shot evaluation. It answers
whether the TL-to-CF reversal is an IID-FedAvg peculiarity. It does not
answer whether a different FL algorithm would reverse, and it does not
make federated averaging a proposed defense.

## F. Camera-ready space

A three-number sentence (non-IID TL AUC `0.7297`, CF AUC `0.3507`, and
positive ΔAUC on all five seeds) can fit in six pages. A new table would
displace masking, the protocol crosstab, or the centralized control, which
are stronger mechanism evidence.

**Recommendation: retain separately.** Optionally add one sentence if space
remains after the Phase A material. Do not replace Table III.

## Per-seed non-IID zero-shot metrics

| seed | TL AUC | CF AUC | MLCVE AUC | TL MCC | CF MCC | ΔAUC | ΔMCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.7436 | 0.3556 | 0.3363 | 0.2218 | -0.2135 | 0.3880 | 0.4353 |
| 23 | 0.7144 | 0.3345 | 0.3243 | 0.2090 | -0.2374 | 0.3799 | 0.4464 |
| 42 | 0.7267 | 0.3554 | 0.3389 | 0.2287 | -0.2399 | 0.3713 | 0.4687 |
| 77 | 0.7127 | 0.3530 | 0.3430 | 0.1849 | -0.2777 | 0.3597 | 0.4626 |
| 101 | 0.7514 | 0.3549 | 0.3497 | 0.2292 | -0.2068 | 0.3965 | 0.4360 |

## Input hashes

- `insdn`: `c05f1f40dceb888aaaaf52b29283054147ed68390ea47997a8539e1505f64a7e`
- `trafficlabelling`: `4ff3075e61dea938892bda5bd90812170e085b225a87d1b0431b4e3c98d089bd`
- `tl_counterfactual`: `957b4170bda3a6f95648548196be35d473a36965a628e2b88580ae218f186b4c`
- `mlcve`: `af5f02ade640caa9870a310c7d95903c5b95a1d2a794c060d0c3932346530a03`

## Limitations

- One linear model, one source corpus, one CICIDS2017 slice, five holdouts.
- Non-IID here is Dirichlet label skew among five InSDN clients, not
  cross-silo feature heterogeneity or a different FL optimizer.
- MLCVE remains an unmatched ecological check.
- No manuscript change in this phase.
