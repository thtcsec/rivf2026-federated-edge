# Camera-ready claim map (Phase B numbers)

Status snapshot for RIVF 2026 #1571348135. This file does **not** publish data.

| Layer | State |
|---|---|
| Camera-ready tag `rivf2026-camera-ready` | Integrated manuscript + Phase B scripts/results |
| Public remote `origin/main` | Accepted submission until the camera-ready PR is merged |

The versioned artifact URL is
https://github.com/thtcsec/rivf2026-federated-edge/tree/rivf2026-camera-ready
A local commit is not that public tag until the tag is pushed.

## Quoted claims

| Claim in `rivf2026.tex` | Source file / key | Script | Data | Git status | Public GitHub |
|---|---|---|---|---|---|
| IID FedAvg TL/CF 0.7357/0.3545 | `results/verified/summary.json` `summary.cross_*` | `run_verified_federated_baseline.py` | prepared CSVs | tracked | yes (`main`) |
| Non-IID TL/CF 0.7297/0.3507; ΔAUC 0.3791 | `results/phase_b/aggregate_noniid_cross.json` `views.*.roc_auc.mean`; `paired_effects.json` `fedavg_noniid.tl_minus_cf.roc_auc.mean` | `run_phase_b_noniid_cross.py` | prepared CSVs | tracked | tag `rivf2026-camera-ready` |
| LR factorial A/B/C/D 0.7357/0.3365/0.7522/0.3545 | `results/phase_b/mechanism/factorial_summary.json` `models.fedavg_iid.conditions_roc_auc` | `run_phase_b_mechanism.py` (replays B.1 checkpoints) | prepared CSVs + `results/phase_b/checkpoints/` | tracked | tag `rivf2026-camera-ready` |
| Nine-field HGB TL/CF/MLCVE 0.9602/0.9080/0.9073 | `results/phase_b/robustness/nonlinear_summary.json` `zero_shot.*.roc_auc.mean` | `run_phase_b_robustness.py` | prepared CSVs | tracked | tag `rivf2026-camera-ready` |
| Nine-field HGB factorial B/C 0.9601/0.9096 | same JSON `factorial.conditions_roc_auc` | `run_phase_b_robustness.py` | prepared CSVs | tracked | tag `rivf2026-camera-ready` |
| Nine-field HGB source 0.9999 | same JSON `source.roc_auc.mean` | `run_phase_b_robustness.py` | prepared CSVs | tracked | tag `rivf2026-camera-ready` |
| Seven-field retrained HGB 0.9603 / MLCVE 0.9592 | `results/phase_b/missingness/policy_comparison_summary.json` `by_model_policy.hgb_categorical.P2` | `run_phase_b_missingness.py` | prepared CSVs | tracked | tag `rivf2026-camera-ready` |
| Source-test overlap 34.86% | same JSON `leakage.mean_test_overlap_frac` | `run_phase_b_missingness.py` | prepared CSVs | tracked | tag `rivf2026-camera-ready` |
| LR seven-field 0.5075 | `results/verified/summary.json` `ablation_intersection_cross_trafficlabelling` | `run_verified_federated_baseline.py` | prepared CSVs | tracked | yes (`main`) |

Prepared CSV SHA-256 values are listed in `data/README.md`. Scripts require Python 3.11, NumPy, pandas, and scikit-learn 1.7.2 as pinned in the manuscript. Phase B must not write `results/verified/`.

## Level A versus Level B

- **Level A (result verification).** A clean checkout of this freeze can check every quoted number against tracked JSON/CSV under `results/verified/` and `results/phase_b/` without raw third-party datasets. `evaluation/verify_paper_numbers.py` skips prepared-CSV row-matching, checksums, and the protocol-label audit when those files are absent.
- **Level B (experimental reproduction).** Regenerating models requires the four prepared CSVs (or the public source corpora plus `evaluation/prepare_public_data.py`). B.2 and LR P0/P1/P3 replay the 30 tracked checkpoints (`results/phase_b/checkpoints/`, about 40 KB). B.1/B.3/B.4 retraining is deterministic under the pinned environment and the documented seeds; it is not required for Level A.

## Not quoted (local artifact only)

Full B.4 imputation matrix, AUROC pair decomposition, numeric HGB, and LR P3/P4. These remain under `results/phase_b/` in this freeze.

## Unseen-row source HGB AUC

The frozen leakage object records `mean_test_overlap_frac` and a qualitative verdict that seven-feature HGB remains near-perfect on unseen source rows. It does **not** store a separate unseen-row ROC-AUC. No such number is quoted in the manuscript.
