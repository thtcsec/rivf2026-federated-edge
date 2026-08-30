# Artifact evaluation guide

**Paper:** *Two Distributions, Opposite Verdicts: Feature Provenance in
Cross-Dataset Federated Intrusion Detection*

**Target track:** Cyber-Security, Cryptography, Blockchain

## Claims supported by this artifact

1. Five-seed centralized, local-only, IID FedAvg, and Dirichlet non-IID
   logistic-regression controls on InSDN, with no FedAvg accuracy advantage.
2. A row-matched counterfactual target view: the same 207,463 `TrafficLabelling`
   test rows in the same order, with identical labels, identical attack
   prevalence and seven bit-identical features, differing only in `ip_proto` and
   `tp_src`. Scoring the same five source models on it moves ROC-AUC from 0.7357
   to 0.3545 and MCC from +0.2511 to -0.1915.
3. Ecological validation on the real `MachineLearningCVE` release (ROC-AUC
   0.3441 / MCC -0.2698) and a centralized control on all three views
   (0.7490 / 0.3544 / 0.3447), showing the effect is neither an artifact of the
   counterfactual nor of federated averaging.
4. A source-mean masking audit and retraining ablations that localize the
   reversal to protocol, whose learned contribution changes sign between
   row-matched views while its marginal KS distance is the smaller of the two,
   and that quantify dropping the unavailable field (0.5157) against
   reconstructing it (0.3545). The schema-intersection setting, which drops both
   fields `MachineLearningCVE` does not supply, reaches 0.5075 and returns the
   same five-seed metrics on TL and CF to four decimals, since removing those two
   columns makes the two views the same data.
5. A label-free preflight check using only source-training statistics and
   unlabelled target values. It flags exactly the two substituted source-port
   columns among 27 column--view pairs, on all five seeds, and does not flag the
   reconstructed protocol; the rejected mass-collapse rule fires on all three
   views, demonstrating that it cannot discriminate.
6. Full regeneration of all prepared views from public source files, with
   per-file field provenance recorded and published SHA-256 checksums.

No deployment, latency, privacy guarantee, secure aggregation, deep model,
reinforcement-learning, or SDN-actuation claim is supported or made.

## Evaluation steps

1. Follow `data/README.md` to prepare or verify the four prepared files. Note
   that the CICIDS2017 side requires **both** the `TrafficLabelling` and the
   `MachineLearningCVE` release, because their difference is the object of
   study; the counterfactual view is derived from the former in the same pass.
2. Verify the SHA-256 checksums in `data/README.md`. Also check
   `field_provenance` in each `dataset_summary.json`: the TL view must report
   `original Protocol column` and `original Source Port column`, the ML view must
   report the inferred and sentinel branches, and the CF view must report
   `row_matched: true` with `columns_changed` equal to `[ip_proto, tp_src]`.
3. Install the pinned packages in `requirements.txt`.
4. Run `python evaluation/run_verified_federated_baseline.py`. A single run
   scores all three views under both the federated and the centralized model,
   and aborts if the counterfactual is not row-matched after subsampling.
5. Run `python evaluation/verify_paper_numbers.py`. It re-reads `rivf2026.tex`
   and asserts that every table cell, prose scalar, count, prevalence, and
   confusion-matrix entry matches `results/verified/`; that all four
   prepared-file SHA-256 digests are the ones published in `data/README.md`; and
   that the committed counterfactual CSV really differs from the TL CSV in
   exactly `ip_proto` and `tp_src` with identical labels and prevalence. It exits
   non-zero on the first mismatch, so a stale table cannot survive a rerun
   unnoticed.
6. For manual inspection, the machine-readable outputs are
   `results/verified/five_seed_metrics.csv`,
   `results/verified/external_ablation_metrics.csv`,
   `results/verified/feature_shift_audit.csv`, and
   `results/verified/provenance_screen.csv`; all but the first are keyed by a
   `mirror` column whose values are `trafficlabelling`, `tl_counterfactual` and
   `mlcve`.
7. Build the PDF twice with `pdflatex`.

The experiment is deterministic for the specified package versions and seeds.
Report platform-dependent floating-point differences rather than silently
overwriting the committed reference outputs.
