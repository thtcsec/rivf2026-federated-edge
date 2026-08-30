# Artifact evaluation guide

**Paper:** *Two Distributions, Opposite Verdicts: Feature Provenance in
Cross-Dataset Federated Intrusion Detection*

**Target track:** Cyber-Security, Cryptography, Blockchain

## Claims supported by this artifact

1. Five-seed centralized, local-only, IID FedAvg, and Dirichlet non-IID
   logistic-regression controls on InSDN, with no FedAvg accuracy advantage.
2. Zero-shot evaluation of the same five source models on two CICIDS2017
   mirrors built from the same capture files, without target-label fitting,
   yielding ROC-AUC 0.7357 / MCC +0.2511 and ROC-AUC 0.3441 / MCC -0.2698
   respectively.
3. A source-mean masking audit and retraining ablations that localize the
   reversal to protocol, whose learned contribution changes sign between the
   two mirrors while its marginal KS distance is the smaller of the two.
4. A label-free screen that uses only source-training statistics and unlabelled
   target values. It flags exactly one of 18 column--mirror pairs, the
   `MachineLearningCVE` source port, on all five seeds, and it does not flag the
   reconstructed protocol; the rejected mass-collapse rule fires on both
   mirrors, demonstrating that it cannot discriminate.
5. Full regeneration of both prepared mirrors from public source files, with
   per-file field provenance recorded and published SHA-256 checksums.

No deployment, latency, privacy guarantee, secure aggregation, deep model,
reinforcement-learning, or SDN-actuation claim is supported or made.

## Evaluation steps

1. Follow `data/README.md` to prepare or verify the three prepared files. Note
   that the CICIDS2017 side requires **both** the `TrafficLabelling` and the
   `MachineLearningCVE` release, because their difference is the object of
   study.
2. Verify the SHA-256 checksums in `data/README.md`. Also check
   `field_provenance` in each `dataset_summary.json`: the TL mirror must report
   `original Protocol column` and `original Source Port column`, and the ML
   mirror must report the inferred and sentinel branches.
3. Install the pinned packages in `requirements.txt`.
4. Run `python evaluation/run_verified_federated_baseline.py`. A single run
   scores both mirrors.
5. Run `python evaluation/verify_paper_numbers.py`. It re-reads `rivf2026.tex`
   and asserts that every table cell, prose scalar, count, prevalence, and
   confusion-matrix entry matches `results/verified/`, and that all three
   prepared-file SHA-256 digests are the ones published in `data/README.md`. It
   exits non-zero on the first mismatch, so a stale table cannot survive a
   rerun unnoticed.
6. For manual inspection, the machine-readable outputs are
   `results/verified/five_seed_metrics.csv`,
   `results/verified/external_ablation_metrics.csv`,
   `results/verified/feature_shift_audit.csv`, and
   `results/verified/provenance_screen.csv`; all but the first are keyed by a
   `mirror` column.
7. Build the PDF twice with `pdflatex`.

The experiment is deterministic for the specified package versions and seeds.
Report platform-dependent floating-point differences rather than silently
overwriting the committed reference outputs.
