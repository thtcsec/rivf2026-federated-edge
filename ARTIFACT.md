# Artifact evaluation guide

**Paper:** *Same Flows, Opposite Verdicts: Feature Reconstruction in
Cross-Dataset Intrusion Detection*

**Authors (IEEE author block):** Tu Hoang Trinh and Thanh Tien Cao

**Status:** Accepted IEEE RIVF 2026 camera-ready candidate (EDAS #1571348135).
The EDAS display name may still require author confirmation. No IEEE Xplore DOI
has been assigned.

**Target track:** Cyber-Security, Cryptography, Blockchain

## Claims supported by this artifact

1. Five-seed centralized, local-only, IID FedAvg, and Dirichlet non-IID
   logistic-regression controls on InSDN, with no FedAvg accuracy advantage.
2. A row-matched counterfactual target view: the same 207,463-row
   `TrafficLabelling` **pool** in the same order, with identical labels, identical
   attack prevalence and seven bit-identical features, differing only in
   `ip_proto` and `tp_src`. Each seed scores a stratified **30,000-row sample**
   of that pool. Scoring the same five source models on those samples moves
   ROC-AUC from 0.7357 to 0.3545 and MCC from +0.2511 to -0.1915.
3. Ecological validation on the real `MachineLearningCVE` release (ROC-AUC
   0.3441 / MCC -0.2698) and a centralized control on all three views
   (0.7490 / 0.3544 / 0.3447), showing the effect is neither an artifact of the
   counterfactual nor of federated averaging.
4. A source-mean masking audit and retraining ablations that localize the
   ranking reversal to protocol (`ΔA` changes sign between row-matched views)
   while protocol total variation is *smaller* for the harmful encoding. Source-port
   masking *gains* 0.0165 ROC-AUC on TL. Dropping the unavailable protocol field
   reaches 0.5157 against 0.3545 for reconstruction. The seven-field intersection
   is provenance-valid but chance-level (0.5075) and returns identical TL/CF
   metrics, confirming those views differ only in the two overwritten columns.
5. A target-trained within-domain logistic-regression control (its own scaler,
   an 80/20 split of that view's sampled rows), not an information-theoretic
   bound: 0.8995 +- 0.0553 on TL, 0.8820 +- 0.0630 on CF and 0.8956 +- 0.0635 on
   MLCVE for the nine-field schema, and 0.7200 +- 0.0931 (TL and CF) against
   0.7117 +- 0.0797 (MLCVE) for the seven-field intersection. The paired
   in-domain gap is 0.0175 (95% CI [0.0069, 0.0280], TL ahead on all five seeds)
   against a zero-shot gap of 0.3813 +- 0.0052.
6. A label-free preflight check using only source-training statistics and
   unlabelled target values. It flags exactly the two substituted source-port
   columns among 27 column--view pairs, on all five seeds, and does not flag the
   reconstructed protocol; the rejected mass-collapse rule fires on all three
   views, demonstrating that it cannot discriminate.
7. Full regeneration of all prepared views from public source files, with
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
   confusion-matrix entry matches `results/verified/`; that quoted Phase B
   numbers match `results/phase_b/`; and, when the prepared CSVs are present,
   that all four SHA-256 digests are the ones published in `data/README.md` and
   that the counterfactual CSV differs from the TL CSV in exactly `ip_proto` and
   `tp_src` with identical labels and prevalence. Without those CSVs, checksum
   and row-match checks are skipped (EXTERNAL DATA REQUIRED) rather than
   reported as passing. The script exits non-zero on the first mismatch.
6. For manual inspection, the machine-readable outputs are
   `results/verified/five_seed_metrics.csv`,
   `results/verified/external_ablation_metrics.csv`,
   `results/verified/feature_shift_audit.csv`, and
   `results/verified/provenance_screen.csv`; all but the first are keyed by a
   `mirror` column whose values are `trafficlabelling`, `tl_counterfactual` and
   `mlcve`. Quoted Phase B keys are listed in
   `evaluation/phase_b_claim_manifest.md`.
7. Build the PDF twice with `pdflatex`.
8. Optional Level B: after checksums match, run the four `run_phase_b_*.py`
   scripts. They must not write `results/verified/`. The 30 tracked checkpoints
   under `results/phase_b/checkpoints/` are required for B.2 replay and for LR
   missingness policies that reuse the nine-field models.

The experiment is deterministic for the specified package versions and seeds.
Report platform-dependent floating-point differences rather than silently
overwriting the committed reference outputs.

Historical Phase B manifests store a git commit and prepared-file hashes. They
do not store executed-source hashes and are not a timestamped preregistration.
Re-running a current script against frozen outputs is a current reproduction,
not proof of the original runtime tree. Do not rewrite those manifests to look
more complete. The leakage audit reports source-split overlap; it does not
compute an unseen-row source AUC. Frozen JSON may still contain an older
qualitative sentence to that effect; that sentence is historical, not a
calculated metric.

`results/verified/summary.json` is the frozen original. It uses Python's
default JSON encoder, so undefined total-variation diagnostics on continuous
features are stored as bare `NaN` tokens. Those tokens are not ROC-AUC values.
A RFC 8259 copy with `null` in those positions is
`results/verified/summary.strict.json`; generating it must not overwrite the
original. Finite metrics in the two files are numerically identical.
