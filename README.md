<p align="center">
  <img src="vinuni.png" height="72" alt="VinUniversity, venue of IEEE RIVF 2026" />
  <br />
  <sub>Conference venue. Authors are affiliated with HUFLIT, not VinUniversity.</sub>
</p>

<h1 align="center">Same Flows, Opposite Verdicts</h1>
<p align="center"><em>Feature reconstruction and provenance in cross-dataset intrusion detection</em></p>

<p align="center">
  <a href="https://rivf2026.org/"><img src="https://img.shields.io/badge/IEEE-RIVF%202026-003147.svg" alt="IEEE RIVF 2026" /></a>
  <a href="https://github.com/thtcsec/rivf2026-federated-edge/tree/rivf2026-camera-ready"><img src="https://img.shields.io/badge/artifact-rivf2026--camera--ready-2ea44f.svg" alt="Camera-ready artifact tag" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code%20license-MIT-blue.svg" alt="MIT license" /></a>
  <a href="evaluation/phase_b_claim_manifest.md"><img src="https://img.shields.io/badge/reproducibility-Level%20A%20verifier-informational.svg" alt="Level A result verification" /></a>
</p>

## Paper metadata

| Field | Record |
|---|---|
| **Title** | *Same Flows, Opposite Verdicts: Feature Reconstruction in Cross-Dataset Intrusion Detection* |
| **Authors** | Tu Hoang Trinh and Thanh Tien Cao* |
| **Affiliation** | Faculty of Information Technology, Ho Chi Minh City University of Foreign Languages — Information Technology |
| **Venue** | IEEE RIVF 2026, VinUniversity, Hanoi, 18–20 December 2026 |
| **Status** | Accepted; camera-ready candidate (EDAS #1571348135). No IEEE Xplore DOI yet. |
| **Track** | Cyber-Security, Cryptography, Blockchain |
| **Artifact tag** | [`rivf2026-camera-ready`](https://github.com/thtcsec/rivf2026-federated-edge/tree/rivf2026-camera-ready) |
| **Code license** | MIT (`LICENSE`). InSDN and CICIDS2017 remain under their owners' terms. |

\*Corresponding author: `thanhct@huflit.edu.vn`

## Overview

CICIDS2017 ships two flow-feature releases that do not expose the same columns. Aligning the MachineLearningCVE schema to a nine-field InSDN model therefore requires dropping Protocol and Source Port or reconstructing them. This artifact evaluates **one specified reconstruction** as a controlled failure mode, not as documented practice.

Federated averaging is the evaluation setting, not the identified cause. Integer protocol codes are empirical labels, not IANA identities.

## Headline comparison

Same InSDN-trained FedAvg IID logistic-regression models, five seeds, no target refit:

| Target view | Protocol / source port | Row-matched | ROC-AUC | MCC |
|---|---|---|---|---|
| `TL` original TrafficLabelling | original columns | — | **0.7357** | **+0.2511** |
| `CF` counterfactual | flag surrogate / sentinel `-1` | **yes** | **0.3545** | **−0.1915** |
| `MLCVE` real release | flag surrogate / sentinel `-1` | **no** | 0.3441 | −0.2698 |

`CF` holds a 207,463-flow pool with labels, prevalence, and seven features fixed and overwrites only two encodings. Each seed scores a stratified 30,000-row sample of that pool. That matched contrast is the causal comparison. `MLCVE` is an unmatched ecological check: it corroborates that a public release can land near the counterfactual, but it is **not** row-matched and is **not** the identified mechanism.

A centralized LR model reverses on the same matched views (0.7490 → 0.3544), so FedAvg aggregation is not required for the sign change.

## Experimental design

```text
InSDN source training (no target labels)
        │  zero-shot scores
        ▼
   ┌────┴──────────────┬────────────────────┐
   │                   │                    │
  TL                  CF                 MLCVE
 original          reconstruction      real release
 row-matched  ←── identical rows ──→   unmatched
                                      ecological
```

- **Source.** Models are trained on InSDN only. Target labels are never used for training, scaling, or threshold selection in the zero-shot evaluation.
- **TL vs CF.** Same rows, same order, same labels. Only `ip_proto` and `tp_src` differ. This is the primary result.
- **MLCVE.** Independent public CSV distribution. Different row set and prevalence. Not a matched intervention.

## Datasets and provenance

Third-party CSVs are **not** in Git. Prepare them with `evaluation/prepare_public_data.py` and the checksums in `data/README.md`. InSDN is the Hugging Face third-party derivative `Sharukesh/INSDN` (revision `f59330c8…`), not an authenticated publisher dump. CICIDS2017 was ingested from Parquet mirrors of the UNB CSVs; filenames are in each `dataset_summary.json`.

| File | Role | Rows | SHA-256 prefix |
|---|---|---|---|
| `data/insdn/flow_stats.csv` | source | 343,889 | `c05f1f40…` |
| `data/cicids2017_mirror/test.csv` | TL | 207,463 | `4ff3075e…` |
| `data/cicids2017_mirror_counterfactual/test.csv` | CF | 207,463 | `957b4170…` |
| `data/cicids2017_mirror_mlcve/test.csv` | MLCVE | 176,036 | `af5f02ad…` |

When Protocol is absent, the specified rule writes 6 if any TCP flag is positive and 0 otherwise; every reconstructed source port is `-1`. The script records the branch taken in `dataset_summary.json`.

## Key robustness findings

Limited to the evaluated models, seeds, and this CICIDS2017 slice:

- Protocol-only replacement reverses the tested LR (factorial 0.7357 → 0.3365). Source-port substitution does not (0.7522).
- Dirichlet non-IID FedAvg (`α=0.5`) also reverses on the matched views (0.7297 / 0.3507).
- A **centralized** nine-field categorical HistGradientBoostingClassifier does **not** reverse (0.9602 / 0.9080). It is not a federated model and is not capacity-matched to LR.
- Source-retrained seven-field LR transfers near chance (0.5075). A different source-retrained seven-field HGB remains strong (0.9603 / 0.9592).
- These results do not imply that every reconstruction fails, or that the reversal holds for every classifier.

The claim-to-file map is `evaluation/phase_b_claim_manifest.md`.

## Repository structure

```text
rivf2026-federated-edge/
├── rivf2026.tex / rivf2026.pdf   # IEEE A4 camera-ready manuscript
├── evaluation/                   # runners, verifier, tests
├── results/verified/             # frozen original LR outputs
├── results/phase_b/              # non-IID, factorial, HGB, missingness
├── data/README.md                # provenance and checksums (no CSVs)
├── CITATION.cff
└── ARTIFACT.md
```

## Level A — result verification (no third-party data)

A clean checkout can check every quoted number against **tracked** files under `results/verified/` and `results/phase_b/`. Prepared CSVs are optional; when they are absent the verifier skips row-matching, protocol-label audit, and checksums instead of reporting those checks as passing.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python evaluation/verify_paper_numbers.py
python -m unittest discover -s evaluation -p "test_*.py"
```

`results/verified/summary.json` is the frozen original (Python `NaN` for undefined TV diagnostics). `results/verified/summary.strict.json` is an RFC 8259 copy with `null` in those positions. Do not overwrite the original.

## Level B — experimental reproduction

Requires the four prepared CSVs (or the public InSDN and CICIDS2017 releases). After checksums match `data/README.md`:

```bash
python evaluation/run_verified_federated_baseline.py
python evaluation/run_phase_b_noniid_cross.py
python evaluation/run_phase_b_mechanism.py      # replays tracked checkpoints
python evaluation/run_phase_b_robustness.py     # retrains centralized HGB
python evaluation/run_phase_b_missingness.py
python evaluation/verify_paper_numbers.py
```

Phase B must write only under `results/phase_b/`. B.2 and some missingness policies replay the 30 tracked checkpoints; B.1, B.3, and B.4 retrain source-only models.

## Limitations

One source corpus, one CICIDS2017 slice, five repeated holdouts, one reconstruction rule, and the tested LR/HGB configurations. Near-ceiling source HGB AUC is not external validity. About 34.86% of source-test rows match a training feature vector; that overlap is label-consistent and is not target-label leakage.

## Citation

```bibtex
@inproceedings{trinh2026sameflows,
  author    = {Tu Hoang Trinh and Thanh Tien Cao},
  title     = {Same Flows, Opposite Verdicts: Feature Reconstruction in Cross-Dataset Intrusion Detection},
  booktitle = {Proc. IEEE RIVF},
  year      = {2026},
  note      = {Accepted camera-ready manuscript. Artifact: https://github.com/thtcsec/rivf2026-federated-edge/tree/rivf2026-camera-ready}
}
```

Machine-readable citation: [`CITATION.cff`](CITATION.cff). Evaluation protocol: [`ARTIFACT.md`](ARTIFACT.md).

## License

Code is MIT. InSDN and CICIDS2017 must be obtained from their original providers and must not be redistributed from this repository.
