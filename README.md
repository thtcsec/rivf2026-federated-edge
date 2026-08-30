# Two Distributions, Opposite Verdicts

Reproduction package for the IEEE RIVF 2026 submission by Hoang Tu Trinh and
Tien Thanh Cao. The repository contains only the logistic-regression FedAvg
audit reported in the paper; it makes no TCN/GRU, reinforcement-learning,
OpenFlow, latency, resource, or physical-deployment claim.

## Headline result

CICIDS2017 is published in two flow-feature distributions that expose different
columns. Holding the source model, seeds, preprocessing, and target traffic
fixed, and changing only which distribution supplied the target features, the
cross-dataset verdict reverses:

| Target mirror | Protocol / source port | ROC-AUC | MCC |
|---|---|---|---|
| `TrafficLabelling` | original columns | 0.7357 | +0.2511 |
| `MachineLearningCVE` | inferred from TCP flags / sentinel `-1` | 0.3441 | -0.2698 |

Masking localizes the reversal to protocol, whose contribution changes sign
(-0.1438 vs +0.2103 ROC-AUC) even though its marginal KS distance is the
*smaller* of the two.

A label-free screen (source-training statistics vs unlabelled target values)
flags the substituted port column on all five seeds with no false positive among
18 column-mirror pairs, but **cannot** flag the reconstructed protocol: it stays
in range with a plausible variance, and every distance-based rule we tested
ranks the harmful encoding as the closer one. Placeholder substitution is
mechanically detectable; semantic reconstruction has to be declared.

## What is evaluated

- centralized, five-client local-only, IID FedAvg, and Dirichlet non-IID
  FedAvg controls over five fixed seeds;
- leakage-safe source-fitted preprocessing;
- zero-shot transfer from InSDN to **both** documented CICIDS2017 mirrors,
  scored by the same models in a single run;
- port/protocol retraining ablations against both mirrors;
- a feature-level audit using two-sample KS distance and source-mean masking,
  reported per mirror; and
- a label-free provenance screen scored against that audit.

## Reproduce

Python 3.11 is recommended.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python evaluation/prepare_public_data.py --insdn-src /path/to/Dataset.csv
python evaluation/prepare_public_data.py \
  --cicids-root /path/to/cicids2017/TrafficLabelling \
  --cicids-out-name cicids2017_mirror
python evaluation/prepare_public_data.py \
  --cicids-root /path/to/cicids2017/MachineLearningCVE \
  --cicids-out-name cicids2017_mirror_mlcve

python evaluation/run_verified_federated_baseline.py
python evaluation/verify_paper_numbers.py
pdflatex -interaction=nonstopmode rivf2026.tex
pdflatex -interaction=nonstopmode rivf2026.tex
```

`verify_paper_numbers.py` re-reads `rivf2026.tex` and fails if any table cell,
prose figure, count, or published checksum disagrees with `results/verified/`.

Data licenses prevent redistribution. See `data/README.md` for source files,
feature conversion, per-file field provenance, expected rows, and SHA-256
checksums. Prepared files may also be supplied using `INSDN_CSV`,
`CICIDS2017_MIRROR_CSV`, and `CICIDS2017_MLCVE_MIRROR_CSV`.

## Outputs

The evaluation writes all computed artifacts under `results/verified/` and the
convergence figure under `paper/figures/verified/`. The machine-readable outputs
are `five_seed_metrics.csv`, `external_ablation_metrics.csv`,
`feature_shift_audit.csv`, `provenance_screen.csv`, and `summary.json`. All but
the first carry a `mirror` column or key so that every number can be traced to
one provenance.

## Scope and interpretation

Both mirrors contain only Monday benign, Friday DDoS, and Friday PortScan
traffic, so the external experiment is a feature-provenance interoperability
audit rather than a CICIDS2017 leaderboard. The below-random result on the
`MachineLearningCVE` mirror is evidence about harmonization, not a deployment
benchmark, and the paper does not present it as a property of the traffic.

## License

Code is released under the MIT License. InSDN and CICIDS2017 remain subject to
their respective owners' terms.
