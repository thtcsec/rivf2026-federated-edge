# Two Distributions, Opposite Verdicts

Reproduction package for the IEEE RIVF 2026 submission by Hoang Tu Trinh and
Tien Thanh Cao. The repository contains only the logistic-regression FedAvg
audit reported in the paper; it makes no TCN/GRU, reinforcement-learning,
OpenFlow, latency, resource, or physical-deployment claim.

## Headline result

CICIDS2017 is published in two flow-feature distributions that expose different
columns. The primary experiment is a **row-matched counterfactual**: the same
207,463 target flows in the same order, with identical labels, identical attack
prevalence and seven bit-identical features, in which only `Protocol` and
`Source Port` are replaced by the surrogates the `MachineLearningCVE` release
forces. That is enough to reverse the cross-dataset verdict.

| Target view | Protocol / source port | Row-matched | ROC-AUC | MCC |
|---|---|---|---|---|
| `TL` original | original columns | — | 0.7357 | +0.2511 |
| `CF` counterfactual | flag surrogate / sentinel `-1` | **yes** | 0.3545 | -0.1915 |
| `ML` real release | flag surrogate / sentinel `-1` | no | 0.3441 | -0.2698 |

Because `CF` shares its rows, labels and prevalence with `TL`, the difference
cannot be attributed to attack mix, labeling, collection environment or
sampling. The real `ML` release lands next to the counterfactual, so the
mechanism isolated on matched rows is what dominates in an actual public
release. A centralized model shows the same pattern (0.7490 / 0.3544 / 0.3447),
so this is not an artifact of FedAvg aggregation.

Masking localizes the reversal to protocol, whose contribution changes sign
(-0.1438 on `TL` vs +0.2647 on `CF`) even though its marginal KS distance is the
*smaller* of the two. On those identical rows, **dropping** the unavailable field
reaches 0.5157 while **reconstructing** it reaches only 0.3545: incorrect
reconstruction is worse than admitting the field is missing.

A label-free preflight check (source-training statistics vs unlabelled target
values) flags the substituted port on all five seeds and nothing else among the
27 column-view pairs, but it does **not** flag the reconstructed protocol, which
stays in range with a plausible variance. None of the marginal distance rules we
tested separates the two protocol encodings. Placeholder substitution is
statistically obvious; semantic reconstruction has to be declared.

## What is evaluated

- centralized, five-client local-only, IID FedAvg, and Dirichlet non-IID
  FedAvg controls over five fixed seeds;
- leakage-safe source-fitted preprocessing;
- zero-shot transfer from InSDN to **all three** CICIDS2017 views, scored by the
  same models in a single run, plus the centralized model on the same views;
- port/protocol retraining ablations against every view, which quantify dropping
  a field against reconstructing it;
- a feature-level audit using two-sample KS distance and source-mean masking,
  reported per view; and
- a label-free preflight provenance check, reported against that audit.

The counterfactual view is asserted to be row-matched at three independent
points: when it is written, after seeded subsampling in the evaluation, and
against the committed CSVs in `verify_paper_numbers.py`.

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
  --cicids-out-name cicids2017_mirror \
  --cicids-counterfactual-name cicids2017_mirror_counterfactual
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
`CICIDS2017_MIRROR_CSV`, `CICIDS2017_COUNTERFACTUAL_CSV`, and
`CICIDS2017_MLCVE_MIRROR_CSV`.

## Outputs

The evaluation writes all computed artifacts under `results/verified/`. The
machine-readable outputs are `five_seed_metrics.csv`,
`external_ablation_metrics.csv`, `feature_shift_audit.csv`,
`provenance_screen.csv`, and `summary.json`. All but the first carry a `mirror`
column or key so that every number can be traced to one provenance.

## Scope and interpretation

All three views contain only Monday benign, Friday DDoS, and Friday PortScan
traffic, so the external experiment is a feature-provenance interoperability
audit rather than a CICIDS2017 leaderboard. The below-random results are evidence
about harmonization, not a deployment benchmark, and the paper does not present
them as a property of the traffic. The counterfactual establishes what these two
fields do to this model on these rows; it does not establish that provenance is
the largest source of cross-dataset error in general.

## License

Code is released under the MIT License. InSDN and CICIDS2017 remain subject to
their respective owners' terms.
