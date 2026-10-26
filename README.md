# Same Flows, Opposite Verdicts

Reproduction package for the IEEE RIVF 2026 camera-ready candidate
*Same Flows, Opposite Verdicts: Feature Reconstruction in Cross-Dataset
Intrusion Detection* by Hoang Tu Trinh and Tien Thanh Cao. The working title
may require EDAS/TPC confirmation. This freeze contains the logistic-regression
FedAvg audit plus the Phase B robustness controls quoted in the camera-ready
manuscript. It makes no TCN/GRU, reinforcement-learning, OpenFlow, latency,
resource, or physical-deployment claim. Do not treat integer protocol codes as
established IANA identities. The public GitHub `main` branch still hosts the
accepted submission until this commit is pushed.

## Headline result

CICIDS2017 is published in two flow-feature releases that expose different
columns. The primary experiment is a **row-matched counterfactual**: the same
207,463 target flows in the same order, with identical labels, identical attack
prevalence and seven bit-identical features, in which only `Protocol` and
`Source Port` are replaced by one specified MachineLearningCVE-compatible
surrogate. That reconstruction is a controlled failure mode, not documented
practice. It reverses the cross-dataset ranking of an InSDN-trained linear model.

| Target view | Protocol / source port | Row-matched | ROC-AUC | MCC |
|---|---|---|---|---|
| `TL` original | original columns | — | 0.7357 | +0.2511 |
| `CF` counterfactual | flag surrogate / sentinel `-1` | **yes** | 0.3545 | -0.1915 |
| `MLCVE` real release | flag surrogate / sentinel `-1` | no | 0.3441 | -0.2698 |

Because `CF` shares its rows, labels and prevalence with `TL`, the difference
cannot be attributed to attack mix, labeling, collection environment or
sampling. The real `MLCVE` release lands next to the counterfactual, so the
mechanism isolated on matched rows is what dominates in an actual public
release. A centralized model shows the same pattern (0.7490 / 0.3544 / 0.3447),
so this is not an artifact of FedAvg aggregation.

In this CICIDS2017 slice every code-17 (UDP-coded) flow is benign, while InSDN
associates code 0 with 99.7% attacks; every protocol change in `CF` is `17→0`.
Three PortScan rows have code 0; they are not recoded.
Masking localizes the ranking reversal to protocol (`ΔA = -0.1438` on `TL` vs
`+0.2647` on `CF`). Source-port masking *gains* 0.0165 ROC-AUC on `TL`; it does
not cost ranking quality. Protocol total variation is smaller for the harmful
encoding than for the original. On those identical rows, **dropping** the unavailable field
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
  a field against reconstructing it, including the schema-intersection setting
  that drops both fields the `MachineLearningCVE` release does not supply;
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

**Level A (no third-party data).** A clean checkout can check every quoted
number against tracked files under `results/verified/` and `results/phase_b/`
with `python evaluation/verify_paper_numbers.py`. Prepared-CSV checksums and
the protocol-label audit are skipped when those files are absent. See
`evaluation/phase_b_claim_manifest.md`.

**Level B (experimental reproduction).** Regenerating models needs the four
prepared CSVs. After checksums match `data/README.md`:

```bash
python evaluation/run_phase_b_noniid_cross.py
python evaluation/run_phase_b_mechanism.py
python evaluation/run_phase_b_robustness.py
python evaluation/run_phase_b_missingness.py
```

Phase B writes only under `results/phase_b/` and must not overwrite
`results/verified/`. B.2 replays the tracked B.1 checkpoints; it does not
retrain the nine-field LR models.

`verify_paper_numbers.py` re-reads `rivf2026.tex` and fails if any table cell,
prose figure, count, or published checksum disagrees with `results/verified/`
and the quoted Phase B JSON keys.

Data licenses prevent redistribution. See `data/README.md` for source files,
feature conversion, per-file field provenance, expected rows, and SHA-256
checksums. Prepared files may also be supplied using `INSDN_CSV`,
`CICIDS2017_MIRROR_CSV`, `CICIDS2017_COUNTERFACTUAL_CSV`, and
`CICIDS2017_MLCVE_MIRROR_CSV`.

## Outputs

The original evaluation writes under `results/verified/`. The machine-readable
outputs are `five_seed_metrics.csv`, `external_ablation_metrics.csv`,
`feature_shift_audit.csv`, `provenance_screen.csv`, and `summary.json`. All but
the first carry a `mirror` column or key so that every number can be traced to
one provenance. Phase B writes only under `results/phase_b/` (non-IID scoring,
factorial, HGB robustness, missingness, and 40 KB of LR checkpoints).

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
