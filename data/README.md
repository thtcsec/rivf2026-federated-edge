# Data preparation and provenance

Third-party datasets are intentionally not committed. Download them from their
original providers and run `evaluation/prepare_public_data.py`.

## Inputs

- **InSDN binary source**: `Dataset.csv`, with the columns named in the
  preparation script. The study uses 343,889 cleaned rows (68,424 normal and
  275,465 attack). Its `ip_proto` stores integer codes `{0, 6, 17}` as they
  appear in the release CSV. Code 0 is not treated as an IANA protocol identity;
  in the prepared file every such row has source and destination port 0.
  Dataset paper: <https://doi.org/10.1109/ACCESS.2020.3022633>.

  The CSV used here is the **third-party Hugging Face derivative**
  [`Sharukesh/INSDN`](https://huggingface.co/datasets/Sharukesh/INSDN) at
  revision `f59330c8ff0f1b231b382e05fc41b1e8fb0df6b4`
  (`Dataset.csv` SHA-256
  `a93d51e1c2debc7246728578f22d84a442f156628be64b8c9bae5ccc5dd88a42`).
  That file is not an authenticated dump from the original publisher.
  Prepared InSDN `data/insdn/flow_stats.csv` is a cleaned derivative of that
  CSV (SHA-256 `c05f1f40…` below).
- **CICIDS2017**, Monday, Friday DDoS, and Friday PortScan. Obtain them from
  <https://www.unb.ca/cic/datasets/ids-2017.html>. CSV and Parquet inputs are
  both accepted.

  This preparation ingested Parquet mirrors of the UNB CSVs. The committed
  `dataset_summary.json` files record these source filenames (no independent
  SHA-256 of those Parquet archives is published here):

  - `Monday-WorkingHours.pcap_ISCX.csv.parquet`
  - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet`
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet`

  CICIDS2017 is published in two flow-feature distributions that do **not**
  expose the same columns, and the paper's central result is that the choice
  between them decides the cross-dataset verdict:

  | Release | `Protocol` | `Source Port` |
  |---|---|---|
  | `TrafficLabelling` / `GeneratedLabelledFlows` | present | present |
  | `MachineLearningCVE` / `MachineLearningCSV` | absent | absent |

  When a column is absent the script reconstructs it: protocol becomes TCP (6)
  if any TCP flag is positive and 0 otherwise, and every source port becomes the
  sentinel `-1`. The script never guesses silently; it records the branch taken
  for each input file under `field_provenance` in `dataset_summary.json`.

## The three target views

The paper evaluates three views of the same capture files.

| View | Built from | Protocol | Source port | Row-matched to TL |
|---|---|---|---|---|
| `TL` | `TrafficLabelling` | original column | original column | — |
| `CF` | the finished `TL` rows | flag surrogate | sentinel `-1` | **yes** |
| `MLCVE` | `MachineLearningCVE` | flag surrogate | sentinel `-1` | no |

`CF` is the controlled comparison. It is **not** produced by re-running the
cleaner with the two columns suppressed: both the missing-value filter and the
de-duplication key on those columns, so a rerun would silently select a
different row set. Instead the script derives `CF` from the finished `TL` test
split, keeping the same rows in the same order with the same labels and the same
seven remaining features, and overwrites only `ip_proto` and `tp_src`. It then
asserts that exactly two columns differ and that the rest are bit-identical, and
aborts otherwise. `evaluation/run_verified_federated_baseline.py` repeats that
assertion after seeded subsampling, and `evaluation/verify_paper_numbers.py`
repeats it against the prepared CSVs when those files are present. They are
not committed; Level A verification skips the CSV checks rather than treating
absence as a pass.

`MLCVE` is an unmatched ecological check. It differs from `TL` in row set and
attack prevalence, so it can corroborate that a public release lands near the
counterfactual. It is not row-matched and is not the identified causal effect.

## Commands

```powershell
# source domain
python evaluation/prepare_public_data.py --insdn-src C:\path\to\Dataset.csv

# target views TL and CF in one pass; CF must be derived from the same TL rows
python evaluation/prepare_public_data.py `
  --cicids-root C:\path\to\cicids2017\TrafficLabelling `
  --cicids-out-name cicids2017_mirror `
  --cicids-counterfactual-name cicids2017_mirror_counterfactual

# target view ML: both columns absent, fields reconstructed
python evaluation/prepare_public_data.py `
  --cicids-root C:\path\to\cicids2017\MachineLearningCVE `
  --cicids-out-name cicids2017_mirror_mlcve
```

Either half can be rebuilt on its own, so a verifier who already holds one
checksum-matching file does not need both raw corpora.

The evaluation expects `data/insdn/flow_stats.csv`,
`data/cicids2017_mirror/test.csv`,
`data/cicids2017_mirror_counterfactual/test.csv`, and
`data/cicids2017_mirror_mlcve/test.csv`. These can be overridden with
`INSDN_CSV`, `CICIDS2017_MIRROR_CSV`, `CICIDS2017_COUNTERFACTUAL_CSV`, and
`CICIDS2017_MLCVE_MIRROR_CSV`.

## Verified prepared-file checksums

| File | Rows | SHA-256 |
|---|---|---|
| InSDN `insdn/flow_stats.csv` | 343,889 | `c05f1f40dceb888aaaaf52b29283054147ed68390ea47997a8539e1505f64a7e` |
| TL view `cicids2017_mirror/test.csv` | 207,463 | `4ff3075e61dea938892bda5bd90812170e085b225a87d1b0431b4e3c98d089bd` |
| CF view `cicids2017_mirror_counterfactual/test.csv` | 207,463 | `957b4170bda3a6f95648548196be35d473a36965a628e2b88580ae218f186b4c` |
| ML view `cicids2017_mirror_mlcve/test.csv` | 176,036 | `af5f02ade640caa9870a310c7d95903c5b95a1d2a794c060d0c3932346530a03` |

All views are built from the same three capture files with the same fixed
stratified split (`random_state=42`). `TL` and `CF` have identical row counts
and identical labels by construction; the `MLCVE` difference comes from
de-duplication, which collapses far more rows once real source ports are absent.
Against `TL`, the `CF` surrogate assigns a different protocol to 30.32% of rows,
because code 17 has no representation in the flag-derived rule. These views are
an interoperability audit, not a complete CICIDS2017 leaderboard.

## Excluded column

All prepared CSVs contain `flow_duration`, which the preparation script derives
from the same `Flow Duration` field as `duration_sec` and which is therefore
identical to it in every row. This is an instance of the feature duplication
reported for these releases by Rosay et al. (ICISSP 2022). The evaluation script
reads only the nine independent features listed in the paper and ignores
`flow_duration`. The checksums above cover the prepared files as written,
including that column.

## Phase B robustness controls (camera-ready)

Quoted Phase B numbers are generated under `results/phase_b/` and must not
overwrite `results/verified/`. After the four prepared CSVs match the checksums
above, a third party with this camera-ready commit reproduces them with:

```powershell
python evaluation/run_phase_b_noniid_cross.py
python evaluation/run_phase_b_mechanism.py
python evaluation/run_phase_b_robustness.py
python evaluation/run_phase_b_missingness.py
python evaluation/verify_paper_numbers.py
```

B.1 retrains the source-only LR models and writes small checkpoints used by the
fixed-model factorial (B.2) and by LR missingness policies that replay the
nine-field models (B.4). B.3 retrains a centralized nine-field categorical
`HistGradientBoostingClassifier`; it is not federated. B.4 also retrains
seven-field models (policy P2). Do not fit on target labels, do not tune after
inspecting target metrics, and do not call `run_verified_federated_baseline.py`
from Phase B.

**Publication state.** Quoted numbers can be checked against tracked JSON
without redistributing third-party CSVs (Level A). Regenerating models from
source files is Level B and needs the prepared datasets. B.2 and some
missingness policies replay tracked checkpoints; B.1/B.3/B.4 retrain source-only
models. The scientific freeze tag is `rivf2026-camera-ready` and must not be
moved. Historical Phase B manifests record a git commit SHA and prepared-CSV
hashes; they do **not** record executed-source hashes and are not a timestamped
preregistration. Re-running a current script against frozen outputs is a
**current reproduction**, not proof of the original runtime tree. The claim map
is `evaluation/phase_b_claim_manifest.md`. The manuscript quotes only the subset
checked by `evaluation/verify_paper_numbers.py`; the full missingness matrix
stays in `results/phase_b/missingness/`.
