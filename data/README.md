# Data preparation and provenance

Third-party datasets are intentionally not committed. Download them from their
original providers and run `evaluation/prepare_public_data.py`.

## Inputs

- **InSDN binary source**: `Dataset.csv`, with the columns named in the
  preparation script. The study uses 343,889 cleaned rows (68,424 normal and
  275,465 attack). Its `ip_proto` carries true IANA numbers `{0, 6, 17}`.
  Dataset paper: <https://doi.org/10.1109/ACCESS.2020.3022633>.
- **CICIDS2017**, Monday, Friday DDoS, and Friday PortScan. Obtain them from
  <https://www.unb.ca/cic/datasets/ids-2017.html>. CSV and Parquet inputs are
  both accepted.

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
| `ML` | `MachineLearningCVE` | flag surrogate | sentinel `-1` | no |

`CF` is the controlled comparison. It is **not** produced by re-running the
cleaner with the two columns suppressed: both the missing-value filter and the
de-duplication key on those columns, so a rerun would silently select a
different row set. Instead the script derives `CF` from the finished `TL` test
split, keeping the same rows in the same order with the same labels and the same
seven remaining features, and overwrites only `ip_proto` and `tp_src`. It then
asserts that exactly two columns differ and that the rest are bit-identical, and
aborts otherwise. `evaluation/run_verified_federated_baseline.py` repeats that
assertion after seeded subsampling, and `evaluation/verify_paper_numbers.py`
repeats it against the committed CSVs.

`ML` differs from `TL` in row set and attack prevalence as well, so it
establishes practical relevance rather than the mechanism.

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
and identical labels by construction; the `ML` difference comes from
de-duplication, which collapses far more rows once real source ports are absent.
Against `TL`, the `CF` surrogate assigns a different protocol to 30.32% of rows,
because UDP (17) has no representation in the flag-derived rule. These views are
an interoperability audit, not a complete CICIDS2017 leaderboard.

## Excluded column

All prepared CSVs contain `flow_duration`, which the preparation script derives
from the same `Flow Duration` field as `duration_sec` and which is therefore
identical to it in every row. This is an instance of the feature duplication
reported for these releases by Rosay et al. (ICISSP 2022). The evaluation script
reads only the nine independent features listed in the paper and ignores
`flow_duration`. The checksums above cover the prepared files as written,
including that column.
