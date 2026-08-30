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
  between them decides the cross-dataset verdict. Both must be prepared:

  | Release | `Protocol` | `Source Port` |
  |---|---|---|
  | `TrafficLabelling` / `GeneratedLabelledFlows` | present | present |
  | `MachineLearningCVE` / `MachineLearningCSV` | absent | absent |

  When a column is absent the script reconstructs it: protocol becomes TCP (6)
  if any TCP flag is positive and 0 otherwise, and every source port becomes the
  sentinel `-1`. The script never guesses silently; it records the branch taken
  for each input file under `field_provenance` in `dataset_summary.json`.

## Commands

```powershell
# source domain
python evaluation/prepare_public_data.py --insdn-src C:\path\to\Dataset.csv

# target mirror TL: original Protocol and Source Port
python evaluation/prepare_public_data.py `
  --cicids-root C:\path\to\cicids2017\TrafficLabelling `
  --cicids-out-name cicids2017_mirror

# target mirror ML: both columns absent, fields reconstructed
python evaluation/prepare_public_data.py `
  --cicids-root C:\path\to\cicids2017\MachineLearningCVE `
  --cicids-out-name cicids2017_mirror_mlcve
```

Either half can be rebuilt on its own, so a verifier who already holds one
checksum-matching file does not need both raw corpora.

The evaluation expects `data/insdn/flow_stats.csv`,
`data/cicids2017_mirror/test.csv`, and
`data/cicids2017_mirror_mlcve/test.csv`. These can be overridden with
`INSDN_CSV`, `CICIDS2017_MIRROR_CSV`, and `CICIDS2017_MLCVE_MIRROR_CSV`.

## Verified prepared-file checksums

| File | Rows | SHA-256 |
|---|---|---|
| InSDN `insdn/flow_stats.csv` | 343,889 | `c05f1f40dceb888aaaaf52b29283054147ed68390ea47997a8539e1505f64a7e` |
| TL mirror `cicids2017_mirror/test.csv` | 207,463 | `4ff3075e61dea938892bda5bd90812170e085b225a87d1b0431b4e3c98d089bd` |
| ML mirror `cicids2017_mirror_mlcve/test.csv` | 176,036 | `af5f02ade640caa9870a310c7d95903c5b95a1d2a794c060d0c3932346530a03` |

Both mirrors are built from the same three capture files with the same fixed
stratified split (`random_state=42`); the row-count difference comes from
deduplication, which collapses far more rows once real source ports are absent.
The mirrors are an interoperability audit, not a complete CICIDS2017
leaderboard.

## Excluded column

All prepared CSVs contain `flow_duration`, which the preparation script derives
from the same `Flow Duration` field as `duration_sec` and which is therefore
identical to it in every row. This is an instance of the feature duplication
reported for these releases by Rosay et al. (ICISSP 2022). The evaluation script
reads only the nine independent features listed in the paper and ignores
`flow_duration`. The checksums above cover the prepared files as written,
including that column.
