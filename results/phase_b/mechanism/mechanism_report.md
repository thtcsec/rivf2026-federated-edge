# Phase B.2 mechanism report

Matched 2×2 factorial and exact ROC-AUC pair decomposition on frozen
Phase B.1 checkpoints. No retraining. The camera-ready manuscript was
not modified.

- Git commit: `ed880b51214a3b7906898b9086b5aa341600afbe`
- Checkpoint replay: **PASS** (330 checks). A reproduces stored TL; D reproduces stored CF.
- AUC reconstruction vs sklearn: **PASS** (15/15, tolerance 1e-12).
- Groups: original TL protocol codes `{0, 6, 17}` on the seeded 30,000-row sample.
- CF protocol in B and D is copied from matched CF rows, not re-inferred from flags.

## Mean ROC-AUC by condition

| model | A (TL) | B (CF proto, TL port) | C (TL proto, CF port) | D (CF) | D−A |
|---|---:|---:|---:|---:|---:|
| FedAvg IID | 0.7357 | 0.3365 | 0.7522 | 0.3545 | -0.3813 |
| FedAvg non-IID | 0.7297 | 0.3386 | 0.7416 | 0.3507 | -0.3791 |
| Centralized | 0.7490 | 0.3457 | 0.7576 | 0.3544 | -0.3946 |

A matches Phase B.1 TL; D matches Phase B.1 CF (combined D−A = −(TL−CF gap)).

## Protocol-only effect (B−A), original ports

Replacing only the protocol column with the matched CF values, keeping TL source ports.

| model | mean | SD | all five seeds |
|---|---:|---:|---|
| FedAvg IID | -0.3992 | 0.0071 | all negative |
| FedAvg non-IID | -0.3912 | 0.0172 | all negative |
| Centralized | -0.4033 | 0.0104 | all negative |

B is below 0.5 for every model. Protocol replacement **alone** produces the ranking reversal.

## Port-only effect (C−A), original protocol

Replacing TL's *variable* source-port column with the CF sentinel `-1`, keeping TL protocol.

| model | mean | SD | all five seeds |
|---|---:|---:|---|
| FedAvg IID | +0.0165 | 0.0050 | all positive |
| FedAvg non-IID | +0.0119 | 0.0116 | all positive |
| Centralized | +0.0086 | 0.0099 | all positive |

C stays above 0.5. Source-port substitution does **not** reverse ranking. It slightly *raises* AUC, consistent in sign with the paper's TL source-port masking gain (`+0.0165` for IID). These are still different operations (see below).

## Interaction (D−C−B+A) and combined effect

| model | interaction mean | combined D−A |
|---|---:|---:|
| FedAvg IID | +0.0014 | -0.3813 |
| FedAvg non-IID | +0.0002 | -0.3791 |
| Centralized | +0.0001 | -0.3946 |

The interaction is two orders of magnitude smaller than the protocol effect. On ROC-AUC the two interventions are approximately additive for this linear model: D−A ≈ (B−A)+(C−A). That is an observed numerical fact here, not a general claim that metrics must be additive.

## Masking versus replacement

Paper **masking** zeros one already-standardized column (the source-training mean) on a finished TL or CF matrix. A constant CF source-port column cannot reorder scores when masked; the paper reports CF source-port masking ΔAUC ≈ 0.

Factorial **C/D** replace TL's originally variable source-port column with that sentinel *before* scaling. Replacement can change ranking even though later masking of the resulting constant cannot.

These statements must not be conflated. IID C−A happens to equal the published TL masking gain at four decimals (`+0.0165`); that is a similar-magnitude, same-sign operating-point/ranking effect of discarding TL source-port variation, not proof that the two interventions are identical.

Protocol **masking** on TL (`ΔA = -0.1438`) removes a useful standardized column. Protocol **replacement** (B−A ≈ `-0.39`) writes InSDN's attack-associated code 0 onto the originally code-17 population. Zeroing cannot create that collision; substituting the colliding code can. The two numbers are not required to match, and they do not.

The seven-field intersection (`AUC ≈ 0.5075`) *retrains* without the two fields. Factorial cells score a model that still uses those fields. They are different experiments.

## Exact AUROC pair decomposition

weight(g,h) = n_pos(g)·n_neg(h) / (N_pos·N_neg). Pairwise AUC uses 0.5 tie credit.
Empty cells (no positives in g or no negatives in h) are not estimable and contribute 0. Within-group AUC for all-benign code 17 is not replaced by 0.5 or 0.

On the non-IID model, seed 11 (representative):

| pos g | neg h | weight | AUC_A | AUC_D | Δ contribution (A−D) | code-17? |
|---|---|---:|---:|---:|---:|---|
| 6 | 17 | 0.419 | 0.944 | 0.007 | +0.392 | yes |
| 6 | 6 | 0.581 | 0.600 | 0.607 | −0.004 | no |
| other cells | | ≈0 | n.e. or ~0 | | ~0 | mixed |

Code-17 positives do not exist in this slice, so (17, ·) cells have zero weight.

Mean over five non-IID seeds:

- Signed code-17 cell contribution to A−D: `+0.3913`
- Reconstructed A−D gap: `+0.3791`
- Accounting share: **103.2%** (per-seed 101.1%–108.6%)

The share exceeds 100% because the residual (mostly positives-6 vs negatives-6) moves slightly *against* the global gap (CF a bit better within code 6). Shares can exceed 100% or be negative when cells cancel. This is an accounting identity, not a unique causal attribution.

IID share 104.8%; centralized 102.3%. Same pattern.

The factorial says protocol replacement causes the reversal under a controlled intervention. The decomposition says the ranking comparisons that move are those between original code-6 attacks and original code-17 benign flows. Keep those claims separate.

## Seed-level variation

Non-IID protocol-only B−A ranges from `-0.366` (seed 77) to `-0.410` (seed 101), always a reversal. Non-IID port-only C−A ranges from `+0.0038` to `+0.0313` (seed 42 is the high outlier), always a small gain. No seed fails the protocol-dominance pattern.

## Contradiction check

**No contradiction** with the manuscript claim that protocol localizes the ranking reversal.

|B−A| ≈ 0.39 ≫ |C−A| ≈ 0.01 for all three models. Port substitution does not reverse ranking. Protocol substitution does, even with original TL ports left in place.

## Camera-ready

A single sentence can fit: protocol replacement alone reverses ranking; source-port substitution does not. A new factorial table would displace Table III / the protocol crosstab. **Retain this directory as a supplement** unless space remains after Phase A material.

## Limitations

- Linear logistic models only; one CICIDS2017 slice; five holdouts from one corpus.
- Pair decomposition is descriptive accounting, not a unique causal identification.
- Code-0 attack mass is tiny after 30k sampling, so (0, ·) cells are often not estimable.
- Does not test other imputations, trees, or other dataset pairs.
