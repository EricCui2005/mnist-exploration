# 03: Removal test

**Status:** Done for test image 0 (seed 0) and 6 misclassified 7s (seeds 0, 1, 2).
**Question:** Does the removal of the top-k examples of each ranking raise the loss on the test image more than a random removal?
**Motivation:** A ranking is evidence only if the removal of its top examples changes the model.

**Setup:**
- Rankings: raw attention, cosine attention and attention × error from [experiment 02](02-attention-ranking.md), plus random sets. Later sections add the error-norm control, TracIn and full-network scores.
- Retrain: the step 1 setup, the same batch order. Removed examples leave their batch. Each batch loss divides by 128. The rankings always come from the seed 0 run.
- Measure: the loss on the test image, against a full retrain with the same seed and rules.

**Test image 0, k = 100, seed 0.** Full retrain loss = 0.00132.

| Removed top 100 of | Loss change | Ratio |
|---|---|---|
| Raw attention | +0.00001 | 1.01x |
| Cosine attention | +0.00008 | 1.06x |
| Attention × error | +0.00035 | 1.27x |
| Random | +0.00005 | 1.04x |

**Noise:** `--random 20` removes 20 random sets of 100. Random removal changes the loss by +0.00001 ± 0.00006 (range −0.00022 to +0.00007).

| Removed top 100 of | z against random | Beats random draws |
|---|---|---|
| Raw attention | +0.0 | 40% |
| Cosine attention | +1.1 | 100% |
| Attention × error | +5.6 | 100% |

**k = 1000:** `--random 20 --k 1000`. Random removal changes the loss by +0.00005 ± 0.00013.

| Removed top 1000 of | Loss change | Ratio | z against random |
|---|---|---|---|
| Raw attention | −0.00002 | 0.98x | −0.6 |
| Cosine attention | +0.00017 | 1.13x | +1.0 |
| Attention × error | +0.00094 | 1.71x | +7.1 |

**Control: error norm.** The 1000 training 7s with the largest `|e_i|`, with no query. Loss change +0.00079 (z = +5.9), against +0.00094 for attention × error. The two sets share 810 examples.

**Misclassified test image 4433** (a 7 predicted as 2, `p(7)` = 0.26), `--test 4433`, k = 1000. Full retrain loss = 1.178.

| Removed top 1000 of | Loss change | z against random |
|---|---|---|
| Random (20 draws) | +0.01 ± 0.10 | — |
| Raw attention | +0.16 | +1.4 |
| Cosine attention | +2.27 | +22.5 |
| Error norm (7s, no query) | +2.87 | +28.5 |
| Attention × error | +4.14 | +41.1 |

**Seeds** (`--seed S`; the rankings stay from seed 0). Test image 4433, k = 1000:

| Removed top 1000 of | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Random | +0.01 ± 0.10 | +0.04 ± 0.09 | +0.04 ± 0.08 |
| Raw attention | +0.16 | +0.23 | +0.16 |
| Cosine attention | +2.27 | +1.97 | +1.36 |
| Error norm (7s) | +2.87 | +3.62 | +3.37 |
| Attention × error | +4.14 | +4.79 | +3.63 |

**TracIn (last layer)** `sum eta (k_i^T q_t)(e_i^T e_q,t)`: the test side at the time of each write. Test image 4433, k = 1000:

| Removed top 1000 of | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Random (mean of 20 draws) | +0.01 | +0.04 | +0.04 |
| TracIn (last layer) | +4.01 | +4.22 | +3.07 |
| Attention × error | +4.14 | +4.79 | +3.63 |

Correlation 0.89; top-1000 overlap 795.

**Full network** (`--full`): the scores sum the terms of all 3 linear layers (weights only). Test image 4433, k = 1000:

| Removed top 1000 of | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| Raw attention, last / full | +0.16 / +0.98 | +0.23 / +1.22 | +0.16 / +0.90 | +0.18 / +1.03 |
| Error norm (7s) | +2.87 | +3.62 | +3.37 | +3.29 |
| TracIn, last / full | +4.01 / +5.50 | +4.22 / +4.68 | +3.07 / +3.78 | +3.77 / +4.65 |
| Attention × error, last / full | +4.14 / +6.08 | +4.79 / +5.04 | +3.63 / +4.44 | +4.19 / +5.19 |

The layer-3 terms reproduce the last-layer scores (r = 1). The 3 layers have similar shares of the mean |term|.

**Five more misclassified 7s** (10th to 90th loss percentile), full network, mean of 3 seeds:

| Test image | TracIn last / full | Att. × error last / full | Dual − TracIn (full), mean |
|---|---|---|---|
| 1206 | +3.65 / +5.02 | +4.66 / +5.34 | +0.33 |
| 2325 | +2.32 / +3.28 | +2.37 / +3.24 | −0.04 |
| 3189 | +4.13 / +5.20 | +4.17 / +5.20 | +0.01 |
| 7432 | +3.65 / +4.73 | +3.65 / +4.86 | +0.13 |
| 3838 | +4.10 / +5.22 | +4.36 / +5.36 | +0.14 |

**Interpretation:**
- The full-network scores beat their last-layer versions in all 18 runs (6 test images × 3 seeds).
- Attention × error (full) beats TracIn (full) on 4433 (mean +0.54). On the 5 other test images, the two are approximately equal (mean +0.11; the dual form wins 8 of 15 runs).
- Attention × error is far outside the random spread: 5 times the largest random change.
- On test image 0, the error norm alone gives 84% of the attention × error effect: most of the effect comes from the removal of hard 7s.
- On the misclassified test image 4433, the error norm gives 69%. The gap is positive for seeds 0, 1 and 2 (+1.27, +1.18, +0.26), but it varies.
- Attention × error is equal to or better than last-layer TracIn for all 3 seeds (mean difference +0.42).
- The order of the methods is the same for all 3 seeds. The rankings from seed 0 transfer to other training runs.
- The attention × error effect grows with k: 1.27x at k = 100, 1.71x at k = 1000.
- Raw attention is equal to random at both k. Cosine is weak at k = 100 and inside the random spread at k = 1000.
- On test image 0, the test loss is near zero, so the effects are small in absolute size.

**Entries:** [009](../../entries/009-2026-11-20-step3-removal-test.md), [010](../../entries/010-2026-11-20-step3b-removal-noise.md), [011](../../entries/011-2026-11-20-step3c-removal-k1000.md), [012](../../entries/012-2026-11-20-step3d-error-norm-control.md), [013](../../entries/013-2026-11-20-step3e-misclassified-test-image.md), [014](../../entries/014-2026-11-20-step3f-seeds.md), [015](../../entries/015-2026-11-20-step3g-tracin.md), [016](../../entries/016-2026-11-21-step3h-full-network.md), [018](../../entries/018-2026-11-21-step3i-five-misclassified.md).
**Code:** [step3_removal.py](../../../step3_removal.py).
