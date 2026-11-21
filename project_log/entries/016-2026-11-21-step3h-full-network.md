# 016: Step 3h — full-network scores beat their last-layer versions
Date: 2026-11-21   Follows: [015](015-2026-11-20-step3g-tracin.md)

**What:** `step3_removal.py --full` adds three full-network rankings: raw attention, attention × error and TracIn, summed over all 3 linear layers. Run on test image 4433, k = 1000, seeds 0, 1, 2, 20 random draws.
**Why:** All earlier scores used only the last layer. The hidden layers learn the features, and their writes are missing from the scores.
**Decisions:**
- Weights only, no bias terms — the same as the last-layer versions.
- Per layer, the key is the layer input and the error is `dLoss/d(layer output)`. Layer 1 keys are normalized pixels.
- One replay of the seed 0 run gives both test sides; the script stores no logs.
- Raw attention (full) — Eric's request: `sum_l k_il^T q_l`, with no error.

**Sanity checks:** the replay gives the step 1 weights exactly (gap 0). The layer-3 terms reproduce the last-layer scores (r = 1.000000).

**Result:** loss change on test image 4433.

| Removed top 1000 of | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| Random (std) | 0.10 | 0.09 | 0.08 | — |
| Raw attention, last / full | +0.16 / +0.98 | +0.23 / +1.22 | +0.16 / +0.90 | +0.18 / +1.03 |
| Error norm (7s, no query) | +2.87 | +3.62 | +3.37 | +3.29 |
| TracIn, last / full | +4.01 / +5.50 | +4.22 / +4.68 | +3.07 / +3.78 | +3.77 / +4.65 |
| Attention × error, last / full | +4.14 / +6.08 | +4.79 / +5.04 | +3.63 / +4.44 | +4.19 / +5.19 |

Each layer has a similar share of the mean |term| (layer 1: 38%, layers 2 and 3: 29–33%). Full and last-layer attention × error have r = 0.92 and share 778 of the top 1000.

**Interpretation:** The full-network versions beat the last-layer versions for all 3 seeds, for all three scores. Attention × error (full) beats TracIn (full) for all 3 seeds (mean +0.54). Its gap over the error-norm control grows from +0.90 to +1.90. Raw attention (full) now beats random, but it stays far below the scores with the error. One test image only.
**Next:** Several misclassified test images.
