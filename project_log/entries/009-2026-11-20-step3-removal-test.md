# 009: Step 3 — removal test at k = 100 on test image 0
Date: 2026-11-20   Follows: [008](008-2026-11-20-step2c-attention-times-error.md)

**What:** `step3_removal.py` retrains the step 1 model without the top 100 examples of each ranking. It reports the change in the loss on test image 0 against a full retrain.
**Why:** A ranking is evidence only if the removal of its top examples changes the model. The random removal gives the baseline.
**Decisions:**
- Rankings for test image 0, not for all 7s — the scores are per test image (Eric).
- Test loss, not accuracy — the loss is continuous and shows small effects (Eric).
- The same seed for all runs — same initial weights and same batch order (Eric).
- Removed examples leave their batch, and each batch loss divides by 128 — every other example keeps its weight.
- One random draw, one seed — the first version only.

**Result:** full retrain loss on test 0 = 0.00132.

| Removed top 100 of | Loss change | Ratio to full retrain |
|---|---|---|
| Raw attention | +0.00001 | 1.01x |
| Cosine attention | +0.00008 | 1.06x |
| Attention × error | +0.00035 | 1.27x |
| Random | +0.00005 | 1.04x |

The full retrain differs from the step 1 model (0.00144). The cause: the last batch has 96 examples, and step 1 divided its loss by 96, not by 128.

**Interpretation:** Only attention × error moves the loss clearly more than random. Raw and cosine attention are equal to random. The noise level is unknown: one random draw and one seed only. The loss is near zero, so all effects are small in absolute size.
**Next:** Measure the noise with several random draws, or use a test image with a larger loss.
