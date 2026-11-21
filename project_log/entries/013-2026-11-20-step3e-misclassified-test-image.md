# 013: Step 3e — on a misclassified test 7, the query adds a clear effect
Date: 2026-11-20   Follows: [012](012-2026-11-20-step3d-error-norm-control.md)

**What:** `step3_removal.py --random 20 --k 1000 --test 4433` repeats step 3d on test image 4433.
**Why:** Test image 0 has a loss near 0, so its error points only at "more 7". A misclassified 7 has an error with a specific direction, so the query has room to matter.
**Decisions:**
- Test image 4433: a 7 that the model predicts as a 2 (`p(7)` = 0.26). The most frequent mistake for 7s is 2 (26 of 46); its loss (1.35) is near the middle of the misclassified 7s.
- Test error from the true label (7) — the score explains "what makes the answer correct".

**Result:** full retrain loss = 1.178. Test error `e_q`: +0.53 on 2, −0.74 on 7.

| Removed top 1000 of | Loss change | z against random |
|---|---|---|
| Random (20 draws) | +0.01 ± 0.10 | — |
| Raw attention | +0.16 | +1.4 |
| Cosine attention | +2.27 | +22.5 |
| Error norm (7s, no query) | +2.87 | +28.5 |
| Attention × error | +4.14 | +41.1 |

The attention × error and error-norm sets share 697 examples. The attention × error set has 791 7s and no 2s.

**Interpretation:** The error-norm control gives 69% of the attention × error effect (84% on test image 0). The gap (+1.27) is 13 times the random spread, so the query adds a clear part here. Cosine attention now has a large effect. One seed and one test image only.
**Next:** Several misclassified test images, or several seeds.
