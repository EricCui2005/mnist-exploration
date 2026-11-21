# 012: Step 3d — the error norm alone gives most of the attention × error effect
Date: 2026-11-20   Follows: [011](011-2026-11-20-step3c-removal-k1000.md)

**What:** `step3_removal.py --random 20 --k 1000` adds a control ranking: the training 7s with the largest error `|e_i|`.
**Why:** Attention × error removes hard 7s. The effect can be generic (hard 7s help all 7s) or specific to test image 0. The control uses no query.
**Decisions:** Only training 7s in the control — attention × error removes 93% 7s, so the class mix is similar.

**Result:**

| Removed top 1000 of | Loss change | z against random |
|---|---|---|
| Attention × error | +0.00094 | +7.1 |
| Error norm (7s, no query) | +0.00079 | +5.9 |

The two top-1000 sets share 810 examples.

**Interpretation:** The control gives 84% of the attention × error effect. Most of the effect is generic: the removal of hard 7s. The query adds a small part (+0.00015). One seed and one test image only; the noise of this difference is unknown.
**Next:** Measure the loss on many test 7s, to show whether the effect is generic.
