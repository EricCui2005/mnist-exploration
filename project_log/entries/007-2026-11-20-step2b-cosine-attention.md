# 007: Step 2b — cosine attention removes the size bias
Date: 2026-11-20   Follows: [006](006-2026-11-20-step2-result.md)

**What:** `step2_attention.py --cosine` ranks the training examples by `k^T q / (|k| |q|)` for test image 0.
**Why:** Raw attention favored thick strokes (step 2). Eric's hypothesis: cosine removes this bias and favors 7s with a similar shape.
**Decisions:**
- One flag on the step 2 script, not a new script — the only change is the normalization.
- Mean pixel cosine of the top 100 with the test image — a direct measure of shape similarity.

**Result:** [step2_attention_cosine.png](../../figures/step2_attention_cosine.png)

| Measure | Raw attention | Cosine attention |
|---|---|---|
| Top 100 with label 7 | 100% | 100% |
| Correlation with `|k|` | 0.53 | 0.02 |
| Mean pixel cosine of top 100 | 0.751 | 0.829 |
| Top-100 overlap with raw | — | 9% |

All 7s have a mean pixel cosine of 0.562. The bottom 8 are now all 6s. The minimum cosine is 0.12, so all scores are still positive.

**Interpretation:** The hypothesis is correct. Cosine removes the size bias and selects 7s with the shape of the test image. Cosine is not a term of the dual form: it does not add up to the layer output. Cosine shows similarity only; it cannot identify opponents.
**Next:** Add the error term and compare it with both attention rankings.
