# 008: Step 2c — attention × error finds hard examples and opponents
Date: 2026-11-20   Follows: [007](007-2026-11-20-step2b-cosine-attention.md)

**What:** `step2_attention.py --error` ranks the training examples by `eta (k^T q)(e_i^T e_q)` for test image 0.
**Why:** Attention is always positive and has no link to the loss. The error term gives each write a sign: positive means the write lowers the test loss (first order).
**Decisions:**
- Test error from the true label — for test image 0, the true label and the predicted label are the same (7).
- Raw attention, not cosine — the raw term is the term of the dual form.
- One flag on the step 2 script — the only change is the error factor.

**Result:** [step2_attention_error.png](../../figures/step2_attention_error.png)

| Measure | Raw attention | Attention × error |
|---|---|---|
| Top 100 with label 7 | 100% | 100% |
| Correlation with `|k|` | 0.53 | -0.01 |
| Mean pixel cosine of top 100 (all 7s: 0.562) | 0.751 | 0.450 |
| Mean `p(7)` at write time, top 100 (all 7s: 0.84) | — | 0.15 |
| Negative scores | 0% | 39% |
| Top-100 overlap with raw / cosine | — | 0% / 0% |

The top 8 are unusual 7s (crossed, curved). The bottom 100 are mostly 2s (47), 9s (32) and 3s (18).

**Interpretation:** The error term changes the ranking completely. It favors 7s that the model got wrong at write time, because these writes have a large error. It also identifies opponents: 2s, 3s and 9s with a shape near a 7. One test image only. The result does not show faithfulness.
**Next:** A removal test, or a repeat on a misclassified test image.
