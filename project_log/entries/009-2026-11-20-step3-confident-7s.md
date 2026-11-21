# 009: Step 3 — the most confident 7s are thick, upright 7s
Date: 2026-11-20   Follows: [008](008-2026-11-20-step2c-attention-times-error.md)

**What:** `step3_confident_7s.py` ranks the training 7s by the 7-logit margin of the final model.
**Why:** Step 2c showed that the score depends on how typical a 7 is to the model. This step shows what the model treats as a typical 7.
**Decisions:**
- The 7-logit margin, not `p(7)` — `p(7)` is 0.99999 for all top 8, so it cannot separate them.
- Real training images, not activation maximization — real images are easy to read.

**Result:** [step3_confident_7s.png](../../figures/step3_confident_7s.png)

| Measure | Value |
|---|---|
| Top-8 margins | 12.1 to 12.9 |
| Median margin of the training 7s | 5.9 |
| Margin of test image 0 | 7.0 (67th percentile) |

The top 8 are thick, upright 7s with a flat top bar.

**Interpretation:** The margin also has the size bias: logits are `W k`, so large keys give large margins. The ranking mixes confidence and stroke thickness. Test image 0 is a typical 7 to the model, which agrees with its small `e_q`.
**Next:** A size-free measure of a typical 7 (for example, the margin divided by `|k|`), or a misclassified test image.
