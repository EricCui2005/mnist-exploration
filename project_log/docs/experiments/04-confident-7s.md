# 04: The most confident training 7s

**Status:** Done.
**Question:** Which training 7s does the final model treat as the most typical 7s?
**Motivation:** Attention × error ([experiment 02](02-attention-ranking.md)) favors 7s that are not typical to the model. This experiment shows the typical 7s.

**Setup:**
- Model: from [experiment 01](01-weight-rebuild.md).
- Score: the 7-logit margin (the 7 logit minus the highest other logit), for all training 7s.

**Result:** [step3_confident_7s.png](../../../figures/step3_confident_7s.png)

| Measure | Value |
|---|---|
| Top-8 margins | 12.1 to 12.9 |
| Median margin of the training 7s | 5.9 |
| Margin of test image 0 | 7.0 (67th percentile) |

The top 8 are thick, upright 7s with a flat top bar.

**Interpretation:**
- The margin has the size bias of raw attention: large keys give large logits.
- Test image 0 is a typical 7 to the model. This agrees with its small `e_q`.
- The margin shows examples, not the internal template of the model.

**Entries:** [009](../../entries/009-2026-11-20-step3-confident-7s.md).
**Code:** [step3_confident_7s.py](../../../step3_confident_7s.py).
