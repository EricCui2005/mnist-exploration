# 02: Rank training examples by attention

**Status:** Done.
**Question:** For one test image, which training examples get the highest last-layer attention `k^T q`? Do they look meaningful?
**Motivation:** Attention is the simplest reading of the writes. Irie et al. (2022) use only attention.

**Setup:**
- Model and logs: from [experiment 01](01-weight-rebuild.md) (1 epoch, so one write per training example).
- Test image: MNIST test image 0, a 7, classified correctly.
- Score: raw `k^T q` in the last layer, with no normalization.

**Result:** [step2_attention.png](../../../figures/step2_attention.png)

| Measure | Value |
|---|---|
| Top-100 examples with the test label | 100% |
| Correlation of attention with `|k|` | 0.53 |
| Minimum attention | 18 (all scores are positive) |

The top 8 are thick, bright 7s. The bottom 8 are mostly thin 1s.

**Variant: cosine attention** `k^T q / (|k| |q|)`, with the `--cosine` flag. Result: [step2_attention_cosine.png](../../../figures/step2_attention_cosine.png)

| Measure | Raw | Cosine |
|---|---|---|
| Top 100 with label 7 | 100% | 100% |
| Correlation with `|k|` | 0.53 | 0.02 |
| Mean pixel cosine of top 100 (all 7s: 0.562) | 0.751 | 0.829 |
| Top-100 overlap with raw | — | 9% |

The cosine top 8 are thin 7s with the shape of the test image. The cosine bottom 8 are all 6s.

**Variant: attention × error** `eta (k^T q)(e_i^T e_q)`, with the `--error` flag. The test error uses the true label (7, also the predicted label). Result: [step2_attention_error.png](../../../figures/step2_attention_error.png)

| Measure | Raw | Attention × error |
|---|---|---|
| Correlation with `|k|` | 0.53 | -0.01 |
| Mean pixel cosine of top 100 | 0.751 | 0.450 |
| Mean `p(7)` at write time, top 100 (all 7s: 0.84) | — | 0.15 |
| Negative scores | 0% | 39% |
| Top-100 overlap with raw / cosine | — | 0% / 0% |

The top 8 are unusual 7s that the model got wrong at write time. The bottom 100 are mostly 2s, 9s and 3s.

**Interpretation:**
- Attention × error favors writes with a large error: hard 7s, not typical 7s.
- Attention × error identifies opponents: digits with a shape near a 7.
- Attention finds the test class.
- Cosine removes the size bias and selects examples with a similar shape. Cosine is not a term of the dual form, so it does not add up to the output.
- Attention also measures key size, so thick strokes rank high and thin strokes rank low.
- Attention alone is never negative after a ReLU, so it cannot identify opponents.
- One test image only. The result does not show faithfulness.

**Entries:** [005](../../entries/005-2026-11-20-step2-design.md), [006](../../entries/006-2026-11-20-step2-result.md), [007](../../entries/007-2026-11-20-step2b-cosine-attention.md), [008](../../entries/008-2026-11-20-step2c-attention-times-error.md).
**Code:** [step2_attention.py](../../../step2_attention.py).
