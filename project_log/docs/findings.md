# Findings

## Confirmed in the current code
- **The dual form is exact under plain SGD.** The stored writes rebuild the last-layer weights with a relative error of 7e-7. Evidence: [experiment 01](experiments/01-weight-rebuild.md).

- **Attention finds the test class but also tracks key size.** For one test 7, the top 100 examples are all 7s, but thick strokes rank highest (correlation with `|k|` 0.53). Evidence: [experiment 02](experiments/02-attention-ranking.md).
- **Cosine attention removes the size bias.** It selects 7s with the shape of the test image (mean pixel cosine 0.829 against 0.751 for raw). Cosine is not a term of the dual form. Evidence: [experiment 02](experiments/02-attention-ranking.md).
- **Attention × error favors hard examples and identifies opponents.** Its top 100 are 7s with a mean `p(7)` of 0.15 at write time. 39% of scores are negative; the strongest are 2s, 3s and 9s. Its top 100 do not overlap with the raw or cosine top 100. Evidence: [experiment 02](experiments/02-attention-ranking.md).
- **The most confident 7s are thick, upright 7s.** The 7-logit margin has the size bias of raw attention. Test image 0 is at the 67th percentile of the training 7s. Evidence: [experiment 04](experiments/04-confident-7s.md).
- **Attention cannot identify opponents.** Keys and queries come after a ReLU, so every raw or cosine attention score is positive. Evidence: [experiment 02](experiments/02-attention-ranking.md).
- **Removal test: only attention × error clearly beats random.** Removal of its top 100 raises the test-0 loss by 27% (z = +5.6 against 20 random draws). At k = 1000 the effect is 71% (z = +7.1). Raw attention is equal to random at both k. Cosine is weak or equal to random. A control with the error norm alone (no query) gives 84% of this effect, so most of the effect is generic. On a misclassified test 7 (4433), the control gives 69%, and the query adds a part: +1.27, +1.18 and +0.26 loss for seeds 0, 1 and 2. The method order is the same for all 3 seeds. Test images 0 and 4433 only. Evidence: [experiment 03](experiments/03-removal-test.md).
- **In the last layer, attention × error is equal to or better than TracIn.** On 4433 it beats TracIn on all 3 seeds (+0.13 to +0.58). On 5 other misclassified 7s, the mean differences are +1.01, +0.05, +0.04, 0.00 and +0.26. Evidence: [experiment 03](experiments/03-removal-test.md).
- **Full-network scores beat their last-layer versions.** This holds in all 18 runs: 6 misclassified test 7s × 3 seeds, k = 1000. On 4433, the mean loss change rises from +4.19 to +5.19 (attention × error) and from +3.77 to +4.65 (TracIn).
- **In the full network, dual form and TracIn are approximately equal.** The dual form is better on 4433 (+0.54). On 5 other misclassified 7s, the mean difference is +0.11, and the dual form wins 8 of 15 runs. Evidence: [experiment 03](experiments/03-removal-test.md).

## Preliminary (prototype, code removed)
These results came from the prototype in [entry 001](../entries/001-2026-11-20-prototype-pipeline.md). The current code does not reproduce them yet.
- Attention alone was equal to random in a removal-and-retrain test. It showed a bias toward bright images.
- Attention × error and TracIn-Ideal were much better than random, and equal to each other within the error bars.
- All of the signal came from 10 misclassified test points. Correct test points had a loss near 0.002, so removal did not change them.
