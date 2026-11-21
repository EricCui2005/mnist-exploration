# 010: Step 3b — the attention × error effect is far outside the random noise
Date: 2026-11-20   Follows: [009](009-2026-11-20-step3-removal-test.md)

**What:** `step3_removal.py --random 20` removes 20 different random sets of 100 examples. It compares each ranking with this spread.
**Why:** Step 3 had one random draw, so the noise level was unknown.
**Decisions:**
- 20 random draws — enough for a mean and a spread; each retrain takes about 1.5 s.
- Seed 0 for all runs — this measures the noise from the choice of removed examples only, not the seed noise.

**Result:** random removal changes the test-0 loss by +0.00001 ± 0.00006 (range −0.00022 to +0.00007).

| Removed top 100 of | Loss change | z against random | Beats random draws |
|---|---|---|---|
| Raw attention | +0.00001 | +0.0 | 40% |
| Cosine attention | +0.00008 | +1.1 | 100% |
| Attention × error | +0.00035 | +5.6 | 100% |

**Interpretation:** Attention × error is far outside the random spread: 5 times the largest random change. Raw attention is equal to random. Cosine is just above the largest random change, so its effect is weak. The seed noise is not measured.
**Next:** A test image with a larger loss, or several seeds.
