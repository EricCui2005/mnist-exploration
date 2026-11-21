# 018: Step 3i — on 5 more misclassified 7s, full > last layer holds; dual ≈ TracIn
Date: 2026-11-21   Follows: [017](017-2026-11-21-log-corrections.md)

**What:** The step 3h run (`--full --random 20 --k 1000`, seeds 0, 1, 2) on 5 more misclassified test 7s.
**Why:** Step 3h showed full > last layer and dual form > TracIn on one test image (4433). This step tests whether these results generalize.
**Decisions:** The 5 test images are the misclassified 7s at the 10th, 30th, 50th, 70th and 90th loss percentiles (46 misclassified 7s in all; 4433 is at rank 12).

**Result:** mean loss change over 3 seeds (random std: 0.08 to 0.12).

| Test image (pred., loss) | Error norm | TracIn last / full | Att. × error last / full | Dual − TracIn (full), per seed |
|---|---|---|---|---|
| 1206 (2, 0.93) | +3.15 | +3.65 / +5.02 | +4.66 / +5.34 | +0.35, +0.45, +0.18 |
| 2325 (2, 1.37) | +1.70 | +2.32 / +3.28 | +2.37 / +3.24 | −0.07, −0.10, +0.05 |
| 3189 (4, 2.07) | +2.94 | +4.13 / +5.20 | +4.17 / +5.20 | −0.11, −0.03, +0.16 |
| 7432 (2, 2.73) | +3.02 | +3.65 / +4.73 | +3.65 / +4.86 | −0.02, −0.01, +0.41 |
| 3838 (1, 4.58) | +3.12 | +4.10 / +5.22 | +4.36 / +5.36 | +0.31, +0.30, −0.19 |

**Interpretation:**
- Full network > last layer for both scores in all 15 runs (5 images × 3 seeds).
- Dual form vs TracIn (full): the dual form wins 8 of 15 runs; the mean difference is +0.11. Most differences are near the random std. With 4433, the dual form wins 11 of 18 runs.
- The large dual-form advantage on 4433 does not generalize. In the full network, the two scores are approximately equal on misclassified 7s.
**Next:** A setup where the two scores must disagree (for example, a backdoor early in training), or more test images to measure a small difference.
