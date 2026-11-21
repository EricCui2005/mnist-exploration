# 001: Prototype attribution pipeline (code later removed)
Date: 2026-11-20

**What:** A first prototype compared three attribution scores on a 784-800-800-10 MLP on MNIST. The prototype used a removal-and-retrain test with 370 retrains.
**Why:** Omri asked if the error term adds information to the attention of Irie et al.
**Decisions:**
- Plain SGD, lr 0.1, batch 128, 3 epochs — the dual form is exact only under plain SGD.
- Exact TracIn-Ideal from test-side logs at every step — removes the checkpoint approximation.
- True-label loss for the test error — a choice, not tested against the predicted label.

**Result:** Removal of the top 1000 proponents changed the mean test loss as follows (n = 20 test points):

| Score | Proponents removed | Opponents removed |
|---|---|---|
| Random | -0.02 | -0.02 |
| Attention only | -0.04 | -0.05 |
| Attention × error | +2.50 | -0.67 |
| TracIn-Ideal | +2.47 | -0.59 |

**Interpretation:** The error term made the score faithful. Attention alone was equal to random and showed a bias toward bright images. Attention × error and TracIn were equal within the error bars. All of the signal came from the 10 misclassified test points.
**Status:** The prototype code is not in the project. Treat these results as preliminary. See [002](002-2026-11-20-project-restart.md).
