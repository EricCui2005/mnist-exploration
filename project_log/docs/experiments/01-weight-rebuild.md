# 01: Rebuild the last layer from stored writes

**Status:** Done.
**Question:** Do the stored keys and errors rebuild the trained weights exactly?
**Motivation:** All attribution in this project depends on an exact dual form.

**Setup:**
- Model: MLP 784-800-800-10, PyTorch default initialization, seed 0.
- Training: plain SGD, lr 0.1, batch 128, 1 epoch, 60,000 MNIST images.
- Logs: key and error of each write to the last layer.
- Test: `W_init - lr * E^T K` against the trained `W`.

**Result:** The relative error is 7e-7. The norm of the training change is 2.56. The norm of the gap is 1.8e-6.

**Interpretation:** The gap is float32 rounding, so the dual form is exact. The result does not tell which examples are important. The result applies only to plain SGD.

**Entries:** [003](../../entries/003-2026-11-20-step1-design.md), [004](../../entries/004-2026-11-20-step1-result.md).
**Code:** [step1_dual_form.py](../../../step1_dual_form.py).
