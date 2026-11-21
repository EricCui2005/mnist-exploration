# Open questions

## Next-step options
- Find a size-free measure of a typical 7, for example the margin divided by `|k|`.

## Open problems
- **Test label:** the test error needs a label. The true label explains "what makes the answer correct." The predicted label explains "why the model gave this answer." The project has no decision yet.
- **Test points with no signal:** correct test points have a loss near zero. Removal tests on them show nothing. Low-margin or misclassified test points give more signal.
- **Dual form or TracIn:** on 6 misclassified 7s, the two are approximately equal in the full network (entry 018). A setup where the two scores must disagree is necessary, for example a backdoor that occurs only early in training.

## Risks
- **Optimizer:** Adam and momentum break the exact rank-1 sum. All runs must use plain SGD.
- **Storage:** the logs grow with examples × epochs × layer widths. The prototype used 2.9 GB for 3 epochs of this MLP.

## Later directions
- MNIST backdoor: a corner patch on some training images, relabeled as 0 (Omri's proposal).
- Waterbirds: attribution and post-hoc edits of a spurious correlation (Omri's proposal). This work has its own folder: [../../../waterbirds/](../../../waterbirds/).
