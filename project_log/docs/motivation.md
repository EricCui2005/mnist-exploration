# Motivation

## Goals (from Omri)
1. Find out if the stored dual form gives insights into model behavior that methods on the trained weights cannot give.
2. Find out if the method can scale efficiently and keep these gains.

The project stays in usual deep learning (MLPs, CNNs, SGD). It starts with small models, where the full dual form fits on disk.

## The first hypothesis
For one linear layer, the TracIn gradient product factors into two terms:

`g_q · g_i = (q^T k_i) (e_q^T e_i)`

- The first term is attention. Irie et al. (2022) use only this term.
- The second term is the alignment of the two errors.

Omri's hypothesis: the error term adds information. A score with both terms identifies the training examples that cause a prediction better than attention alone.

## How the dual form differs from TracIn
Both scores use the same training-side key and error. The test side is different:
- TracIn measures the test example at the model of each step. It divides the loss decrease along the training path.
- The dual form measures the test example at the final model. It divides the output of the final model.

Neither score is the true counterfactual (retrain without the example). A removal-and-retrain test is the judge.

## Key papers
- Irie et al. 2022, the dual form of neural networks (arXiv 2202.05798).
- Pruthi et al. 2020, TracIn (arXiv 2002.08484).
- Koh & Liang 2017, influence functions (arXiv 1703.04730).
