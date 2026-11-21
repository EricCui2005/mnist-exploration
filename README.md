# MNIST exploration: dual-form data attribution

Under plain SGD, each linear layer equals its initial weights plus one rank-1 write `-η e kᵀ` per training example and step (`k`: layer input, `e`: loss gradient at the layer output). This repo stores these writes for an MLP on MNIST and asks which training examples cause a prediction, comparing attention `kᵀq`, attention × error `(kᵀq)(eᵀe_q)` and TracIn with removal-and-retrain tests.

Mentor: Omri Uzan. Collaborator: Amir Zur.

## Structure

```
step1_dual_form.py     train the 784-800-800-10 MLP (1 epoch, SGD), store keys/errors, rebuild the weights
step2_attention.py     rank training images for one test image (--cosine, --error)
step3_confident_7s.py  the training 7s with the largest 7-logit margin
step3_removal.py       remove the top-k of each ranking, retrain, measure the test loss (--full, --test, --seed, --k, --random)
figures/               output figures
project_log/
  docs/                current state: findings, experiments, codebase, open questions
  entries/             append-only history, one file per step
```

Start with [project_log/docs/README.md](project_log/docs/README.md), then [findings.md](project_log/docs/findings.md).

## Setup

Python 3.12 with torch, torchvision, numpy and matplotlib (the `uv` environment lives in the parent folder).

```bash
uv run python step1_dual_form.py   # downloads MNIST to data/, writes step1_logs.npz (~190 MB) and step1_model.pt
uv run python step3_removal.py --full --random 20 --k 1000 --test 4433
```

Run scripts from this folder; they use relative paths. `data/` and `step1_logs.npz` are not tracked.
