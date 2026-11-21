# Codebase

## Setup
- The project is in `omri/mnist_exploration/`. The `uv` environment (Python 3.12, torch, torchvision, numpy) is in `omri/` and all experiment folders share it.
- Run a script from `mnist_exploration/` with `uv run python <script>.py`. The scripts use relative paths.
- MNIST downloads to `data/` on the first run.

## Scripts
| Script | What it does | Outputs |
|---|---|---|
| [step1_dual_form.py](../../step1_dual_form.py) | Trains the 784-800-800-10 MLP for 1 epoch with plain SGD. Stores the key and the error of each last-layer write. Rebuilds the last-layer weights and prints the error. | `step1_logs.npz`: `keys` [60000, 800], `errors` [60000, 10], `example_ids` [60000]; `step1_model.pt`: the trained model |
| [step2_attention.py](../../step2_attention.py) | Scores all training examples by last-layer attention `k^T q` for one test image. Shows the top 8 and the bottom 8. The `--cosine` flag divides by `|k| |q|`. The `--error` flag multiplies by `eta e_i^T e_q`. | `figures/step2_attention.png`, `figures/step2_attention_cosine.png`, `figures/step2_attention_error.png` |
| [step3_confident_7s.py](../../step3_confident_7s.py) | Ranks the training 7s by the 7-logit margin of the final model. Prints the margin of the test image. | `figures/step3_confident_7s.png` |
| [step3_removal.py](../../step3_removal.py) | Retrains the step 1 model without the top 100 examples of each ranking for test image 0, plus a random 100. Prints the change in the test-0 loss against a full retrain. The `--random N` flag uses N random draws and prints a z-score for each ranking. It also ranks the training 7s by the error norm `|e_i|` as a control with no query. It also ranks by last-layer TracIn: it replays the seed 0 run to get the test query and test error before each write. The `--seed S` flag sets the retrain seed (the rankings stay from seed 0). The `--test N` flag sets the test image (default 0). The `--k N` flag sets the number of removed examples (default 100). The `--full` flag adds full-network raw attention, attention × error and TracIn (all 3 linear layers, weights only), from one replay of the seed 0 run. | Printed only |

## Conventions
- Each error includes the 1/B factor of the mean batch loss.
- Rows in the logs are in training order: one row per write. `example_ids` gives the training image of each row.
- Seed 0 makes `step1_dual_form.py` give the same run each time.
- All figures go in `figures/`.
