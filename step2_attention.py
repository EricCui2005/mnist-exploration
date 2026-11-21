"""Step 2: for one test image, which training examples get the most attention in the last layer?

Attention of a write = k^T q
  k = the stored key of the training example (input to the last layer during training)
  q = the query: the test image's input to the last layer, from the final model
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

TEST_ID = 0  # first MNIST test image
COSINE = "--cosine" in sys.argv  # divide by |k| |q|: compares direction only, not size
ERROR = "--error" in sys.argv    # multiply by e_i^T e_q: positive = the write lowers the test loss
LR = 0.1                         # step 1 learning rate

# Load the trained model and the stored writes from step 1
hidden = nn.Sequential(nn.Linear(784, 800), nn.ReLU(), nn.Linear(800, 800), nn.ReLU())
hidden.load_state_dict(torch.load("step1_model.pt")["hidden"])
logs = np.load("step1_logs.npz")
K, E, ids = logs["keys"], logs["errors"], logs["example_ids"]

# Query: the test image's input to the last layer
test = datasets.MNIST("data", train=False)
train = datasets.MNIST("data", train=True)
x = test.data[TEST_ID].float().div(255).sub(0.1307).div(0.3081).view(1, 784)
with torch.no_grad():
    q = hidden(x).numpy()[0]
    last = nn.Linear(800, 10); last.load_state_dict(torch.load("step1_model.pt")["last"])
    p = torch.softmax(last(torch.tensor(q)), 0).numpy()
e_q = p - np.eye(10)[test.targets[TEST_ID].item()]  # test error, true label (= predicted label here)

attention = K @ q                       # one score per write (= per training image, 1 epoch)
raw_order = np.argsort(-attention)
if COSINE:
    attention = attention / (np.linalg.norm(K, axis=1) * np.linalg.norm(q))
cos_order = np.argsort(-attention / np.linalg.norm(K, axis=1))
if ERROR:
    # first-order drop in the test loss from this write: eta (k^T q)(e_i^T e_q)
    attention = LR * attention * (E @ e_q)
order = np.argsort(-attention)
top, bottom = order[:8], order[-8:][::-1]

# How much does attention just track the size of the key?
key_norm = np.linalg.norm(K, axis=1)
corr = np.corrcoef(attention, key_norm)[0, 1]
labels = train.targets.numpy()[ids]
print(f"test image {TEST_ID}: label {test.targets[TEST_ID].item()}")
print(f"top-100 by attention: {np.mean(labels[order[:100]] == test.targets[TEST_ID].item()):.0%} share the test label")
print(f"correlation of attention with key size |k|: {corr:.2f}")

# Do the top examples match the test image in raw pixels? (mean pixel cosine of the top 100)
pix = train.data.numpy()[ids].reshape(len(ids), -1).astype(float)
t = test.data[TEST_ID].numpy().reshape(-1).astype(float)
pix_cos = pix @ t / (np.linalg.norm(pix, axis=1) * np.linalg.norm(t))
print(f"mean pixel cosine with the test image, top 100: {pix_cos[order[:100]].mean():.3f} "
      f"(all 7s: {pix_cos[labels == 7].mean():.3f})")
if ERROR:
    print(f"negative scores (opponents): {np.mean(attention < 0):.1%}")
    print(f"overlap of top 100 with raw / cosine attention: "
          f"{len(set(order[:100]) & set(raw_order[:100]))}% / {len(set(order[:100]) & set(cos_order[:100]))}%")
    print(f"bottom-100 labels: {np.bincount(labels[order[-100:]], minlength=10)}")
if COSINE:
    print(f"overlap of top 100 with raw attention: {len(set(order[:100]) & set(raw_order[:100]))}%")

# Figure: test image, top 8, bottom 8
fig, axes = plt.subplots(2, 9, figsize=(11, 3))
for row, (name, rows) in enumerate([("Top 8", top), ("Bottom 8", bottom)]):
    axes[row, 0].imshow(test.data[TEST_ID], cmap="gray")
    axes[row, 0].set_title(f"test {TEST_ID}", fontsize=8)
    axes[row, 0].set_ylabel(name, fontsize=9)
    for j, r in enumerate(rows):
        axes[row, j + 1].imshow(train.data[ids[r]], cmap="gray")
        axes[row, j + 1].set_title(f"y={labels[r]}  {attention[r]:.1e}" if ERROR else f"y={labels[r]}  {attention[r]:.2f}" if COSINE else f"y={labels[r]}  {attention[r]:.0f}", fontsize=8)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Training images ranked by last-layer " + ("attention x error eta (k^T q)(e_i^T e_q)" if ERROR else"cosine attention k^T q / |k||q|" if COSINE else "attention k^T q"), fontsize=10)
fig.tight_layout()
fig.savefig("figures/" + ("step2_attention_error.png" if ERROR else "step2_attention_cosine.png" if COSINE else "step2_attention.png"), dpi=130)
