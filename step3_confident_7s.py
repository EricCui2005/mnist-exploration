"""Step 3: which training 7s is the final model most confident about?

Confidence = 7-logit margin: the 7 logit minus the highest other logit.
p(7) saturates near 1 for the top examples, so the margin separates them better.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

TEST_ID = 0

sd = torch.load("step1_model.pt")
hidden = nn.Sequential(nn.Linear(784, 800), nn.ReLU(), nn.Linear(800, 800), nn.ReLU())
hidden.load_state_dict(sd["hidden"])
last = nn.Linear(800, 10)
last.load_state_dict(sd["last"])
norm = lambda d: d.float().div(255).sub(0.1307).div(0.3081).view(-1, 784)

train = datasets.MNIST("data", train=True)
test = datasets.MNIST("data", train=False)
X = train.data[train.targets == 7]


def margin_of(images):
    with torch.no_grad():
        z = last(hidden(norm(images)))
    others = z.clone(); others[:, 7] = -float("inf")
    return (z[:, 7] - others.max(1).values).numpy()


margin = margin_of(X)
order = np.argsort(-margin)
test_margin = margin_of(test.data[TEST_ID:TEST_ID + 1])[0]
print(f"top-8 margins: {np.round(margin[order[:8]], 1)}")
print(f"test image {TEST_ID} margin: {test_margin:.1f} (median training 7: {np.median(margin):.1f}, "
      f"percentile {np.mean(margin < test_margin):.0%})")

# Figure: test image, top 8, pixel mean of the top 100
fig, axes = plt.subplots(1, 10, figsize=(13, 1.9))
axes[0].imshow(test.data[TEST_ID], cmap="gray"); axes[0].set_title(f"test {TEST_ID}", fontsize=8)
for j in range(8):
    axes[j + 1].imshow(X[order[j]], cmap="gray")
    axes[j + 1].set_title(f"margin {margin[order[j]]:.1f}", fontsize=8)
axes[9].imshow(X[order[:100]].float().mean(0), cmap="gray"); axes[9].set_title("mean top 100", fontsize=8)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Training 7s ranked by the final model's 7-logit margin", fontsize=10)
fig.tight_layout()
fig.savefig("figures/step3_confident_7s.png", dpi=130)
