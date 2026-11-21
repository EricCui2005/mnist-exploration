"""Step 3: does removing the top-100 examples of each ranking raise the loss on test image 0?

For each ranking (raw attention, cosine attention, attention x error, random) we:
  1. take the 100 highest-scoring training examples for test image 0,
  2. retrain the step 1 model from the same seed without them,
  3. report the change in the loss on test image 0 against the full retrain.

Same seed => same initial weights and same batch order. Removed examples are dropped from
their batch, and each batch loss is divided by 128 (not by the new batch size), so every
remaining example has the same weight as in the full run.
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

LR, BATCH = 0.1, 128
TEST_ID = int(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 0  # test image
K_REMOVE = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 100  # examples removed
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0  # retrain seed; rankings stay from seed 0
FULL = "--full" in sys.argv  # add full-network rankings (all 3 linear layers, weights only)
N_RANDOM = int(sys.argv[sys.argv.index("--random") + 1]) if "--random" in sys.argv else 1  # random draws

mnist = datasets.MNIST("data", train=True)
x = mnist.data.float().div(255).sub(0.1307).div(0.3081).view(-1, 784)
y = mnist.targets
test = datasets.MNIST("data", train=False)
x_test = test.data[TEST_ID].float().div(255).sub(0.1307).div(0.3081).view(1, 784)
y_test = test.targets[TEST_ID].view(1)


def train(removed):
    """Step 1 training, without the training images in `removed`. Returns the loss on the test image."""
    torch.manual_seed(SEED)
    hidden = nn.Sequential(nn.Linear(784, 800), nn.ReLU(), nn.Linear(800, 800), nn.ReLU())
    last = nn.Linear(800, 10)
    opt = torch.optim.SGD(list(hidden.parameters()) + list(last.parameters()), lr=LR)
    keep = torch.ones(len(x), dtype=torch.bool)
    keep[torch.as_tensor(sorted(removed), dtype=torch.long)] = False
    for idx in torch.randperm(len(x)).split(BATCH):
        idx = idx[keep[idx]]
        loss = F.cross_entropy(last(hidden(x[idx])), y[idx], reduction="sum") / BATCH
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return F.cross_entropy(last(hidden(x_test)), y_test).item()


# Scores from step 2 (one write per training image, 1 epoch)
logs = np.load("step1_logs.npz")
Kk, E, ids = logs["keys"], logs["errors"], logs["example_ids"]
sd = torch.load("step1_model.pt")
hidden = nn.Sequential(nn.Linear(784, 800), nn.ReLU(), nn.Linear(800, 800), nn.ReLU())
hidden.load_state_dict(sd["hidden"])
last = nn.Linear(800, 10); last.load_state_dict(sd["last"])
with torch.no_grad():
    q = hidden(x_test)[0].numpy()
    e_q = torch.softmax(last(torch.tensor(q)), 0).numpy() - np.eye(10)[y_test.item()]
att = Kk @ q
scores = {
    "raw attention": att,
    "cosine attention": att / np.linalg.norm(Kk, axis=1),
    "attention x error": LR * att * (E @ e_q),
    # control: the training 7s with the largest error |e_i|; uses no query, so knows nothing about the test image
    "error norm (7s)": np.where(y.numpy()[ids] == y_test.item(), np.linalg.norm(E, axis=1), -np.inf),
}


def tracin_test_side():
    """Replay the step 1 run (seed 0) and record the test query q_t and test error e_q,t before each write."""
    torch.manual_seed(0)
    h = nn.Sequential(nn.Linear(784, 800), nn.ReLU(), nn.Linear(800, 800), nn.ReLU())
    l = nn.Linear(800, 10)
    opt = torch.optim.SGD(list(h.parameters()) + list(l.parameters()), lr=LR)
    qs, eqs = [], []
    for idx in torch.randperm(len(x)).split(BATCH):
        with torch.no_grad():
            q_t = h(x_test)[0]
            e_t = torch.softmax(l(q_t), 0) - F.one_hot(y_test[0], 10)
        qs += [q_t.numpy()] * len(idx); eqs += [e_t.numpy()] * len(idx)
        loss = F.cross_entropy(l(h(x[idx])), y[idx])  # as in step 1
        opt.zero_grad(); loss.backward(); opt.step()
    return np.array(qs), np.array(eqs)


# TracIn (last layer): the same training side (k_i, e_i), but the test side at the time of each write
Q_t, Eq_t = tracin_test_side()
scores["TracIn (last layer)"] = LR * (Kk * Q_t).sum(1) * (E * Eq_t).sum(1)
r = np.corrcoef(scores["TracIn (last layer)"], scores["attention x error"])[0, 1]


def layer_terms(h, l, xb, yb, mean=True):
    """Per-example keys k_l (layer inputs) and errors e_l (dLoss/d layer output) for the 3 linear layers."""
    z1 = h[0](xb); a1 = torch.relu(z1); z2 = h[2](a1); a2 = torch.relu(z2); z3 = l(a2)
    for z in (z1, z2, z3):
        z.retain_grad()
    loss = F.cross_entropy(z3, yb, reduction="mean" if mean else "sum")
    loss.backward()
    return [xb, a1.detach(), a2.detach()], [z1.grad, z2.grad, z3.grad], loss


def full_network_terms():
    """Replay the step 1 run (seed 0). For every write and every layer, return k_i.q and e_i.e_q,
    with the test side from the final model (dual form) and from the model before the write (TracIn)."""
    # test side from the final model (the step 1 model)
    h_f = nn.Sequential(nn.Linear(784, 800), nn.ReLU(), nn.Linear(800, 800), nn.ReLU()); h_f.load_state_dict(sd["hidden"])
    l_f = nn.Linear(800, 10); l_f.load_state_dict(sd["last"])
    q_f, eq_f, _ = layer_terms(h_f, l_f, x_test, y_test)
    torch.manual_seed(0)
    h = nn.Sequential(nn.Linear(784, 800), nn.ReLU(), nn.Linear(800, 800), nn.ReLU())
    l = nn.Linear(800, 10)
    params = list(h.parameters()) + list(l.parameters())
    opt = torch.optim.SGD(params, lr=LR)
    out = {key: [[], [], []] for key in ("att_f", "err_f", "att_t", "err_t")}
    for idx in torch.randperm(len(x)).split(BATCH):
        opt.zero_grad()
        q_t, eq_t, _ = layer_terms(h, l, x_test, y_test)   # test side before this write
        opt.zero_grad()
        keys, errs, _ = layer_terms(h, l, x[idx], y[idx])   # training side, mean loss as in step 1
        for i in range(3):
            out["att_f"][i].append((keys[i] @ q_f[i][0]).numpy()); out["err_f"][i].append((errs[i] @ eq_f[i][0]).numpy())
            out["att_t"][i].append((keys[i] @ q_t[i][0]).numpy()); out["err_t"][i].append((errs[i] @ eq_t[i][0]).numpy())
        opt.step()
    gap = max((a - b).abs().max().item() for a, b in zip(params, list(h_f.parameters()) + list(l_f.parameters())))
    print(f"replay vs step 1 model: max weight gap {gap:.1e}")
    return {key: np.array([np.concatenate(v) for v in val]) for key, val in out.items()}  # [3 layers, 60000]


if FULL:
    T = full_network_terms()
    scores["raw attention (full)"] = T["att_f"].sum(0)
    scores["attention x error (full)"] = LR * (T["att_f"] * T["err_f"]).sum(0)
    scores["TracIn (full)"] = LR * (T["att_t"] * T["err_t"]).sum(0)
    # sanity: the last-layer terms must reproduce the last-layer scores
    for name, key_a, key_e in [("attention x error", "att_f", "err_f"), ("TracIn (last layer)", "att_t", "err_t")]:
        print(f"sanity: layer-3 term vs {name}: r = {np.corrcoef(LR * T[key_a][2] * T[key_e][2], scores[name])[0, 1]:.6f}")
    # share of each layer in the full scores (mean |term|)
    for name, key_a, key_e in [("attention x error", "att_f", "err_f"), ("TracIn", "att_t", "err_t")]:
        m = np.abs(T[key_a] * T[key_e]).mean(1)
        print(f"{name} (full): layer share of mean |term|: " + ", ".join(f"L{i+1} {v / m.sum():.0%}" for i, v in enumerate(m)))
    for a, b in [("attention x error (full)", "attention x error"), ("TracIn (full)", "TracIn (last layer)"),
                 ("attention x error (full)", "TracIn (full)")]:
        print(f"{a} vs {b}: r = {np.corrcoef(scores[a], scores[b])[0, 1]:.2f}")

top = {name: set(ids[np.argsort(-s)[:K_REMOVE]].tolist()) for name, s in scores.items()}

if FULL:
    for a_, b_ in [("attention x error (full)", "attention x error"), ("TracIn (full)", "TracIn (last layer)"),
                   ("raw attention (full)", "raw attention")]:
        print(f"top-{K_REMOVE} overlap {a_} / {b_}: {len(top[a_] & top[b_])}")
print(f"correlation of TracIn with attention x error: {r:.2f}; top-{K_REMOVE} overlap: "
      f"{len(top['TracIn (last layer)'] & top['attention x error'])}")
base = train(set())
with torch.no_grad():
    step1_loss = F.cross_entropy(last(hidden(x_test)), y_test).item()
print(f"seed {SEED}, test {TEST_ID} (label {y_test.item()}) full retrain: loss {base:.5f} (step 1 model: {step1_loss:.5f})")
top_rand = {"random": set(np.random.default_rng(0).choice(len(x), K_REMOVE, replace=False).tolist())}
for name, removed in ({**top, **top_rand}.items() if N_RANDOM == 1 else []):
    loss = train(removed)
    print(f"{name:18s}: loss {loss:.5f}  change {loss - base:+.5f}  ({loss / base:.2f}x)")

# Noise: how much does the loss change when we remove 100 random examples?
if N_RANDOM > 1:
    rng = np.random.default_rng(0)
    changes = np.array([train(set(rng.choice(len(x), K_REMOVE, replace=False).tolist())) - base
                        for _ in range(N_RANDOM)])
    print(f"random x{N_RANDOM}: change mean {changes.mean():+.5f}  std {changes.std(ddof=1):.5f}  "
          f"min {changes.min():+.5f}  max {changes.max():+.5f}")
    for name, removed in top.items():
        c = train(removed) - base
        print(f"{name:18s}: change {c:+.5f}  z = {(c - changes.mean()) / changes.std(ddof=1):+.1f}  "
              f"beats {np.mean(c > changes):.0%} of random draws")
