"""Step 1: can we rebuild the last layer's trained weights from stored keys and errors?

Under plain SGD, each training example adds a rank-1 "write" to a linear layer:
    W  <-  W - lr * e k^T
  k = the layer's input for that example        (the key)
  e = dLoss/d(layer output) for that example    (the error; v = -e is the value)
So after training:  W_final = W_init - sum over all writes of lr * e k^T.
We store every k and e for the last layer, rebuild W_final from them, and compare.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

torch.manual_seed(0)
LR, BATCH, EPOCHS = 0.1, 128, 1

# MNIST, flattened and normalized
mnist = datasets.MNIST("data", train=True, download=True)
x = mnist.data.float().div(255).sub(0.1307).div(0.3081).view(-1, 784)
y = mnist.targets

# Irie et al.'s MLP: 784 -> 800 -> 800 -> 10
hidden = nn.Sequential(nn.Linear(784, 800), nn.ReLU(), nn.Linear(800, 800), nn.ReLU())
last = nn.Linear(800, 10)
params = list(hidden.parameters()) + list(last.parameters())
opt = torch.optim.SGD(params, lr=LR)  # plain SGD: no momentum, no weight decay

W_init = last.weight.detach().clone()
keys, errors, example_ids = [], [], []  # one row per (example, step)

for epoch in range(EPOCHS):
    for idx in torch.randperm(len(x)).split(BATCH):
        k = hidden(x[idx])                 # key: input to the last layer
        out = last(k)
        out.retain_grad()
        loss = F.cross_entropy(out, y[idx])
        opt.zero_grad()
        loss.backward()
        keys.append(k.detach())
        errors.append(out.grad.detach())   # error: dLoss/d(output), per example
        example_ids.append(idx)            # which training image each row is
        opt.step()

K = torch.cat(keys).double()     # [60000, 800]
E = torch.cat(errors).double()   # [60000, 10]
print(f"stored {len(K)} writes: keys {tuple(K.shape)}, errors {tuple(E.shape)}")

# Rebuild: W_final = W_init - lr * sum_i e_i k_i^T   (E^T K does the whole sum at once)
W_rebuilt = W_init.double() - LR * E.T @ K
W_true = last.weight.detach().double()

change = (W_true - W_init.double()).norm()
gap = (W_rebuilt - W_true).norm()
print(f"size of training's change to W : {change:.4f}")
print(f"gap between rebuilt and true W : {gap:.2e}")
print(f"relative error                 : {gap / change:.2e}")

np.savez("step1_logs.npz", keys=K.float().numpy(), errors=E.float().numpy(),
         example_ids=torch.cat(example_ids).numpy())
torch.save({"hidden": hidden.state_dict(), "last": last.state_dict()}, "step1_model.pt")
