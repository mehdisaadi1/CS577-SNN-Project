"""Why does SELU's variance grow after training?

The SNN fixed-point argument assumes the inputs of a unit are
independent, so that   Var(z_k) = sum_j w_kj^2 Var(x_j).
In general              Var(z_k) = w_k^T Cov(x) w_k.

For each hidden layer we print the ratio  actual / independence-prediction
(averaged over units), before and after training. A ratio near 1 means the
independence assumption holds; a ratio > 1 means positively correlated
features inflate the variance, which the fixed-point analysis does not cover.
"""

import torch
import torch.nn.functional as F

from data import make_loaders
from experiments import CONFIG
from model import SNN


def variance_ratios(model, x):
    ratios = []
    x = x.reshape(len(x), 784)
    with torch.no_grad():
        for layer in model.hidden_layers:
            xc = x - x.mean(dim=0)
            cov = xc.T @ xc / len(x)              # (fan_in, fan_in)
            W = layer.weight                       # (128, fan_in)
            actual = ((W @ cov) * W).sum(dim=1)    # w_k^T Cov w_k
            indep = (W ** 2 * torch.diag(cov)).sum(dim=1)
            ratios.append((actual / indep).mean().item())
            x = model.activation(layer(x))
    return ratios


def main(activation="selu", seed=0):
    train_loader, val_loader = make_loaders(CONFIG["batch_size"], seed)
    x_val = val_loader.dataset.tensors[0]

    torch.manual_seed(seed)
    model = SNN(activation=activation)
    before = variance_ratios(model, x_val)

    opt = torch.optim.SGD(model.parameters(), lr=CONFIG["lr"])
    for _ in range(CONFIG["epochs"]):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()
    after = variance_ratios(model, x_val)

    print(f"{activation}, seed {seed}: Var actual / Var under independence")
    print("layer   " + " ".join(f"{i:6d}" for i in range(1, 9)))
    print("init    " + " ".join(f"{r:6.2f}" for r in before))
    print("trained " + " ".join(f"{r:6.2f}" for r in after))


if __name__ == "__main__":
    main("selu")
    main("elu")
