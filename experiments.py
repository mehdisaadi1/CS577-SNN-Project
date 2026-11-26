"""SELU vs ELU matched ablation.

Everything except the activation is fixed: architecture, 1/fan_in
initialization, data split, optimizer, batch size, seeds and number of
updates. For a given seed, both activations start from the *same*
initial weights and see the *same* mini-batch order.

Usage:
    python experiments.py            # 2 activations x 3 seeds
    python experiments.py --pilot    # 1 epoch per config, timing check
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from activation_stats import measure_activation_stats
from data import make_loaders
from model import SNN

CONFIG = {
    "batch_size": 128,
    "epochs": 20,
    "lr": 0.01,
    "optimizer": "sgd",
    "seeds": [0, 1, 2],
    "activations": ["selu", "elu"],
}

RESULTS_DIR = "results/runs"


def evaluate(model, loader):
    """Mean cross-entropy and accuracy over a loader."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def snapshot(model, val_loader, epoch, train_loss, n_updates, epoch_time):
    """Everything we log at the end of an epoch (epoch 0 = initialization)."""
    stats, D = measure_activation_stats(model, val_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    return {
        "epoch": epoch,
        "updates": n_updates,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "D": D,
        "layer_mean": [m for m, _ in stats],
        "layer_var": [v for _, v in stats],
        "epoch_time_s": epoch_time,
    }


def run(activation, seed, config):
    train_loader, val_loader = make_loaders(config["batch_size"], seed)

    # Same seed -> same initial weights for SELU and ELU.
    torch.manual_seed(seed)
    model = SNN(activation=activation)

    if config["optimizer"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=config["lr"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    history = [snapshot(model, val_loader, 0, None, 0, 0.0)]
    n_updates = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
            n_updates += 1
        epoch_time = time.perf_counter() - t0

        history.append(
            snapshot(model, val_loader, epoch, loss_sum / n,
                     n_updates, epoch_time)
        )
        h = history[-1]
        print(f"  [{activation} seed={seed}] epoch {epoch:2d} | "
              f"train {h['train_loss']:.4f} | val {h['val_loss']:.4f} | "
              f"acc {100 * h['val_acc']:.2f}% | D {h['D']:.4f}")

    return {
        "activation": activation,
        "seed": seed,
        "config": {k: v for k, v in config.items()
                   if k not in ("seeds", "activations")},
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="1 epoch per config, results not saved")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = dict(CONFIG)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.pilot:
        config["epochs"] = 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.perf_counter()

    for activation in config["activations"]:
        for seed in config["seeds"]:
            t0 = time.perf_counter()
            result = run(activation, seed, config)
            elapsed = time.perf_counter() - t0
            result["wall_time_s"] = elapsed
            print(f"{activation} seed={seed}: {elapsed:.1f}s, "
                  f"D init={result['history'][0]['D']:.4f} -> "
                  f"final={result['history'][-1]['D']:.4f}")

            if not args.pilot:
                path = os.path.join(RESULTS_DIR,
                                    f"{activation}_seed{seed}.json")
                with open(path, "w") as f:
                    json.dump(result, f, indent=1)

    print(f"Total: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
