"""Fixed, standardized MNIST subset used by every experiment.

The train/validation split depends only on SPLIT_SEED, never on the
training seed, so SELU and ELU runs (and all seeds) see exactly the
same 10,000 training and 2,000 validation images.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

N_TRAIN = 10_000
N_VAL = 2_000
SPLIT_SEED = 0
DATA_ROOT = "./data"  # ignored by git, kept outside the submission archive


def load_mnist_subset(root=DATA_ROOT):
    """Return (x_train, y_train), (x_val, y_val), (mean, std).

    Images are (N, 1, 28, 28) float tensors standardized with the
    statistics of the training subset only (no information from the
    validation images leaks into the preprocessing).
    """
    mnist = datasets.MNIST(root=root, train=True, download=True)

    # uint8 (60000, 28, 28) -> float (60000, 1, 28, 28) in [0, 1]
    x = mnist.data.unsqueeze(1).float() / 255.0
    y = mnist.targets.clone()

    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(x), generator=g)
    train_idx = perm[:N_TRAIN]
    val_idx = perm[N_TRAIN:N_TRAIN + N_VAL]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    # Zero mean / unit variance input: SELU's fixed point assumes it.
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    return (x_train, y_train), (x_val, y_val), (mean.item(), std.item())


def make_loaders(batch_size, seed, root=DATA_ROOT, val_batch_size=500):
    """Train loader shuffled by `seed`; validation loader in fixed order."""
    (x_train, y_train), (x_val, y_val), _ = load_mnist_subset(root)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=val_batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    (xt, yt), (xv, yv), (m, s) = load_mnist_subset()
    print(f"train: {tuple(xt.shape)}  val: {tuple(xv.shape)}")
    print(f"standardization constants: mean={m:.4f}, std={s:.4f}")
    print(f"train mean/var after standardization: "
          f"{xt.mean():.4f} / {xt.var():.4f}")
    print(f"val   mean/var after standardization: "
          f"{xv.mean():.4f} / {xv.var():.4f}")
    print(f"train class counts: {torch.bincount(yt).tolist()}")
