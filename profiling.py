"""Lightweight CPU efficiency profile.

Reports hardware, parameter count, batch size, warm-up, mean inference
time over several short runs, and peak process memory (Linux RSS).
"""

import json
import os
import platform
import resource
import statistics as st
import time

import torch

from data import make_loaders
from experiments import CONFIG
from model import SNN

WARMUP_ITERS = 10
N_RUNS = 5


def time_inference(model, loader):
    """Seconds to run the model over the whole loader once."""
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, _ in loader:
            model(x)
    return time.perf_counter() - t0


def main():
    batch_size = CONFIG["batch_size"]
    _, val_loader = make_loaders(batch_size, seed=0,
                                 val_batch_size=batch_size)
    n_samples = len(val_loader.dataset)

    print(f"Hardware: {platform.processor() or platform.machine()}, "
          f"{os.cpu_count()} logical CPUs, "
          f"torch threads={torch.get_num_threads()}")
    print(f"Python {platform.python_version()}, torch {torch.__version__}")
    print(f"Batch size: {batch_size}, warm-up: {WARMUP_ITERS} batches, "
          f"timed runs: {N_RUNS} x {n_samples} validation images")

    report = {}
    for activation in CONFIG["activations"]:
        torch.manual_seed(0)
        model = SNN(activation=activation).eval()
        n_params = sum(p.numel() for p in model.parameters())

        # Warm-up: first calls pay for allocation / kernel selection.
        x0 = next(iter(val_loader))[0]
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                model(x0)

        times = [time_inference(model, val_loader) for _ in range(N_RUNS)]
        m, s = st.mean(times), st.stdev(times)
        per_batch_ms = 1000 * m / len(val_loader)
        throughput = n_samples / m

        report[activation] = {
            "params": n_params,
            "val_pass_s_mean": m,
            "val_pass_s_std": s,
            "per_batch_ms": per_batch_ms,
            "images_per_s": throughput,
        }
        print(f"{activation.upper()}: {n_params:,} params | "
              f"{1000 * m:.1f} ± {1000 * s:.1f} ms per validation pass | "
              f"{per_batch_ms:.2f} ms/batch | {throughput:,.0f} images/s")

    # ru_maxrss is in KB on Linux: peak memory of the whole Python process,
    # including the loaded dataset, not just the model.
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak process RSS: {peak_mb:.0f} MB")
    report["peak_rss_mb"] = peak_mb
    report["hardware"] = {
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/profiling.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
