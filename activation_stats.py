
import torch


def measure_activation_stats(model, loader):
    model.eval()
    layer_values = [[] for _ in range(8)]

    with torch.no_grad():
        for images, _ in loader:
            _, activations = model(
                images, return_activations=True
            )

            for i, a in enumerate(activations):
                layer_values[i].append(a.flatten())

    stats = []

    for chunks in layer_values:
        values = torch.cat(chunks)
        mean = values.mean().item()
        var = values.var(unbiased=False).item()
        stats.append((mean, var))

    D = sum(
        abs(mu) + abs(var - 1)
        for mu, var in stats
    ) / len(stats)

    return stats, D
