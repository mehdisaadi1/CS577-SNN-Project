
import torch
import torch.nn as nn


class SNN(nn.Module):
    def __init__(self, activation="selu"):
        super().__init__()

        if activation not in ["selu", "elu"]:
            raise ValueError("Activation must be selu or elu")

        self.activation = (
            nn.SELU() if activation == "selu"
            else nn.ELU()
        )

        self.hidden_layers = nn.ModuleList()

        input_dim = 784

        # Eight hidden layers, each with 128 neurons
        for _ in range(8):
            layer = nn.Linear(input_dim, 128)

            # Variance = 1 / fan_in
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=(1 / input_dim) ** 0.5
            )
            nn.init.zeros_(layer.bias)

            self.hidden_layers.append(layer)
            input_dim = 128

        # Output layer for MNIST (10 classes)
        self.output_layer = nn.Linear(128, 10)

        nn.init.normal_(
            self.output_layer.weight,
            mean=0.0,
            std=(1 / 128) ** 0.5
        )
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, return_activations=False):
        x = x.reshape(x.shape[0], 784)

        activations = []

        for layer in self.hidden_layers:
            x = self.activation(layer(x))

            if return_activations:
                activations.append(x)

        logits = self.output_layer(x)

        if return_activations:
            return logits, activations

        return logits
