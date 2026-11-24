
import torch
import torch.nn.functional as F
from model import SNN


def test_shapes():
    model = SNN(activation="selu")
    x = torch.randn(4, 1, 28, 28)

    logits, activations = model(
        x, return_activations=True
    )

    assert logits.shape == (4, 10)
    assert len(activations) == 8

    for a in activations:
        assert a.shape == (4, 128)

    print("Shape tests passed!")


def test_gradients():
    model = SNN(activation="selu")

    x = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])

    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name

    print("Gradient tests passed!")


def test_selu():
    x = torch.tensor([-1.0, 1.0], requires_grad=True)

    y = F.selu(x)

    alpha = 1.6732632423543772
    scale = 1.0507009873554805

    expected = torch.where(
        x > 0,
        scale * x,
        scale * alpha * (torch.exp(x) - 1)
    )

    assert torch.allclose(y, expected, atol=1e-6)

    y.sum().backward()

    expected_grad = torch.where(
        x > 0,
        torch.full_like(x, scale),
        scale * alpha * torch.exp(x)
    )

    assert torch.allclose(
        x.grad, expected_grad, atol=1e-6
    )

    print("SELU value and gradient tests passed!")


if __name__ == "__main__":
    test_shapes()
    test_gradients()
    test_selu()

    print("All initial tests passed!")
