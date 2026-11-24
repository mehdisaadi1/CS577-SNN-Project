
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import SNN

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

images = torch.stack([dataset[i][0] for i in range(16)])
labels = torch.tensor([dataset[i][1] for i in range(16)])

model = SNN(activation="selu")
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for step in range(1, 401):
    optimizer.zero_grad()

    logits = model(images)
    loss = F.cross_entropy(logits, labels)

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        predictions = model(images).argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

    if step == 1 or step % 50 == 0:
        print(
            f"Step {step:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy * 100:.1f}%"
        )

    if accuracy == 1.0 and loss.item() < 0.02:
        print(f"PASSED at step {step}!")
        break
else:
    raise AssertionError("Tiny-batch overfit target not reached")
