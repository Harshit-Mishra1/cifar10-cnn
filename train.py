"""Train and evaluate a CNN on CIFAR-10 using PyTorch."""

from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class CNN(nn.Module):
    """A compact 3-block CNN for 32x32 RGB CIFAR-10 images."""

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_layers(x)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_dir = Path("data")
    trainset = CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    testset = CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)

    trainloader = DataLoader(
        trainset, batch_size=64, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    testloader = DataLoader(
        testset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1:02d}/{epochs} | training loss: {average_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"\nTest accuracy: {accuracy:.2f}% ({correct}/{total})")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), artifacts_dir / "cifar10_cnn.pth")
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"dataset": "CIFAR-10", "epochs": epochs, "test_accuracy_percent": round(accuracy, 4)},
            f, indent=2
        )
    print(f"Saved model and metrics to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
