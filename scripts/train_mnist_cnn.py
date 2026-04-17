"""
Train a CNN on MNIST.

Key ideas:
- The model outputs logits (raw class scores).
- We use multiclass CrossEntropyLoss, which internally applies softmax + log.
- We train offline (PC/laptop). GPU is allowed if available, but CPU also works.
- We save ONLY the trained weights (state_dict). The microcontroller later receives
  the trained/compressed weights for inference only (no training, no loss, no optimizer).
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nn_compression.cnn import CNN
from nn_compression.exp_utils import get_device
from nn_compression.training import evaluate, train_one_epoch


def plot_training_curves(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
):
    epochs = range(1, len(train_losses) + 1)

    os.makedirs("results/training", exist_ok=True)

    # Loss plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/training/cnn_training_loss.pdf", bbox_inches="tight")
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accuracies, label="Train accuracy")
    plt.plot(epochs, test_accuracies, label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/training/cnn_training_accuracy.pdf", bbox_inches="tight")
    plt.close()


def main():
    device = get_device()
    print("Training device:", device)

    # Dataset (MNIST)
    # MNIST images are 28x28 grayscale and labels are 0..9
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Model
    model = CNN(
        in_channels=1,
        input_height=28,
        input_width=28,
        conv_channels=[8, 16],
        kernel_size=3,
        pool_kernel_size=2,
        fc_hidden_dims=[64],
        out_dim=10,
    ).to(device)

    # Loss + Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 5
    best_test_accuracy = 0.0

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.4f}, accuracy {train_accuracy:.4f} | "
            f"Test loss {test_loss:.4f}, accuracy {test_accuracy:.4f}"
        )

        # Save the best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), "cnn_mnist_best.pt")

    # Save final model
    torch.save(model.state_dict(), "cnn_mnist_final.pt")
    print("Saved: cnn_mnist_best.pt and cnn_mnist_final.pt")

    # Plot training curves
    plot_training_curves(
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
    )


if __name__ == "__main__":
    main()