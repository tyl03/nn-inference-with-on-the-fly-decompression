"""
Train an FCN on MNIST (dataset).

Key ideas:
- The model outputs logits (raw class scores).
- We use multiclass CrossEntropyLoss, which internally applies softmax + log.
- We train offline (PC/laptop). GPU is allowed if available, but CPU also works.
- We save ONLY the trained weights (state_dict). The microcontroller later receives
  the trained/compressed weights for inference only (no training, no loss, no optimizer).
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.fcn import FCN
from src.training import get_device, train_one_epoch, evaluate


def plot_training_curves(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
):
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=200)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accuracies, label="Train accuracy")
    plt.plot(epochs, test_accuracies, label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_accuracy.png", dpi=200)
    plt.close()
    
    
# Main training script
def main():
    device = get_device()
    print("Training device: ", device)
    
    # Dataset (MNIST)
    # MNIST images are 28x28 and labels are 0..9 (total of 10 classes)
    transform = transforms.ToTensor()
    
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    # Model
    in_dim = 28 * 28
    out_dim = 10
    hidden_dims = [512, 256]
    
    model = FCN(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim).to(device)
    
    # Loss + Optimizer
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # lr = learning rate
    
    
    # Training loop
    epochs = 3 # After reviewing the plots, epochs = 3 was the best option for this dataset
    best_test_accuracy = 0.0
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        test_loss, test_accuracy = evaluate(
            model, test_loader, loss_fn, device
        )
        
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
            torch.save(model.state_dict(), "fcn_mnist_best.pt")
            
    # Save final model
    torch.save(model.state_dict(), "fcn_mnist_final.pt")
    print("Saved: fcn_mnist_best.pt and fcn_mnist_final.pt")
    
    # Plot training curves
    plot_training_curves(
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
    )
    
if __name__ == "__main__":
    main()
    