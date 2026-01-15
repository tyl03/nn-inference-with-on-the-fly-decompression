"""
Pruning experiment:
- Load a trained FCN (MNIST)
- Apply magnitude pruning with different prune amounts
- Make pruning permanent
- Measure sparsity and accuracy drop
- Print results as a table
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.fcn import FCN
from src.training import get_device, evaluate
from src.pruning import (
    magnitude_prune_linear_layers,
    make_pruning_permanent,
    model_sparsity,
)


def load_test_loader():
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=256, shuffle=False)


def build_model(device: torch.device):
    return FCN(in_dim=28 * 28, hidden_dims=[512, 256], out_dim=10).to(device)


def load_weights(model: nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


def main():
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader()
    
    ckpt_path = "fcn_mnist_best.pt"
    prune_amounts = [0.4, 0.5, 0.6, 0.65, 0.7]
    
    results = []
    
    for amount in prune_amounts:
        # Enures a fresh model for each amount
        model = build_model(device)
        load_weights(model, ckpt_path, device)
        
        # Baseline evaluation (same weight each time)
        base_loss, base_accuracy = evaluate(model, test_loader, loss_fn, device)
        
        # Prune and make it permanent
        magnitude_prune_linear_layers(model, amount=amount)
        make_pruning_permanent(model)
        
        sp = model_sparsity(model)
        
        # Evaluate after pruning
        pr_loss, pr_accuracy = evaluate(model, test_loader, loss_fn, device)
        
        results.append(
            (amount, sp * 100.0, base_accuracy, pr_accuracy, base_accuracy - pr_accuracy, base_loss, pr_loss)
        )
        
    
    # Print results as a table
    print("\nPruning Sweep Results\n")
    header = f"{'amount':>8} | {'sparsity(%)':>9} | {'accuracy_before':>10} | {'accuracy_after':>9} | {'drop':>0}"
    print(header)
    print("-" * len(header))
    
    for amount, sp_pct, accuracy_b, accuracy_a, drop, loss_b, loss_a in results:
        print(f"{amount:8.2f} | {sp_pct:9.2f} | {accuracy_b:10.4f} | {accuracy_a:9.4f} | {drop:8.4f}")
            
    
if __name__ == "__main__":
    main()