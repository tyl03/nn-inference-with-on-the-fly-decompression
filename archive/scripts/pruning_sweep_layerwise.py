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

from src.exp_utils import (
    get_device,
    load_test_loader,
    build_model,
    load_weights,
)
from src.training import evaluate
from src.pruning import (
    magnitude_prune_linear_layers,
    make_pruning_permanent,
    model_sparsity,
)


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
    header = f"{'amount':>8} | {'sparsity(%)':>11} | {'accuracy_before':>10} | {'accuracy_after':>9} | {'drop':>8}"
    print(header)
    print("-" * len(header))
    
    for amount, sp_pct, accuracy_b, accuracy_a, drop, loss_b, loss_a in results:
        print(f"{amount:8.2f} | {sp_pct:11f} | {accuracy_b:15f} | {accuracy_a:14f} | {drop:8.4f}")
            
    
if __name__ == "__main__":
    main()