"""
Pruning sweep experiment (global pruning):
- Load a trained FCN (MNIST)
- Apply GLOBAL magnitude pruning with different prune amounts
- Make pruning permanent
- Measure sparsity and accuracy drop
- Print results as a table
"""

import os

import matplotlib.pyplot as plt
import torch.nn as nn

from nn_compression.exp_utils import (
    build_model,
    get_device,
    load_test_loader,
    load_weights,
)
from nn_compression.pruning import (
    global_magnitude_prune_linear_layers,
    make_pruning_permanent,
    model_sparsity,
    per_layer_sparsity,
)
from nn_compression.training import evaluate


def main():
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader()

    ckpt_path = "fcn_mnist_best.pt"
    prune_amounts = [0.5, 0.6, 0.7, 0.8, 0.9]
    # prune_amounts = [0.8, 0.825, 0.85, 0.875, 0.9]

    os.makedirs("results/pruning", exist_ok=True)

    # Baseline
    base_model = build_model(device)
    load_weights(base_model, ckpt_path, device)
    base_loss, base_accuracy = evaluate(base_model, test_loader, loss_fn, device)

    results = []

    for amount in prune_amounts:
        # Enures a fresh model for each amount
        model = build_model(device)
        load_weights(model, ckpt_path, device)

        # Global prune + make permanent
        global_magnitude_prune_linear_layers(model, amount=amount)
        make_pruning_permanent(model)

        sp = model_sparsity(model)

        # Evaluate after pruning
        pr_loss, pr_accuracy = evaluate(model, test_loader, loss_fn, device)

        results.append(
            (
                amount,
                sp * 100.0,
                base_accuracy,
                pr_accuracy,
                base_accuracy - pr_accuracy,
                base_loss,
                pr_loss,
            )
        )

        # Debug to see how global pruning distributes sparsity
        print("Per-layer sparsity: ", per_layer_sparsity(model))

    # Print results as a table
    print("\nGlobal Pruning Sweep Results\n")
    header = f"{'amount':>8} | {'sparsity(%)':>11} | {'acc_before':>10} | {'acc_after':>9} | {'drop':>8}"
    print(header)
    print("-" * len(header))

    for amount, sp_pct, acc_b, acc_a, drop, loss_b, loss_a in results:
        print(
            f"{amount:8.2f} | {sp_pct:11.2f} | {acc_b:10.4f} | {acc_a:9.4f} | {drop:8.4f}"
        )

    # Plot
    sparsities = [r[1] for r in results]
    drops = [r[3] for r in results]

    plt.figure()
    plt.plot(sparsities, drops, marker="o")
    plt.xlabel("Model sparsity (%)")
    plt.ylabel("Accuracy drop")
    plt.title("Accuracy Drop vs Sparsity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/pruning/accuracy_drop_vs_sparsity.png", dpi=200)

    print("\nSaved pruning plot in results/pruning/")


if __name__ == "__main__":
    main()
