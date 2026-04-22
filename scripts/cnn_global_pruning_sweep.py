"""
CNN global pruning sweep experiment (MNIST).

Purpose:
- Find a reasonable global pruning amount for CNNs before running the final
  blockwise compression experiment.

Compares:
- Baseline FP32 CNN
- Global Conv+FC pruning

For each pruning amount, reports:
- Accuracy
- Sparsity

Also saves:
- Accuracy vs pruning amount plot
- Sparsity vs pruning amount plot
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from nn_compression.exp_utils import (
    build_model,
    get_device,
    load_test_loader,
    load_weights,
)
from nn_compression.pruning import (
    global_magnitude_prune_conv_and_linear_layers,
    make_pruning_permanent,
    model_sparsity,
)
from nn_compression.training import evaluate


def save_accuracy_plot(
    *,
    prune_amounts: list[float],
    accuracies: list[float],
    baseline_accuracy: float,
    out_path: str,
) -> None:
    """
    Save plot of test accuracy vs pruning amount.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(prune_amounts, accuracies, marker="o", label="Global Conv+FC pruning")
    plt.axhline(
        y=baseline_accuracy,
        linestyle="--",
        label=f"Baseline ({baseline_accuracy:.4f})",
    )
    plt.xlabel("Prune amount")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_sparsity_plot(
    *,
    prune_amounts: list[float],
    sparsities: list[float],
    out_path: str,
) -> None:
    """
    Save plot of model sparsity vs pruning amount.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(prune_amounts, sparsities, marker="o", label="Global Conv+FC pruning")
    plt.xlabel("Prune amount")
    plt.ylabel("Global sparsity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def print_summary_table(
    *,
    prune_amounts: list[float],
    baseline_accuracy: float,
    accuracies: list[float],
    sparsities: list[float],
) -> None:
    """
    Print a readable summary table to the terminal.
    """
    print("\n" + "=" * 86)
    print("CNN Global Pruning Sweep (MNIST)".center(86))
    print("=" * 86)

    print("\n[Baseline]")
    print(f"  Accuracy: {baseline_accuracy:.4f}")

    print("\n[Results]")
    header = (
        f"{'Prune amount':>14}"
        f"{'Sparsity':>14}"
        f"{'Accuracy':>14}"
        f"{'Acc. drop':>14}"
    )
    print(header)
    print("-" * len(header))

    for i, amount in enumerate(prune_amounts):
        acc = accuracies[i]
        sp = sparsities[i] * 100.0

        print(
            f"{amount:>14.2f}"
            f"{sp:>13.2f}%"
            f"{acc:>14.4f}"
            f"{(baseline_accuracy - acc):>+14.4f}"
        )

    print("=" * 86 + "\n")


def main():
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader(batch_size=1)

    ckpt_path = "cnn_mnist_best.pt"

    # Pruning amounts to test
    prune_amounts = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # 1) Baseline CNN
    baseline_model = build_model(device, model_type="cnn")
    load_weights(baseline_model, ckpt_path, device)
    _, baseline_accuracy = evaluate(baseline_model, test_loader, loss_fn, device)

    print(f"Baseline accuracy: {baseline_accuracy:.4f}")

    # Store results
    accuracies = []
    sparsities = []

    for amount in prune_amounts:
        print(f"\nTesting prune amount = {amount:.2f}")

        model = build_model(device, model_type="cnn")
        load_weights(model, ckpt_path, device)

        global_magnitude_prune_conv_and_linear_layers(model, amount=amount)
        make_pruning_permanent(model)

        _, accuracy = evaluate(model, test_loader, loss_fn, device)
        sparsity = model_sparsity(model)

        accuracies.append(accuracy)
        sparsities.append(sparsity)

        print(
            f"  Global Conv+FC | sparsity {sparsity*100:.2f}% | accuracy {accuracy:.4f}"
        )

    # Print final summary
    print_summary_table(
        prune_amounts=prune_amounts,
        baseline_accuracy=baseline_accuracy,
        accuracies=accuracies,
        sparsities=sparsities,
    )

    # Save plots
    save_accuracy_plot(
        prune_amounts=prune_amounts,
        accuracies=accuracies,
        baseline_accuracy=baseline_accuracy,
        out_path="results/pruning/cnn_global_pruning_sweep_accuracy.pdf",
    )
    print("Saved: results/pruning/cnn_global_pruning_sweep_accuracy.pdf")

    save_sparsity_plot(
        prune_amounts=prune_amounts,
        sparsities=sparsities,
        out_path="results/pruning/cnn_global_pruning_sweep_sparsity.pdf",
    )
    print("Saved: results/pruning/cnn_global_pruning_sweep_sparsity.pdf")


if __name__ == "__main__":
    main()