"""
Global unstructured magnitude pruning utilities for Conv2d + Linear layers.

- Global pruning: prunes the smallest weights across all Conv2d and Linear layers combined.
- Permanent pruning: removes the mask reparameterization so zeros are stored in weight.
"""


import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def global_magnitude_prune_conv_and_linear_layers(
    model: nn.Module, amount: float
) -> nn.Module:
    """
    Global unstructured magnitude pruning across ALL nn.Conv2d and nn.Linear layer weights.
    """
    if not (0.0 <= amount <= 1.0):
        raise ValueError("The wanted amount must be between 0.0 and 1.0")

    parameters_to_prune = [
        (layer, "weight")
        for layer in model.modules()
        if isinstance(layer, (nn.Conv2d, nn.Linear))
    ]

    if not parameters_to_prune:
        raise ValueError("No nn.Conv2d or nn.Linear layers found in the model to prune.")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    return model


def global_magnitude_prune_linear_layers(model: nn.Module, amount: float) -> nn.Module:
    """
    Global unstructured magnitude pruning across ALL nn.Linear layer weights.
    """
    if not (0.0 <= amount <= 1.0):
        raise ValueError("The wanted amount must be between 0.0 and 1.0")

    parameters_to_prune = [
        (layer, "weight") for layer in model.modules() if isinstance(layer, nn.Linear)
    ]

    if not parameters_to_prune:
        raise ValueError("No nn.Linear layers found in the model to prune.")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    return model


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Since PyTorch pruning represents pruned weights as weight = weight_orig * weight_mask,
    calling prune.remove materializes this computation by storing the result directly in weight,
    thereby converting masked pruning into permanent zero-valued weights.
    """
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)) and hasattr(layer, "weight_orig"):
            prune.remove(layer, "weight")

    return model


def model_sparsity(model: nn.Module) -> float:
    """
    Fraction of weights that are exactly 0.0 across all Conv2d and Linear layers.
    """
    zeros = 0
    total = 0

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            w = layer.weight.detach()
            total += w.numel()
            zeros += w.numel() - torch.count_nonzero(w).item()

    return zeros / total if total > 0 else 0.0


def per_layer_sparsity(model: nn.Module):
    """
    List of (layer_name, sparsity) for each Conv2d / Linear layer.
    Useful to see how global pruning distributes sparsity.
    """
    out = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            w = layer.weight.detach()
            sparsity = (w == 0).float().mean().item()
            out.append((name, sparsity))

    return out
