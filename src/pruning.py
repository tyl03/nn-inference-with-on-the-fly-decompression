"""
Takes a model, and prunes its weights.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def magnitude_prune_linear_layers(model: nn.Module, amount: float) -> nn.Module:
    """
    Unstructured magnitude pruning for neural networks with linear layers.
    It sets the fraction of smallest |weights| to 0 using a mask.
    """
    if not (0.0 <= amount <= 1.0):
        raise ValueError("The wanted amount must be between 0.0 and 1.0")
    
    for layer in model.modules(): # gives every sub-layer inside the model
        if isinstance(layer, nn.Linear):
            prune.l1_unstructured(layer, name="weight", amount=amount)
    
    return model
        
        
def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Since PyTorch pruning represents pruned weights as weight = weight_orig * weight_mask, 
    calling prune.remove materializes this computation by storing the result directly in weight, 
    thereby converting masked pruning into permanent zero-valued weights.
    
    When calling `prune.remove(layer, "weight)`, PyTorch internally does:
    - new_weight = weight_orig * weight_mask
    - delete weight_orig
    - delete weight_mask
    - register new_weight as layer.weight
    
    """
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if hasattr(layer, "weight_orig"):
                prune.remove(layer, "weight")
                
    return model
            

def linear_layer_sparsity(layer: nn.Linear) -> float:
    """
    The fraction of weights that are exactly 0.0 in this layer.
    """
    w = layer.weight.detach()
    return (w == 0).float().mean().item()


def model_sparsity(model: nn.Module) -> float:
    """
    The fraction of weights that are exatly 0.0 across all linear layers.
    """
    zeros = 0
    total = 0
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach()
            zeros += (w == 0).sum().item()
            total += w.numel()
            
    return zeros / total if total > 0 else 0.0