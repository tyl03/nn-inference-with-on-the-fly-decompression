"""
Unit tests for pruning utilities.

What these tests check:
- magnitude_prune_linear_layers rejects invalid prune amounts
- magnitude pruning adds PyTorch pruning attributes (weight_orig and weight_mask)
- make_pruning_permanent removes those pruning attributes
- model_sparsity increases after pruning is made permanent

What these tests do NOT check:
- model accuracy after pruning (that is part of an experiment, not a unit test)
"""

import pytest
import torch
import torch.nn as nn

from src.fcn import FCN
from src.pruning import (
    magnitude_prune_linear_layers,
    make_pruning_permanent,
    model_sparsity,
)


def _get_linear_layers(model: nn.Module):
    """
    Helper to return all linear layers in the model.
    """
    return [layer for layer in model.modules() if isinstance(layer, nn.Linear)]


def test_prune_amount_out_of_range_raises():
    model = FCN(in_dim=28 * 28, hidden_dims=[64], out_dim=10)
    
    with pytest.raises(ValueError):
        magnitude_prune_linear_layers(model, amount=-0.1)
        
    with pytest.raises(ValueError):
        magnitude_prune_linear_layers(model, amount=1.1)
        
        
def test_magnitude_pruning_adds_mask_attributions():
    model = FCN(in_dim=28 * 28, hidden_dims=[64], out_dim=10)
    
    magnitude_prune_linear_layers(model, amount=0.5)
    
    linear_layers = _get_linear_layers(model)
    assert len(linear_layers) > 0 # sanity check
    
    # After pruning, PyTorch usually adds `weight_orig` and `weight_mask`
    for layer in linear_layers:
        assert hasattr(layer, "weight_orig")
        assert hasattr(layer, "weight_mask")
        
        
def test_make_pruning_permanent_removes_mask_attributes():
    model = FCN(in_dim=28 * 28, hidden_dims=[64], out_dim=10)
    
    magnitude_prune_linear_layers(model, amount=0.5)
    make_pruning_permanent(model)
    
    linear_layers = _get_linear_layers(model)
    
    # After having run `prune.remove`, `weight_orig` and `weight_mask` should be gone
    for layer in linear_layers:
        assert not hasattr(layer, "weight_orig")
        assert not hasattr(layer, "weight_mask")


def test_model_sparsity_increases_after_pruning_is_permanent():
    model = FCN(in_dim=28 * 28, hidden_dims=[64], out_dim=10)
    
    before = model_sparsity(model)
    assert 0.0 <= before <= 1.0
    
    magnitude_prune_linear_layers(model, amount=0.5)
    make_pruning_permanent(model)
    
    after = model_sparsity(model)
    assert 0.0 <= after <= 1.0
    
    # We check if the sparsity increases
    assert after > before
    