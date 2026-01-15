"""
These tests verify the basic structure and behavior of the FCN model.

What these tests check:
- The model accepts image-shaped input (N, 1, 28, 28) and produces output
  with shape (N, number_of_classes)
- The model also accepts flat vector input (N, in_dim) and produces correct
  output shape
- The model contains the expected number of Linear layers based on the
  specified hidden dimensions

These tests ensure that the FCN forward pass and network structure behave
as expected.
"""


import torch
import torch.nn as nn
from src.fcn import FCN

def test_forward_shape_for_img_input():
    # Create the model
    model = FCN(in_dim=28 * 28, hidden_dims=[256, 128], out_dim=10)
    
    # Create fake input data (a batch)
    x = torch.randn(32, 1, 28, 28)
    
    # Run the model
    y = model(x)
    assert y.shape == (32, 10)
    

def test_forward_shape_for_vector_input():
    model = FCN(in_dim=100, hidden_dims=[64], out_dim=3)
    x = torch.randn(8, 100)
    y = model(x)
    assert y.shape == (8, 3)
    

def test_model_contains_linear_layers():
    # in_dims -> 5 -> 4 -> out_dim
    hidden_dims= [5, 4]
    model = FCN(in_dim=10, hidden_dims=hidden_dims, out_dim=2)
    
    linear_layers = [
        layer for layer in model.net 
        if isinstance(layer, nn.Linear)
    ]
    
    expected = len(hidden_dims) + 1
    assert len(linear_layers) == expected