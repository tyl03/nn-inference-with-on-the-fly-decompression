"""
Defines the neural network.
"""

import torch
import torch.nn as nn

class FCN(nn.Module):
    """
    Fully Connected Network
    - Works for any input size (as long as it flattens to `in_dim`)
    - Hidden layers is defined as `hidden_dims`
    - Output size is defined as `out_dim`
    """
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        
        self.in_dim = in_dim
        self.flatten = nn.Flatten()
        
        layers = []
        prev_layer_size = in_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_layer_size, h))
            layers.append(nn.ReLU())
            prev_layer_size = h
            
        layers.append(nn.Linear(prev_layer_size, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.flatten(x) # works for images or vectors
        
        # Error checking
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Expected input with {self.in_dim} features. Got {x.shape[1]}.")
        
        return self.net(x)