"""
Defines a Convolutional Neural Network (CNN).
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        input_height: int,
        input_width: int,
        conv_channels: list[int],
        kernel_size: int,
        pool_kernel_size: int,
        fc_hidden_dims: list[int],
        out_dim: int
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.pool_kernel = pool_kernel_size
        self.fc_hidden_dims = fc_hidden_dims
        self.out_dim = out_dim
        
        # Convolutional feature extractor
        feature_layers = []
        
        current_channels = in_channels
        current_height = input_height
        current_width = input_width

        for out_channels in conv_channels:
            feature_layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size)
            )
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.MaxPool2d(pool_kernel_size))

            current_channels = out_channels

            # Keep track of spatial size after Conv + Pool
            current_height = current_height - kernel_size + 1
            current_width = current_width - kernel_size + 1

            current_height = current_height // pool_kernel_size
            current_width = current_width // pool_kernel_size

        self.features = nn.Sequential(*feature_layers)
        
        # Compute flattened feature size after conv layers to determine input size for FCN
        flattened_dim = current_channels * current_height * current_width
        self.flatten_dim = flattened_dim
        
        fully_connected_layers = []

        prev_dim = flattened_dim
        for h in fc_hidden_dims:
            fully_connected_layers.append(nn.Linear(prev_dim, h))
            fully_connected_layers.append(nn.ReLU())
            prev_dim = h

        fully_connected_layers.append(nn.Linear(prev_dim, out_dim))

        self.fully_connected = nn.Sequential(*fully_connected_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten before fully connected layers
        x = self.fully_connected(x)
        return x