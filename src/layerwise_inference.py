"""
Layerwise inference on a compressed model representation.

During inference, we only decompress ONE layer at a time:
    1) load compressed weights for one layer
    2) dequantize (decompress) just that layer
    3) compute the layer output
    4) discard the decompressed weights
    
This reduces peak memory usage because we never hold the full decompressed model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .quantization import symmetric_dequantization


def _flatten_input(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


@torch.no_grad()
def layerwise_forward(compressed: dict, x: torch.Tensor) -> torch.Tensor:
    """
    Runs a forward pass (logits output) using layer-by-layer decompression.

    Returns:
    - logits (float32): shape [batch_size, out_dim]
    """
    # 1) Flatten input
    input_flat = _flatten_input(x)
    
    # 2) Validate input size
    expected_in_dim = int(compressed["in_dim"])
    if input_flat.shape[1] != expected_in_dim:
        raise ValueError(
            f"Expected input with {expected_in_dim} features, got {input_flat.shape[1]}"
        )
    
    # 3) Apply each layer entry in order
    for entry in compressed["layers"]:
        layer_type = entry["type"]
        
        if layer_type == "linear":
            # Load compressed weights
            W_q = entry["W_q"]
            s_w = entry["s_w"]
            b = entry["b"]
            
            # Decompress just this layer
            W = symmetric_dequantization(W_q, s_w)
            
            # Compute: x <- x @ W^T + b
            input_flat = F.linear(input_flat, W, b)
            
            # Discard decompressed weights
            del W
            
        elif layer_type == "relu":
            input_flat = F.relu(input_flat)
            
        else:
            raise ValueError(f"Unknown layer type in compressed model: {layer_type}")
        
    return input_flat # logits


@torch.no_grad()
def layerwise_predict(compressed: dict, x: torch.Tensor) -> torch.Tensor:
    logits = layerwise_forward(compressed, x)
    return logits.argmax(dim=1)


@torch.no_grad()
def layerwise_evaluate_accuracy(compressed: dict, loader, device: torch.device) -> float:
    """
    Evaluates accuracy on a DataLoader using layerwise inference.

    Notes:
    - We keep the ACTIVATIONS on the chosen device.
    - The compressed layer weights are dequantized on-the-fly.

    Returns:
    - accuracy in [0, 1]
    """
    correct = 0
    total = 0
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        preds = layerwise_predict(compressed, x)
        correct += (preds == y).sum().item()
        total += y.numel()
        
    return correct / total if total > 0 else 0.0