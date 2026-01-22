"""
Exports a trained FCN into a compressed, layer-by-layer format.

Why this file exists:
- In normal PyTorch inference, the entire model (all layers) is loaded at once.
- In this bachelor project, we want to store the model in a compressed form.
- During inference, we only "open" (decompress) ONE layer at a time, use it,
  then discard it before moving to the next layer.
- This file creates the "packed bags" (compressed layers) that layerwise inference can use.

What we store for each Linear layer:
- W_q: int8 weights (smaller than float32)
- s_w: scale for dequantization (float)
- b: bias (float32, small, easy to keep)
- in_features / out_features (so we know the shape)

We also store ReLU entries so the layer order is preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .quantization import compute_scale, symmetric_quantization


def export_fcn_to_compressed(model: nn.Module) -> dict:
    """
    Converts a FCN to a compressed representation:
    - A list of layer entries in forward order.
    
    This assumes that our model has:
    - model.in_dim 
    - model.net as nn.Sequential that contains [Linear/ReLU/Linear/ReLU/.../Linear]
    
    It returns a Python dict that can be saved with torch.save().
    """
    if not hasattr(model, "in_dim"):
        raise ValueError("Model must have 'in_dim'.")
    if not hasattr(model, "net"):
        raise ValueError("Model must have 'net'.")
    
    layers_out: list[dict] = []
    
    # We export layers in the exact order they are used in forward().
    for layer in model.net:
        # 1) Linear layer: store quantized weight, scale and bias
        if isinstance(layer, nn.Linear):
            # Move to CPU so the exported file is device-independent;
            # Meaning it doesn't depend on having a GPU available.
            W = layer.weight.detach().cpu()
            b = layer.bias.detach().cpu() if layer.bias is not None else None
            
            # Compute per-layer scale and quantize to int8
            s_w = compute_scale(W) # float
            W_q = symmetric_quantization(W, s_w) # int8 tensor
            
            layers_out.append({
                "type": "linear",
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "W_q": W_q, # int8 weights
                "s_w": float(s_w), # scale needed for dequantization
                "b": b, # bias kept as FP32
            })
            
        # 2) ReLU layer: store marker only
        elif isinstance(layer, nn.ReLU):
            layers_out.append({
                "type": "relu"
            })
            
        # 3) Anything else: not supported in this version
        else:
            raise ValueError(f"Unsupported layer type in export: {type(layer)}")
        
    
    compressed = {
        "format_version": 1,
        "model_type": "FCN",
        "in_dim": int(model.in_dim),
        "layers": layers_out,
    }
    
    return compressed


def save_compressed(compressed: dict, path: str) -> None:
    """
    Saves the compressed model to disk.
    torch.save is fine for experiments.
    """
    torch.save(compressed, path)
    
    
def load_compressed(path: str) -> dict:
    """
    Loads a compressed model from disk (saved by save_compressed).
    """
    return torch.load(path, map_location="cpu")


def estimate_compressed_weight_bytes(compressed: dict) -> int:
    """
    Estimates storage size for weights + scales (and bias) in the compressed dict.

    - W_q stored as int8 -> 1 byte per weight
    - s_w stored as FP32 -> 4 bytes per Linear layer
    - bias stored as FP32 -> 4 bytes per bias value
    """
    total_bytes = 0
    
    for entry in compressed["layers"]:
        if entry["type"] == "linear":
            W_q = entry["W_q"]
            b = entry["b"]
            
            # int8 weights
            total_bytes += W_q.numel() * 1
            
            # One FP32 scale
            total_bytes += 1 * 4
            
            # bias FP32 (if present)
            if b is not None:
                total_bytes += b.numel() * 4
                
    return total_bytes