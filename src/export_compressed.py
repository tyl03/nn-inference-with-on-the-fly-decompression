"""
Export a trained FCN into a Zstandard-compressed, layer-by-layer format.

Goal:
- Store the model on disk in compressed form (Zstandard).
- Later, inference can decompress ONE layer at a time, compute, discard, repeat.

What is stored:
- For each Linear layer:
    - W_zstd: zstd-compressed bytes of FP32 weights
    - b_zstd: zstd-compressed bytes of FP32 bias (optional)
    - weight_shape, bias_shape
    - in_features, out_features
- For each ReLU:
    - marker only, to preserve order

Notes:
- Zstd compression is lossless: decompression restores the exact FP32 values.
- We compress raw FP32 bytes.
"""

from __future__ import annotations
from .zstd_utils import zstd_compress, zstd_decompress

import torch
import torch.nn as nn
import numpy as np

FORMAT_VERSION = 1


def _to_fp32_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert a PyTorch tensor to raw FP32 bytes.
    """
    t_cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return t_cpu.numpy().tobytes()


def _from_fp32_bytes(data: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    """
    Convert raw FP32 bytes back to a PyTorch tensor with the given shape.
    """
    arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())
    

def export_fcn_to_compressed(model: nn.Module, *, zstd_level: int = 3) -> dict:
    """
    Export model to a dict where ALL Linear weights/bias are zstd-compressed.

    zstd_level:
        - Lower = faster, larger output
        - Higher = slower, smaller output
        - Default 3 is a good baseline.
    """
    if not hasattr(model, "in_dim"):
        raise ValueError("Model must have 'in_dim'.")
    if not hasattr(model, "net"):
        raise ValueError("Model must have 'net' (nn.Sequential).")
    if not isinstance(model.net, nn.Sequential):
        raise ValueError("Model 'net' must be an nn.Sequential.")
    if not isinstance(zstd_level, int):
        raise ValueError("zstd_level must be an integer.")
    
    layers_out: list[dict] = []
    
    # We export layers in the exact order they are used in forward().
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            W = layer.weight
            b = layer.bias
            
            W_raw = _to_fp32_bytes(W)
            b_raw = _to_fp32_bytes(b) if b is not None else None
            
            W_zstd = zstd_compress(W_raw, level=zstd_level)
            b_zstd = zstd_compress(b_raw, level=zstd_level) if b_raw is not None else None
            
            layers_out.append(
                {
                    "type": "linear",
                    "in_features": int(layer.in_features),
                    "out_features": int(layer.out_features),

                    "dtype": "float32",
                    "weight_shape": tuple(W.shape),
                    "bias_shape": tuple(b.shape) if b is not None else None,

                    "zstd_level": int(zstd_level),

                    # compressed payloads (bytes)
                    "W_zstd": W_zstd,
                    "b_zstd": b_zstd,

                    # optional debug metadata (helps validate compression ratio)
                    "W_raw_nbytes": len(W_raw),
                    "W_zstd_nbytes": len(W_zstd),
                    "b_raw_nbytes": len(b_raw) if b_raw is not None else 0,
                    "b_zstd_nbytes": len(b_zstd) if b_zstd is not None else 0,
                }
            )
            
        # ReLU layer: store marker only
        elif isinstance(layer, nn.ReLU):
            layers_out.append({
                "type": "relu"
            })
            
        else:
            raise ValueError(f"Unsupported layer type in export: {type(layer)}")
        
    
    compressed = {
        "format_version": FORMAT_VERSION,
        "model_type": "FCN",
        "in_dim": int(model.in_dim),
        "compression": "zstd",
        "layers": layers_out,
    }
    
    return compressed


def save_compressed(compressed: dict, path: str) -> None:
    """
    Save exported dict to disk.
    """
    torch.save(compressed, path)
    
    
def load_compressed(path: str) -> dict:
    """
    Load exported dict from disk.
    """
    return torch.load(path, map_location="cpu")


def estimate_compressed_payload_bytes(compressed: dict) -> int:
    """
    Sum of stored compressed payload bytes (W_zstd + b_zstd).
    """
    total = 0
    
    for entry in compressed["layers"]:
        if entry["type"] == "linear":
            total += len(entry["W_zstd"])
            if entry["b_zstd"] is not None:
                total += len(entry["b_zstd"])
                
    return total


def decompress_linear_layer(entry: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Decompress a single linear layer entry back to weight and bias tensors.
    Useful for inference where we want to load one layer at a time.
    """
    if entry.get("type") != "linear":
        raise ValueError("Entry must be of type 'linear' to decompress.")

    W_raw = zstd_decompress(entry["W_zstd"])
    W = _from_fp32_bytes(W_raw, tuple(entry["weight_shape"]))

    b_zstd = entry["b_zstd"]
    if b_zstd is not None:
        b_raw = zstd_decompress(b_zstd)
        b = _from_fp32_bytes(b_raw, tuple(entry["bias_shape"]))
    else:
        b = None

    return W, b
