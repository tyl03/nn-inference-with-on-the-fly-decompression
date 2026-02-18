"""
Export a trained FCN into a Zstandard-compressed, layer-by-layer blockwise format.

Goal:
- Store the model on disk in compressed form (Zstandard).
- During inference, decompress ONE weight block at a time, compute, discard, repeat.

What is stored:
- For each Linear layer:
    - W_blocks_zstd: list of zstd-compressed FP32 weight blocks (row blocks)
    - block_size
    - in_features, out_features
    - b_zstd: zstd-compressed FP32 bias (one blob, optional)
    - bias_shape (for reconstruction)
- For each ReLU:
    - marker only

Notes:
- Zstd compression is lossless: decompression restores exact FP32 values.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blockwise_utils import compress_weight_blocks_fp32
from .tensor_bytes_utils import to_fp32_bytes
from .zstd_utils import zstd_compress

FORMAT_VERSION = 2


def export_fcn_to_compressed(
    model: nn.Module, *, zstd_level: int = 3, block_size: int = 64
) -> dict:
    """
    Export model to a dict where ALL Linear weights are stored as Zstd-compressed FP32 blocks.
    """
    if not hasattr(model, "in_dim"):
        raise ValueError("Model must have 'in_dim'.")
    if not hasattr(model, "net"):
        raise ValueError("Model must have 'net' (nn.Sequential).")
    if not isinstance(model.net, nn.Sequential):
        raise ValueError("Model 'net' must be an nn.Sequential.")
    if not isinstance(zstd_level, int):
        raise ValueError("zstd_level must be an integer.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    layers_out: list[dict] = []

    # We export layers in the exact order they are used in forward().
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            W = layer.weight
            b = layer.bias

            # Weights: store as multiple compressed blocks (Zstd of raw FP32 bytes)
            W_blocks_zstd = compress_weight_blocks_fp32(
                W, block_size=block_size, zstd_level=zstd_level
            )

            # Bias: store as one compressed blob (Zstd of raw FP32 bytes), if it exists
            if b is not None:
                b_raw = to_fp32_bytes(b)
                b_zstd = zstd_compress(b_raw, level=zstd_level)
                bias_shape = tuple(b.shape)
            else:
                b_zstd = None
                bias_shape = None

            layers_out.append(
                {
                    "type": "linear",
                    "storage": "blockwise",
                    "in_features": int(layer.in_features),
                    "out_features": int(layer.out_features),
                    "block_size": int(block_size),
                    "dtype": "float32",
                    "bias_shape": bias_shape,
                    "zstd_level": int(zstd_level),
                    # payloads
                    "W_blocks_zstd": W_blocks_zstd,
                    "b_zstd": b_zstd,
                }
            )

        # ReLU layer: store marker only
        elif isinstance(layer, nn.ReLU):
            layers_out.append({"type": "relu"})

        else:
            raise ValueError(f"Unsupported layer type in export: {type(layer)}")

    compressed = {
        "format_version": FORMAT_VERSION,
        "model_type": "FCN",
        "in_dim": int(model.in_dim),
        "compression": "zstd",
        "storage_layout": "blockwise",
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
    Sum of stored compressed payload bytes (sum(W_blocks_zstd) + b_zstd).
    """
    total = 0

    for entry in compressed["layers"]:
        if entry["type"] == "linear":
            total += sum(len(block) for block in entry["W_blocks_zstd"])
            if entry["b_zstd"] is not None:
                total += len(entry["b_zstd"])

    return total
