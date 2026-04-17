"""
Export a trained model into a Zstandard-compressed, blockwise format.

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

- For each Conv2d layer:
    - W_blocks_zstd: list of zstd-compressed FP32 filter blocks
    - block_size
    - in_channels, out_channels
    - kernel_size, stride, padding
    - b_zstd: zstd-compressed FP32 bias (one blob, optional)
    - bias_shape (for reconstruction)

- For ReLU / MaxPool2d / Flatten:
    - marker + necessary metadata only

Notes:
- Zstd compression is lossless: decompression restores exact FP32 values.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blockwise_utils import (
    compress_conv_weight_blocks_fp32,
    compress_weight_blocks_fp32,
)
from .tensor_bytes_utils import to_fp32_bytes
from .zstd_utils import zstd_compress

FORMAT_VERSION = 3


def _normalize_2tuple(value) -> tuple[int, int]:
    """
    Convert an int or tuple-like value into a 2-tuple of ints.
    """
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


def export_model_to_compressed(
    model: nn.Module, *, zstd_level: int = 3, block_size: int = 64
) -> dict:
    """
    Export model to a dict where supported learned layers are stored as
    Zstd-compressed FP32 blocks.

    Supports:
    - Linear
    - Conv2d
    - ReLU
    - MaxPool2d
    - Flatten
    """
    if not isinstance(zstd_level, int):
        raise ValueError("zstd_level must be an integer.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    layers_out: list[dict] = []

    # Support either:
    #   - model.net   (FCN-style)
    # or a CNN-style model with:
    #   - model.features
    #   - model.fully_connected
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        layer_sequence = list(model.net)

    elif (
        hasattr(model, "features")
        and isinstance(model.features, nn.Sequential)
        and hasattr(model, "fully_connected")
        and isinstance(model.fully_connected, nn.Sequential)
    ):
        layer_sequence = (
            list(model.features)
            + [nn.Flatten()]   # NEW: explicit flatten marker between CNN features and fully_connected
            + list(model.fully_connected)
        )
    else:
        raise ValueError(
            "Model must either have 'net' (nn.Sequential) or "
            "'features' + 'fully_connected' (both nn.Sequential)."
        )

    # Export layers in forward order
    for layer in layer_sequence:
        if isinstance(layer, nn.Linear):
            W = layer.weight
            b = layer.bias

            W_blocks_zstd = compress_weight_blocks_fp32(
                W, block_size=block_size, zstd_level=zstd_level
            )

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
                    "W_blocks_zstd": W_blocks_zstd,
                    "b_zstd": b_zstd,
                }
            )

        elif isinstance(layer, nn.Conv2d):
            # Export Conv2d in filter blocks along output channels
            W = layer.weight
            b = layer.bias

            W_blocks_zstd = compress_conv_weight_blocks_fp32(
                W, block_size=block_size, zstd_level=zstd_level
            )

            if b is not None:
                b_raw = to_fp32_bytes(b)
                b_zstd = zstd_compress(b_raw, level=zstd_level)
                bias_shape = tuple(b.shape)
            else:
                b_zstd = None
                bias_shape = None

            layers_out.append(
                {
                    "type": "conv",
                    "storage": "blockwise",
                    "in_channels": int(layer.in_channels),
                    "out_channels": int(layer.out_channels),
                    "kernel_size": _normalize_2tuple(layer.kernel_size),
                    "stride": _normalize_2tuple(layer.stride),
                    "padding": _normalize_2tuple(layer.padding),
                    "block_size": int(block_size),
                    "dtype": "float32",
                    "bias_shape": bias_shape,
                    "zstd_level": int(zstd_level),
                    "W_blocks_zstd": W_blocks_zstd,
                    "b_zstd": b_zstd,
                }
            )

        elif isinstance(layer, nn.ReLU):
            layers_out.append({"type": "relu"})

        elif isinstance(layer, nn.MaxPool2d):
            # Export pooling marker + needed metadata
            kernel_size = layer.kernel_size
            stride = layer.stride if layer.stride is not None else layer.kernel_size

            layers_out.append(
                {
                    "type": "pool",
                    "kernel_size": _normalize_2tuple(kernel_size),
                    "stride": _normalize_2tuple(stride),
                }
            )

        elif isinstance(layer, nn.Flatten):
            # Export flatten marker
            layers_out.append({"type": "flatten"})

        else:
            raise ValueError(f"Unsupported layer type in export: {type(layer)}")

    compressed = {
        "format_version": FORMAT_VERSION,
        "compression": "zstd",
        "storage_layout": "blockwise",
        "block_size": int(block_size),
        "layers": layers_out,

        # Old experiments
        "model_type": model.__class__.__name__,
    }

    # Old FCN
    if hasattr(model, "in_dim"):
        compressed["in_dim"] = int(model.in_dim)

    return compressed


def export_fcn_to_compressed(
    model: nn.Module, *, zstd_level: int = 3, block_size: int = 64
) -> dict:
    return export_model_to_compressed(
        model, zstd_level=zstd_level, block_size=block_size
    )


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
    Sum of stored compressed payload bytes:
    sum(W_blocks_zstd) + b_zstd over learned layers.
    """
    total = 0

    for entry in compressed["layers"]:
        # Count both linear and conv payloads
        if entry["type"] in {"linear", "conv"}:
            total += sum(len(block) for block in entry["W_blocks_zstd"])
            if entry["b_zstd"] is not None:
                total += len(entry["b_zstd"])

    return total