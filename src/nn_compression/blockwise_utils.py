"""
Block-wise utilities for FP32 weight storage + reconstruction.

Purpose:
- Split a linear layer weight matrix W into blocks of rows.
- Compress each block with Zstd.
- Decompress one block at a time during layerwise inference.
"""

from __future__ import annotations

import torch

from .tensor_bytes_utils import from_fp32_bytes, to_fp32_bytes
from .zstd_utils import zstd_compress, zstd_decompress


def iter_row_blocks(W: torch.Tensor, block_size: int):
    """
    Yield blocks of rows from the weight matrix W.

    Yields: (start, end, W_block) where W_block is the slice of W from rows [start:end, :].
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    out_features = int(W.shape[0])
    for start in range(0, out_features, block_size):
        end = min(start + block_size, out_features)
        yield start, end, W[start:end]


def num_blocks(out_features: int, block_size: int) -> int:
    """
    Calculate the number of blocks needed to cover out_features with given block_size.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    return (out_features + block_size - 1) // block_size


def block_shape(
    in_features: int, out_features: int, block_size: int, block_idx: int
) -> tuple[int, int]:
    """
    Calculate the shape of a specific block given the overall in/out features and block size.
    """
    if block_idx < 0:
        raise ValueError("block_idx must be non-negative.")

    start = block_idx * block_size
    if start >= out_features:
        raise ValueError(
            "block_idx is out of range for the given out_features and block_size."
        )

    block_out = min(block_size, out_features - start)
    return (block_out, in_features)


def compress_weight_blocks_fp32(
    W: torch.Tensor, *, block_size: int, zstd_level: int
) -> list[bytes]:
    """
    Split W into row blocks, convert each block to FP32 bytes, compress each block with Zstd.

    Returns a list of compressed byte strings, one per block.
    """
    if not isinstance(zstd_level, int):
        raise ValueError("zstd_level must be an integer.")

    compressed_blocks: list[bytes] = []

    for _start, _end, W_block in iter_row_blocks(W, block_size):
        fp32_bytes = to_fp32_bytes(W_block)
        compressed = zstd_compress(fp32_bytes, level=zstd_level)
        compressed_blocks.append(compressed)

    return compressed_blocks


def decompress_weight_block_fp32(entry: dict, block_idx: int) -> torch.Tensor:
    """
    Decompress one weight block from a blockwise-compressed linear layer entry.
    """
    if entry.get("type") != "linear":
        raise ValueError("Entry must be of type 'linear'.")
    if entry.get("storage") != "blockwise":
        raise ValueError("Entry must have 'storage' set to 'blockwise'.")

    W_blocks = entry.get("W_blocks_zstd")
    if not isinstance(W_blocks, list):
        raise ValueError(
            "Entry must contain 'W_blocks_zstd' as a list of compressed blocks."
        )
    if block_idx < 0 or block_idx >= len(W_blocks):
        raise ValueError(
            "block_idx is out of range for the number of blocks in 'W_blocks_zstd'."
        )

    in_features = int(entry["in_features"])
    out_features = int(entry["out_features"])
    block_size = int(entry["block_size"])

    raw = zstd_decompress(W_blocks[block_idx])
    shape = block_shape(in_features, out_features, block_size, block_idx)
    return from_fp32_bytes(raw, shape)


def decompress_bias_fp32(entry: dict) -> torch.Tensor | None:
    """
    Decompress the bias vector from a blockwise-compressed linear layer entry, if it exists.
    """
    if entry.get("type") != "linear":
        raise ValueError("Entry must be of type 'linear'.")

    b_zstd = entry.get("b_zstd")
    if b_zstd is None:
        return None

    bias_shape = entry.get("bias_shape")
    if bias_shape is None:
        raise ValueError("Bias shape information is missing in the entry.")

    raw = zstd_decompress(b_zstd)
    return from_fp32_bytes(raw, tuple(bias_shape))
