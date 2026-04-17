"""
Block-wise utilities for FP32 weight storage + reconstruction.

Purpose:
- Split a linear layer weight matrix W into blocks of rows.
- Split a convolution layer weight tensor W into blocks of output filters.
- Compress each block with Zstd.
- Decompress one block at a time during blockwise inference.
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
    if entry.get("type") not in {"linear", "conv"}:
        raise ValueError("Entry must be of type 'linear' or 'conv'.")

    b_zstd = entry.get("b_zstd")
    if b_zstd is None:
        return None

    bias_shape = entry.get("bias_shape")
    if bias_shape is None:
        raise ValueError("Bias shape information is missing in the entry.")

    raw = zstd_decompress(b_zstd)
    return from_fp32_bytes(raw, tuple(bias_shape))


def iter_conv_filter_blocks(W: torch.Tensor, block_size: int):
    """
    Yield blocks of output filters from a Conv2d weight tensor.

    Conv2d weight shape:
        W = [out_channels, in_channels, kernel_h, kernel_w]

    Each block contains a subset of output channels / filters.

    Yields:
        (start, end, W_block)
    where W_block = W[start:end, :, :, :]
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    out_channels = int(W.shape[0])
    for start in range(0, out_channels, block_size):
        end = min(start + block_size, out_channels)
        yield start, end, W[start:end]
        
        
def conv_block_shape(
    in_channels: int,
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    block_size: int,
    block_idx: int,
) -> tuple[int, int, int, int]:
    """
    Shape of one convolution block:
        [block_out_channels, in_channels, kernel_h, kernel_w]
    """
    if block_idx < 0:
        raise ValueError("block_idx must be non-negative.")

    start = block_idx * block_size
    if start >= out_channels:
        raise ValueError(
            "block_idx is out of range for the given out_channels and block_size."
        )

    block_out = min(block_size, out_channels - start)
    return (block_out, in_channels, kernel_h, kernel_w)


def compress_conv_weight_blocks_fp32(
    W: torch.Tensor, *, block_size: int, zstd_level: int
) -> list[bytes]:
    """
    Split a Conv2d weight tensor into blocks of output filters, convert each block
    to FP32 bytes, and compress each block with Zstd.

    Input shape:
        W = [out_channels, in_channels, kernel_h, kernel_w]
    """
    if not isinstance(zstd_level, int):
        raise ValueError("zstd_level must be an integer.")

    compressed_blocks: list[bytes] = []

    for _start, _end, W_block in iter_conv_filter_blocks(W, block_size):
        fp32_bytes = to_fp32_bytes(W_block)
        compressed = zstd_compress(fp32_bytes, level=zstd_level)
        compressed_blocks.append(compressed)

    return compressed_blocks


def decompress_conv_weight_block_fp32(entry: dict, block_idx: int) -> torch.Tensor:
    """
    Decompress one weight block from a blockwise-compressed convolution layer entry.
    """
    if entry.get("type") != "conv":
        raise ValueError("Entry must be of type 'conv'.")
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

    in_channels = int(entry["in_channels"])
    out_channels = int(entry["out_channels"])

    # Expect kernel size to be stored in entry, e.g. kernel_size=(3, 3)
    kernel_size = entry.get("kernel_size")
    if kernel_size is None:
        raise ValueError("Conv entry must contain 'kernel_size'.")

    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h = int(kernel_size[0])
        kernel_w = int(kernel_size[1])

    block_size = int(entry["block_size"])

    raw = zstd_decompress(W_blocks[block_idx])
    shape = conv_block_shape(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        block_size=block_size,
        block_idx=block_idx,
    )
    return from_fp32_bytes(raw, shape)
