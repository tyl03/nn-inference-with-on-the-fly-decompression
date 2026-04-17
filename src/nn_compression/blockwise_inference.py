"""
Blockwise inference utilities for Zstd-compressed exports.

Supports:
- Linear layers with blockwise decompression
- Convolution layers with blockwise decompression over output filters
- ReLU
- MaxPool2d
- Flatten

Key idea:
- During inference, decompress ONE block at a time, compute, discard it,
  and continue until the full layer output has been reconstructed.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from nn_compression.blockwise_utils import (
    decompress_bias_fp32,
    decompress_conv_weight_block_fp32,
    decompress_weight_block_fp32,
)
from nn_compression.preprocess import preprocess_input


@torch.no_grad()
def blockwise_forward(
    compressed: dict, x: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Forward pass using a Zstd-compressed model dict.
    Decompresses one block at a time.
    """
    # If a single sample comes in, add batch dimension for consistent processing
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = preprocess_input(x).to(device=device, dtype=torch.float32)
    
    first_layer_type = compressed["layers"][0]["type"]
    if first_layer_type == "linear" and x.dim() > 2:
        x = x.view(x.size(0), -1)

    # Process layers in order
    for entry in compressed["layers"]:
        layer_type = entry["type"]

        if layer_type == "linear":
            out_features = int(entry["out_features"])
            block_size = int(entry["block_size"])

            # Bias
            b = decompress_bias_fp32(entry)
            b = b.to(device) if b is not None else None

            # Allocate output buffer for this layer
            # x shape: [batch_size, in_features]
            batch_size = x.size(0)
            buffer = torch.empty(
                (batch_size, out_features), device=device, dtype=x.dtype
            )

            # Loop weight blocks for this layer
            W_blocks_zstd = entry["W_blocks_zstd"]
            for block_idx in range(len(W_blocks_zstd)):
                W_block = decompress_weight_block_fp32(entry, block_idx).to(device)

                # W_block shape: [block_out_features, in_features]
                block_out_features = W_block.size(0)

                start = block_idx * block_size
                end = (
                    start + block_out_features
                )  # handles last block which may be smaller

                # Compute block output: y_block = x @ W_block^T + b_block
                bias_slice = b[start:end] if b is not None else None
                y_block = F.linear(x, W_block, bias_slice)
                buffer[:, start:end] = (
                    y_block  # write block output to correct slice of y
                )

                # Discard decompressed block
                del W_block, y_block

            # After processing all blocks, y contains the full output of this Linear layer
            x = buffer  # set input for next layer
            del buffer, b  # free bias and output buffer for this layer
            
        elif layer_type == "conv":
            # Blockwise convolution over output-channel blocks / filter blocks
            out_channels = int(entry["out_channels"])

            # Decompress full bias only once for the whole conv layer
            b_full = decompress_bias_fp32(entry)
            b_full = b_full.to(device) if b_full is not None else None

            batch_size = x.size(0)
            output = None
            start = 0

            W_blocks_zstd = entry["W_blocks_zstd"]
            for block_idx in range(len(W_blocks_zstd)):
                # Use conv-specific block decompression
                W_block = decompress_conv_weight_block_fp32(entry, block_idx).to(device)

                block_out_channels = W_block.size(0)
                end = start + block_out_channels

                b_block = b_full[start:end] if b_full is not None else None

                y_block = F.conv2d(
                    x,
                    W_block,
                    bias=b_block,
                    stride=entry.get("stride", 1),
                    padding=entry.get("padding", 0),
                )

                # Allocate output tensor once we know spatial output size
                if output is None:
                    output = torch.empty(
                        (batch_size, out_channels, y_block.size(2), y_block.size(3)),
                        device=device,
                        dtype=x.dtype,
                    )

                # Write into explicit channel slice
                output[:, start:end, :, :] = y_block

                start = end

                # Discard decompressed block
                del W_block, y_block

            x = output
            del output, b_full

        elif layer_type == "relu":
            x = F.relu(x)
            
        elif layer_type == "pool":
            # Ensure tuple format for consistency
            kernel_size = tuple(entry.get("kernel_size", (2, 2)))
            stride = tuple(entry.get("stride", kernel_size))

            x = F.max_pool2d(
                x,
                kernel_size=kernel_size,
                stride=stride,
            )
            
        elif layer_type == "flatten":
            x = x.view(x.size(0), -1)

        else:
            raise ValueError(f"Unknown layer type in compressed model: {layer_type}")

    return x  # logits


@torch.no_grad()
def blockwise_evaluate_accuracy(
    compressed: dict, loader, device: torch.device
) -> float:
    """
    Accuracy evaluation for a Zstd-compressed model using blockwise inference.
    """
    correct = 0
    total = 0

    with torch.inference_mode():
        for x, y in loader:
            logits = blockwise_forward(compressed, x, device)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == y).sum().item()
            total += y.numel()

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def measure_blockwise_inference_time(
    compressed: dict,
    loader,
    device: torch.device,
    warmup_batches: int = 5,
    timed_batches: int = 30,
) -> float:
    """
    Returns avg forward time per batch (seconds).
    """
    it = iter(loader)

    # warmup
    for _ in range(warmup_batches):
        x, _ = next(it)
        _ = blockwise_forward(compressed, x, device)

    # timed
    times = []
    for _ in range(timed_batches):
        x, _ = next(it)
        t0 = time.perf_counter()
        _ = blockwise_forward(compressed, x, device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times)
