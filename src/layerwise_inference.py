"""
Layerwise inference utilities for Zstd-compressed FCN exports.

Assumptions:
- Compressed format produced by src.export_compressed.export_fcn_to_compressed()
- Model is an FCN with model.net = [Linear/ReLU/.../Linear]
- Linear payload stores FP32 bytes compressed with zstd (lossless)

Key idea:
- During inference, decompress ONE layer at a time, compute, discard.
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from src.export_compressed import decompress_linear_layer

def _flatten_input(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


@torch.no_grad()
def layerwise_forward(compressed: dict, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Forward pass using a Zstd-compressed model dict.
    Decompresses one Linear layer at a time.
    """
    # 1) Flatten input
    if x.dim() > 2:
        x = _flatten_input(x)
    
    x = x.to(device)
    
    # 2) Process layers in order
    for entry in compressed["layers"]:
        layer_type = entry["type"]
        
        if layer_type == "linear":
            # Load compressed weights
            W, b = decompress_linear_layer(entry)
            W = W.to(device)
            b = b.to(device) if b is not None else None
            
            # Compute: x <- x @ W^T + b
            x = F.linear(x, W, b)
            
            # Discard decompressed weights
            del W, b
            
        elif layer_type == "relu":
            x = F.relu(x)
            
        else:
            raise ValueError(f"Unknown layer type in compressed model: {layer_type}")
        
    return x # logits


@torch.no_grad()
def layerwise_evaluate_accuracy(compressed: dict, loader, device: torch.device) -> float:
    """
    Accuracy evaluation for a Zstd-compressed model using layerwise inference.
    """
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for x, y in loader:
            logits = layerwise_forward(compressed, x, device)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == y).sum().item()
            total += y.numel()
        
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def measure_layerwise_inference_time(
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
        _ = layerwise_forward(compressed, x, device)

    # timed
    times = []
    for _ in range(timed_batches):
        x, _ = next(it)
        t0 = time.perf_counter()
        _ = layerwise_forward(compressed, x, device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times)