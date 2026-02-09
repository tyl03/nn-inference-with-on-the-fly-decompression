"""
Layerwise inference on a compressed model representation (int8 weights + scale).

Decode ONE layer at a time:
- dequantize W_q using s_w
- compute in FP32
- discard decompressed weights
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import time

from .quantization import symmetric_dequantization


def _flatten_input(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


@torch.no_grad()
def layerwise_forward(compressed: dict, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Runs a forward pass (logits output) using layer-by-layer decompression.

    Returns:
    - logits (float32): shape [batch_size, out_dim]
    """
    # 1) Flatten input
    x = x.to(device)
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
            W_q = entry["W_q"].to(device)  # int8 quantized weights
            s_w = float(entry["s_w"]) # scale factor
            b = entry["b"]
            b = b.to(device) if b is not None else None
            
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
    
    with torch.inference_mode():
        for x, y in loader:
            y = y.to(device)
            logits = layerwise_forward(compressed, x, device)
            preds = logits.argmax(dim=1)
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