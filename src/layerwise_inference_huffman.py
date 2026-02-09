"""
Layerwise inference where weights are stored as Huffman-compressed int8.

Per Linear layer:
- Huffman decode -> W_q (int8)
- dequantize -> FP32
- compute
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import time

from .huffman import huff_decompress_int8_tensor


@torch.no_grad()
def layerwise_forward_int8_huffman(compressed_huff: dict, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    x = x.view(x.size(0), -1)

    expected_in_dim = int(compressed_huff["in_dim"])
    if x.shape[1] != expected_in_dim:
        raise ValueError(f"Expected input with {expected_in_dim} features, got {x.shape[1]}")

    for entry in compressed_huff["layers"]:
        if entry["type"] == "linear":
            W_q = huff_decompress_int8_tensor(entry["W_q_huff"]).to(device)
            s_w = float(entry["s_w"])
            W = W_q.to(torch.float32) * s_w

            b = entry["b"]
            b = b.to(device) if b is not None else None

            x = F.linear(x, W, b)

            del W_q, W, b

        elif entry["type"] == "relu":
            x = F.relu(x)

        else:
            raise ValueError(f"Unknown layer type: {entry['type']}")

    return x


@torch.no_grad()
def layerwise_evaluate_accuracy_int8_huffman(compressed_huff: dict, loader, device: torch.device) -> float:
    correct = 0
    total = 0

    for x, y in loader:
        y = y.to(device)
        logits = layerwise_forward_int8_huffman(compressed_huff, x, device)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def measure_layerwise_inference_time_huffman(
    compressed_huff: dict,
    loader,
    device: torch.device,
    warmup_batches: int = 5,
    timed_batches: int = 30,
) -> float:
    it = iter(loader)

    for _ in range(warmup_batches):
        x, _ = next(it)
        _ = layerwise_forward_int8_huffman(compressed_huff, x, device)

    times = []
    for _ in range(timed_batches):
        x, _ = next(it)
        t0 = time.perf_counter()
        _ = layerwise_forward_int8_huffman(compressed_huff, x, device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times)