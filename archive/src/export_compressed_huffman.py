"""
Converts the existing int8 compressed model format into int8+Huffman format.

Input format: (from export_compressed.py)
- per linear layer: W_q (int8 tensor), s_w (float), b (fp32)

Output format:
- per linear layer: W_q_huff (Huffman package), s_w, b
"""

from __future__ import annotations

import torch

from .huffman import huff_compress_int8_tensor, estimate_huff_pkg_bytes


def compress_int8_model_with_huffman(compressed_int8: dict) -> dict:
    out_layers = []

    for entry in compressed_int8["layers"]:
        if entry["type"] == "linear":
            out_layers.append({
                "type": "linear",
                "in_features": entry["in_features"],
                "out_features": entry["out_features"],
                "W_q_huff": huff_compress_int8_tensor(entry["W_q"]),
                "s_w": float(entry["s_w"]),
                "b": entry["b"],  # keep bias FP32
            })
        else:
            out_layers.append(entry)

    return {
        "format_version": 1,
        "model_type": compressed_int8.get("model_type", "FCN"),
        "compression": "int8+huffman",
        "in_dim": int(compressed_int8["in_dim"]),
        "layers": out_layers,
    }


def save_int8_huffman(compressed_huff: dict, path: str) -> None:
    torch.save(compressed_huff, path)


def load_int8_huffman(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def estimate_int8_huffman_weight_bytes(compressed_huff: dict) -> int:
    total = 0

    for entry in compressed_huff["layers"]:
        if entry["type"] != "linear":
            continue

        total += estimate_huff_pkg_bytes(entry["W_q_huff"])  # Huffman weights
        total += 4  # scale FP32
        b = entry["b"]
        if b is not None:
            total += b.numel() * 4  # bias FP32

    return total