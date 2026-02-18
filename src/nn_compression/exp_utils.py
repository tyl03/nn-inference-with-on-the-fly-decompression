"""
Shared experiment utilities.

This file only contains helpers that are repeated across multiple
experiment scripts.
No new functionality is introduced here.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .export_compressed import (
    estimate_compressed_payload_bytes,
    export_fcn_to_compressed,
    load_compressed,
    save_compressed,
)
from .fcn import FCN


# DEVICE
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DATA LOADING (MNIST)
def load_test_loader(batch_size: int = 256) -> DataLoader:
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# MODEL HELPERS
def build_model(device: torch.device) -> FCN:
    return FCN(
        in_dim=28 * 28,
        hidden_dims=[512, 256],
        out_dim=10,
    ).to(device)


def load_weights(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


# STORAGE ESTIMATES
def estimate_fp32_weight_bytes(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.numel() * 4  # float32 = 4 bytes

    return total


def estimate_peak_decompressed_layer_bytes(model: nn.Module) -> int:
    peak = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            peak = max(peak, m.weight.numel() * 4)

    return peak


def fmt_bytes(b: int) -> str:
    kb = b / 1024.0
    mb = kb / 1024.0
    if mb >= 1:
        return f"{b:,} B ({mb:.2f} MB)"
    return f"{b:,} B ({kb:.2f} KB)"


# @torch.no_grad()
# def apply_qdq_to_linear_weights_inplace(model: nn.Module):
#     """
#     Quantize -> dequantize each nn.Linear weight and write the FP32 result back.

#     Purpose:
#     - Simulate the accuracy impact of int8 quantization (quantization noise),
#       while still using normal FP32 PyTorch inference.
#     - This is NOT the stored-format model. For storage numbers, use
#       estimate_compressed_storage_bytes_from_model(...).

#     Returns:
#         list of (layer_name, scale) for debugging/inspection.
#     """
#     scales = []
#     for name, m in model.named_modules():
#         if isinstance(m, nn.Linear):
#             W = m.weight.data
#             s = compute_scale(W)
#             W_q = symmetric_quantization(W, s)      # int8
#             W_fp = symmetric_dequantization(W_q, s) # float32
#             m.weight.data.copy_(W_fp)
#             scales.append((name, float(s)))
#     return scales


# COMPRESSED (STORED) INT8 MODEL HELPERS
def export_compressed_model(model: nn.Module) -> dict:
    """
    Returns the compressed representation used for storage / layerwise inference.
    This dict is what we store on disk (torch.save()).
    """
    return export_fcn_to_compressed(model)


def save_compressed_model(model: nn.Module, path: str) -> None:
    """
    Exports + saves the compressed representation.
    """
    compressed = export_compressed_model(model)
    save_compressed(compressed, path)


def load_compressed_model(path: str) -> dict:
    """
    Loads a compressed model (CPU dict).
    """
    return load_compressed(path)


def estimate_compressed_storage_bytes_from_model(model: nn.Module) -> int:
    """
    Estimates stored bytes for int8 weights + FP32 scales + FP32 bias
    using the compressed format.
    """
    compressed = export_compressed_model(model)
    return estimate_compressed_payload_bytes(compressed)


def estimate_compressed_storage_bytes_from_file(path: str) -> int:
    """
    Loads a saved compressed model and estimates stored bytes.
    Useful when the FP32 model isn't available anymore.
    """
    compressed = load_compressed_model(path)
    return estimate_compressed_payload_bytes(compressed)


# Converts bytes into kilobytes
def fmt_kb(b: int) -> float:
    return b / 1024.0
