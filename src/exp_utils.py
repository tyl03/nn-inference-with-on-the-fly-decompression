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

from .fcn import FCN
from .quantization import compute_scale, symmetric_quantization, symmetric_dequantization


# DEVICE
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DATA LOADING (MNIST)
def load_test_loader(batch_size: int = 256) -> DataLoader:
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST(
        root="data", 
        train=False, 
        download=True, 
        transform=transform
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
            total += m.weight.numel() * 4 # float32 = 4 bytes
            
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



# QUANTIZE HELPERS
@torch.no_grad()
def quantize_dequantize_linear_weights_inplace(model: nn.Module):
    """
    Replaces each nn.Linear weight with a dequantized copy of its int8 quantized version.
    Returns a list of (layer_name, weight_scale) so we can estimate storage.
    """
    scales = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            W = m.weight.data
            s = compute_scale(W)
            W_q = symmetric_quantization(W, s) # int8
            W_fp = symmetric_dequantization(W_q, s) # float32
            m.weight.data.copy_(W_fp)
            scales.append((name, s))
    
    return scales


def estimate_int8_weight_bytes_plus_scales(model: nn.Module, num_scales: int) -> int:
    """
    int8 weights: 1 byte each
    scales: store as float32 per layer (4 bytes each)
    To enable dequantization we need both the weight and the scale.
    """
    int8_bytes = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            int8_bytes += m.weight.numel() * 1
            
    scale_bytes = num_scales * 4
    return int8_bytes, scale_bytes


# Converts bytes into kilobytes
def fmt_kb(b: int) -> float:
    return b / 1024.0