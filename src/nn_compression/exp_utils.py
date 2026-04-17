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

from .blockwise_export_compressed import (
    estimate_compressed_payload_bytes,
    export_fcn_to_compressed,
    load_compressed,
    save_compressed,
)
from .cnn import CNN
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
def build_model(device: torch.device, model_type: str = "fcn") -> nn.Module:
    """
    Build either an FCN or CNN model for MNIST.
    """
    if model_type == "fcn":
        model = FCN(
            in_dim=28 * 28,
            hidden_dims=[512, 256],
            out_dim=10,
        )

    elif model_type == "cnn":
        model = CNN(
            in_channels=1,
            input_height=28,
            input_width=28,
            conv_channels=[8, 16],
            kernel_size=3,
            pool_kernel_size=2,
            fc_hidden_dims=[64],
            out_dim=10,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


def load_weights(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


# STORAGE ESTIMATES
def estimate_fp32_weight_bytes(model: nn.Module) -> int:
    """
    Estimate total FP32 weight storage in bytes for learned layers.
    Supports Linear and Conv2d.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel() * 4  # float32 = 4 bytes

    return total


def estimate_peak_decompressed_layer_bytes(model: nn.Module) -> int:
    """
    Estimate peak decompressed weight size for layerwise inference.
    Supports Linear and Conv2d.
    """
    peak = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            peak = max(peak, m.weight.numel() * 4)

    return peak


def estimate_peak_runtime_layerwise_bytes(
    model: nn.Module, batch_size: int
) -> int:
    """
    Estimate peak runtime RAM for layerwise inference.

    Linear layer:
    - input activation x:        [B, in_features]
    - output tensor y:           [B, out_features]
    - full decompressed weight:  [out_features, in_features]
    - bias:                      [out_features] (optional)

    Conv2d layer:
    - input activation x:        [B, in_channels, H, W]
    - output tensor y:           [B, out_channels, H_out, W_out]
    - full decompressed weight:  [out_channels, in_channels, k_h, k_w]
    - bias:                      [out_channels] (optional)

    Returns the maximum estimated peak across supported layers.
    """
    peak = 0
    B = int(batch_size)

    # Start from MNIST image size
    current_h = 28
    current_w = 28

    for m in model.modules():
        if isinstance(m, nn.Linear):
            in_features = int(m.in_features)
            out_features = int(m.out_features)

            x_bytes = B * in_features * 4
            y_bytes = B * out_features * 4
            W_bytes = out_features * in_features * 4
            bias_bytes = out_features * 4 if m.bias is not None else 0

            layer_peak = x_bytes + y_bytes + W_bytes + bias_bytes
            peak = max(peak, layer_peak)

        elif isinstance(m, nn.Conv2d):
            in_channels = int(m.in_channels)
            out_channels = int(m.out_channels)

            if isinstance(m.kernel_size, int):
                k_h = k_w = m.kernel_size
            else:
                k_h, k_w = m.kernel_size

            if isinstance(m.padding, int):
                p_h = p_w = m.padding
            else:
                p_h, p_w = m.padding

            if isinstance(m.stride, int):
                s_h = s_w = m.stride
            else:
                s_h, s_w = m.stride

            H_out = (current_h + 2 * p_h - k_h) // s_h + 1
            W_out = (current_w + 2 * p_w - k_w) // s_w + 1

            x_bytes = B * in_channels * current_h * current_w * 4
            y_bytes = B * out_channels * H_out * W_out * 4
            W_bytes = out_channels * in_channels * k_h * k_w * 4
            bias_bytes = out_channels * 4 if m.bias is not None else 0

            layer_peak = x_bytes + y_bytes + W_bytes + bias_bytes
            peak = max(peak, layer_peak)

            current_h, current_w = H_out, W_out

        elif isinstance(m, nn.MaxPool2d):
            if isinstance(m.kernel_size, int):
                pool_k = m.kernel_size
            else:
                pool_k = m.kernel_size[0]

            current_h = current_h // pool_k
            current_w = current_w // pool_k

    return peak


def estimate_peak_runtime_blockwise_bytes(
    model: nn.Module, block_size: int, batch_size: int
) -> int:
    """
    Estimate peak runtime RAM for blockwise inference.

    Linear layer block:
    - input activation x:             [B, in_features]
    - output buffer for full layer:   [B, out_features]
    - decompressed weight block:      [block_out, in_features]
    - temporary block output:         [B, block_out]
    - bias slice:                     [block_out] (optional)

    Conv2d layer block:
    - input activation x:             [B, in_channels, H, W]
    - output buffer for full layer:   [B, out_channels, H_out, W_out]
    - decompressed weight block:      [block_out, in_channels, k_h, k_w]
    - temporary block output:         [B, block_out, H_out, W_out]
    - bias slice:                     [block_out] (optional)

    Returns the maximum estimated peak across supported layers.
    """
    peak = 0
    B = int(batch_size)

    current_h = 28
    current_w = 28

    for m in model.modules():
        if isinstance(m, nn.Linear):
            in_features = int(m.in_features)
            out_features = int(m.out_features)
            block_out = min(block_size, out_features)

            x_bytes = B * in_features * 4
            buffer_bytes = B * out_features * 4
            W_block_bytes = block_out * in_features * 4
            y_block_bytes = B * block_out * 4
            bias_bytes = block_out * 4 if m.bias is not None else 0

            layer_peak = (
                x_bytes
                + buffer_bytes
                + W_block_bytes
                + y_block_bytes
                + bias_bytes
            )
            peak = max(peak, layer_peak)

        elif isinstance(m, nn.Conv2d):
            in_channels = int(m.in_channels)
            out_channels = int(m.out_channels)
            block_out = min(block_size, out_channels)

            if isinstance(m.kernel_size, int):
                k_h = k_w = m.kernel_size
            else:
                k_h, k_w = m.kernel_size

            if isinstance(m.padding, int):
                p_h = p_w = m.padding
            else:
                p_h, p_w = m.padding

            if isinstance(m.stride, int):
                s_h = s_w = m.stride
            else:
                s_h, s_w = m.stride

            H_out = (current_h + 2 * p_h - k_h) // s_h + 1
            W_out = (current_w + 2 * p_w - k_w) // s_w + 1

            x_bytes = B * in_channels * current_h * current_w * 4
            buffer_bytes = B * out_channels * H_out * W_out * 4
            W_block_bytes = block_out * in_channels * k_h * k_w * 4
            y_block_bytes = B * block_out * H_out * W_out * 4
            bias_bytes = block_out * 4 if m.bias is not None else 0

            layer_peak = (
                x_bytes
                + buffer_bytes
                + W_block_bytes
                + y_block_bytes
                + bias_bytes
            )
            peak = max(peak, layer_peak)

            current_h, current_w = H_out, W_out

        elif isinstance(m, nn.MaxPool2d):
            if isinstance(m.kernel_size, int):
                pool_k = m.kernel_size
            else:
                pool_k = m.kernel_size[0]

            current_h = current_h // pool_k
            current_w = current_w // pool_k

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
