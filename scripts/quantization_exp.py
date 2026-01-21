"""
Quantization experiment:
- Load a trained FCN (MNIST)
- Quantize weights of Linear layers to int8 (symmetric, [-127,127])
- Dequantize back to FP32 (simulates on-the-fly dequant inference)
- Measure accuracy drop
- Measure storage size (FP32 vs int8 + scale metadata)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.fcn import FCN
from src.training import get_device, evaluate
from src.quantization import compute_scale, symmetric_quantization, symmetric_dequantization


def load_test_loader():
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=256, shuffle=False)


def build_model(device: torch.device):
    return FCN(in_dim=28 * 28, hidden_dims=[512, 256], out_dim=10).to(device)


def load_weights(model: nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model
    

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


def estimate_fp32_weight_bytes(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.numel() * 4 # float32 = 4 bytes
            
    return total


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


def main():
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader()
    
    ckpt_path = "fcn_mnist_best.pt"
    
    # Baseline FP32
    model = build_model(device)
    load_weights(model, ckpt_path, device)
    base_loss, base_accuracy = evaluate(model, test_loader, loss_fn, device)
    
    # Quantized (weights int8 -> dequant FP32)
    qmodel = build_model(device)
    load_weights(qmodel, ckpt_path, device)
    scales = quantize_dequantize_linear_weights_inplace(qmodel)
    q_loss, q_accuracy = evaluate(qmodel, test_loader, loss_fn, device)
    
    # Storage estimates (weights only)
    fp32_bytes = estimate_fp32_weight_bytes(model)
    int8_weight_bytes, scale_bytes = estimate_int8_weight_bytes_plus_scales(model, num_scales=len(scales))
    total_int8_bytes = int8_weight_bytes + scale_bytes

    
    # Converts bytes into kilobytes
    def fmt_kb(b): return b / 1024.0
    
    print("\nQuantization Results (weights only, symmetric int8)\n")
    print(f"FP32 accuracy: {base_accuracy:.4f}   loss: {base_loss:.4f}")
    print(f"INT8 accuracy: {q_accuracy:.4f}   loss: {q_loss:.4f}")
    print(f"Accuracy drop: {base_accuracy - q_accuracy:.4f}\n")

    print("Storage estimate (weights only):")
    print(f"FP32 weights: {fp32_bytes} bytes ({fmt_kb(fp32_bytes):.2f} KB)")
    print(f"INT8 weights + scales: {total_int8_bytes} bytes ({fmt_kb(total_int8_bytes):.2f} KB)")
    print(f"    - weights: {int8_weight_bytes} bytes ({fmt_kb(int8_weight_bytes):.2f} KB)")
    print(f"    - scales:  {scale_bytes} bytes ({fmt_kb(scale_bytes):.2f} KB)")
    print(f"Compression ratio: {fp32_bytes / total_int8_bytes:.2f}x")

    
    
if __name__ == "__main__":
    main()