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

from src.exp_utils import (
    get_device,
    load_test_loader,
    build_model,
    load_weights,
    estimate_fp32_weight_bytes,
    quantize_dequantize_linear_weights_inplace,
    estimate_int8_weight_bytes_plus_scales,
    fmt_kb
)
from src.training import evaluate


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