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
    estimate_compressed_storage_bytes_from_model,
    fmt_bytes,
    apply_qdq_to_linear_weights_inplace
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
    
    # Noise from simulating QDQ
    qmodel = build_model(device)
    load_weights(qmodel, ckpt_path, device)
    apply_qdq_to_linear_weights_inplace(qmodel)
    q_loss, q_accuracy = evaluate(qmodel, test_loader, loss_fn, device)
    
    # Storage estimates (weights only)
    fp32_bytes = estimate_fp32_weight_bytes(model)
    compressed_bytes = estimate_compressed_storage_bytes_from_model(model)
        
    print("\nQuantization Storage Results (symmetric int8 stored format)\n")
    print(f"FP32 accuracy: {base_accuracy:.4f}   loss: {base_loss:.4f}\n")
    print(f"QDQ accuracy (simulated): {q_accuracy:.4f}   loss: {q_loss:.4f}")
    print(f"Accuracy drop vs FP32: {base_accuracy - q_accuracy:.4f}")

    print("Storage estimate (weights + scale + bias):")
    print(f"FP32 weights (RAM/reference): {fmt_bytes(fp32_bytes)}")
    print(f"Compressed stored (int8+meta): {fmt_bytes(compressed_bytes)}")
    print(f"Compression ratio: {fp32_bytes / compressed_bytes:.2f}x")

    
    
if __name__ == "__main__":
    main()