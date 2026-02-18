"""
Quantization experiment:
- Load a trained FCN (MNIST)
- Quantize weights of Linear layers to int8 (symmetric, [-127,127])
- Dequantize back to FP32 (simulates on-the-fly dequant inference)
- Measure accuracy drop
- Measure storage size (FP32 vs int8 + scale metadata)
"""

import torch.nn as nn

from nn_compression.exp_utils import (
    apply_qdq_to_linear_weights_inplace,
    build_model,
    estimate_compressed_storage_bytes_from_model,
    estimate_fp32_weight_bytes,
    fmt_bytes,
    get_device,
    load_test_loader,
    load_weights,
)
from nn_compression.training import evaluate


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

    # Storage estimates (stored format: int8 weights + scale + bias)
    fp32_bytes = estimate_fp32_weight_bytes(model)
    compressed_bytes = estimate_compressed_storage_bytes_from_model(model)

    acc_drop = base_accuracy - q_accuracy
    ratio = (fp32_bytes / compressed_bytes) if compressed_bytes >= 0 else float("inf")

    print("\n" + "=" * 70)
    print("Quantization Experiment (MNIST)".center(70))
    print("=" * 70)

    print("\n[Setup]")
    print(f"  Checkpoint     : {ckpt_path}")
    print("  Method         : symmetric int8 (stored), QDQ simulated (FP32 compute)")

    print("\n[Accuracy]")
    print(f"  FP32 baseline  : {base_accuracy:.4f}   (loss {base_loss:.4f})")
    print(f"  QDQ simulated  : {q_accuracy:.4f}   (loss {q_loss:.4f})")
    print(f"  Accuracy drop vs FP32   : {acc_drop:+.4f}")

    print("\n[Storage]")
    print(f"  FP32 weights (B)        : {fmt_bytes(fp32_bytes)}")
    print(f"  Stored int8+meta (B)    : {fmt_bytes(compressed_bytes)}")
    print(f"  Compression ratio       : {ratio:.2f}x")
    print("  (int8+meta = int8 weights + FP32 scale + FP32 bias + layer metadata)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
