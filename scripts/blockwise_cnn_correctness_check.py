"""
Correctness check for blockwise CNN inference on MNIST.

Purpose:
- Verify that the trained CNN loads correctly
- Verify that blockwise export works for CNN
- Verify that blockwise inference runs end-to-end
- Compare baseline FP32 accuracy vs blockwise compressed accuracy
- Print exported layer order for inspection
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nn_compression.blockwise_export_compressed import (
    export_fcn_to_compressed,
)
from nn_compression.blockwise_inference import (
    blockwise_evaluate_accuracy,
)
from nn_compression.exp_utils import (
    build_model,
    get_device,
    load_test_loader,
    load_weights,
)
from nn_compression.training import evaluate


def main():
    device = get_device()
    infer_device = torch.device("cpu")
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader(batch_size=1)

    ckpt_path = "cnn_mnist_best.pt"
    block_size = 32
    zstd_level = 16

    # Build and load trained CNN
    model = build_model(device, model_type="cnn")
    load_weights(model, ckpt_path, device)

    # Baseline FP32 accuracy
    _, base_acc = evaluate(model, test_loader, loss_fn, device)

    # Export compressed CNN
    compressed = export_fcn_to_compressed(
        model,
        zstd_level=zstd_level,
        block_size=block_size,
    )

    # Print exported layer types for sanity check
    print("\nExported layer order:")
    for i, entry in enumerate(compressed["layers"]):
        print(f"  Layer {i:02d}: {entry['type']}")

    # Blockwise compressed accuracy
    bw_acc = blockwise_evaluate_accuracy(compressed, test_loader, infer_device)

    print("\nCorrectness check")
    print("-" * 50)
    print(f"Checkpoint           : {ckpt_path}")
    print(f"Block size           : {block_size}")
    print(f"Zstd level           : {zstd_level}")
    print(f"Baseline FP32 acc    : {base_acc:.4f}")
    print(f"Blockwise CNN acc    : {bw_acc:.4f}")
    print(f"Accuracy difference  : {base_acc - bw_acc:+.6f}")
    print("-" * 50)

    # Simple warning if something looks off
    diff = abs(base_acc - bw_acc)
    if diff > 1e-3:
        print("WARNING: Accuracy difference is larger than expected.")
    else:
        print("OK: Blockwise CNN matches baseline very closely.")


if __name__ == "__main__":
    main()