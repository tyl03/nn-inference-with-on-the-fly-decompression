"""
Layerwise inference experiment (MNIST):

Compares:
- Baseline FP32 inference
- Layerwise int8 inference
- Layerwise int8 + Huffman inference

Focus:
- Accuracy impact
- Storage footprint
- Inference time (avg per batch)
"""

import torch
import torch.nn as nn
import time

from src.exp_utils import (
    get_device,
    load_test_loader,
    build_model,
    load_weights,
    estimate_fp32_weight_bytes,
    estimate_peak_decompressed_layer_bytes,
    fmt_bytes,
    save_compressed_model,
    load_compressed_model,
    estimate_compressed_storage_bytes_from_file,
)
from src.training import evaluate
from src.pruning import magnitude_prune_linear_layers, make_pruning_permanent, model_sparsity

from src.layerwise_inference import layerwise_evaluate_accuracy, measure_layerwise_inference_time
from src.export_compressed_huffman import (
    compress_int8_model_with_huffman,
    save_int8_huffman,
    load_int8_huffman,
    estimate_int8_huffman_weight_bytes,
)
from src.layerwise_inference_huffman import layerwise_evaluate_accuracy_int8_huffman, measure_layerwise_inference_time_huffman


@torch.no_grad()
def measure_baseline_inference_time(model, loader, device, warmup_batches=5, timed_batches=30):
    model.eval()
    it = iter(loader)

    for _ in range(warmup_batches):
        x, _ = next(it)
        _ = model(x.to(device))

    times = []
    for _ in range(timed_batches):
        x, _ = next(it)
        t0 = time.perf_counter()
        _ = model(x.to(device))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times)


def print_report(
    ckpt_path: str,
    prune_amount: float,
    sparsity: float,
    base_acc: float,
    pruned_acc: float,
    lw_acc_int8: float,
    lw_acc_int8_huff: float,
    fp32_bytes: int,
    int8_bytes: int,
    int8_huff_bytes: int,
    peak_layer_bytes: int,
    t_base: float,
    t_int8: float,
    t_int8_huff: float,
    save_path_int8: str,
    save_path_int8_huff: str,
) -> None:
    ratio_int8 = fp32_bytes / int8_bytes if int8_bytes > 0 else float("inf")
    ratio_huff = fp32_bytes / int8_huff_bytes if int8_huff_bytes > 0 else float("inf")

    print("\n" + "=" * 78)
    print("Layerwise Inference Experiment (MNIST)".center(78))
    print("=" * 78)

    print("\n[Setup]")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Prune amount   : {prune_amount:.2f}")
    print(f"  Sparsity       : {sparsity:.2f}%")

    print("\n[Accuracy]")
    print(f"  Baseline FP32          : {base_acc:.4f}")
    print(f"  Pruned FP32            : {pruned_acc:.4f}")
    print(f"  Layerwise int8         : {lw_acc_int8:.4f} (drop {base_acc - lw_acc_int8:+.4f})")
    print(f"  Layerwise int8+Huffman : {lw_acc_int8_huff:.4f} (drop {base_acc - lw_acc_int8_huff:+.4f})")

    print("\n[Storage (weights/scales/bias)]")
    print(f"  FP32 weights (dense)       : {fmt_bytes(fp32_bytes)}")
    print(f"  Stored int8+meta (est.)    : {fmt_bytes(int8_bytes)}   ({ratio_int8:.2f}x)")
    print(f"  Stored int8+Huffman (est.) : {fmt_bytes(int8_huff_bytes)}   ({ratio_huff:.2f}x)")

    print("\n[Peak decompressed weights]")
    print(f"  Largest layer (FP32)       : {fmt_bytes(peak_layer_bytes)}")

    print("\n[Timing - avg per batch, CPU]")
    print(f"  Baseline FP32 forward      : {t_base*1000:.3f} ms")
    print(f"  Layerwise int8 forward     : {t_int8*1000:.3f} ms")
    print(f"  Layerwise int8+Huffman     : {t_int8_huff*1000:.3f} ms")

    print("\n[Artifacts]")
    print(f"  Saved int8 model           : {save_path_int8}")
    print(f"  Saved int8+Huffman model   : {save_path_int8_huff}")
    print("=" * 78 + "\n")


def main():
    device = get_device()
    layer_device = torch.device("cpu")  # Layerwise inference on CPU to highlight time differences
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader()

    ckpt_path = "fcn_mnist_best.pt"
    prune_amount = 0.5

    # 1) Baseline FP32
    base_model = build_model(device)
    load_weights(base_model, ckpt_path, device)
    _, base_accuracy = evaluate(base_model, test_loader, loss_fn, device)

    # 2) Prune model and evaluate pruned FP32
    pruned_model = build_model(device)
    load_weights(pruned_model, ckpt_path, device)

    if prune_amount > 0.0:
        magnitude_prune_linear_layers(pruned_model, amount=prune_amount)
        make_pruning_permanent(pruned_model)

    _, pruned_accuracy = evaluate(pruned_model, test_loader, loss_fn, device)
    sparsity = model_sparsity(pruned_model) * 100.0

    # 3) Export int8 compressed model
    save_path_int8 = f"fcn_mnist_pruned_{int(prune_amount*100)}_int8_compressed.pt"
    save_compressed_model(pruned_model, save_path_int8)
    compressed_int8 = load_compressed_model(save_path_int8)

    # 4) Layerwise inference accuracy (int8 only)
    layer_device = torch.device("cpu")
    lw_accuracy_int8 = layerwise_evaluate_accuracy(compressed_int8, test_loader, layer_device)

    # 5) Storage + peak memory estimates
    fp32_weight_bytes = estimate_fp32_weight_bytes(pruned_model)
    int8_bytes_est = estimate_compressed_storage_bytes_from_file(save_path_int8)
    peak_layer_fp32_bytes = estimate_peak_decompressed_layer_bytes(pruned_model)

    # 6) Build + save int8+Huffman model
    compressed_int8_huff = compress_int8_model_with_huffman(compressed_int8)
    save_path_int8_huff = f"fcn_mnist_pruned_{int(prune_amount*100)}_int8_huffman.pt"
    save_int8_huffman(compressed_int8_huff, save_path_int8_huff)

    # Load (just to prove load works)
    compressed_int8_huff_loaded = load_int8_huffman(save_path_int8_huff)

    # 7) Layerwise inference accuracy (int8 + Huffman)
    lw_accuracy_int8_huff = layerwise_evaluate_accuracy_int8_huffman(
        compressed_int8_huff_loaded, test_loader, layer_device
    )

    # 8) Storage estimate for Huffman format
    int8_huff_bytes_est = estimate_int8_huffman_weight_bytes(compressed_int8_huff_loaded)

    # 9) Timing
    t_base = measure_baseline_inference_time(base_model.to(layer_device), test_loader, layer_device)
    t_int8 = measure_layerwise_inference_time(compressed_int8, test_loader, layer_device)
    t_int8_huff = measure_layerwise_inference_time_huffman(compressed_int8_huff_loaded, test_loader, layer_device)

    # Report
    print_report(
        ckpt_path=ckpt_path,
        prune_amount=prune_amount,
        sparsity=sparsity,
        base_acc=base_accuracy,
        pruned_acc=pruned_accuracy,
        lw_acc_int8=lw_accuracy_int8,
        lw_acc_int8_huff=lw_accuracy_int8_huff,
        fp32_bytes=fp32_weight_bytes,
        int8_bytes=int8_bytes_est,
        int8_huff_bytes=int8_huff_bytes_est,
        peak_layer_bytes=peak_layer_fp32_bytes,
        t_base=t_base,
        t_int8=t_int8,
        t_int8_huff=t_int8_huff,
        save_path_int8=save_path_int8,
        save_path_int8_huff=save_path_int8_huff,
    )


if __name__ == "__main__":
    main()
