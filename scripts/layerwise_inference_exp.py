"""
Layerwise inference experiment (MNIST) using Zstandard-compressed FP32 layers.

Compares:
- Baseline FP32 inference
- Pruned FP32 inference
- Layerwise Zstd inference (decompress one layer at a time)

Focus:
- Accuracy
- Storage footprint (compressed payload bytes)
- Peak decompressed layer bytes (RAM estimate)
- Inference time (avg per batch, CPU)
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from nn_compression.exp_utils import (
    build_model,
    estimate_fp32_weight_bytes,
    estimate_peak_decompressed_layer_bytes,
    fmt_bytes,
    get_device,
    load_test_loader,
    load_weights,
)
from nn_compression.export_compressed import (
    estimate_compressed_payload_bytes,
    export_fcn_to_compressed,
    load_compressed,
    save_compressed,
)
from nn_compression.layerwise_inference import (
    layerwise_evaluate_accuracy,
    measure_layerwise_inference_time,
)
from nn_compression.pruning import (
    global_magnitude_prune_linear_layers,
    make_pruning_permanent,
    model_sparsity,
)
from nn_compression.training import evaluate


@torch.no_grad()
def measure_baseline_inference_time(
    model, loader, device, warmup_batches=5, timed_batches=30
):
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
    sparsity_pct: float,
    zstd_level: int,
    base_acc: float,
    pruned_acc: float,
    lw_acc_zstd: float,
    fp32_bytes: int,
    zstd_payload_bytes: int,
    file_bytes: int,
    overhead_bytes: int,
    peak_layer_fp32_bytes: int,
    t_base: float,
    t_lw: float,
    save_path: str,
) -> None:
    ratio_payload = (
        fp32_bytes / zstd_payload_bytes if zstd_payload_bytes > 0 else float("inf")
    )
    ratio_file = fp32_bytes / file_bytes if file_bytes > 0 else float("inf")

    payload_fraction = (zstd_payload_bytes / file_bytes) if file_bytes > 0 else 0.0

    print("\n" + "=" * 78)
    print("Layerwise Inference Experiment (MNIST) — Zstd FP32".center(78))
    print("=" * 78)

    print("\n[Setup]")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Prune amount   : {prune_amount:.2f}")
    print(f"  Sparsity       : {sparsity_pct:.2f}%")
    print(f"  Zstd level     : {zstd_level}")

    print("\n[Accuracy]")
    print(f"  Baseline FP32          : {base_acc:.4f}")
    print(f"  Pruned FP32            : {pruned_acc:.4f}")
    print(
        f"  Layerwise Zstd (FP32)  : {lw_acc_zstd:.4f} (drop vs pruned {pruned_acc - lw_acc_zstd:+.4f})"
    )

    print("\n[Storage]")
    print(f"  FP32 weights (dense)       : {fmt_bytes(fp32_bytes)}")
    print(
        f"  Zstd payload (W+b only)    : {fmt_bytes(zstd_payload_bytes)}   ({ratio_payload:.2f}x smaller)"
    )
    print(
        f"  Total file size (on disk)  : {fmt_bytes(file_bytes)}   ({ratio_file:.2f}x smaller)"
    )
    print(f"  Overhead (meta)            : {fmt_bytes(overhead_bytes)}")
    print(f"  Payload fraction           : {payload_fraction*100:.2f}%")

    print("\n[Peak decompressed weights (RAM estimate)]")
    print(f"  Largest layer (FP32)       : {fmt_bytes(peak_layer_fp32_bytes)}")

    print("\n[Timing - avg per batch, CPU]")
    print(f"  Baseline FP32 forward      : {t_base*1000:.3f} ms")
    print(f"  Layerwise Zstd forward     : {t_lw*1000:.3f} ms")

    print("\n[Artifacts]")
    print(f"  Saved compressed model     : {save_path}")
    print("=" * 78 + "\n")


def main():
    device = get_device()
    layer_device = torch.device(
        "cpu"
    )  # Layerwise inference on CPU to highlight time differences
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader(batch_size=1)

    ckpt_path = "fcn_mnist_best.pt"

    prune_amount = 0.85
    zstd_level = 7

    # 1) Baseline FP32 accuracy
    base_model = build_model(device)
    load_weights(base_model, ckpt_path, device)
    _, base_accuracy = evaluate(base_model, test_loader, loss_fn, device)

    # 2) Pruned FP32 accuracy
    pruned_model = build_model(device)
    load_weights(pruned_model, ckpt_path, device)

    global_magnitude_prune_linear_layers(pruned_model, amount=prune_amount)
    make_pruning_permanent(pruned_model)

    _, pruned_accuracy = evaluate(pruned_model, test_loader, loss_fn, device)
    sparsity_pct = model_sparsity(pruned_model) * 100.0

    # 3) Export zstd compressed model
    os.makedirs("results/zstd", exist_ok=True)
    save_path = (
        f"results/zstd/fcn_mnist_pruned_{int(prune_amount*100)}_zstd_lvl{zstd_level}.pt"
    )

    packed = export_fcn_to_compressed(pruned_model, zstd_level=zstd_level)
    save_compressed(packed, save_path)

    compressed = load_compressed(save_path)

    # Measure compressed payload bytes and overhead
    zstd_payload_bytes = estimate_compressed_payload_bytes(compressed)
    file_bytes = os.path.getsize(save_path)
    overhead_bytes = file_bytes - zstd_payload_bytes

    # 4) Layerwise inference accuracy
    lw_accuracy_zstd = layerwise_evaluate_accuracy(
        compressed, test_loader, layer_device
    )

    # 5) Storage + RAM estimates
    fp32_bytes = estimate_fp32_weight_bytes(pruned_model)
    zstd_payload_bytes = estimate_compressed_payload_bytes(compressed)
    peak_layer_fp32_bytes = estimate_peak_decompressed_layer_bytes(pruned_model)

    # 6) Timing
    t_base = measure_baseline_inference_time(
        base_model.to(layer_device), test_loader, layer_device
    )
    t_lw = measure_layerwise_inference_time(compressed, test_loader, layer_device)

    # Report
    print_report(
        ckpt_path=ckpt_path,
        prune_amount=prune_amount,
        sparsity_pct=sparsity_pct,
        zstd_level=zstd_level,
        base_acc=base_accuracy,
        pruned_acc=pruned_accuracy,
        lw_acc_zstd=lw_accuracy_zstd,
        fp32_bytes=fp32_bytes,
        zstd_payload_bytes=zstd_payload_bytes,
        file_bytes=file_bytes,
        overhead_bytes=overhead_bytes,
        peak_layer_fp32_bytes=peak_layer_fp32_bytes,
        t_base=t_base,
        t_lw=t_lw,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
