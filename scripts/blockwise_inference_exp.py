"""
Layerwise vs Blockwise inference experiment (MNIST) using Zstandard-compressed FP32 weights.

Compares:
- Baseline FP32 inference
- Pruned FP32 inference
- Layerwise Zstd inference (decompress one layer at a time)
- Blockwise Zstd inference (decompress one block at a time)

Focus:
- Accuracy
- Storage footprint (compressed payload bytes)
- Total file size + overhead bytes
- Estimated peak runtime RAM
- Inference time (avg per batch, CPU)
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Blockwise exports
from nn_compression.blockwise_export_compressed import (
    estimate_compressed_payload_bytes as estimate_blockwise_payload_bytes,
)
from nn_compression.blockwise_export_compressed import (
    export_fcn_to_compressed as export_blockwise_compressed,
)
from nn_compression.blockwise_export_compressed import (
    load_compressed as load_blockwise_compressed,
)
from nn_compression.blockwise_export_compressed import (
    save_compressed as save_blockwise_compressed,
)
from nn_compression.blockwise_inference import (
    blockwise_evaluate_accuracy,
    measure_blockwise_inference_time,
)
from nn_compression.exp_utils import (
    build_model,
    estimate_fp32_weight_bytes,
    estimate_peak_runtime_blockwise_bytes,
    estimate_peak_runtime_layerwise_bytes,
    fmt_bytes,
    get_device,
    load_test_loader,
    load_weights,
)

# Layerwise exports
from nn_compression.export_compressed import (
    estimate_compressed_payload_bytes as estimate_layerwise_payload_bytes,
)
from nn_compression.export_compressed import (
    export_fcn_to_compressed as export_layerwise_compressed,
)
from nn_compression.export_compressed import (
    load_compressed as load_layerwise_compressed,
)
from nn_compression.export_compressed import (
    save_compressed as save_layerwise_compressed,
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


def save_ram_vs_latency_barchart(
    *,
    out_path: str,
    lw_latency_ms_sample: float,
    bw_latency_ms_sample: float,
    lw_peak_kb: float,
    bw_peak_kb: float,
    title: str = "Layerwise vs Blockwise: Latency vs Estimated Runtime RAM (CPU)",
) -> None:
    """
    Grouped bar chart with two y-axes:
      - Left axis: latency (ms/sample)
      - Right axis: estimated runtime RAM (KB)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    labels = ["Layerwise", "Blockwise"]
    latency = [lw_latency_ms_sample, bw_latency_ms_sample]
    peak_kb = [lw_peak_kb, bw_peak_kb]

    x = list(range(len(labels)))
    width = 0.38

    fig, ax1 = plt.subplots()

    # Latency bars (left axis)
    ax1.bar(
        [i - width / 2 for i in x],
        latency,
        width=width,
        color="tab:blue",
        alpha=0.85,
        label="Latency (ms/sample)",
    )
    ax1.set_ylabel("Latency (ms/sample)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    # Peak RAM bars (right axis)
    ax2 = ax1.twinx()
    ax2.bar(
        [i + width / 2 for i in x],
        peak_kb,
        width=width,
        color="tab:orange",
        alpha=0.85,
        label="Estimated runtime RAM (KB)",
    )
    ax2.set_ylabel("Estimated runtime RAM (KB)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # One combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


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


def _file_stats(path: str, payload_bytes: int) -> tuple[int, int, float]:
    """Return (file_bytes, overhead_bytes, payload_fraction)."""
    file_bytes = os.path.getsize(path)
    overhead_bytes = file_bytes - payload_bytes
    payload_fraction = payload_bytes / file_bytes if file_bytes > 0 else 0.0
    return file_bytes, overhead_bytes, payload_fraction


def print_report(
    *,
    ckpt_path: str,
    prune_amount: float,
    sparsity_pct: float,
    zstd_level: int,
    block_size: int,
    base_acc: float,
    pruned_acc: float,
    lw_acc: float,
    bw_acc: float,
    fp32_bytes: int,
    lw_payload_bytes: int,
    lw_file_bytes: int,
    lw_overhead_bytes: int,
    lw_payload_fraction: float,
    bw_payload_bytes: int,
    bw_file_bytes: int,
    bw_overhead_bytes: int,
    bw_payload_fraction: float,
    peak_layer_runtime_bytes: int,
    peak_block_runtime_bytes: int,
    runtime_batch_size: int,
    t_base_ms_sample: float,
    t_lw_ms_sample: float,
    t_bw_ms_sample: float,
    save_path_lw: str,
    save_path_bw: str,
) -> None:
    def ratio(a: int, b: int) -> float:
        return (a / b) if b > 0 else float("inf")

    print("\n" + "=" * 94)
    print("Layerwise vs Blockwise Inference Experiment (MNIST) — Zstd FP32".center(94))
    print("=" * 94)

    print("\n[Setup]")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Prune amount   : {prune_amount:.2f}")
    print(f"  Sparsity       : {sparsity_pct:.2f}%")
    print(f"  Zstd level     : {zstd_level}")
    print(f"  Block size     : {block_size}")

    print("\n[Accuracy]")
    print(f"  Baseline FP32             : {base_acc:.4f}")
    print(f"  Pruned FP32               : {pruned_acc:.4f}")
    print(
        f"  Layerwise Zstd (FP32)     : {lw_acc:.4f} (drop vs pruned {pruned_acc - lw_acc:+.4f})"
    )
    print(
        f"  Blockwise Zstd (FP32)     : {bw_acc:.4f} (drop vs pruned {pruned_acc - bw_acc:+.4f})"
    )

    print("\n[Storage — reference]")
    print(f"  FP32 weights (dense)      : {fmt_bytes(fp32_bytes)}")

    print("\n[Storage — layerwise]")
    print(
        f"  Payload (W+b only)        : {fmt_bytes(lw_payload_bytes)}   ({ratio(fp32_bytes, lw_payload_bytes):.2f}x smaller)"
    )
    print(
        f"  Total file size           : {fmt_bytes(lw_file_bytes)}      ({ratio(fp32_bytes, lw_file_bytes):.2f}x smaller)"
    )
    print(f"  Overhead (meta)           : {fmt_bytes(lw_overhead_bytes)}")
    print(f"  Payload fraction          : {lw_payload_fraction*100:.2f}%")
    print(f"  Saved                      : {save_path_lw}")

    print("\n[Storage — blockwise]")
    print(
        f"  Payload (W+b only)        : {fmt_bytes(bw_payload_bytes)}   ({ratio(fp32_bytes, bw_payload_bytes):.2f}x smaller)"
    )
    print(
        f"  Total file size           : {fmt_bytes(bw_file_bytes)}      ({ratio(fp32_bytes, bw_file_bytes):.2f}x smaller)"
    )
    print(f"  Overhead (meta)           : {fmt_bytes(bw_overhead_bytes)}")
    print(f"  Payload fraction          : {bw_payload_fraction*100:.2f}%")
    print(f"  Saved                      : {save_path_bw}")

    print("\n[Estimated Peak Runtime RAM]")
    print(f"  Layerwise inference       : {fmt_bytes(peak_layer_runtime_bytes)}")
    print(f"  Blockwise inference       : {fmt_bytes(peak_block_runtime_bytes)}")
    print(f"  RAM batch size used       : {runtime_batch_size}")

    print("\n[Timing — average inference latency per sample (CPU)]")
    print(f"  Baseline FP32 forward     : {t_base_ms_sample:.4f} ms/sample")
    print(f"  Layerwise Zstd forward    : {t_lw_ms_sample:.4f} ms/sample")
    print(f"  Blockwise Zstd forward    : {t_bw_ms_sample:.4f} ms/sample")

    print("=" * 94 + "\n")


def main():
    device = get_device()
    infer_device = torch.device("cpu")  # Run decompression/inference
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader()

    ckpt_path = "fcn_mnist_best.pt"

    prune_amount = 0.85
    zstd_level = 16
    block_size = 32

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

    # 3) Export both compressed formats
    os.makedirs("results/zstd", exist_ok=True)
    save_path_lw = f"results/zstd/fcn_mnist_pruned_{int(prune_amount*100)}_layerwise_zstd{zstd_level}.pt"
    save_path_bw = f"results/zstd/fcn_mnist_pruned_{int(prune_amount*100)}_blockwise_bs{block_size}_zstd{zstd_level}.pt"

    packed_lw = export_layerwise_compressed(pruned_model, zstd_level=zstd_level)
    save_layerwise_compressed(packed_lw, save_path_lw)
    compressed_lw = load_layerwise_compressed(save_path_lw)

    packed_bw = export_blockwise_compressed(
        pruned_model, zstd_level=zstd_level, block_size=block_size
    )
    save_blockwise_compressed(packed_bw, save_path_bw)
    compressed_bw = load_blockwise_compressed(save_path_bw)

    # 4) Storage stats
    fp32_bytes = estimate_fp32_weight_bytes(pruned_model)

    lw_payload_bytes = estimate_layerwise_payload_bytes(compressed_lw)
    lw_file_bytes, lw_overhead_bytes, lw_payload_fraction = _file_stats(
        save_path_lw, lw_payload_bytes
    )

    bw_payload_bytes = estimate_blockwise_payload_bytes(compressed_bw)
    bw_file_bytes, bw_overhead_bytes, bw_payload_fraction = _file_stats(
        save_path_bw, bw_payload_bytes
    )

    # 5) Accuracy via compressed inference
    lw_acc = layerwise_evaluate_accuracy(compressed_lw, test_loader, infer_device)
    bw_acc = blockwise_evaluate_accuracy(compressed_bw, test_loader, infer_device)

    # 6) Estimated peak runtime RAM for both methods
    runtime_batch_size = test_loader.batch_size  # CHANGED: use the same batch size as the MNIST test loader

    if runtime_batch_size is None or runtime_batch_size <= 0:  # NEW: safety check
        raise ValueError(
            "test_loader.batch_size is None/invalid. Set a valid batch_size in load_test_loader()."
        )

    peak_layer_runtime_bytes = estimate_peak_runtime_layerwise_bytes(
        pruned_model, runtime_batch_size
    )

    peak_block_runtime_bytes = estimate_peak_runtime_blockwise_bytes(
        pruned_model, block_size, runtime_batch_size
    )

    # 7) Timing
    t_base = measure_baseline_inference_time(
        base_model.to(infer_device), test_loader, infer_device
    )
    t_lw = measure_layerwise_inference_time(compressed_lw, test_loader, infer_device)
    t_bw = measure_blockwise_inference_time(compressed_bw, test_loader, infer_device)

    batch_size = test_loader.batch_size
    if batch_size is None or batch_size <= 0:
        raise ValueError(
            "test_loader.batch_size is None/invalid. Set a batch_size in load_test_loader()."
        )

    t_base_ms_sample = (t_base * 1000.0) / batch_size
    t_lw_ms_sample = (t_lw * 1000.0) / batch_size
    t_bw_ms_sample = (t_bw * 1000.0) / batch_size

    # Peak RAM numbers in KB (for plot)
    lw_peak_kb = peak_layer_runtime_bytes / 1024.0
    bw_peak_kb = peak_block_runtime_bytes / 1024.0

    plot_path = "results/zstd/plots/runtime_ram_vs_latency_layerwise_vs_blockwise_bars.png"
    save_ram_vs_latency_barchart(
        out_path=plot_path,
        lw_latency_ms_sample=t_lw_ms_sample,
        bw_latency_ms_sample=t_bw_ms_sample,
        lw_peak_kb=lw_peak_kb,
        bw_peak_kb=bw_peak_kb,
        title="Layerwise vs Blockwise: Latency vs Estimated Runtime RAM (ms/sample, CPU)",
    )
    print(f"Saved plot: {plot_path}")

    # Report
    print_report(
        ckpt_path=ckpt_path,
        prune_amount=prune_amount,
        sparsity_pct=sparsity_pct,
        zstd_level=zstd_level,
        block_size=block_size,
        base_acc=base_accuracy,
        pruned_acc=pruned_accuracy,
        lw_acc=lw_acc,
        bw_acc=bw_acc,
        fp32_bytes=fp32_bytes,
        lw_payload_bytes=lw_payload_bytes,
        lw_file_bytes=lw_file_bytes,
        lw_overhead_bytes=lw_overhead_bytes,
        lw_payload_fraction=lw_payload_fraction,
        bw_payload_bytes=bw_payload_bytes,
        bw_file_bytes=bw_file_bytes,
        bw_overhead_bytes=bw_overhead_bytes,
        bw_payload_fraction=bw_payload_fraction,
        peak_layer_runtime_bytes=peak_layer_runtime_bytes,
        peak_block_runtime_bytes=peak_block_runtime_bytes,
        runtime_batch_size=runtime_batch_size,
        t_base_ms_sample=t_base_ms_sample,
        t_lw_ms_sample=t_lw_ms_sample,
        t_bw_ms_sample=t_bw_ms_sample,
        save_path_lw=save_path_lw,
        save_path_bw=save_path_bw,
    )


if __name__ == "__main__":
    main()
