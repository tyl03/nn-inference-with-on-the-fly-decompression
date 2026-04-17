"""
Baseline vs Blockwise CNN inference experiment (MNIST) using Zstandard-compressed FP32 weights.

CNN version.

Compares:
- Baseline FP32 CNN inference
- Blockwise Zstd CNN inference (decompress one block at a time)

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
    fmt_bytes,
    get_device,
    load_test_loader,
    load_weights,
)
from nn_compression.training import evaluate


def save_ram_vs_latency_barchart(
    *,
    out_path: str,
    base_latency_ms_sample: float,
    bw_latency_ms_sample: float,
    base_peak_kb: float,
    bw_peak_kb: float,
    title: str = "CNN Baseline vs Blockwise: Latency vs Estimated Runtime RAM (CPU)",
) -> None:
    """
    Grouped bar chart with two y-axes:
      - Left axis: latency (ms/sample)
      - Right axis: estimated runtime RAM (KB)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    labels = ["Baseline", "Blockwise"]
    latency = [base_latency_ms_sample, bw_latency_ms_sample]
    peak_kb = [base_peak_kb, bw_peak_kb]

    x = list(range(len(labels)))
    width = 0.38

    fig, ax1 = plt.subplots()

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
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def measure_baseline_inference_time(
    model, loader, device, warmup_batches=5, timed_batches=30
):
    """
    Returns avg forward time per batch (seconds) for the baseline FP32 model.
    """
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
    file_bytes = os.path.getsize(path)
    overhead_bytes = file_bytes - payload_bytes
    payload_fraction = payload_bytes / file_bytes if file_bytes > 0 else 0.0
    return file_bytes, overhead_bytes, payload_fraction


def estimate_baseline_runtime_bytes(model: nn.Module, batch_size: int) -> int:
    """
    Estimate peak runtime RAM for the baseline FP32 CNN model.

    This is a rough estimate for comparison against blockwise inference.

    Supports:
    - Conv2d
    - Linear

    Assumes MNIST input size: 1 x 28 x 28
    """
    peak = 0
    B = int(batch_size)

    current_h = 28
    current_w = 28

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
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

        elif isinstance(m, nn.Linear):
            in_features = int(m.in_features)
            out_features = int(m.out_features)

            x_bytes = B * in_features * 4
            y_bytes = B * out_features * 4
            W_bytes = out_features * in_features * 4
            bias_bytes = out_features * 4 if m.bias is not None else 0

            layer_peak = x_bytes + y_bytes + W_bytes + bias_bytes
            peak = max(peak, layer_peak)

    return peak


def print_report(
    *,
    ckpt_path: str,
    zstd_level: int,
    block_size: int,
    base_acc: float,
    bw_acc: float,
    fp32_bytes: int,
    bw_payload_bytes: int,
    bw_file_bytes: int,
    bw_overhead_bytes: int,
    bw_payload_fraction: float,
    peak_base_runtime_bytes: int,
    peak_block_runtime_bytes: int,
    runtime_batch_size: int,
    t_base_ms_sample: float,
    t_bw_ms_sample: float,
    save_path_bw: str,
) -> None:
    def ratio(a: int, b: int) -> float:
        return (a / b) if b > 0 else float("inf")

    print("\n" + "=" * 94)
    print("Baseline vs Blockwise CNN Inference Experiment (MNIST) — Zstd FP32".center(94))
    print("=" * 94)

    print("\n[Setup]")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Zstd level     : {zstd_level}")
    print(f"  Block size     : {block_size}")

    print("\n[Accuracy]")
    print(f"  Baseline FP32             : {base_acc:.4f}")
    print(
        f"  Blockwise Zstd (FP32)     : {bw_acc:.4f} (drop vs baseline {base_acc - bw_acc:+.4f})"
    )

    print("\n[Storage — reference]")
    print(f"  FP32 weights (dense)      : {fmt_bytes(fp32_bytes)}")

    print("\n[Storage — blockwise]")
    print(
        f"  Payload (W+b only)        : {fmt_bytes(bw_payload_bytes)}   ({ratio(fp32_bytes, bw_payload_bytes):.2f}x smaller)"
    )
    print(
        f"  Total file size           : {fmt_bytes(bw_file_bytes)}      ({ratio(fp32_bytes, bw_file_bytes):.2f}x smaller)"
    )
    print(f"  Overhead (meta)           : {fmt_bytes(bw_overhead_bytes)}")
    print(f"  Payload fraction          : {bw_payload_fraction*100:.2f}%")
    print(f"  Saved                     : {save_path_bw}")

    print("\n[Estimated Peak Runtime RAM]")
    print(f"  Baseline FP32 inference   : {fmt_bytes(peak_base_runtime_bytes)}")
    print(f"  Blockwise inference       : {fmt_bytes(peak_block_runtime_bytes)}")
    print(f"  RAM batch size used       : {runtime_batch_size}")

    print("\n[Timing — average inference latency per sample (CPU)]")
    print(f"  Baseline FP32 forward     : {t_base_ms_sample:.4f} ms/sample")
    print(f"  Blockwise Zstd forward    : {t_bw_ms_sample:.4f} ms/sample")

    print("=" * 94 + "\n")


def main():
    device = get_device()
    infer_device = torch.device("cpu")
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader(batch_size=1)

    ckpt_path = "cnn_mnist_best.pt"

    # Use the selected block size from the sweep here
    zstd_level = 16
    block_size = 4

    # 1) Baseline FP32 accuracy
    base_model = build_model(device, model_type="cnn")
    load_weights(base_model, ckpt_path, device)
    _, base_accuracy = evaluate(base_model, test_loader, loss_fn, device)

    # 2) Export blockwise compressed CNN
    os.makedirs("results/zstd", exist_ok=True)
    save_path_bw = (
        f"results/zstd/cnn_mnist_blockwise_bs{block_size}_zstd{zstd_level}.pt"
    )

    packed_bw = export_blockwise_compressed(
        base_model, zstd_level=zstd_level, block_size=block_size
    )
    save_blockwise_compressed(packed_bw, save_path_bw)
    compressed_bw = load_blockwise_compressed(save_path_bw)

    # 3) Storage stats
    fp32_bytes = estimate_fp32_weight_bytes(base_model)

    bw_payload_bytes = estimate_blockwise_payload_bytes(compressed_bw)
    bw_file_bytes, bw_overhead_bytes, bw_payload_fraction = _file_stats(
        save_path_bw, bw_payload_bytes
    )

    # 4) Accuracy via blockwise compressed inference
    bw_acc = blockwise_evaluate_accuracy(compressed_bw, test_loader, infer_device)

    # 5) Estimated peak runtime RAM
    runtime_batch_size = test_loader.batch_size
    if runtime_batch_size is None or runtime_batch_size <= 0:
        raise ValueError(
            "test_loader.batch_size is None/invalid. Set a valid batch_size in load_test_loader()."
        )

    peak_base_runtime_bytes = estimate_baseline_runtime_bytes(
        base_model, runtime_batch_size
    )
    peak_block_runtime_bytes = estimate_peak_runtime_blockwise_bytes(
        base_model, block_size, runtime_batch_size
    )

    # 6) Timing
    t_base = measure_baseline_inference_time(
        base_model.to(infer_device), test_loader, infer_device
    )
    t_bw = measure_blockwise_inference_time(compressed_bw, test_loader, infer_device)

    batch_size = test_loader.batch_size
    if batch_size is None or batch_size <= 0:
        raise ValueError(
            "test_loader.batch_size is None/invalid. Set a batch_size in load_test_loader()."
        )

    t_base_ms_sample = (t_base * 1000.0) / batch_size
    t_bw_ms_sample = (t_bw * 1000.0) / batch_size

    # Peak RAM numbers in KB (for plot)
    base_peak_kb = peak_base_runtime_bytes / 1024.0
    bw_peak_kb = peak_block_runtime_bytes / 1024.0

    plot_path = "results/zstd/plots/cnn_runtime_ram_vs_latency_baseline_vs_blockwise_bars.pdf"
    save_ram_vs_latency_barchart(
        out_path=plot_path,
        base_latency_ms_sample=t_base_ms_sample,
        bw_latency_ms_sample=t_bw_ms_sample,
        base_peak_kb=base_peak_kb,
        bw_peak_kb=bw_peak_kb,
        title="CNN Baseline vs Blockwise: Latency vs Estimated Runtime RAM (ms/sample, CPU)",
    )
    print(f"Saved plot: {plot_path}")

    print_report(
        ckpt_path=ckpt_path,
        zstd_level=zstd_level,
        block_size=block_size,
        base_acc=base_accuracy,
        bw_acc=bw_acc,
        fp32_bytes=fp32_bytes,
        bw_payload_bytes=bw_payload_bytes,
        bw_file_bytes=bw_file_bytes,
        bw_overhead_bytes=bw_overhead_bytes,
        bw_payload_fraction=bw_payload_fraction,
        peak_base_runtime_bytes=peak_base_runtime_bytes,
        peak_block_runtime_bytes=peak_block_runtime_bytes,
        runtime_batch_size=runtime_batch_size,
        t_base_ms_sample=t_base_ms_sample,
        t_bw_ms_sample=t_bw_ms_sample,
        save_path_bw=save_path_bw,
    )


if __name__ == "__main__":
    main()