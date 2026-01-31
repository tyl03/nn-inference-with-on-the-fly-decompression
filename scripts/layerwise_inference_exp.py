"""
Layerwise inference experiment (MNIST):

Purpose:
- Demonstrate the bachelor-project idea:
  Store the model in a compressed form, then during inference only decompress
  ONE layer at a time (compute -> discard -> next layer).

What this script does:
    1) Load MNIST test set
    2) Load a trained FCN checkpoint
    3) Evaluate baseline accuracy (normal PyTorch inference)
    4) Optionally prune + make pruning permanent, then evaluate pruned accuracy (still normal)
    5) Export the pruned model to a compressed, per-layer format (int8 weights + scale + bias)
    6) Run layerwise inference on the compressed file and measure accuracy
    7) Print storage estimates and peak decompressed-weight estimate
"""

import torch
import torch.nn as nn

from src.exp_utils import (
    get_device,
    load_test_loader,
    build_model,
    load_weights,
    estimate_fp32_weight_bytes,
    estimate_peak_decompressed_layer_bytes,
    fmt_bytes
)
from src.training import evaluate
from src.pruning import magnitude_prune_linear_layers, make_pruning_permanent, model_sparsity
from src.export_compressed import export_fcn_to_compressed, save_compressed, load_compressed, estimate_compressed_weight_bytes
from src.layerwise_inference import layerwise_evaluate_accuracy


def print_report(
    ckpt_path: str,
    prune_amount: float,
    sparsity: float,
    base_acc: float, base_loss: float,
    pruned_acc: float, pruned_loss: float,
    lw_acc: float,
    fp32_bytes: int,
    compressed_bytes: int,
    peak_layer_bytes: int,
    save_path: str,
) -> None:
    drop = base_acc - lw_acc
    ratio = (fp32_bytes / compressed_bytes) if compressed_bytes > 0 else float("inf")

    print("\n" + "=" * 70)
    print("Layerwise Inference Experiment (MNIST)".center(70))
    print("=" * 70)

    print("\n[Setup]")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Prune amount   : {prune_amount:.2f}")
    print(f"  Sparsity       : {sparsity:.2f}%")

    print("\n[Accuracy]")
    print(f"  Baseline FP32  : {base_acc:.4f}   (loss {base_loss:.4f})")
    print(f"  Pruned FP32    : {pruned_acc:.4f}   (loss {pruned_loss:.4f})")
    print(f"  Layerwise int8 : {lw_acc:.4f}")
    print(f"  Drop vs base   : {drop:.4f}")

    print("\n[Storage (weights/scales/bias)]")
    print(f"  FP32 weights           : {fmt_bytes(fp32_bytes)}")
    print(f"  Compressed (int8+meta) : {fmt_bytes(compressed_bytes)}")
    print(f"  Compression ratio      : {ratio:.2f}x")

    print("\n[Peak decompressed weights]")
    print(f"  Largest layer (FP32)   : {fmt_bytes(peak_layer_bytes)}")
    print("  (Temporary RAM used for the decompressed layer during inference.)")

    print("\n[Artifacts]")
    print(f"  Saved compressed model : {save_path}")
    print("=" * 70 + "\n")



def main():
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader()
    
    ckpt_path = "fcn_mnist_best.pt"
    
    # Choose pruning amount based on the sweep results
    prune_amount = 0.5
    
    # 1) Baseline FP32 accuracy
    base_model = build_model(device)
    load_weights(base_model, ckpt_path, device)
    base_loss, base_accuracy = evaluate(base_model, test_loader, loss_fn, device)
    
    # 2) pruning + permanent zeros
    pruned_model = build_model(device)
    load_weights(pruned_model, ckpt_path, device)

    if prune_amount > 0.0:
        magnitude_prune_linear_layers(pruned_model, amount=prune_amount)
        make_pruning_permanent(pruned_model)
        
    pruned_loss, pruned_accuracy = evaluate(pruned_model, test_loader, loss_fn, device)
    sparsity = model_sparsity(pruned_model) * 100.0
    
    # 3) Export compressed model
    # OBS: export from the pruned_model, so the compressed file reflects pruning
    compressed = export_fcn_to_compressed(pruned_model)
    
    save_path = f"fcn_mnist_pruned_{int(prune_amount*100)}_int8_compressed.pt"
    save_compressed(compressed, save_path)
    
    # Load the compressed model
    compressed_loaded = load_compressed(save_path)
    
    # 4) Layerwise inference accuracy
    layer_device = torch.device("cpu")
    lw_accuracy = layerwise_evaluate_accuracy(compressed_loaded, test_loader, layer_device)
    
    # 5) Storage + peak memory estimates
    fp32_weight_bytes = estimate_fp32_weight_bytes(pruned_model)
    compressed_bytes = estimate_compressed_weight_bytes(compressed_loaded)
    peak_layer_fp32_bytes = estimate_peak_decompressed_layer_bytes(pruned_model)
    
    
    # Print Summary
    print_report(
        ckpt_path=ckpt_path,
        prune_amount=prune_amount,
        sparsity=sparsity,
        base_acc=base_accuracy, base_loss=base_loss,
        pruned_acc=pruned_accuracy, pruned_loss=pruned_loss,
        lw_acc=lw_accuracy,
        fp32_bytes=fp32_weight_bytes,
        compressed_bytes=compressed_bytes,
        peak_layer_bytes=peak_layer_fp32_bytes,
        save_path=save_path,
    )
    

if __name__ == "__main__":
    main()