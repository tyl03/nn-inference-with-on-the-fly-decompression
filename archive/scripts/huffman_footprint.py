"""
Huffman Footprint Experiment (MNIST)

Purpose:
- Measure storage footprint of Huffman-compressed model weights.
- Also perform a sanity check that Huffman compression is lossless by
  reconstructing a model from compressed weights and re-evaluating accuracy.

Important idea:
- Huffman coding works on symbols and their frequencies.
- Neural network weights are FP32 tensors, so we convert each FP32 value into a
  stable, hashable symbol by reinterpreting its raw 32-bit bit-pattern (uint32).
  This is NOT rounding or changing values; it is a bit-level view of the same data.
"""

import pickle
import torch
import torch.nn as nn

from src.exp_utils import (
    get_device,
    load_test_loader,
    build_model,
    load_weights,
    estimate_fp32_weight_bytes,
    fmt_bytes,
)
from src.training import evaluate
from src.pruning import magnitude_prune_linear_layers, make_pruning_permanent, model_sparsity
from src.huffman import huffman_compress_tensor, huffman_decompress_tensor


def estimate_huffman_storage_bytes(model: nn.Module) -> int:
    """
    Estimate how many bytes it would take to store the model weights using Huffman coding.

    What we count per tensor:
    - encoded bitstream bytes: len(encoded)
    - codebook bytes (freqs): size of serialized frequency table
    - shape bytes: size of serialized shape metadata

    Why serialize with pickle?
    - freqs and shape are Python objects. Serializing them gives a realistic
      "if we saved this to disk" size estimate.
    """
    total_bytes = 0
    # sd is the state_dict of the model, which is a dictionary mapping parameter names to their corresponding tensors.
    sd = model.state_dict()
    
    for name, t in sd.items():
        # Only compress floating tensors (weights/biases)
        if not t.dtype.is_floating_point:
            continue
        
        compressed = huffman_compress_tensor(t)
        
        encoded_bytes = len(compressed["encoded"])
        freqs_bytes = len(pickle.dumps(compressed["freqs"], protocol=pickle.HIGHEST_PROTOCOL))
        shape_bytes = len(pickle.dumps(compressed["shape"], protocol=pickle.HIGHEST_PROTOCOL))
        
        total_bytes += encoded_bytes + freqs_bytes + shape_bytes
        
    return total_bytes


def reconstruct_model_from_huffman(pruned_model: nn.Module, device: torch.device) -> nn.Module:
    """
    Build a NEW model with the same architecture and fill it using Huffman-decoded tensors.

    Why do this?
    - To verify the full "round-trip": tensor -> Huffman -> tensor
    - And to verify that we can rebuild a working PyTorch model from the compressed form.

    Important:
    - build_model(device) creates the architecture but with freshly initialized weights.
      It does NOT contain trained weights until we load a state_dict into it.
    """
    rebuilt = build_model(device)
    rebuilt_sd = rebuilt.state_dict()
    
    src_sd = pruned_model.state_dict()
    for name, t in src_sd.items():
        if t.dtype.is_floating_point:
            compressed = huffman_compress_tensor(t)
            decompressed = huffman_decompress_tensor(compressed)
            rebuilt_sd[name] = decompressed.to(device)
        else:
            rebuilt_sd[name] = t.to(device)
            
    rebuilt.load_state_dict(rebuilt_sd, strict=True)
    return rebuilt


def print_report(
    ckpt_path: str,
    prune_amount: float,
    sparsity_pct: float,
    base_acc: float, base_loss: float,
    pruned_acc: float, pruned_loss: float,
    huff_acc: float, huff_loss: float,
    fp32_bytes: int,
    huff_bytes: int,
) -> None:
    ratio = (fp32_bytes / huff_bytes) if huff_bytes > 0 else float("inf")

    print("\n" + "=" * 70)
    print("Huffman Footprint Experiment (MNIST)".center(70))
    print("=" * 70)

    print("\n[Setup]")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Prune amount   : {prune_amount:.2f}")
    print(f"  Sparsity       : {sparsity_pct:.2f}%")

    print("\n[Accuracy]")
    print(f"  Baseline FP32          : {base_acc:.4f} (loss {base_loss:.4f})")
    print(f"  Pruned FP32            : {pruned_acc:.4f} (loss {pruned_loss:.4f})")
    print(f"  Huffman->FP32 (decoded) : {huff_acc:.4f} (loss {huff_loss:.4f})")
    print(f"  Drop (baseline - huff) : {(base_acc - huff_acc):.6f}")

    print("\n[Storage (weights/bias only)]")
    print(f"  FP32 bytes (pruned, dense) : {fmt_bytes(fp32_bytes)}")
    print(f"  Huffman bytes (+overhead)  : {fmt_bytes(huff_bytes)}")
    print(f"  Compression ratio          : {ratio:.2f}x")

    print("=" * 70 + "\n")
    

def main():
    """
    Main flow:
    1) Load test set + checkpoint
    2) Evaluate baseline accuracy
    3) Prune model and evaluate pruned accuracy
    4) Compute storage:
       - FP32 dense weight size (zeros still take space)
       - Huffman compressed size (encoded + codebook + metadata)
    5) Sanity-check losslessness:
       - Rebuild a model by Huffman compress+decompress of each tensor
       - Re-evaluate accuracy (should match pruned accuracy)
    """
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader()
    
    ckpt_path = "fcn_mnist_best.pt"
    prune_amount = 0.5
    
    # 1) Baseline FP32 accuracy
    base_model = build_model(device)
    load_weights(base_model, ckpt_path, device)
    base_loss, base_acc = evaluate(base_model, test_loader, loss_fn, device)
    
    # 2) Prune model
    pruned_model = build_model(device)
    load_weights(pruned_model, ckpt_path, device)
    
    if prune_amount > 0.0:
        magnitude_prune_linear_layers(pruned_model, amount=prune_amount)
        make_pruning_permanent(pruned_model)
        
    pruned_loss, pruned_acc = evaluate(pruned_model, test_loader, loss_fn, device)
    sparsity_pct = model_sparsity(pruned_model) * 100
    
    # 3) Huffman footprint estimate
    fp32_bytes = estimate_fp32_weight_bytes(pruned_model)
    huff_bytes = estimate_huffman_storage_bytes(pruned_model)
    
    # 4) Rebuild model from Huffman-compressed weights and evaluate to confirm losslessness
    huff_model = reconstruct_model_from_huffman(pruned_model, device)
    huff_loss, huff_acc = evaluate(huff_model, test_loader, loss_fn, device)
    
    # Summary
    print_report(
        ckpt_path=ckpt_path,
        prune_amount=prune_amount,
        sparsity_pct=sparsity_pct,
        base_acc=base_acc, base_loss=base_loss,
        pruned_acc=pruned_acc, pruned_loss=pruned_loss,
        huff_acc=huff_acc, huff_loss=huff_loss,
        fp32_bytes=fp32_bytes,
        huff_bytes=huff_bytes,
    )
    
    
if __name__ == "__main__":
    main()