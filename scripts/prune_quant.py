"""
Prune + Quantization experiment:

- Load a trained FCN (MNIST)
- Apply magnitude pruning with different prune amounts
- Make pruning permanent
- Quantize Linear weights to int8 (symmetric, [-127,127]) with per-layer scale
- Simulate on-the-fly dequant inference by replacing weights with quant->dequant copies (FP32 compute)
- Measure sparsity, accuracy drop, and storage size (weights only)

Outputs a table per prune amount.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.fcn import FCN
from src.training import get_device, evaluate
from src.pruning import magnitude_prune_linear_layers, make_pruning_permanent, model_sparsity
from src.quantization import compute_scale, symmetric_quantization, symmetric_dequantization


def load_test_loader():
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=256, shuffle=False)


def build_model(device: torch.device):
    return FCN(in_dim=28 * 28, hidden_dims=[512, 256], out_dim=10).to(device)


def load_weights(model: nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model
    

@torch.no_grad()
def quantize_dequantize_linear_weights_inplace(model: nn.Module):
    """
    Replaces each nn.Linear weight with a dequantized copy of its int8 quantized version.
    Returns a list of (layer_name, weight_scale) so we can estimate storage.
    """
    scales = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            W = m.weight.data
            s = compute_scale(W)
            W_q = symmetric_quantization(W, s) # int8
            W_fp = symmetric_dequantization(W_q, s) # float32
            m.weight.data.copy_(W_fp)
            scales.append((name, s))
    
    return scales


def estimate_fp32_weight_bytes(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.numel() * 4 # float32 = 4 bytes
            
    return total


def estimate_int8_weight_bytes_plus_scales(model: nn.Module, num_scales: int) -> int:
    """
    int8 weights: 1 byte each
    scales: store as float32 per layer (4 bytes each)
    To enable dequantization we need both the weight and the scale.
    """
    int8_bytes = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            int8_bytes += m.weight.numel() * 1
            
    scale_bytes = num_scales * 4
    return int8_bytes, scale_bytes


# Converts bytes into kilobytes
def fmt_kb(b: int) -> float:
    return b / 1024.0


def main():
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    test_loader = load_test_loader()
    
    ckpt_path = "fcn_mnist_best.pt"
    prune_amounts = [0.0, 0.4, 0.5, 0.6, 0.65, 0.7]
    
    # Baseline FP32
    base_model = build_model(device)
    load_weights(base_model, ckpt_path, device)
    base_loss, base_accuracy = evaluate(base_model, test_loader, loss_fn, device)
    
    print(f"\nFP32 baseline accuracy: {base_accuracy:.4f} (loss {base_loss:.4f})\n")
    
    # The drop is "how much accuracy there is lost compared to the original FP32 model"
    header = (
        f"{'prune':>6} | {'sparsity%':>9} | {'acc_pruned':>10} | {'acc_prune+quant':>15} | {'drop':>11}"
    )
    print(header)
    print("-" * len(header))
    
    
    for amount in prune_amounts:
        # Prune only
        prune_model = build_model(device)
        load_weights(prune_model, ckpt_path, device)
        
        if amount > 0.0:
            magnitude_prune_linear_layers(prune_model, amount=amount)
            make_pruning_permanent(prune_model)
            
        sparsity_pct = model_sparsity(prune_model) * 100.0
        prune_loss, prune_accuracy = evaluate(prune_model, test_loader, loss_fn, device)
        
        
        # Prune + quant
        pq_model = build_model(device)
        load_weights(pq_model, ckpt_path, device)
        
        if amount > 0.0:
            magnitude_prune_linear_layers(pq_model, amount=amount)
            make_pruning_permanent(pq_model)
            
        quantize_dequantize_linear_weights_inplace(pq_model)
        pq_loss, pq_accuracy = evaluate(pq_model, test_loader, loss_fn, device)
        
        print(f"{amount:6.2f} | {sparsity_pct:9.2f} | {prune_accuracy:10.4f} | {pq_accuracy:15.4f} | {base_accuracy - pq_accuracy:11.4f}")
    
    
if __name__ == "__main__":
    main()