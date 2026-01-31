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

from src.exp_utils import (
    get_device,
    load_test_loader,
    build_model,
    load_weights,
    quantize_dequantize_linear_weights_inplace,
)
from src.training import evaluate
from src.pruning import magnitude_prune_linear_layers, make_pruning_permanent, model_sparsity


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