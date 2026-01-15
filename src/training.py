"""
Helper functions for training and evaluating a model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# One training epoch
def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Trains for 1 epoch and returns:
    - average loss (float)
    - accuracy (float in [0, 1])
    
    x = input data
    y = target / label
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in loader:
        x = x.to(device) # moves data to where computations happens
        y = y.to(device)
        
        # 1) Forward pass (logits)
        logits = model(x)
        
        # 2) Loss (compares logits vs true class index)
        loss = loss_fn(logits, y)
        
        # 3) Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 4) Update weights
        optimizer.step()
        
        # Track stats
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# Evaluation (no gradients)
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluates model and returns:
    - average loss
    - accuracy
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
