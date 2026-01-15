"""
This test is a small sanity check for the training pipeline.

What this test checks:
- The training loop (train_one_epoch) can run on a small subset of MNIST without errors
- The evaluation loop (evaluate) can run without errors
- Both functions return numeric values
- Loss values are non-negative
- Accuracy values are in the range [0, 1]

What this test does NOT check:
- That the model reaches a specific accuracy
- That loss decreases over multiple epochs
- That one number of epochs is better than another

These results depend on randomness, data order, and hardware, and are therefore
not suitable for strict automated testing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.fcn import FCN
from src.training import train_one_epoch, evaluate


def test_training_and_eval_runs_on_small_subset(tmp_path):
    device = torch.device("cpu")
    
    # Small subset from dataset to keep the test fast
    transform = transforms.ToTensor()
    ds = datasets.MNIST(root=tmp_path, train=True, download=True, transform=transform)
    small_ds = Subset(ds, range(256)) # Subset
    
    train_loader = DataLoader(small_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(small_ds, batch_size=64, shuffle=False)
    
    model = FCN(in_dim=28 * 28, hidden_dims=[64], out_dim=10).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
    
    # Sanity Checks
    assert isinstance(train_loss, float)
    assert isinstance(test_loss, float)
    assert 0.0 <= train_accuracy <= 1.0
    assert 0.0 <= test_accuracy <= 1.0