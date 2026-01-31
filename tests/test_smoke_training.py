"""
Smoke test for the training and evaluation pipeline.

What this test checks:
- train_one_epoch runs without errors
- evaluate runs without errors
- Returned loss values are floats and non-negative
- Returned accuracy values are in the range [0, 1]

What this test does NOT check:
- Model convergence
- Absolute accuracy values
- Loss decrease over epochs

This test uses FakeData instead of MNIST to:
- Avoid network downloads in CI
- Run fast and deterministically
- Work reliably on Windows, WSL, and Linux
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

from src.fcn import FCN
from src.training import train_one_epoch, evaluate


def test_training_and_eval_runs_on_small_subset():
    # Force CPU for consistency in CI
    device = torch.device("cpu")

    # Make results deterministic
    torch.manual_seed(0)

    # Fake MNIST-like dataset
    # - 256 samples
    # - 1 channel (grayscale)
    # - 28x28 images
    # - 10 classes (0..9)
    dataset = FakeData(
        size=256,
        image_size=(1, 28, 28),
        num_classes=10,
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = FCN(
        in_dim=28 * 28,
        hidden_dims=[64],
        out_dim=10,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run one training epoch
    train_loss, train_accuracy = train_one_epoch(
        model, train_loader, optimizer, loss_fn, device
    )

    # Run evaluation
    test_loss, test_accuracy = evaluate(
        model, test_loader, loss_fn, device
    )

    # Sanity checks (NOT performance checks)
    assert isinstance(train_loss, float)
    assert isinstance(test_loss, float)

    assert train_loss >= 0.0
    assert test_loss >= 0.0

    assert 0.0 <= train_accuracy <= 1.0
    assert 0.0 <= test_accuracy <= 1.0