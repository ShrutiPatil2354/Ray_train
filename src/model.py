import csv
import os

import torch
from torch import nn


class SimpleRegressionNet(nn.Module):
    """Small synthetic regression model for Ray Train demonstration."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def generate_synthetic_dataset(num_samples: int = 2000, num_features: int = 10, seed: int = 42):
    """Create a deterministic synthetic regression dataset."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, num_features, generator=generator)
    y = 2.0 * x[:, 0] + 3.0 * x[:, 1] + 1.0 + 0.1 * torch.randn(num_samples, generator=generator)
    return x, y


def save_synthetic_dataset_csv(path: str, num_samples: int = 2000, num_features: int = 10, seed: int = 42):
    """Persist a real dataset to CSV so it can be versioned and tracked by DVC."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x, y = generate_synthetic_dataset(num_samples=num_samples, num_features=num_features, seed=seed)

    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = [f"feature_{index}" for index in range(num_features)] + ["target"]
        writer.writerow(header)

        for row_index in range(x.shape[0]):
            row = [float(value) for value in x[row_index].tolist()] + [float(y[row_index].item())]
            writer.writerow(row)

    return x, y
