import torch
from torch import nn


class IrisClassifier(nn.Module):
    """Compact multilayer classifier for the four-feature Iris dataset."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
