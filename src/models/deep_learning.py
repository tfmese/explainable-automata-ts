from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.config import ProjectConfig


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        return self.classifier(hidden[-1]).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        return self.classifier(hidden[-1]).squeeze(-1)


class CNN1DClassifier(nn.Module):
    def __init__(self, input_size: int, channels: int = 64, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.transpose(1, 2)).squeeze(-1)


def resolve_model_architecture(config: ProjectConfig, model_name: str) -> dict[str, Any]:
    normalized = model_name.lower()
    if normalized in {"cnn1d", "1d-cnn", "cnn"}:
        normalized = "cnn1d"
    return dict(config.get("deep_learning", normalized, default={}) or {})


def build_deep_model(name: str, input_size: int, **kwargs) -> nn.Module:
    normalized = name.lower()
    if normalized == "lstm":
        return LSTMClassifier(input_size=input_size, **kwargs)
    if normalized == "gru":
        return GRUClassifier(input_size=input_size, **kwargs)
    if normalized in {"cnn1d", "1d-cnn", "cnn"}:
        return CNN1DClassifier(input_size=input_size, **kwargs)
    raise ValueError(f"Unsupported deep learning model: {name}")
