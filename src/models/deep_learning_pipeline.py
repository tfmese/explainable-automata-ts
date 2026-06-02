from __future__ import annotations

import copy
import logging
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import ProjectConfig
from src.evaluation.metrics import classification_metrics
from src.models.deep_learning import build_deep_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Zaman serisi verilerini kayan pencere (sliding window) seklinde hazırlayan dataset sınıfımız
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        if len(self.X) < self.window_size:
            return 0
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_window = self.X[idx : idx + self.window_size]
        y_target = self.y[idx + self.window_size - 1]
        return x_window, y_target


# Aşırı öğrenmeyi engellemek amacıyla validation loss'u izleyen erken durdurma sınıfımız
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state: dict[str, Any] | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.praise_patience():
                self.early_stop = True

    def praise_patience(self) -> int:
        return self.patience


class DeepLearningPipeline:
    def __init__(
        self,
        config: ProjectConfig,
        model_name: str,
        input_size: int,
        window_size: int = 4,
    ):
        self.config = config
        self.model_name = model_name
        self.input_size = input_size
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_deep_model(self.model_name, input_size=self.input_size).to(self.device)

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x_batch)
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader, criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * len(x_batch)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
        return avg_loss, np.asarray(all_targets), np.asarray(all_preds)

    def fit(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
    ) -> DeepLearningPipeline:
        max_epochs = self.config.get("deep_learning", "max_epochs", default=50)
        batch_size = self.config.get("deep_learning", "batch_size", default=32)
        patience = self.config.get("deep_learning", "early_stopping", "patience", default=5)

        train_dataset = TimeSeriesDataset(train_X, train_y, self.window_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_X is not None and val_y is not None:
            val_dataset = TimeSeriesDataset(val_X, val_y, self.window_size)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)

            if val_loader is not None:
                val_loss, _, _ = self.evaluate(val_loader, criterion)
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    if early_stopping.best_state is not None:
                        self.model.load_state_dict(early_stopping.best_state)
                    break
            else:
                # Validation seti yoksa (örneğin SKAB GroupKFold fold'larında), train loss üzerinden takip ediyoruz
                early_stopping(train_loss, self.model)

        # Eğer eğitim bittiyse ve elimizde kaydedilmiş en iyi ağırlıklar varsa onları yüklüyoruz
        if val_loader is None and early_stopping.best_state is not None:
            self.model.load_state_dict(early_stopping.best_state)

        return self

    def predict_proba(self, test_X: np.ndarray) -> np.ndarray:
        test_y_placeholder = np.zeros(len(test_X))
        test_dataset = TimeSeriesDataset(test_X, test_y_placeholder, self.window_size)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for x_batch, _ in test_loader:
                x_batch = x_batch.to(self.device)
                logits = self.model(x_batch)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())

        return np.asarray(all_probs, dtype=float)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(test_X) >= 0.5).astype(int)

    def predict_and_evaluate(self, test_X: np.ndarray, test_y: np.ndarray) -> dict[str, float]:
        test_dataset = TimeSeriesDataset(test_X, test_y, self.window_size)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        criterion = nn.BCEWithLogitsLoss()

        _, y_true, y_pred = self.evaluate(test_loader, criterion)
        return classification_metrics(y_true, y_pred)
