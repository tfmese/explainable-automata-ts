from __future__ import annotations

import string

import numpy as np
from scipy.stats import norm

from src.features.paa import paa


def _validate_alphabet_size(alphabet_size: int) -> None:
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least 2")
    if alphabet_size > len(string.ascii_lowercase):
        raise ValueError("alphabet_size exceeds supported lowercase alphabet")


def _gaussian_breakpoints(alphabet_size: int) -> np.ndarray:
    return norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])


def sax(sequence: np.ndarray, alphabet_size: int) -> str:
    """Stateless SAX (yalnızca testler ve geriye dönük uyumluluk için)."""
    values = np.asarray(sequence, dtype=float).reshape(-1)
    _validate_alphabet_size(alphabet_size)

    std = values.std()
    normalized = values - values.mean()
    if std > 0:
        normalized = normalized / std

    breakpoints = _gaussian_breakpoints(alphabet_size)
    symbols = string.ascii_lowercase[:alphabet_size]
    indexes = np.digitize(normalized, breakpoints)
    return "".join(symbols[index] for index in indexes)


class TrainableSAXEncoder:
    """Eğitim verisinden öğrenilen SAX sözlüğü; testte yalnızca bu istatistikler kullanılır."""

    def __init__(self) -> None:
        self.mean_: float = 0.0
        self.std_: float = 1.0
        self.breakpoints_: np.ndarray | None = None
        self.alphabet_size_: int = 3
        self.symbols_: str = "abc"
        self.window_size_: int = 4
        self.paa_segments_: int = 4
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        sequence: np.ndarray,
        *,
        window_size: int,
        paa_segments: int,
        alphabet_size: int,
    ) -> "TrainableSAXEncoder":
        _validate_alphabet_size(alphabet_size)
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if paa_segments <= 0:
            raise ValueError("paa_segments must be positive")

        from src.features.windowing import sliding_windows

        values = np.asarray(sequence, dtype=float).reshape(-1)
        paa_values: list[float] = []
        for window in sliding_windows(values, window_size):
            paa_values.extend(paa(window, paa_segments).tolist())

        if not paa_values:
            raise ValueError("Training sequence is too short to fit SAX encoder")

        train_paa = np.asarray(paa_values, dtype=float)
        self.mean_ = float(train_paa.mean())
        self.std_ = float(train_paa.std())
        if self.std_ <= 0 or np.isnan(self.std_):
            self.std_ = 1.0

        self.alphabet_size_ = alphabet_size
        self.symbols_ = string.ascii_lowercase[:alphabet_size]
        self.window_size_ = window_size
        self.paa_segments_ = paa_segments
        self.breakpoints_ = _gaussian_breakpoints(alphabet_size)
        self._is_fitted = True
        return self

    def encode_paa(self, paa_values: np.ndarray) -> str:
        if not self._is_fitted or self.breakpoints_ is None:
            raise RuntimeError("TrainableSAXEncoder must be fit before encoding")

        normalized = (np.asarray(paa_values, dtype=float).reshape(-1) - self.mean_) / self.std_
        indexes = np.digitize(normalized, self.breakpoints_)
        return "".join(self.symbols_[index] for index in indexes)

    def encode_window(self, window: np.ndarray) -> str:
        return self.encode_paa(paa(window, self.paa_segments_))

    def transform(self, sequence: np.ndarray) -> list[str]:
        if not self._is_fitted:
            raise RuntimeError("TrainableSAXEncoder must be fit before transform")

        from src.features.windowing import sliding_windows

        values = np.asarray(sequence, dtype=float).reshape(-1)
        return [self.encode_window(window) for window in sliding_windows(values, self.window_size_)]
