from __future__ import annotations

import string

import numpy as np
from scipy.stats import norm


def sax(sequence: np.ndarray, alphabet_size: int) -> str:
    values = np.asarray(sequence, dtype=float).reshape(-1)
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least 2")
    if alphabet_size > len(string.ascii_lowercase):
        raise ValueError("alphabet_size exceeds supported lowercase alphabet")

    std = values.std()
    normalized = values - values.mean()
    if std > 0:
        normalized = normalized / std

    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    symbols = string.ascii_lowercase[:alphabet_size]
    indexes = np.digitize(normalized, breakpoints)
    return "".join(symbols[index] for index in indexes)
