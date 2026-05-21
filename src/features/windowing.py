from __future__ import annotations

import numpy as np

from src.features.paa import paa
from src.features.sax import sax


def sliding_windows(sequence: np.ndarray, window_size: int) -> list[np.ndarray]:
    values = np.asarray(sequence, dtype=float).reshape(-1)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(values) < window_size:
        return []
    return [values[start : start + window_size] for start in range(0, len(values) - window_size + 1)]


def extract_sax_patterns(
    sequence: np.ndarray,
    window_size: int,
    paa_segments: int | None,
    alphabet_size: int,
) -> list[str]:
    resolved_segments = paa_segments if paa_segments is not None else window_size
    patterns: list[str] = []
    for window in sliding_windows(sequence, window_size):
        patterns.append(sax(paa(window, resolved_segments), alphabet_size))
    return patterns
