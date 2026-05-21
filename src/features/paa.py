from __future__ import annotations

import numpy as np


def paa(sequence: np.ndarray, segments: int) -> np.ndarray:
    values = np.asarray(sequence, dtype=float).reshape(-1)
    if segments <= 0:
        raise ValueError("segments must be positive")
    if len(values) < segments:
        raise ValueError("sequence length must be at least the number of PAA segments")

    boundaries = np.linspace(0, len(values), segments + 1)
    reduced = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        left = int(np.floor(start))
        right = int(np.floor(end))
        if right <= left:
            right = left + 1
        reduced.append(values[left:right].mean())
    return np.asarray(reduced)
