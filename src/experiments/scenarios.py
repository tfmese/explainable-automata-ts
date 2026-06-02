from __future__ import annotations

import itertools
import string

import numpy as np


def add_gaussian_noise(values: np.ndarray, mean: float, std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values + rng.normal(loc=mean, scale=std, size=values.shape)


def _choose_unseen_pattern(
    *,
    alphabet_size: int,
    length: int,
    forbidden_patterns: set[str],
    fallback: str = "zzzz",
) -> str:
    if alphabet_size < 2:
        return fallback

    symbols = string.ascii_lowercase[:alphabet_size]
    # length küçük olduğu için (window_size) tam arama yapılabilir.
    for combo in itertools.product(symbols, repeat=length):
        candidate = "".join(combo)
        if candidate not in forbidden_patterns:
            return candidate
    return fallback


def inject_unseen_pattern(
    patterns: list[str],
    replacement: str = "zzzz",
    forbidden_patterns: set[str] | None = None,
    alphabet_size: int | None = None,
) -> list[str]:
    if not patterns:
        return []

    mutated = list(patterns)
    idx = len(mutated) // 2

    if forbidden_patterns is not None and alphabet_size is not None:
        replacement = _choose_unseen_pattern(
            alphabet_size=alphabet_size,
            length=len(mutated[idx]),
            forbidden_patterns=forbidden_patterns,
            fallback=replacement,
        )

    mutated[idx] = replacement
    return mutated
