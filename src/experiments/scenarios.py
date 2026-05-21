from __future__ import annotations

import numpy as np


def add_gaussian_noise(values: np.ndarray, mean: float, std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values + rng.normal(loc=mean, scale=std, size=values.shape)


def inject_unseen_pattern(patterns: list[str], replacement: str = "zzzz") -> list[str]:
    if not patterns:
        return []
    mutated = list(patterns)
    mutated[len(mutated) // 2] = replacement
    return mutated
