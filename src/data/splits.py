from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray | None
    test: np.ndarray


def make_skab_group_folds(
    df: pd.DataFrame,
    target_column: str,
    group_column: str,
    n_splits: int,
    seed: int,
    prefer_stratified: bool = True,
) -> list[SplitIndices]:
    y = df[target_column].to_numpy()
    groups = df[group_column].to_numpy()
    x_placeholder = np.zeros(len(df))

    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds unique SKAB source files={len(unique_groups)}")

    if prefer_stratified and StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        raw_splits = splitter.split(x_placeholder, y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        raw_splits = splitter.split(x_placeholder, y, groups)

    return [SplitIndices(train=train_idx, validation=None, test=test_idx) for train_idx, test_idx in raw_splits]


def make_batadal_chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> SplitIndices:
    total = train_ratio + validation_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"BATADAL split ratios must sum to 1.0, got {total}")

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    validation_end = train_end + int(n_rows * validation_ratio)

    return SplitIndices(
        train=np.arange(0, train_end),
        validation=np.arange(train_end, validation_end),
        test=np.arange(validation_end, n_rows),
    )


def assert_no_group_leakage(df: pd.DataFrame, split: SplitIndices, group_column: str) -> None:
    train_groups = set(df.iloc[split.train][group_column])
    test_groups = set(df.iloc[split.test][group_column])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise AssertionError(f"Group leakage detected for {group_column}: {sorted(overlap)}")
