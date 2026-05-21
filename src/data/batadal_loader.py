from __future__ import annotations

from pathlib import Path

import pandas as pd


COMMON_LABEL_NAMES = (
    "ATT_FLAG",
    "Attack",
    "attack",
    "Label",
    "label",
    "anomaly",
    "target",
)


def load_batadal_training_dataset_2(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"BATADAL Training Dataset 2 not found: {csv_path}")
    return pd.read_csv(csv_path, sep=None, engine="python")


def infer_batadal_target_column(df: pd.DataFrame, configured_target: str | None = None) -> str:
    if configured_target:
        if configured_target not in df.columns:
            raise ValueError(f"Configured BATADAL target column is missing: {configured_target}")
        return configured_target

    matches = [column for column in COMMON_LABEL_NAMES if column in df.columns]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple possible BATADAL target columns found: {matches}")

    raise ValueError(
        "BATADAL target column could not be inferred. Set datasets.batadal.target_column "
        "after checking the Training Dataset 2 file."
    )


def batadal_feature_columns(df: pd.DataFrame, target_column: str, time_columns: list[str]) -> list[str]:
    excluded = set(time_columns)
    excluded.add(target_column)
    return [column for column in df.columns if column not in excluded]
