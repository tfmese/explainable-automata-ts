from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_skab(root: str | Path, enabled_groups: list[str] | tuple[str, ...]) -> pd.DataFrame:
    root_path = Path(root)
    frames: list[pd.DataFrame] = []

    for group in enabled_groups:
        group_dir = root_path / group
        if not group_dir.exists():
            raise FileNotFoundError(f"SKAB group directory not found: {group_dir}")

        for csv_path in sorted(group_dir.glob("*.csv")):
            frame = pd.read_csv(csv_path, sep=None, engine="python")
            frame["source_group"] = group
            frame["source_file"] = csv_path.name
            frames.append(frame)

    if not frames:
        raise ValueError(f"No SKAB CSV files found under {root_path} for groups {enabled_groups}")

    return pd.concat(frames, ignore_index=True)


def skab_feature_columns(df: pd.DataFrame, target_column: str, excluded_columns: list[str]) -> list[str]:
    excluded = set(excluded_columns)
    excluded.add(target_column)
    return [column for column in df.columns if column not in excluded]
