from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SUPPORTED_MISSING_STRATEGIES = ("none", "median", "mean", "forward_fill")


@dataclass
class LeakageSafePreprocessor:
    use_standard_scaler: bool = True
    pca_components: int | None = None
    missing_strategy: str = "none"

    def __post_init__(self) -> None:
        if self.missing_strategy not in SUPPORTED_MISSING_STRATEGIES:
            raise ValueError(
                f"Unsupported missing_strategy={self.missing_strategy!r}. "
                f"Choose from {SUPPORTED_MISSING_STRATEGIES}"
            )
        self.scaler_: StandardScaler | None = StandardScaler() if self.use_standard_scaler else None
        self.pca_: PCA | None = PCA(n_components=self.pca_components) if self.pca_components else None
        self.feature_columns_: list[str] | None = None
        self._fill_values_: dict[str, float] = {}

    def _impute_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns_ is None:
            raise RuntimeError("Preprocessor must be fit before imputation.")
        if self.missing_strategy == "none":
            return df

        frame = df.copy()
        columns = self.feature_columns_
        if self.missing_strategy == "forward_fill":
            frame[columns] = frame[columns].ffill()
            for column in columns:
                if frame[column].isna().any():
                    frame[column] = frame[column].fillna(self._fill_values_[column])
            return frame

        for column in columns:
            frame[column] = frame[column].fillna(self._fill_values_[column])
        return frame

    def _learn_fill_values(self, train_df: pd.DataFrame) -> None:
        columns = self.feature_columns_
        if columns is None:
            raise RuntimeError("feature_columns must be set before learning fill values.")

        self._fill_values_ = {}
        if self.missing_strategy == "none":
            return

        for column in columns:
            series = train_df[column]
            if self.missing_strategy == "median":
                value = float(series.median())
            elif self.missing_strategy == "mean":
                value = float(series.mean())
            elif self.missing_strategy == "forward_fill":
                non_null = series.dropna()
                value = float(non_null.iloc[0]) if len(non_null) else 0.0
            else:
                value = 0.0
            if np.isnan(value):
                value = 0.0
            self._fill_values_[column] = value

    def fit(self, train_df: pd.DataFrame, feature_columns: list[str]) -> "LeakageSafePreprocessor":
        self.feature_columns_ = list(feature_columns)
        self._learn_fill_values(train_df)
        imputed = self._impute_frame(train_df)
        values = imputed[self.feature_columns_].to_numpy(dtype=float)
        if self.scaler_ is not None:
            values = self.scaler_.fit_transform(values)
        if self.pca_ is not None:
            self.pca_.fit(values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_columns_ is None:
            raise RuntimeError("Preprocessor must be fit before transform.")

        imputed = self._impute_frame(df)
        values = imputed[self.feature_columns_].to_numpy(dtype=float)
        if self.scaler_ is not None:
            values = self.scaler_.transform(values)
        if self.pca_ is not None:
            values = self.pca_.transform(values)
        return values

    def fit_transform(self, train_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        self.fit(train_df, feature_columns)
        return self.transform(train_df)


def build_leakage_safe_preprocessor(
    config: Any,
    *,
    for_automata: bool = False,
) -> LeakageSafePreprocessor:
    """Config'ten ön işlemci oluşturur.

    for_automata=False (varsayılan): DL için çok değişkenli girdi korunur, PCA uygulanmaz.
    for_automata=True: PDF gereği otomata hattında PC1 kullanılır.
    """
    from src.config import ProjectConfig

    if not isinstance(config, ProjectConfig):
        raise TypeError("config must be a ProjectConfig instance")

    missing_cfg = config.get("preprocessing", "missing_values", default={}) or {}
    resolved_pca = (
        config.get("preprocessing", "automata_pca_components") if for_automata else None
    )
    return LeakageSafePreprocessor(
        use_standard_scaler=config.get("preprocessing", "normalization") == "standard",
        pca_components=resolved_pca,
        missing_strategy=missing_cfg.get("strategy", "none"),
    )
