from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class LeakageSafePreprocessor:
    use_standard_scaler: bool = True
    pca_components: int | None = None

    def __post_init__(self) -> None:
        self.scaler_: StandardScaler | None = StandardScaler() if self.use_standard_scaler else None
        self.pca_: PCA | None = PCA(n_components=self.pca_components) if self.pca_components else None
        self.feature_columns_: list[str] | None = None

    def fit(self, train_df: pd.DataFrame, feature_columns: list[str]) -> "LeakageSafePreprocessor":
        self.feature_columns_ = list(feature_columns)
        values = train_df[self.feature_columns_].to_numpy(dtype=float)
        if self.scaler_ is not None:
            values = self.scaler_.fit_transform(values)
        if self.pca_ is not None:
            self.pca_.fit(values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_columns_ is None:
            raise RuntimeError("Preprocessor must be fit before transform.")

        values = df[self.feature_columns_].to_numpy(dtype=float)
        if self.scaler_ is not None:
            values = self.scaler_.transform(values)
        if self.pca_ is not None:
            values = self.pca_.transform(values)
        return values

    def fit_transform(self, train_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        self.fit(train_df, feature_columns)
        return self.transform(train_df)
