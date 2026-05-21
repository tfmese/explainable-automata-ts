from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import ProjectConfig
from src.data.preprocessing import LeakageSafePreprocessor
from src.explainability.automata_explainer import explain_automata_decision
from src.features.windowing import extract_sax_patterns
from src.models.automata import ProbabilisticAutomata


@dataclass
class AutomataPipeline:
    config: ProjectConfig
    window_size: int
    alphabet_size: int

    def __post_init__(self) -> None:
        self.preprocessor = LeakageSafePreprocessor(
            use_standard_scaler=self.config.get("preprocessing", "normalization") == "standard",
            pca_components=self.config.get("preprocessing", "automata_pca_components"),
        )
        self.automata = ProbabilisticAutomata(
            smoothing=self.config.get("automata", "smoothing"),
            anomaly_quantile=self.config.get("automata", "anomaly_quantile"),
        )
        self.paa_segments = self.config.get("automata", "paa_segments")

    def fit(self, train_df: pd.DataFrame, feature_columns: list[str]) -> "AutomataPipeline":
        train_values = self.preprocessor.fit_transform(train_df, feature_columns).reshape(-1)
        patterns = extract_sax_patterns(
            train_values,
            window_size=self.window_size,
            paa_segments=self.paa_segments,
            alphabet_size=self.alphabet_size,
        )
        self.automata.fit(patterns)
        return self

    def transform_patterns(self, df: pd.DataFrame) -> list[str]:
        values = self.preprocessor.transform(df).reshape(-1)
        return extract_sax_patterns(
            values,
            window_size=self.window_size,
            paa_segments=self.paa_segments,
            alphabet_size=self.alphabet_size,
        )

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        return self.automata.predict_sequence(self.transform_patterns(df))

    def explain(self, df: pd.DataFrame, time_step: int | None = None) -> dict[str, Any]:
        return explain_automata_decision(self.automata, self.transform_patterns(df), time_step=time_step)


def build_fixed_automata_pipeline(config: ProjectConfig) -> AutomataPipeline:
    fixed = config.get("automata", "fixed_comparison")
    return AutomataPipeline(
        config=config,
        window_size=fixed["window_size"],
        alphabet_size=fixed["alphabet_size"],
    )
