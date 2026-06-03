from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.pipeline import AutomataPipeline


def run_parameter_grid(
    config: ProjectConfig,
    *,
    dataset_name: str,
    train_test_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_columns: list[str],
    target_column: str,
) -> list[dict[str, Any]]:
    window_grid = config.get("automata", "parameter_grid", "window_size", default=[3, 4, 5, 6])
    alphabet_grid = config.get("automata", "parameter_grid", "alphabet_size", default=[3, 4, 5, 6])

    variation_results: list[dict[str, Any]] = []
    for window_size in window_grid:
        for alphabet_size in alphabet_grid:
            f1_scores: list[float] = []
            state_counts: list[int] = []
            transition_densities: list[float] = []

            for train_df, test_df in train_test_splits:
                pipe = AutomataPipeline(config=config, window_size=window_size, alphabet_size=alphabet_size)
                pipe.fit(train_df, feature_columns)
                test_patterns = pipe.transform_patterns(test_df)
                y_test_aligned = test_df[target_column].to_numpy()[window_size - 1 :]

                y_pred: list[int] = []
                for idx in range(1, len(test_patterns) + 1):
                    decision_dict = pipe.automata.predict_sequence(test_patterns[:idx])
                    y_pred.append(1 if decision_dict["decision"] == "anomaly" else 0)

                from src.evaluation.metrics import classification_metrics

                metrics = classification_metrics(y_test_aligned, np.asarray(y_pred))
                f1_scores.append(metrics["f1"])
                state_counts.append(pipe.automata.state_count)
                transition_densities.append(pipe.automata.transition_density)

            variation_results.append(
                {
                    "dataset": dataset_name,
                    "window_size": window_size,
                    "alphabet_size": alphabet_size,
                    "f1_mean": float(np.mean(f1_scores)),
                    "f1_std": float(np.std(f1_scores)),
                    "states_mean": float(np.mean(state_counts)),
                    "density_mean": float(np.mean(transition_densities)),
                }
            )

    return variation_results
