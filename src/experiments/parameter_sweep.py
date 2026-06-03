from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.evaluation.metrics import classification_metrics
from src.pipeline import AutomataPipeline

logger = logging.getLogger(__name__)


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
    total_combinations = len(window_grid) * len(alphabet_grid)

    variation_results: list[dict[str, Any]] = []
    combination_idx = 0

    for window_size in window_grid:
        for alphabet_size in alphabet_grid:
            combination_idx += 1
            logger.info(
                "Parameter grid %s (%d/%d): window_size=%s, alphabet_size=%s",
                dataset_name,
                combination_idx,
                total_combinations,
                window_size,
                alphabet_size,
            )

            f1_scores: list[float] = []
            state_counts: list[int] = []
            transition_densities: list[float] = []

            for fold_idx, (train_df, test_df) in enumerate(train_test_splits, start=1):
                pipe = AutomataPipeline(config=config, window_size=window_size, alphabet_size=alphabet_size)
                pipe.fit(train_df, feature_columns)
                test_patterns = pipe.transform_patterns(test_df)
                y_test_aligned = test_df[target_column].to_numpy()[window_size - 1 :]
                y_pred = pipe.automata.predict_prefix_labels(test_patterns)

                metrics = classification_metrics(y_test_aligned, y_pred)
                f1_scores.append(metrics["f1"])
                state_counts.append(pipe.automata.state_count)
                transition_densities.append(pipe.automata.transition_density)

                logger.info(
                    "  fold %d/%d f1=%.4f states=%d",
                    fold_idx,
                    len(train_test_splits),
                    metrics["f1"],
                    pipe.automata.state_count,
                )

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
