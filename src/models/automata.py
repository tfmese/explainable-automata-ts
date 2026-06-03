from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from Levenshtein import distance as levenshtein_distance


@dataclass(frozen=True)
class PatternMapping:
    original: str
    mapped: str
    status: str
    distance: int


class ProbabilisticAutomata:
    def __init__(self, smoothing: float = 1e-12, anomaly_quantile: float = 0.05) -> None:
        self.smoothing = smoothing
        self.anomaly_quantile = anomaly_quantile
        self.states_: set[str] = set()
        self.transition_counts_: dict[str, Counter[str]] = defaultdict(Counter)
        self.transition_probabilities_: dict[str, dict[str, float]] = {}
        self.threshold_: float | None = None

    def fit(self, patterns: list[str]) -> "ProbabilisticAutomata":
        if len(patterns) < 2:
            raise ValueError("At least two patterns are required to fit automata transitions")

        self.states_ = set(patterns)
        self.transition_counts_ = defaultdict(Counter)
        for source, target in zip(patterns[:-1], patterns[1:]):
            self.transition_counts_[source][target] += 1

        self.transition_probabilities_ = {}
        for source, counts in self.transition_counts_.items():
            total = sum(counts.values())
            self.transition_probabilities_[source] = {
                target: count / total for target, count in counts.items()
            }

        train_scores = []
        path_probability = 1.0
        for idx in range(1, len(patterns)):
            prob = self.transition_probability(patterns[idx - 1], patterns[idx])
            path_probability *= prob
            train_scores.append(path_probability)
        self.threshold_ = float(np.quantile(train_scores, self.anomaly_quantile))
        return self

    def map_pattern(self, pattern: str) -> PatternMapping:
        if pattern in self.states_:
            return PatternMapping(original=pattern, mapped=pattern, status="seen", distance=0)
        if not self.states_:
            raise RuntimeError("Automata must be fit before mapping patterns")

        nearest = min(self.states_, key=lambda state: (levenshtein_distance(pattern, state), state))
        return PatternMapping(
            original=pattern,
            mapped=nearest,
            status="unseen",
            distance=levenshtein_distance(pattern, nearest),
        )

    def transition_probability(self, source: str, target: str) -> float:
        value: Any = self.transition_probabilities_.get(source, {}).get(target, self.smoothing)
        # Bazı edge-case'lerde (örn. tip dönüşümleri / serileştirme) olasılık değeri float dışı gelebilir.
        # Otomata olasılık hesapları deterministik ve sayısal olmalı; bu yüzden güvenli biçimde float'a zorluyoruz.
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(self.smoothing)

    def path_probability(self, patterns: list[str]) -> float:
        if len(patterns) < 2:
            return 1.0
        mapped = [self.map_pattern(pattern).mapped if self.states_ else pattern for pattern in patterns]
        probability: float = 1.0
        for source, target in zip(mapped[:-1], mapped[1:]):
            probability = float(probability) * self.transition_probability(str(source), str(target))
        return float(probability)

    def predict_sequence(self, patterns: list[str]) -> dict[str, Any]:
        probability = self.path_probability(patterns)
        threshold = self.threshold_ if self.threshold_ is not None else self.smoothing
        decision = "anomaly" if probability <= threshold else "normal"
        return {
            "probability": probability,
            "threshold": threshold,
            "decision": decision,
            "confidence": probability,
        }

    def predict_prefix_labels(self, patterns: list[str]) -> np.ndarray:
        """Her zaman adımı için prefix olasılığına göre etiket üretir (O(n))."""
        if not patterns:
            return np.asarray([], dtype=int)

        threshold = self.threshold_ if self.threshold_ is not None else self.smoothing
        mappings = [self.map_pattern(pattern) for pattern in patterns]
        path_probability = 1.0
        predictions: list[int] = []

        for idx, _current in enumerate(mappings):
            if idx > 0:
                previous = mappings[idx - 1]
                current = mappings[idx]
                path_probability *= self.transition_probability(previous.mapped, current.mapped)
            predictions.append(1 if path_probability <= threshold else 0)

        return np.asarray(predictions, dtype=int)

    @property
    def state_count(self) -> int:
        return len(self.states_)

    @property
    def transition_density(self) -> float:
        n_states = len(self.states_)
        if n_states <= 1:
            return 0.0
        transition_count = sum(len(targets) for targets in self.transition_probabilities_.values())
        return transition_count / (n_states * (n_states - 1))
