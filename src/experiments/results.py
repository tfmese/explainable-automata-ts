from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentResult:
    dataset: str
    model: str
    scenario: str
    seed: int
    parameters: dict[str, Any]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "scenario": self.scenario,
            "seed": self.seed,
            "parameters": self.parameters,
            "metrics": self.metrics,
        }
