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
