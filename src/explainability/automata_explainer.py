from __future__ import annotations

from typing import Any

from src.models.automata import ProbabilisticAutomata


def explain_automata_decision(
    automata: ProbabilisticAutomata,
    patterns: list[str],
    time_step: int | None = None,
) -> dict[str, Any]:
    if not patterns:
        raise ValueError("patterns must not be empty")

    mappings = [automata.map_pattern(pattern) for pattern in patterns]
    transitions = []
    path_probability = 1.0

    for previous, current in zip(mappings[:-1], mappings[1:]):
        probability = automata.transition_probability(previous.mapped, current.mapped)
        path_probability *= probability
        transitions.append(
            {
                "from": previous.mapped,
                "to": current.mapped,
                "probability": probability,
            }
        )

    prediction = automata.predict_sequence(patterns)
    current = mappings[-1]
    return {
        "time_step": time_step,
        "state": current.mapped,
        "pattern": current.original,
        "status": current.status,
        "mapped_to": current.mapped if current.status == "unseen" else None,
        "mapping_distance": current.distance,
        "transitions": transitions,
        "probability": path_probability,
        "decision": prediction["decision"],
        "confidence": prediction["confidence"],
        "threshold": prediction["threshold"],
    }
