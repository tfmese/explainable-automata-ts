from __future__ import annotations

from typing import Any

from Levenshtein import distance as levenshtein_distance
from src.models.automata import ProbabilisticAutomata


def find_counterfactual(
    automata: ProbabilisticAutomata,
    patterns: list[str],
    current_decision: str,
    threshold: float,
) -> dict[str, Any] | None:
    if not patterns or not automata.states_:
        return None

    last_pattern = patterns[-1]
    best_counterfactual = None
    min_dist = float("inf")

    # Arama uzayı olarak eğitime katılan tüm benzersiz durumları (pattern'ları) tarıyoruz.
    # Amacımız, son zaman adımındaki pattern yerine adayın konması durumunda kararın değişip değişmeyeceğini görmek.
    for candidate in sorted(automata.states_):
        if candidate == last_pattern:
            continue

        modified_patterns = list(patterns)
        modified_patterns[-1] = candidate

        new_prob = automata.path_probability(modified_patterns)
        new_decision = "anomaly" if new_prob <= threshold else "normal"

        if new_decision != current_decision:
            dist = levenshtein_distance(last_pattern, candidate)
            if dist < min_dist:
                min_dist = dist
                best_counterfactual = {
                    "pattern": candidate,
                    "distance": dist,
                    "probability": new_prob,
                    "decision": new_decision,
                }
            elif dist == min_dist:
                # Eşit mesafe durumunda: kararı değiştiren daha güvenli alternatifi seçiyoruz.
                if current_decision == "anomaly":  # normal yapmak istiyoruz, olasılığı en yüksek olanı seçelim
                    if best_counterfactual is None or new_prob > best_counterfactual["probability"]:
                        best_counterfactual = {
                            "pattern": candidate,
                            "distance": dist,
                            "probability": new_prob,
                            "decision": new_decision,
                        }
                else:  # anomali yapmak istiyoruz, olasılığı en düşük olanı seçelim
                    if best_counterfactual is None or new_prob < best_counterfactual["probability"]:
                        best_counterfactual = {
                            "pattern": candidate,
                            "distance": dist,
                            "probability": new_prob,
                            "decision": new_decision,
                        }

    return best_counterfactual


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
    
    counterfactual = find_counterfactual(
        automata,
        patterns,
        prediction["decision"],
        prediction["threshold"],
    )

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
        "counterfactual": counterfactual,
    }

