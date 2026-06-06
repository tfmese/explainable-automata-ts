from __future__ import annotations

from typing import Any


def select_explanation_samples(
    explanations: list[dict[str, Any]],
    scenario: str,
    max_count: int,
) -> list[dict[str, Any]]:
    """Unseen senaryosunda en az bir unseen örneği ve bağlamını export'a dahil eder."""
    if max_count <= 0 or not explanations:
        return []

    if scenario != "unseen":
        return explanations[:max_count]

    unseen_indices = [index for index, item in enumerate(explanations) if item.get("status") == "unseen"]
    if not unseen_indices:
        return explanations[:max_count]

    selected_indices: list[int] = []
    for index in unseen_indices:
        for neighbor in (index - 1, index, index + 1):
            if 0 <= neighbor < len(explanations) and neighbor not in selected_indices:
                selected_indices.append(neighbor)

    for index in range(len(explanations)):
        if len(selected_indices) >= max_count:
            break
        if index not in selected_indices:
            selected_indices.append(index)

    return [explanations[index] for index in selected_indices[:max_count]]
