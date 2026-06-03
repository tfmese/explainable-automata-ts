from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.results import ExperimentResult


class ExperimentLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "experiments.jsonl"
        self.summary_path = self.log_dir / "run_summary.json"

    def log(self, record: ExperimentResult | dict[str, Any]) -> None:
        payload = record.to_dict() if isinstance(record, ExperimentResult) else record
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_run_summary(self, summary: dict[str, Any]) -> None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_experiment_record(
    *,
    dataset: str,
    model: str,
    scenario: str,
    seed: int,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    fold: int | None = None,
    extra: dict[str, Any] | None = None,
) -> ExperimentResult:
    merged_params = dict(parameters)
    if fold is not None:
        merged_params["fold"] = fold
    if extra:
        merged_params.update(extra)
    return ExperimentResult(
        dataset=dataset,
        model=model,
        scenario=scenario,
        seed=seed,
        parameters=merged_params,
        metrics=metrics,
    )
