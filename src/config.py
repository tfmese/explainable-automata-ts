from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    values: dict[str, Any]
    path: Path

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self.values
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def load_config(path: str | Path = "config/config.yaml") -> ProjectConfig:
    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        values = yaml.safe_load(raw)
    except ModuleNotFoundError:
        values = json.loads(raw)

    if not isinstance(values, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    return ProjectConfig(values=values, path=config_path)
