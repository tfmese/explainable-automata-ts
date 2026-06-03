from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def prepare_results_dir(base_dir: str | Path, *, timestamped: bool = False) -> Path:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    if not timestamped:
        return base_path

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json_result(
    results_dir: Path,
    filename: str,
    payload: Any,
    *,
    backup_existing: bool = True,
) -> Path:
    target = results_dir / filename
    if backup_existing and target.exists():
        backups_dir = results_dir / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backups_dir / f"{target.stem}_{stamp}{target.suffix}"
        shutil.copy2(target, backup_path)

    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return target
