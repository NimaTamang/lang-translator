from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
