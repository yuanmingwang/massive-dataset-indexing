from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path(filename: str) -> Path:
    return project_root() / "configs" / filename


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in config file: {config_path}")
    return data


def merge_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
