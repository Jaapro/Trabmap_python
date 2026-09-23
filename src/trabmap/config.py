"""Configuration helpers for trabmap."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


def _read_yaml_or_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
    except Exception:
        import json
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping/dictionary: {path}")
    return data


def load_config(path: str | Path, section: str | None = None) -> dict[str, Any]:
    """Load a YAML/JSON config file and optionally return one section."""
    data = _read_yaml_or_json(path)
    if section is None:
        return data
    section_data = data.get(section, {})
    if section_data is None:
        return {}
    if not isinstance(section_data, dict):
        raise ValueError(f"Config section {section!r} must be a mapping/dictionary.")
    return section_data


def apply_config_to_module(module: Any, config: Mapping[str, Any]) -> None:
    """Set existing module-level constants from the config dictionary.

    Keys are matched case-insensitively, so `runner_csv` updates `RUNNER_CSV` for example.
    Unknown keys are ignored deliberately to keep configs portable.
    """
    name_lookup = {name.lower(): name for name in dir(module)}
    for key, value in config.items():
        attr = name_lookup.get(str(key).lower())
        if attr is not None:
            setattr(module, attr, value)


def namespace_from_defaults(defaults: Any, overrides: Mapping[str, Any]) -> SimpleNamespace:
    """Create a SimpleNamespace by copying defaults and applying overrides."""
    data = dict(vars(defaults)) if hasattr(defaults, "__dict__") else dict(defaults)
    data.update(dict(overrides))
    return SimpleNamespace(**data)
