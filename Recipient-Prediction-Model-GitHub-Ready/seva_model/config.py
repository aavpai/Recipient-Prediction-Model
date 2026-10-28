from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def special_flags(month: int, drive_position: int, config: dict) -> dict[str, int]:
    flags = {name: 0 for name in config.get("special_overlays", {})}
    for name, spec in config.get("special_overlays", {}).items():
        flags[name] = int(
            int(spec["month"]) == int(month)
            and int(spec["drive_position"]) == int(drive_position)
        )
    return flags
