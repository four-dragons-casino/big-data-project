import json
import re
from pathlib import Path
from typing import Any, Dict


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    """Convert feature names into safe snake_case column names."""
    lowered = value.lower()
    slug = re.sub(r"[^0-9a-z]+", "_", lowered).strip("_")
    return slug or "feature"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

