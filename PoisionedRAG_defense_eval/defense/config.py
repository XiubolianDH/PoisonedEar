from __future__ import annotations

import json
from pathlib import Path

def load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return json.loads(text)
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return payload
