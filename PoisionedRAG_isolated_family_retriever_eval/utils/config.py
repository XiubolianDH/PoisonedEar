from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_project_env(config_path: str | Path | None = None) -> Path | None:
    explicit_env_path = os.getenv("AUDIO_RAG_ENV_FILE")
    candidate_paths: list[Path] = []
    if explicit_env_path:
        candidate_paths.append(Path(explicit_env_path).expanduser())
    if config_path is not None:
        config_dir = Path(config_path).resolve().parent
        candidate_paths.extend(
            [
                config_dir / ".env",
                config_dir.parent / ".env",
            ]
        )
    else:
        candidate_paths.append(Path.cwd() / ".env")

    for candidate in candidate_paths:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return candidate
    return None


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    load_project_env(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
