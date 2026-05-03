from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import ensure_dir


def setup_logger(name: str = "audio_rag", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


class ExperimentLogger:
    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.run_name = run_name
        self.path = self.output_dir / f"{run_name}.jsonl"

    def log(self, event_type: str, payload: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": asdict(payload) if is_dataclass(payload) else payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
