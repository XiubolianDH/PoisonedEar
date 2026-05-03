from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AudioStats:
    duration_seconds: float
    sampling_rate: int
    rms_energy: float
    estimated_snr_db: float
    speech_label: str


@dataclass
class AudioSample:
    sample_id: str
    dataset_name: str
    split: str
    audio_path: str
    text: str
    category: str
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: AudioStats | None = None
    is_poisoned: bool = False
    poison_metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["audio_path"] = str(Path(self.audio_path))
        return data
