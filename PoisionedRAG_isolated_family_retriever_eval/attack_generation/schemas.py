from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CDABStageTexts:
    de_semanticized_text: str | None = None
    physicalized_text: str | None = None
    reframed_text: str | None = None


@dataclass
class CDABAttackRecord:
    source_sample_id: str
    source_text: str
    target_text: str
    anchor_sample_ids: list[str]
    anchor_texts: list[str]
    cluster_id: int | None = None
    stage_texts: CDABStageTexts = field(default_factory=CDABStageTexts)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CDABGenerationResult:
    sample_id: str
    dataset: str
    audio_path: str
    clean_caption: str
    acoustic_description: str
    adversarial_caption: str
    acoustic_consistency: float | None = None
    semantic_distance: float | None = None

    def to_csv_row(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "audio_path": self.audio_path,
            "clean_caption": self.clean_caption,
            "acoustic_description": self.acoustic_description,
            "adversarial_caption": self.adversarial_caption,
            "acoustic_consistency": "" if self.acoustic_consistency is None else f"{self.acoustic_consistency:.6f}",
            "semantic_distance": "" if self.semantic_distance is None else f"{self.semantic_distance:.6f}",
        }
