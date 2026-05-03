from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dataset.types import AudioSample


@dataclass
class RetrievalHit:
    sample: AudioSample
    score: float
    rank: int


class AudioEmbeddingEncoder(ABC):
    name: str

    @abstractmethod
    def encode_audio(self, audio_paths: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not support text embeddings")


class AudioClassifier(ABC):
    @abstractmethod
    def predict_proba(self, audio_paths: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

