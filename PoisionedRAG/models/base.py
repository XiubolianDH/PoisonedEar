from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelInput:
    audio_path: str
    user_query: str
    retrieved_context: str
    prompt: str


@dataclass
class ModelOutput:
    model_name: str
    text: str
    raw_response: object | None = None


class VictimModel(ABC):
    @abstractmethod
    def generate(self, model_input: ModelInput) -> ModelOutput:
        raise NotImplementedError

