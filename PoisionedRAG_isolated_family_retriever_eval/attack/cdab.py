from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CDABPlan:
    source_text: str
    target_text: str
    de_semanticized_text: str | None = None
    physicalized_text: str | None = None
    reframed_text: str | None = None


class CDABPipeline:
    """Structure-only placeholder for future poison generation."""

    def de_semanticize(self, source_text: str) -> str:
        raise NotImplementedError("CDAB generation is intentionally deferred in this stage.")

    def physicalize(self, de_semanticized_text: str) -> str:
        raise NotImplementedError("CDAB generation is intentionally deferred in this stage.")

    def causal_reframe(self, physicalized_text: str, target_text: str) -> str:
        raise NotImplementedError("CDAB generation is intentionally deferred in this stage.")

    def build_plan(self, source_text: str, target_text: str) -> CDABPlan:
        return CDABPlan(source_text=source_text, target_text=target_text)

