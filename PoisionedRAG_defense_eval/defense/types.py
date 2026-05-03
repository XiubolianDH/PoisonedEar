from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievedCaption:
    rank: int
    score: float
    text: str
    sample_id: str
    audio_path: str
    dataset_name: str
    is_malicious: bool
    ppl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSample:
    query_id: str
    audio_path: str
    user_query: str
    clean_caption: str
    adversarial_caption: str


@dataclass
class DefenseResult:
    defense_name: str
    rewritten_query: str
    kept_contexts: list[RetrievedCaption]
    removed_contexts: list[RetrievedCaption]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    model_key: str
    dataset_key: str
    defense_key: str
    query_id: str
    audio_path: str
    original_query: str
    effective_query: str
    clean_caption: str
    adversarial_caption: str
    response: str
    top_k_requested: int
    retrieved_count: int
    malicious_count_before: int
    malicious_count_after: int
    recall_before: float
    recall_after: float
    sim_clean: float
    sim_adv: float
    attack_success: int
    avg_ppl_clean: float | None
    avg_ppl_malicious: float | None
    defense_metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_context_before: list[dict[str, Any]] = field(default_factory=list)
    retrieval_context_after: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    model_key: str
    dataset_key: str
    defense_key: str
    num_queries: int
    recall_before: float
    recall_after: float
    asr: float
    avg_ppl_clean: float | None
    avg_ppl_malicious: float | None
    avg_malicious_count_before: float
    avg_malicious_count_after: float
    delta_asr_vs_no_defense: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
