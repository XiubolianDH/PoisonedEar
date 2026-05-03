from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset.types import AudioSample


@dataclass
class InjectionDecision:
    accepted: list[AudioSample]
    dropped_duplicates: list[AudioSample]
    dropped_filtered: list[AudioSample]


class InjectionSimulator:
    def __init__(self, deduplication_threshold: float = 0.995, filtering_drop_rate: float = 0.0, ranking_noise_std: float = 0.0) -> None:
        self.deduplication_threshold = deduplication_threshold
        self.filtering_drop_rate = filtering_drop_rate
        self.ranking_noise_std = ranking_noise_std

    def deduplicate(self, candidates: list[AudioSample], embeddings: np.ndarray) -> tuple[list[AudioSample], list[AudioSample]]:
        accepted: list[AudioSample] = []
        dropped: list[AudioSample] = []
        accepted_embeddings: list[np.ndarray] = []
        for sample, embedding in zip(candidates, embeddings):
            if accepted_embeddings:
                similarities = np.dot(np.vstack(accepted_embeddings), embedding)
                if float(np.max(similarities)) >= self.deduplication_threshold:
                    dropped.append(sample)
                    continue
            accepted.append(sample)
            accepted_embeddings.append(embedding)
        return accepted, dropped

    def filter_candidates(self, candidates: list[AudioSample]) -> tuple[list[AudioSample], list[AudioSample]]:
        if self.filtering_drop_rate <= 0:
            return candidates, []
        keep_count = max(0, int(round(len(candidates) * (1.0 - self.filtering_drop_rate))))
        return candidates[:keep_count], candidates[keep_count:]

    def apply_ranking_noise(self, scores: np.ndarray) -> np.ndarray:
        if self.ranking_noise_std <= 0:
            return scores
        noise = np.random.normal(loc=0.0, scale=self.ranking_noise_std, size=scores.shape)
        return scores + noise

    def simulate(self, candidates: list[AudioSample], embeddings: np.ndarray) -> InjectionDecision:
        deduped, duplicates = self.deduplicate(candidates, embeddings)
        accepted, filtered = self.filter_candidates(deduped)
        return InjectionDecision(accepted=accepted, dropped_duplicates=duplicates, dropped_filtered=filtered)

