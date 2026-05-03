from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from dataset.types import AudioSample


@dataclass
class AnchorSelectionResult:
    source_sample_id: str
    cluster_id: int
    anchor_sample_ids: list[str]


class AnchorSelector:
    def __init__(self, num_clusters: int = 32) -> None:
        self.num_clusters = num_clusters

    def cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < self.num_clusters:
            self.num_clusters = max(1, len(embeddings))
        model = KMeans(n_clusters=self.num_clusters, n_init="auto", random_state=7)
        return model.fit_predict(embeddings)

    def nearest_neighbors(self, embeddings: np.ndarray, query_index: int, count: int) -> list[int]:
        model = NearestNeighbors(n_neighbors=count + 1, metric="cosine")
        model.fit(embeddings)
        indices = model.kneighbors(embeddings[[query_index]], return_distance=False)[0].tolist()
        return [index for index in indices if index != query_index][:count]

    def select(
        self,
        samples: list[AudioSample],
        embeddings: np.ndarray,
        query_index: int,
        count: int,
    ) -> AnchorSelectionResult:
        cluster_ids = self.cluster_embeddings(embeddings)
        neighbors = self.nearest_neighbors(embeddings, query_index=query_index, count=count)
        return AnchorSelectionResult(
            source_sample_id=samples[query_index].sample_id,
            cluster_id=int(cluster_ids[query_index]),
            anchor_sample_ids=[samples[index].sample_id for index in neighbors],
        )

