from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from tqdm.auto import tqdm

from dataset.types import AudioSample
from retriever.base import AudioEmbeddingEncoder, RetrievalHit

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


@dataclass
class RetrievalIndex:
    encoder_name: str
    dimension: int
    samples: list[AudioSample]
    index: object | None
    embeddings: np.ndarray
    backend: str


class FaissAudioRetriever:
    def __init__(self, encoder: AudioEmbeddingEncoder, similarity: str = "cosine") -> None:
        self.encoder = encoder
        self.similarity = similarity
        self.retrieval_index: RetrievalIndex | None = None

    def build(
        self,
        samples: Sequence[AudioSample],
        show_progress: bool = False,
        progress_desc: str = "Building retrieval index",
        batch_size: int | None = None,
    ) -> RetrievalIndex:
        audio_paths = [sample.audio_path for sample in samples]
        batch_size = max(int(batch_size or len(audio_paths) or 1), 1)
        if show_progress:
            path_iter = tqdm(range(0, len(audio_paths), batch_size), desc=progress_desc, unit="batch", dynamic_ncols=True)
            embeddings_list = []
            for start in path_iter:
                batch_paths = audio_paths[start : start + batch_size]
                embeddings_list.append(self.encoder.encode_audio(batch_paths))
            embeddings = np.vstack(embeddings_list).astype("float32")
        else:
            embeddings = self.encoder.encode_audio(audio_paths).astype("float32")
        dimension = embeddings.shape[1]
        if self.similarity != "cosine":
            raise ValueError("Only cosine similarity is implemented.")
        backend = "sklearn"
        index = None
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        if faiss is not None:
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            backend = "faiss"
        self.retrieval_index = RetrievalIndex(
            encoder_name=self.encoder.name,
            dimension=dimension,
            samples=list(samples),
            index=index,
            embeddings=embeddings,
            backend=backend,
        )
        return self.retrieval_index

    def query(self, audio_path: str, top_k: int) -> list[RetrievalHit]:
        if self.retrieval_index is None:
            raise RuntimeError("Retrieval index has not been built.")
        query_embedding = self.encoder.encode_audio([audio_path]).astype("float32")
        query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12
        if self.retrieval_index.backend == "faiss":
            scores, indices = self.retrieval_index.index.search(query_embedding, top_k)
        else:
            similarities = sklearn_cosine_similarity(query_embedding, self.retrieval_index.embeddings)[0]
            indices = np.argsort(-similarities)[:top_k][None, :]
            scores = similarities[indices]
        hits: list[RetrievalHit] = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            hits.append(
                RetrievalHit(
                    sample=self.retrieval_index.samples[int(index)],
                    score=float(score),
                    rank=rank,
                )
            )
        return hits
