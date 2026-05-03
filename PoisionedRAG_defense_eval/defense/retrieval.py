from __future__ import annotations

from dataclasses import dataclass

from defense.data_types import AudioSample
from defense.types import RetrievedCaption


class AudioOnlyRetriever:
    def __init__(self, encoder_name: str, checkpoint_overrides: dict | None = None, similarity: str = "cosine") -> None:
        self.encoder_name = encoder_name.lower()
        self.checkpoint_overrides = checkpoint_overrides or {}
        self.similarity = similarity
        self.retriever = build_local_retriever(self.encoder_name, self.checkpoint_overrides, similarity)

    def build_index(self, samples: list, batch_size: int | None = None) -> None:
        self.retriever.build(samples, batch_size=batch_size)

    def retrieve(self, audio_path: str, top_k: int) -> list[RetrievedCaption]:
        hits = self.retriever.query(audio_path=audio_path, top_k=top_k)
        results: list[RetrievedCaption] = []
        for hit in hits:
            results.append(
                RetrievedCaption(
                    rank=hit.rank,
                    score=hit.score,
                    text=hit.sample.text,
                    sample_id=hit.sample.sample_id,
                    audio_path=hit.sample.audio_path,
                    dataset_name=hit.sample.dataset_name,
                    is_malicious=bool(hit.sample.is_poisoned),
                    metadata=dict(hit.sample.poison_metadata or {}),
                )
            )
        return results


@dataclass
class LocalRetrievalHit:
    sample: AudioSample
    score: float
    rank: int


class LocalAudioRetriever:
    def __init__(self, encoder: object, similarity: str = "cosine") -> None:
        self.encoder = encoder
        self.similarity = similarity
        self.samples: list[AudioSample] = []
        self.embeddings = None

    def build(self, samples: list[AudioSample], batch_size: int | None = None) -> None:
        import numpy as np

        self.samples = list(samples)
        batch_size = max(int(batch_size or len(self.samples) or 1), 1)
        batches = []
        for start in range(0, len(self.samples), batch_size):
            batch = self.samples[start : start + batch_size]
            batches.append(self.encoder.encode_audio([sample.audio_path for sample in batch]))
        self.embeddings = np.vstack(batches)

    def query(self, audio_path: str, top_k: int) -> list[LocalRetrievalHit]:
        import numpy as np

        if self.embeddings is None:
            raise RuntimeError("Retrieval index has not been built.")
        query = self.encoder.encode_audio([audio_path])[0]
        scores = (self.embeddings @ query) / ((np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query)) + 1e-12)
        best_indices = np.argsort(-scores)[:top_k]
        return [
            LocalRetrievalHit(sample=self.samples[int(index)], score=float(scores[int(index)]), rank=rank)
            for rank, index in enumerate(best_indices, start=1)
        ]


class SpectralAudioEncoder:
    name = "spectral"

    def encode_audio(self, audio_paths: list[str]):
        import numpy as np
        import soundfile as sf

        embeddings = []
        for audio_path in audio_paths:
            waveform, sr = sf.read(audio_path, always_2d=False)
            waveform = np.asarray(waveform, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if waveform.size == 0:
                embeddings.append(np.zeros(16, dtype="float32"))
                continue
            if sr > 16000 and waveform.size > 1:
                step = max(int(round(sr / 16000.0)), 1)
                waveform = waveform[::step]
                sr = max(int(sr / step), 1)

            frame_count = min(8, max(1, waveform.size // 512))
            frames = np.array_split(waveform, frame_count)
            rms = np.array([float(np.sqrt(np.mean(frame * frame) + 1e-12)) for frame in frames], dtype="float32")
            zero_cross = np.array(
                [float(np.mean(np.abs(np.diff(np.signbit(frame.astype("float32")))))) if frame.size > 1 else 0.0 for frame in frames],
                dtype="float32",
            )
            spectrum = np.abs(np.fft.rfft(waveform, n=min(max(len(waveform), 256), 4096))).astype("float32")
            if spectrum.size == 0:
                spectrum = np.zeros(32, dtype="float32")
            band_edges = np.linspace(0, spectrum.size, num=9, dtype=int)
            band_energy = []
            for start, end in zip(band_edges[:-1], band_edges[1:]):
                band = spectrum[start:end]
                band_energy.append(float(np.mean(band)) if band.size else 0.0)

            embedding = np.concatenate(
                [
                    np.array(
                        [
                            float(waveform.mean()),
                            float(waveform.std()),
                            float(np.max(np.abs(waveform))),
                            float(len(waveform) / max(sr, 1)),
                        ],
                        dtype="float32",
                    ),
                    rms[:4],
                    zero_cross[:4],
                    np.asarray(band_energy, dtype="float32"),
                ]
            ).astype("float32")
            embeddings.append(embedding / (np.linalg.norm(embedding) + 1e-12))
        return np.vstack(embeddings)


class ClapAudioEncoder:
    name = "clap"

    def __init__(self, checkpoint: str = "laion/clap-htsat-unfused") -> None:
        from transformers import ClapModel, ClapProcessor
        import torch
        import soundfile as sf
        import numpy as np

        self.torch = torch
        self.sf = sf
        self.np = np
        self.model = ClapModel.from_pretrained(checkpoint, use_safetensors=True)
        self.processor = ClapProcessor.from_pretrained(checkpoint)
        self.model.eval()

    def encode_audio(self, audio_paths: list[str]):
        waveforms = []
        for path in audio_paths:
            waveform, sr = self.sf.read(path, always_2d=False)
            waveform = self.np.asarray(waveform, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if sr != 48000 and waveform.size > 1:
                # Light-weight resampling fallback using numpy indexing to keep this folder self-contained.
                target_length = max(int(round(len(waveform) * 48000.0 / max(sr, 1))), 1)
                indices = self.np.linspace(0, max(len(waveform) - 1, 0), num=target_length)
                waveform = self.np.interp(indices, self.np.arange(len(waveform)), waveform).astype("float32")
            waveforms.append(waveform)
        inputs = self.processor(audios=waveforms, sampling_rate=48000, return_tensors="pt", padding=True)
        with self.torch.no_grad():
            embeddings = self.model.get_audio_features(**inputs)
        array = embeddings.cpu().numpy().astype("float32")
        norms = self.np.linalg.norm(array, axis=1, keepdims=True) + 1e-12
        return array / norms


def build_local_retriever(encoder_name: str, checkpoint_overrides: dict[str, str], similarity: str) -> LocalAudioRetriever:
    normalized = encoder_name.lower()
    if normalized == "spectral":
        return LocalAudioRetriever(encoder=SpectralAudioEncoder(), similarity=similarity)
    if normalized == "clap":
        return LocalAudioRetriever(
            encoder=ClapAudioEncoder(checkpoint=checkpoint_overrides.get("clap", "laion/clap-htsat-unfused")),
            similarity=similarity,
        )
    raise ValueError(f"Unsupported retriever encoder: {encoder_name}")
