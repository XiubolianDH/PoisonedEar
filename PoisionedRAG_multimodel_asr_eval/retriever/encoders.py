from __future__ import annotations

from pathlib import Path
from typing import Sequence
import sys

import librosa
import numpy as np
import torch

from retriever.base import AudioClassifier, AudioEmbeddingEncoder


def _load_audio_batch(audio_paths: Sequence[str], target_sr: int = 16000) -> torch.Tensor:
    waveforms: list[torch.Tensor] = []
    max_length = 0
    for path in audio_paths:
        waveform, _ = librosa.load(Path(path), sr=target_sr, mono=True)
        tensor = torch.tensor(waveform, dtype=torch.float32)
        max_length = max(max_length, tensor.numel())
        waveforms.append(tensor)
    padded = []
    for waveform in waveforms:
        if waveform.numel() < max_length:
            waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.numel()))
        padded.append(waveform)
    return torch.stack(padded, dim=0)


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms


class TransformerAudioEncoder(AudioEmbeddingEncoder):
    name = "transformer_audio_encoder"

    def __init__(self, checkpoint: str, processor_name: str | None = None, pool: str = "mean") -> None:
        from transformers import AutoModel, AutoProcessor

        self.pool = pool
        self.model = AutoModel.from_pretrained(checkpoint)
        self.processor = AutoProcessor.from_pretrained(processor_name or checkpoint)
        self.model.eval()

    def _forward_audio(self, audio_paths: Sequence[str]) -> np.ndarray:
        waveforms = _load_audio_batch(audio_paths)
        inputs = self.processor(audios=waveforms.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1) if self.pool == "mean" else hidden[:, 0, :]
        return _normalize(pooled.cpu().numpy())

    def encode_audio(self, audio_paths: Sequence[str]) -> np.ndarray:
        return self._forward_audio(audio_paths)


class ClapEncoder(AudioEmbeddingEncoder):
    name = "clap"

    def __init__(self, checkpoint: str = "laion/clap-htsat-unfused") -> None:
        from transformers import ClapModel, ClapProcessor

        self.model = ClapModel.from_pretrained(checkpoint, use_safetensors=True)
        self.processor = ClapProcessor.from_pretrained(checkpoint)
        self.model.eval()

    def encode_audio(self, audio_paths: Sequence[str]) -> np.ndarray:
        waveforms = [librosa.load(path, sr=48000, mono=True)[0] for path in audio_paths]
        inputs = self.processor(audios=waveforms, sampling_rate=48000, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = self.model.get_audio_features(**inputs)
        return _normalize(embeddings.cpu().numpy())

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return _normalize(embeddings.cpu().numpy())


class AudioCLIPEncoder(AudioEmbeddingEncoder):
    name = "audioclip"

    def __init__(self, checkpoint: str | None = None) -> None:
        self.checkpoint = checkpoint
        if self.checkpoint is None:
            vendored_checkpoint = (
                Path(__file__).resolve().parents[1]
                / "third_party"
                / "audioclip"
                / "AudioCLIP-Full-Training.pt"
            )
            if vendored_checkpoint.exists():
                self.checkpoint = str(vendored_checkpoint)
        try:
            from audioclip import AudioCLIP  # type: ignore
        except ImportError:
            vendored_repo = Path(__file__).resolve().parents[1] / "third_party" / "AudioCLIP"
            if vendored_repo.exists():
                sys.path.insert(0, str(vendored_repo))
                for module_name in ("model", "utils", "ignite_trainer"):
                    module = sys.modules.get(module_name)
                    if module is not None:
                        module_path = getattr(module, "__file__", "") or ""
                        if str(vendored_repo) not in module_path:
                            sys.modules.pop(module_name, None)
                try:
                    from model.audioclip import AudioCLIP  # type: ignore
                    from utils.transforms import ToTensor1D  # type: ignore
                except ImportError as exc:
                    raise ImportError(
                        "AudioCLIP requires the third-party 'audioclip' package or vendored AudioCLIP source."
                    ) from exc
            else:
                raise ImportError(
                    "AudioCLIP requires the third-party 'audioclip' package or vendored AudioCLIP source."
                )
        if "ToTensor1D" not in locals():
            from utils.transforms import ToTensor1D  # type: ignore
        self.audio_transform = ToTensor1D()
        self.model = AudioCLIP(pretrained=self.checkpoint) if self.checkpoint else AudioCLIP()
        self.model.eval()

    def encode_audio(self, audio_paths: Sequence[str]) -> np.ndarray:
        batch = []
        max_length = 0
        for path in audio_paths:
            waveform, _ = librosa.load(path, sr=44100, mono=True, dtype=np.float32)
            tensor = self.audio_transform(waveform.reshape(1, -1))
            max_length = max(max_length, tensor.shape[-1])
            batch.append(tensor)
        padded_batch = []
        for tensor in batch:
            if tensor.shape[-1] < max_length:
                tensor = torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[-1]))
            padded_batch.append(tensor)
        audio_tensor = torch.stack(padded_batch, dim=0)
        with torch.no_grad():
            ((audio_features, _, _), _), _ = self.model(audio=audio_tensor)
        return _normalize(audio_features.cpu().numpy())


class PANNsEncoder(AudioEmbeddingEncoder, AudioClassifier):
    name = "panns"

    def __init__(self, checkpoint: str | None = None) -> None:
        try:
            from panns_inference import AudioTagging  # type: ignore
        except ImportError as exc:
            raise ImportError("PANNs requires the 'panns-inference' package.") from exc
        self.model = AudioTagging(checkpoint_path=checkpoint) if checkpoint else AudioTagging()

    def encode_audio(self, audio_paths: Sequence[str]) -> np.ndarray:
        waveforms = [librosa.load(path, sr=32000, mono=True)[0] for path in audio_paths]
        embeddings = []
        for waveform in waveforms:
            clipwise_output, embedding = self.model.inference(waveform[None, :])
            _ = clipwise_output
            embeddings.append(embedding[0])
        return _normalize(np.asarray(embeddings))

    def predict_proba(self, audio_paths: Sequence[str]) -> np.ndarray:
        waveforms = [librosa.load(path, sr=32000, mono=True)[0] for path in audio_paths]
        probabilities = []
        for waveform in waveforms:
            clipwise_output, embedding = self.model.inference(waveform[None, :])
            _ = embedding
            probabilities.append(clipwise_output[0])
        return np.asarray(probabilities)


class Wav2Vec2Encoder(AudioEmbeddingEncoder):
    name = "wav2vec2"

    def __init__(
        self,
        checkpoint: str = "facebook/wav2vec2-base",
        device: str | None = None,
        chunk_seconds: float = 30.0,
        max_chunks_per_file: int = 4,
    ) -> None:
        import torchaudio

        _ = checkpoint
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.sample_rate = bundle.sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = bundle.get_model().to(self.device)
        self.model.eval()
        self.chunk_seconds = max(float(chunk_seconds), 1.0)
        self.max_chunks_per_file = max(int(max_chunks_per_file), 1)

    def _chunk_starts(self, num_frames: int, chunk_frames: int) -> list[int]:
        if num_frames <= chunk_frames:
            return [0]
        max_start = num_frames - chunk_frames
        if self.max_chunks_per_file == 1:
            return [0]
        starts = np.linspace(0, max_start, num=self.max_chunks_per_file)
        return [int(round(v)) for v in starts]

    def _encode_waveform(self, waveform: torch.Tensor) -> np.ndarray:
        waveform = waveform.to(self.device)
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveform)
        final_hidden = features[-1].mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        return final_hidden

    def encode_audio(self, audio_paths: Sequence[str]) -> np.ndarray:
        import torchaudio

        embeddings: list[np.ndarray] = []
        for audio_path in audio_paths:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            chunk_frames = int(self.chunk_seconds * self.sample_rate)
            starts = self._chunk_starts(waveform.shape[1], chunk_frames)
            chunk_embeddings: list[np.ndarray] = []
            for start in starts:
                chunk = waveform[:, start : start + chunk_frames]
                chunk_embeddings.append(self._encode_waveform(chunk))
            embeddings.append(np.mean(np.vstack(chunk_embeddings), axis=0).astype(np.float32))
        return _normalize(np.vstack(embeddings))


class SpectralAudioEncoder(AudioEmbeddingEncoder):
    name = "spectral"

    def __init__(self, target_sr: int = 16000, n_mfcc: int = 20) -> None:
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc

    def encode_audio(self, audio_paths: Sequence[str]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for path in audio_paths:
            waveform, _ = librosa.load(path, sr=self.target_sr, mono=True)
            mfcc = librosa.feature.mfcc(y=waveform, sr=self.target_sr, n_mfcc=self.n_mfcc)
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=self.target_sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=self.target_sr)
            zero_crossing = librosa.feature.zero_crossing_rate(y=waveform)
            rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=self.target_sr)
            embedding = np.concatenate(
                [
                    mfcc.mean(axis=1),
                    mfcc.std(axis=1),
                    spectral_centroid.mean(axis=1),
                    spectral_bandwidth.mean(axis=1),
                    zero_crossing.mean(axis=1),
                    rolloff.mean(axis=1),
                ]
            )
            embeddings.append(embedding.astype(np.float32))
        return _normalize(np.vstack(embeddings))


def build_encoder(name: str, checkpoint_overrides: dict[str, str] | None = None) -> AudioEmbeddingEncoder:
    overrides = checkpoint_overrides or {}
    normalized = name.lower()
    if normalized == "spectral":
        return SpectralAudioEncoder()
    if normalized == "clap":
        return ClapEncoder(checkpoint=overrides.get("clap", "laion/clap-htsat-unfused"))
    if normalized == "audioclip":
        return AudioCLIPEncoder(checkpoint=overrides.get("audioclip"))
    if normalized == "panns":
        return PANNsEncoder(checkpoint=overrides.get("panns"))
    if normalized == "wav2vec2":
        return Wav2Vec2Encoder(checkpoint=overrides.get("wav2vec2", "facebook/wav2vec2-base"))
    raise ValueError(f"Unsupported encoder: {name}")
