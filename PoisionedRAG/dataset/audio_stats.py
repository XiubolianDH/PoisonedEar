from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

from dataset.types import AudioSample, AudioStats


def _estimate_snr_db(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        return 0.0
    centered = waveform - np.mean(waveform)
    signal_power = float(np.mean(centered**2) + 1e-12)
    noise = centered - librosa.effects.preemphasis(centered)
    noise_power = float(np.mean(noise**2) + 1e-12)
    return float(10.0 * np.log10(signal_power / noise_power))


def _speech_label_from_metadata(text: str, category: str, metadata: dict[str, object]) -> str:
    haystacks = " ".join(
        [
            text.lower(),
            category.lower(),
            " ".join(str(v).lower() for v in metadata.values()),
        ]
    )
    speech_tokens = ("speech", "speaker", "voice", "talk", "conversation", "vocal", "dialog")
    return "speech" if any(token in haystacks for token in speech_tokens) else "non_speech"


def compute_audio_stats(audio_path: str, text: str, category: str, metadata: dict[str, object]) -> AudioStats:
    waveform, sampling_rate = sf.read(audio_path, always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    waveform = waveform.astype(np.float32)
    duration_seconds = float(len(waveform) / max(sampling_rate, 1))
    rms_energy = float(np.sqrt(np.mean(np.square(waveform)) + 1e-12))
    estimated_snr_db = _estimate_snr_db(waveform)
    speech_label = _speech_label_from_metadata(text=text, category=category, metadata=metadata)
    return AudioStats(
        duration_seconds=duration_seconds,
        sampling_rate=int(sampling_rate),
        rms_energy=rms_energy,
        estimated_snr_db=estimated_snr_db,
        speech_label=speech_label,
    )


def attach_stats(samples: Iterable[AudioSample]) -> list[AudioSample]:
    enriched: list[AudioSample] = []
    for sample in samples:
      sample.stats = compute_audio_stats(
          audio_path=sample.audio_path,
          text=sample.text,
          category=sample.category,
          metadata=sample.metadata,
      )
      enriched.append(sample)
    return enriched


def stats_to_json(samples: Iterable[AudioSample]) -> dict[str, object]:
    items = list(samples)
    durations = [sample.stats.duration_seconds for sample in items if sample.stats]
    snrs = [sample.stats.estimated_snr_db for sample in items if sample.stats]
    rms = [sample.stats.rms_energy for sample in items if sample.stats]
    return {
        "num_samples": len(items),
        "duration_seconds": _summarize_float(durations),
        "estimated_snr_db": _summarize_float(snrs),
        "rms_energy": _summarize_float(rms),
        "speech_counts": {
            "speech": sum(1 for s in items if s.stats and s.stats.speech_label == "speech"),
            "non_speech": sum(1 for s in items if s.stats and s.stats.speech_label == "non_speech"),
        },
        "samples": [sample.to_json() for sample in items],
    }


def _summarize_float(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    array = np.asarray(values, dtype=np.float32)
    return {"min": float(array.min()), "max": float(array.max()), "mean": float(array.mean())}

