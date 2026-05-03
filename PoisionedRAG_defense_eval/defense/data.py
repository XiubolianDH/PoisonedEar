from __future__ import annotations

import json
from pathlib import Path

from defense.data_types import AudioSample, AudioStats
from defense.types import EvaluationSample

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
POISONED_RAG_ROOT = WORKSPACE_ROOT / "PoisionedRAG"


def load_manifest(path: Path) -> list[AudioSample]:
    samples: list[AudioSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            audio_path = Path(str(raw["audio_path"]))
            if not audio_path.is_absolute():
                candidates = [
                    (path.parent / audio_path).resolve(),
                    (WORKSPACE_ROOT / audio_path).resolve(),
                    (POISONED_RAG_ROOT / audio_path).resolve(),
                ]
                for candidate in candidates:
                    if candidate.exists():
                        raw["audio_path"] = str(candidate)
                        break
            stats_raw = raw.pop("stats", None)
            sample = AudioSample(**raw)
            if stats_raw:
                sample.stats = AudioStats(**stats_raw)
            samples.append(sample)
    return samples


def load_query_samples(metadata_path: Path, default_user_query: str) -> list[EvaluationSample]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    base_dir = metadata_path.parent
    samples: list[EvaluationSample] = []
    for row in payload.get("samples", []):
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (base_dir / audio_path).resolve()
        samples.append(
            EvaluationSample(
                query_id=str(row["sample_id"]),
                audio_path=str(audio_path),
                user_query=str(row.get("user_query") or default_user_query),
                clean_caption=str(row["clean_caption"]),
                adversarial_caption=str(row["adversarial_caption"]),
            )
        )
    return samples


def resolve_legacy_path(relative_path: str) -> Path:
    return (POISONED_RAG_ROOT / relative_path).resolve()
