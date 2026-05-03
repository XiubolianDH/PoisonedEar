from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from dataset.types import AudioSample


class DatasetLoaderError(RuntimeError):
    """Raised when a dataset cannot be loaded from the provided metadata."""


def load_configured_datasets(data_config: dict) -> list[AudioSample]:
    samples: list[AudioSample] = []
    datasets = data_config.get("datasets", {})
    limit = data_config.get("max_samples_per_dataset")
    for name, settings in datasets.items():
        if not settings.get("enabled", False):
            continue
        loaded = list(load_dataset(name=name, settings=settings))
        if limit is not None:
            loaded = loaded[: int(limit)]
        samples.extend(loaded)
    return samples


def load_dataset(name: str, settings: dict) -> Iterable[AudioSample]:
    metadata_path = settings.get("metadata_path")
    audio_root = settings.get("audio_root")
    split = settings.get("split", "default")
    if not metadata_path or not audio_root:
        return []
    if name == "wavcaps":
        return load_wavcaps_freesound(metadata_path, audio_root, split)
    if name == "clotho":
        return load_clotho(metadata_path, audio_root, split)
    if name == "audioset":
        return load_audioset(metadata_path, audio_root, split)
    if name == "esc50":
        return load_esc50(metadata_path, audio_root, split)
    raise DatasetLoaderError(f"Unsupported dataset: {name}")


def load_wavcaps_freesound(metadata_path: str, audio_root: str, split: str) -> Iterable[AudioSample]:
    records = _load_json_records(metadata_path)
    for index, record in enumerate(records):
        tags = record.get("tags") or []
        caption = record.get("text") or record.get("caption") or record.get("description") or "No caption available."
        audio_file = record.get("audio_path") or record.get("audio") or record.get("wav_path") or record.get("file_name")
        if not audio_file:
            continue
        audio_path = Path(audio_file)
        if not audio_path.exists():
            audio_path = Path(audio_root) / str(audio_file)
        yield AudioSample(
            sample_id=str(record.get("sample_id") or f"wavcaps-{index}"),
            dataset_name="wavcaps",
            split=split,
            audio_path=str(audio_path),
            text=caption,
            category=str(record.get("category") or (tags[0] if tags else "unknown")),
            metadata={"tags": tags, "source": "freesound", "raw": record},
        )


def load_clotho(metadata_path: str, audio_root: str, split: str) -> Iterable[AudioSample]:
    dataframe = pd.read_csv(metadata_path)
    if "audio_path" in dataframe.columns and "text" in dataframe.columns:
        for _, row in dataframe.iterrows():
            yield AudioSample(
                sample_id=str(row.get("sample_id", f"clotho-{row.name}")),
                dataset_name="clotho",
                split=split,
                audio_path=str(row["audio_path"]),
                text=str(row["text"]),
                category=str(row.get("category", "unknown")),
                metadata={"source_id": row.get("source_id"), "raw": row.to_dict()},
            )
        return
    caption_columns = [column for column in dataframe.columns if column.startswith("caption")]
    for _, row in dataframe.iterrows():
        audio_file = row.get("file_name")
        if not isinstance(audio_file, str):
            continue
        captions = [str(row[column]) for column in caption_columns if pd.notna(row[column])]
        text = captions[0] if captions else "No caption available."
        yield AudioSample(
            sample_id=f"clotho-{audio_file}",
            dataset_name="clotho",
            split=split,
            audio_path=str(Path(audio_root) / audio_file),
            text=text,
            category=str(row.get("sound_id", "unknown")),
            metadata={"captions": captions, "raw": row.to_dict()},
        )


def load_audioset(metadata_path: str, audio_root: str, split: str) -> Iterable[AudioSample]:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            audio_file = row.get("audio_path") or row.get("wav_path") or row.get("file_name") or row.get("youtube_id")
            if not audio_file:
                continue
            label_text = row.get("category") or row.get("display_name") or row.get("label") or "unknown"
            caption = row.get("text") or row.get("caption") or label_text
            if not str(audio_file).endswith((".wav", ".flac")) and not Path(str(audio_file)).exists():
                audio_file = f"{audio_file}.wav"
            audio_path = Path(str(audio_file))
            if not audio_path.exists():
                audio_path = Path(audio_root) / str(audio_file)
            yield AudioSample(
                sample_id=str(row.get("sample_id") or f"audioset-{row.get('youtube_id', audio_file)}"),
                dataset_name="audioset",
                split=split,
                audio_path=str(audio_path),
                text=caption,
                category=label_text,
                metadata={"raw": row},
            )


def load_esc50(metadata_path: str, audio_root: str, split: str) -> Iterable[AudioSample]:
    dataframe = pd.read_csv(metadata_path)
    if "audio_path" in dataframe.columns and "text" in dataframe.columns:
        for _, row in dataframe.iterrows():
            yield AudioSample(
                sample_id=str(row.get("sample_id", f"esc50-{row.name}")),
                dataset_name="esc50",
                split=str(row.get("fold", split)),
                audio_path=str(row["audio_path"]),
                text=str(row["text"]),
                category=str(row.get("category", "unknown")),
                metadata={"source_id": row.get("source_id"), "raw": row.to_dict()},
            )
        return
    for _, row in dataframe.iterrows():
        if split != "all" and int(row.get("fold", -1)) != int(split):
            continue
        audio_file = row.get("filename")
        if not isinstance(audio_file, str):
            continue
        category = str(row.get("category", "unknown"))
        yield AudioSample(
            sample_id=f"esc50-{audio_file}",
            dataset_name="esc50",
            split=str(row.get("fold", split)),
            audio_path=str(Path(audio_root) / audio_file),
            text=category,
            category=category,
            metadata={"target": row.get("target"), "raw": row.to_dict()},
        )


def _load_json_records(path: str) -> list[dict]:
    file_path = Path(path)
    if file_path.suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("data", "audios", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
    if not isinstance(data, list):
        raise DatasetLoaderError(f"Unsupported JSON structure in {path}")
    return data
