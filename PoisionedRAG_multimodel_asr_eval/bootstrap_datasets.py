from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from utils.config import ensure_dir, load_yaml_config
from utils.logging import setup_logger


BOOTSTRAP_SOURCES = {
    "wavcaps": {
        "repo_id": "TwinkStart/wavcaps-freesound",
        "repo_type": "dataset",
        "filename": "data/test-00000-of-00030.parquet",
        "output_audio_dir": "data/bootstrap/wavcaps_audio",
        "output_metadata": "data/bootstrap/wavcaps_freesound_subset.jsonl",
        "format": "jsonl",
    },
    "clotho": {
        "repo_id": "DynamicSuperb/AudioSegmentRetrieval_Clotho",
        "repo_type": "dataset",
        "filename": "data/test-00000-of-00001.parquet",
        "output_audio_dir": "data/bootstrap/clotho_audio",
        "output_metadata": "data/bootstrap/clotho_subset.csv",
        "format": "csv",
    },
    "audioset": {
        "repo_id": "agkphysics/AudioSet",
        "repo_type": "dataset",
        "filename": "data/eval/00.parquet",
        "output_audio_dir": "data/bootstrap/audioset_audio",
        "output_metadata": "data/bootstrap/audioset_subset.csv",
        "format": "csv",
    },
    "esc50": {
        "repo_id": "ashraq/esc50",
        "repo_type": "dataset",
        "filename": "data/train-00000-of-00002-2f1ab7b824ec751f.parquet",
        "output_audio_dir": "data/bootstrap/esc50_audio",
        "output_metadata": "data/bootstrap/esc50_subset.csv",
        "format": "csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap small clean local subsets for Audio RAG datasets.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--samples-per-dataset", type=int, default=12, help="Number of clean samples to stage per dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger()
    cache_dir = ensure_dir(Path(config["data"]["cache_dir"]) / "hf")
    for dataset_name in ("wavcaps", "clotho", "audioset", "esc50"):
        source = BOOTSTRAP_SOURCES[dataset_name]
        logger.info("Bootstrapping %s", dataset_name)
        parquet_path = hf_hub_download(
            repo_id=source["repo_id"],
            repo_type=source["repo_type"],
            filename=source["filename"],
            cache_dir=str(cache_dir),
        )
        records = extract_records_from_parquet(
            dataset_name=dataset_name,
            parquet_path=Path(parquet_path),
            sample_limit=args.samples_per_dataset,
            audio_output_dir=ensure_dir(source["output_audio_dir"]),
            wavcaps_local_audio_root=Path(config["data"]["datasets"]["wavcaps"]["audio_root"]),
        )
        write_records(records=records, path=Path(source["output_metadata"]), format_name=source["format"])
        logger.info("Wrote %s records for %s to %s", len(records), dataset_name, source["output_metadata"])


def extract_records_from_parquet(
    dataset_name: str,
    parquet_path: Path,
    sample_limit: int,
    audio_output_dir: Path,
    wavcaps_local_audio_root: Path,
) -> list[dict[str, Any]]:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    records: list[dict[str, Any]] = []
    seen_audio: set[str] = set()
    for index, row in enumerate(rows):
        record = normalize_record(
            dataset_name=dataset_name,
            row=row,
            index=index,
            audio_output_dir=audio_output_dir,
            wavcaps_local_audio_root=wavcaps_local_audio_root,
        )
        if not record:
            continue
        if record["audio_path"] in seen_audio:
            continue
        seen_audio.add(record["audio_path"])
        records.append(record)
        if len(records) >= sample_limit:
            break
    return records


def normalize_record(
    dataset_name: str,
    row: dict[str, Any],
    index: int,
    audio_output_dir: Path,
    wavcaps_local_audio_root: Path,
) -> dict[str, Any] | None:
    audio_path = materialize_audio(
        dataset_name=dataset_name,
        row=row,
        index=index,
        audio_output_dir=audio_output_dir,
        wavcaps_local_audio_root=wavcaps_local_audio_root,
    )
    if audio_path is None:
        return None
    text = extract_text(dataset_name=dataset_name, row=row, index=index)
    if isinstance(text, list):
        text = ", ".join(str(item) for item in text[:5])
    text = str(text or f"{dataset_name} sample {index}")
    category = extract_category(dataset_name=dataset_name, row=row)
    if isinstance(category, list):
        category = ",".join(str(item) for item in category[:5])
    return {
        "sample_id": f"{dataset_name}-{index}",
        "audio_path": str(audio_path),
        "text": text,
        "category": str(category),
        "source_id": str(row.get("id") or row.get("filename") or row.get("ytid") or index),
    }


def extract_text(dataset_name: str, row: dict[str, Any], index: int) -> str:
    if dataset_name == "clotho":
        instruction = str(row.get("instruction") or "")
        marker = "Caption:"
        if marker in instruction:
            return instruction.split(marker, 1)[1].strip()
    if dataset_name == "audioset":
        human_labels = row.get("human_labels")
        if human_labels:
            return ", ".join(str(item) for item in human_labels)
    return str(
        row.get("caption")
        or row.get("description")
        or row.get("labels")
        or row.get("answer")
        or row.get("sentence")
        or row.get("category")
        or row.get("text")
        or f"{dataset_name} sample {index}"
    )


def extract_category(dataset_name: str, row: dict[str, Any]) -> str | list[str]:
    if dataset_name == "audioset" and row.get("human_labels"):
        return row["human_labels"]
    if dataset_name == "clotho" and row.get("label"):
        return str(row["label"])
    return (
        row.get("category")
        or row.get("label")
        or row.get("labels")
        or row.get("class_label")
        or row.get("target")
        or dataset_name
    )


def materialize_audio(
    dataset_name: str,
    row: dict[str, Any],
    index: int,
    audio_output_dir: Path,
    wavcaps_local_audio_root: Path,
) -> Path | None:
    audio = row.get("audio")
    if dataset_name == "wavcaps":
        for key in ("id", "filename", "audio_id", "wav_id"):
            value = row.get(key)
            if value is not None:
                local_candidate = wavcaps_local_audio_root / f"{value}.flac"
                if local_candidate.exists():
                    return local_candidate
        if isinstance(audio, dict):
            path_value = audio.get("path")
            if path_value:
                local_candidate = wavcaps_local_audio_root / Path(path_value).name
                if local_candidate.exists():
                    return local_candidate
    if isinstance(audio, dict):
        if audio.get("path") and Path(audio["path"]).exists():
            return Path(audio["path"])
        if audio.get("bytes"):
            filename = audio_output_dir / f"{dataset_name}_{index}.wav"
            filename.write_bytes(audio["bytes"])
            return filename
        if audio.get("array") is not None:
            sampling_rate = int(audio.get("sampling_rate") or 16000)
            filename = audio_output_dir / f"{dataset_name}_{index}.wav"
            sf.write(filename, audio["array"], sampling_rate)
            return filename
    for key in ("path", "file", "filename"):
        value = row.get(key)
        if isinstance(value, str) and Path(value).exists():
            candidate = Path(value)
            target = audio_output_dir / candidate.name
            if candidate.resolve() != target.resolve():
                shutil.copy2(candidate, target)
                return target
            return candidate
    return None


def write_records(records: list[dict[str, Any]], path: Path, format_name: str) -> None:
    ensure_dir(path.parent)
    if format_name == "jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
