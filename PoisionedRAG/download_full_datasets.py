from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

from utils.config import ensure_dir
from utils.logging import setup_logger


DATASET_SPECS = {
    "wavcaps_freesound": {
        "repo_id": "TwinkStart/wavcaps-freesound",
        "repo_type": "dataset",
        "file_prefixes": ["data/"],
        "metadata_format": "jsonl",
    },
    "clotho_full": {
        "repo_id": "CLAPv2/clotho_full",
        "repo_type": "dataset",
        "file_prefixes": [""],
        "metadata_format": "csv",
    },
    "esc50": {
        "repo_id": "ashraq/esc50",
        "repo_type": "dataset",
        "file_prefixes": ["data/"],
        "metadata_format": "csv",
    },
    "audioset": {
        "repo_id": "agkphysics/AudioSet",
        "repo_type": "dataset",
        "file_prefixes": ["data/"],
        "metadata_format": "csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and materialize full local audio datasets for Audio RAG.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_SPECS.keys()),
        default=list(DATASET_SPECS.keys()),
        help="Datasets to download.",
    )
    parser.add_argument(
        "--root",
        default="data/full_datasets",
        help="Root directory where datasets will be stored.",
    )
    parser.add_argument(
        "--cache-root",
        default="/scratch/shuhaoz/tmp_cache/audio_rag_hf",
        help="Scratch cache root for parquet downloads. Keeps raw files out of data/.",
    )
    parser.add_argument(
        "--max-parquet-files",
        type=int,
        default=None,
        help="Optional limit for debugging or staged downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    root = ensure_dir(args.root)
    for dataset_name in args.datasets:
        spec = DATASET_SPECS[dataset_name]
        logger.info("Downloading %s", dataset_name)
        dataset_root = ensure_dir(root / dataset_name)
        audio_root = ensure_dir(dataset_root / "audio")
        metadata_path = dataset_root / ("metadata.jsonl" if spec["metadata_format"] == "jsonl" else "metadata.csv")
        cache_root = ensure_dir(Path(args.cache_root) / dataset_name)
        parquet_files = download_parquet_files(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            cache_root=cache_root,
            file_prefixes=spec["file_prefixes"],
            max_parquet_files=args.max_parquet_files,
            logger=logger,
        )
        if args.max_parquet_files is not None:
            parquet_files = parquet_files[: args.max_parquet_files]
        records = materialize_dataset(dataset_name=dataset_name, parquet_files=parquet_files, audio_root=audio_root, logger=logger)
        write_metadata(metadata_path=metadata_path, records=records, metadata_format=spec["metadata_format"])
        logger.info("Finished %s with %s accessible local audio files", dataset_name, len(records))


def download_parquet_files(
    repo_id: str,
    repo_type: str,
    cache_root: Path,
    file_prefixes: list[str],
    max_parquet_files: int | None,
    logger,
) -> list[Path]:
    repo_files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
    parquet_names = [
        file_name
        for file_name in repo_files
        if file_name.endswith(".parquet") and any(file_name.startswith(prefix) for prefix in file_prefixes)
    ]
    parquet_names.sort()
    if max_parquet_files is not None:
        parquet_names = parquet_names[:max_parquet_files]
    downloaded: list[Path] = []
    for file_name in parquet_names:
        logger.info("Downloading file %s", file_name)
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=file_name,
            local_dir=str(cache_root),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        downloaded.append(Path(local_path))
    return downloaded


def materialize_dataset(dataset_name: str, parquet_files: list[Path], audio_root: Path, logger) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for parquet_file in parquet_files:
        logger.info("Processing %s", parquet_file.name)
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                record = convert_row_to_record(dataset_name=dataset_name, row=row, audio_root=audio_root)
                if record is not None:
                    records.append(record)
    return records


def convert_row_to_record(dataset_name: str, row: dict[str, Any], audio_root: Path) -> dict[str, Any] | None:
    audio = row.get("audio")
    if not isinstance(audio, dict):
        return None
    local_audio_path = write_audio_to_wav(dataset_name=dataset_name, row=row, audio=audio, audio_root=audio_root)
    if local_audio_path is None:
        return None
    if dataset_name == "wavcaps_freesound":
        sample_id = str(row["id"])
        text = str(row.get("caption") or row.get("description") or "")
        category = ",".join(row.get("tags") or [])
        split = "test"
    elif dataset_name == "clotho_full":
        sample_id = str(row.get("index"))
        text = str(row.get("text") or "")
        category = row.get("split") or "clotho"
        split = str(row.get("split") or "unknown")
    elif dataset_name == "esc50":
        sample_id = str(row.get("filename"))
        text = str(row.get("category") or "")
        category = str(row.get("category") or "")
        split = f"fold_{row.get('fold')}"
    elif dataset_name == "audioset":
        sample_id = str(row.get("video_id"))
        human_labels = row.get("human_labels") or []
        labels = row.get("labels") or []
        text = ", ".join(human_labels or labels)
        category = ",".join(human_labels or labels)
        split = "unknown"
    else:
        return None
    return {
        "sample_id": sample_id,
        "audio_path": str(local_audio_path),
        "text": text,
        "category": category,
        "split": split,
    }


def write_audio_to_wav(dataset_name: str, row: dict[str, Any], audio: dict[str, Any], audio_root: Path) -> Path | None:
    sample_stem = build_audio_stem(dataset_name=dataset_name, row=row)
    output_path = audio_root / f"{sample_stem}.wav"
    if output_path.exists():
        return output_path
    audio_bytes = audio.get("bytes")
    if not audio_bytes:
        return None
    with io.BytesIO(audio_bytes) as buffer:
        waveform, sampling_rate = sf.read(buffer)
    sf.write(output_path, waveform, sampling_rate)
    return output_path


def build_audio_stem(dataset_name: str, row: dict[str, Any]) -> str:
    if dataset_name == "wavcaps_freesound":
        return str(row.get("id"))
    if dataset_name == "clotho_full":
        return str(row.get("index"))
    if dataset_name == "esc50":
        return Path(str(row.get("filename"))).stem
    if dataset_name == "audioset":
        return str(row.get("video_id"))
    return "sample"


def write_metadata(metadata_path: Path, records: list[dict[str, Any]], metadata_format: str) -> None:
    ensure_dir(metadata_path.parent)
    if metadata_format == "jsonl":
        with metadata_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "audio_path", "text", "category", "split"])
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
