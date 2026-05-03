from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that downloaded dataset audio files are accessible.")
    parser.add_argument("--metadata", required=True, help="Path to dataset metadata file.")
    parser.add_argument("--limit", type=int, default=5, help="Number of files to verify.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    records = load_records(metadata_path)
    for record in records[: args.limit]:
        audio_path = Path(record["audio_path"])
        info = sf.info(str(audio_path))
        print(
            json.dumps(
                {
                    "sample_id": record["sample_id"],
                    "audio_path": str(audio_path),
                    "samplerate": info.samplerate,
                    "frames": info.frames,
                    "duration_seconds": round(info.frames / max(info.samplerate, 1), 4),
                }
            )
        )


def load_records(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
