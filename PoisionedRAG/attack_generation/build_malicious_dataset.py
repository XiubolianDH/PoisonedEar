from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a malicious audio-text dataset from generated CDAB captions.")
    parser.add_argument("--input-csv", required=True, help="Path to generated CDAB CSV.")
    parser.add_argument("--output-root", required=True, help="Output dataset root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    repo_root = input_csv.parents[2]
    output_root = Path(args.output_root).resolve()
    audio_dir = output_root / "audio"
    metadata_path = output_root / "metadata.json"

    output_root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    missing_audio: list[dict[str, str]] = []

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            source_audio_path = (row.get("audio_path") or "").strip()
            adversarial_caption = (row.get("adversarial_caption") or "").strip()
            if not sample_id or not source_audio_path or not adversarial_caption:
                continue

            source_path = Path(source_audio_path)
            if not source_path.is_absolute():
                source_path = (repo_root / source_path).resolve()
            if not source_path.exists():
                missing_audio.append(
                    {
                        "sample_id": sample_id,
                        "source_audio_path": str(source_path),
                    }
                )
                continue

            destination_path = audio_dir / source_path.name
            if not destination_path.exists():
                shutil.copy2(source_path, destination_path)

            records.append(
                {
                    "sample_id": sample_id,
                    "dataset": (row.get("dataset") or "").strip(),
                    "audio_file": destination_path.name,
                    "audio_path": str(Path("audio") / destination_path.name),
                    "clean_caption": (row.get("clean_caption") or "").strip(),
                    "acoustic_description": (row.get("acoustic_description") or "").strip(),
                    "adversarial_caption": adversarial_caption,
                }
            )

    payload = {
        "dataset_name": output_root.name,
        "source_csv": str(input_csv),
        "num_samples": len(records),
        "num_missing_audio": len(missing_audio),
        "samples": records,
    }
    if missing_audio:
        payload["missing_audio"] = missing_audio

    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built malicious dataset at {output_root}")
    print(f"Samples copied: {len(records)}")
    print(f"Metadata: {metadata_path}")
    if missing_audio:
        print(f"Missing audio files: {len(missing_audio)}")


if __name__ == "__main__":
    main()
