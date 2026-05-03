from __future__ import annotations

import csv
import json
import os
from pathlib import Path


INPUTS = [
    ("audioset", Path("data/full_datasets/audioset/metadata.csv")),
    ("clotho", Path("data/full_datasets/clotho_full/metadata.csv")),
    ("esc50", Path("data/full_datasets/esc50/metadata.csv")),
]

OUTPUT_DIR = Path("data/unified_audio_caption_bundle")
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_INDEX = OUTPUT_DIR / "audio_index.jsonl"
OUTPUT_README = OUTPUT_DIR / "README.txt"


def safe_audio_name(source_dataset: str, sample_id: str, audio_path: str) -> str:
    suffix = Path(audio_path).suffix or ".wav"
    base_id = sample_id.replace("/", "_")
    return f"{source_dataset}__{base_id}{suffix}"


def main() -> None:
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUTPUT_INDEX.open("w", encoding="utf-8") as index_handle:
        for source_dataset, metadata_path in INPUTS:
            with metadata_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    source_audio = Path(row["audio_path"])
                    linked_name = safe_audio_name(source_dataset, row["sample_id"], row["audio_path"])
                    linked_path = OUTPUT_AUDIO_DIR / linked_name

                    if linked_path.exists() or linked_path.is_symlink():
                        linked_path.unlink()
                    os.symlink(source_audio.resolve(), linked_path)

                    index_handle.write(
                        json.dumps(
                            {
                                "audio_id": row["sample_id"],
                                "source_dataset": source_dataset,
                                "bundle_audio_path": str(linked_path),
                                "original_audio_path": row["audio_path"],
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    total += 1

    OUTPUT_README.write_text(
        "\n".join(
            [
                "Unified audio bundle for the merged AudioSet/Clotho/ESC-50 JSONL dataset.",
                "Files in audio/ are symbolic links to the original source WAV files.",
                "Use audio_index.jsonl to map each unified row to its bundled audio path.",
                f"Total bundled audio entries: {total}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {total} audio links to {OUTPUT_AUDIO_DIR}")
    print(f"Wrote bundle index to {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
