from __future__ import annotations

import csv
import json
import re
from pathlib import Path


INPUTS = [
    ("audioset", Path("data/full_datasets/audioset/metadata.csv")),
    ("clotho", Path("data/full_datasets/clotho_full/metadata.csv")),
    ("esc50", Path("data/full_datasets/esc50/metadata.csv")),
]

OUTPUT_PATH = Path("data/unified_audio_caption_dataset.jsonl")

# Keep basic punctuation while removing unusual symbols.
ALLOWED_PUNCTUATION = ".,!?;:'\"()-"
DISALLOWED_PATTERN = re.compile(rf"[^a-z0-9\s{re.escape(ALLOWED_PUNCTUATION)}]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_caption(text: str) -> str:
    normalized = text.strip().lower().replace("_", " ")
    normalized = DISALLOWED_PATTERN.sub("", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def iter_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_dataset, metadata_path in INPUTS:
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                caption = normalize_caption(row["text"])
                rows.append(
                    {
                        "audio_id": row["sample_id"],
                        "caption": caption,
                        "source_dataset": source_dataset,
                        "original_label": row["category"],
                    }
                )
    return rows


def main() -> None:
    rows = iter_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
