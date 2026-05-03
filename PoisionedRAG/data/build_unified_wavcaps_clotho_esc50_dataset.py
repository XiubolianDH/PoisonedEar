from __future__ import annotations

import csv
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


INPUT_CSVS = [
    ("clotho", REPO_ROOT / "data/full_datasets/clotho_full/metadata.csv"),
    ("esc50", REPO_ROOT / "data/full_datasets/esc50/metadata.csv"),
]
WAVCAPS_INPUT = ("wavcaps", REPO_ROOT / "data/full_datasets/wavcaps_freesound/metadata.jsonl")

OUTPUT_PATH = REPO_ROOT / "data/unified_wavcaps_clotho_esc50_dataset.jsonl"

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

    for source_dataset, metadata_path in INPUT_CSVS:
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "audio_id": row["sample_id"],
                        "audio_path": row["audio_path"],
                        "caption": normalize_caption(row["text"]),
                        "source_dataset": source_dataset,
                        "original_label": row["category"],
                        "split": row.get("split", ""),
                    }
                )

    source_dataset, metadata_path = WAVCAPS_INPUT
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "audio_id": row["sample_id"],
                    "audio_path": row["audio_path"],
                    "caption": normalize_caption(row["text"]),
                    "source_dataset": source_dataset,
                    "original_label": row["category"],
                    "split": row.get("split", ""),
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
