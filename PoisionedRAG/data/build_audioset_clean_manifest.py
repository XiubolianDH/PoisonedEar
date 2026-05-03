from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_METADATA = REPO_ROOT / "data/full_datasets/audioset/metadata.csv"
OUTPUT_MANIFEST = REPO_ROOT / "outputs/manifests/audioset_clean.jsonl"


def main() -> None:
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with SOURCE_METADATA.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "dataset_name": "audioset",
                    "split": "balanced",
                    "audio_path": str(row["audio_path"]),
                    "text": str(row["text"]),
                    "category": str(row["category"]),
                    "metadata": {"raw": row},
                    "stats": None,
                    "is_poisoned": False,
                    "poison_metadata": {},
                }
            )

    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
