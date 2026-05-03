import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_DATASETS_DIR = REPO_ROOT / "data" / "full_datasets"
CLASSIFIED_DIR = REPO_ROOT / "classified_dataset"
OUTPUT_PATH = REPO_ROOT / "4_dataset_overrallASR_result" / "dataset_and_class_counts.json"

CLASS_ORDER = [
    "other",
    "electronic_synthetic",
    "music_instrument",
    "water_weather",
    "human_voice_speech",
    "machine_mechanical",
    "vehicle_transport",
    "animal_bio",
    "impact_material",
    "ambience_environment",
]


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_class_counts(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["class_name"]: item["count"] for item in payload["classes"]}


def main() -> None:
    dataset_counts = {
        "wavcaps_freesound": count_jsonl_rows(FULL_DATASETS_DIR / "wavcaps_freesound" / "metadata.jsonl"),
        "clotho_full": count_csv_rows(FULL_DATASETS_DIR / "clotho_full" / "metadata.csv"),
        "audioset": count_csv_rows(FULL_DATASETS_DIR / "audioset" / "metadata.csv"),
        "esc50": count_csv_rows(FULL_DATASETS_DIR / "esc50" / "metadata.csv"),
    }

    merged_counts = load_class_counts(CLASSIFIED_DIR / "wavcaps_clotho_esc50" / "class_summary.json")
    audioset_counts = load_class_counts(CLASSIFIED_DIR / "audioset" / "class_summary.json")
    combined_counts = {
        class_name: merged_counts.get(class_name, 0) + audioset_counts.get(class_name, 0)
        for class_name in CLASS_ORDER
    }

    payload = {
        "dataset_counts": dataset_counts,
        "dataset_total": sum(dataset_counts.values()),
        "class_counts_10_classes": combined_counts,
        "class_total": sum(combined_counts.values()),
        "sources": {
            "full_datasets": str(FULL_DATASETS_DIR),
            "wavcaps_clotho_esc50_class_summary": str(CLASSIFIED_DIR / "wavcaps_clotho_esc50" / "class_summary.json"),
            "audioset_class_summary": str(CLASSIFIED_DIR / "audioset" / "class_summary.json"),
        },
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
