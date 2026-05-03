from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


UNIFIED_DATASET = Path("data/unified_audio_caption_dataset.jsonl")
AUDIO_INDEX = Path("data/unified_audio_caption_bundle/audio_index.jsonl")
OUTPUT_DIR = Path("classified_dataset/unified_audio_caption_dataset")
OUTPUT_METADATA = OUTPUT_DIR / "metadata_classified.jsonl"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "class_summary.csv"
OUTPUT_SUMMARY_JSON = OUTPUT_DIR / "class_summary.json"
OUTPUT_README = OUTPUT_DIR / "README.txt"


CLASS_RULES: list[tuple[str, set[str]]] = [
    (
        "human_voice_speech",
        {
            "voice",
            "voices",
            "speech",
            "talk",
            "talking",
            "people",
            "person",
            "human",
            "laugh",
            "laughing",
            "male",
            "female",
            "girl",
            "boy",
            "interview",
            "outtake",
            "vocal",
            "singer",
            "singing",
            "conversation",
            "crowd",
            "speaker",
            "speaking",
        },
    ),
    (
        "music_instrument",
        {
            "music",
            "musical",
            "instrument",
            "orchestra",
            "orchestral",
            "guitar",
            "classical",
            "cello",
            "violoncello",
            "piano",
            "drum",
            "drums",
            "bass",
            "pluck",
            "plucked",
            "strum",
            "strings",
            "brass",
            "woodwind",
            "chord",
            "note",
            "notes",
            "soundtrack",
            "cinematic",
            "bells",
            "chimes",
            "violin",
            "flute",
            "trumpet",
            "saxophone",
        },
    ),
    (
        "animal_bio",
        {
            "bird",
            "birds",
            "birdsong",
            "dog",
            "cat",
            "goat",
            "cow",
            "frog",
            "insect",
            "bee",
            "animal",
            "chirping",
            "bark",
            "barking",
            "meow",
            "rooster",
            "pig",
            "sheep",
        },
    ),
    (
        "water_weather",
        {
            "water",
            "rain",
            "river",
            "dripping",
            "drip",
            "waves",
            "wave",
            "sea",
            "beach",
            "ocean",
            "shore",
            "storm",
            "wind",
            "thunder",
            "weather",
            "surf",
            "gush",
            "pouring",
            "stream",
        },
    ),
    (
        "vehicle_transport",
        {
            "traffic",
            "train",
            "car",
            "engine",
            "vehicle",
            "street",
            "walking",
            "steps",
            "footsteps",
            "arrival",
            "rail",
            "metro",
            "bus",
            "road",
            "truck",
            "motorcycle",
            "airplane",
            "aircraft",
            "helicopter",
        },
    ),
    (
        "machine_mechanical",
        {
            "machine",
            "mechanical",
            "motor",
            "industrial",
            "door",
            "gun",
            "firearm",
            "pistol",
            "metal",
            "electric",
            "device",
            "tool",
            "generator",
            "vacuum",
            "cleaner",
            "alarm",
            "detector",
            "printer",
        },
    ),
    (
        "impact_material",
        {
            "impact",
            "hit",
            "bump",
            "paper",
            "rip",
            "wrapping",
            "smash",
            "breaking",
            "percussive",
            "grainy",
            "wood",
            "crash",
            "bang",
            "break",
            "knock",
            "tap",
            "thud",
        },
    ),
    (
        "electronic_synthetic",
        {
            "electronic",
            "synth",
            "sci",
            "fi",
            "glitch",
            "glitchy",
            "loop",
            "sample",
            "drone",
            "experimental",
            "soundscape",
            "ambient",
            "ambience",
            "atmosphere",
            "noise",
            "white",
            "binaural",
            "generated",
            "beep",
            "buzz",
            "tone",
            "ping",
        },
    ),
    (
        "ambience_environment",
        {
            "field",
            "recording",
            "ambience",
            "ambiance",
            "atmosphere",
            "nature",
            "forest",
            "night",
            "outdoor",
            "interior",
            "landscape",
            "city",
            "quiet",
            "background",
            "environment",
            "room",
            "restaurant",
        },
    ),
]

CLASS_PRIORITY = [
    "vehicle_transport",
    "animal_bio",
    "music_instrument",
    "human_voice_speech",
    "water_weather",
    "machine_mechanical",
    "impact_material",
    "electronic_synthetic",
    "ambience_environment",
    "other",
]


def tokenize(text: str) -> set[str]:
    normalized = (
        text.lower()
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("(", " ")
        .replace(")", " ")
    )
    return {token.strip() for token in normalized.split() if token.strip()}


def classify(caption: str, original_label: str) -> tuple[str, list[str], list[str]]:
    all_tokens = tokenize(caption) | tokenize(original_label)
    matched_classes: list[str] = []
    matched_keywords_by_class: dict[str, list[str]] = {}
    for class_name, keywords in CLASS_RULES:
        matched = sorted(keyword for keyword in keywords if keyword in all_tokens)
        if matched:
            matched_classes.append(class_name)
            matched_keywords_by_class[class_name] = matched

    if not matched_classes:
        return "other", ["other"], []

    candidate_classes = [name for name in matched_classes if name != "ambience_environment"]
    if not candidate_classes:
        candidate_classes = matched_classes
    primary_class = sorted(
        candidate_classes,
        key=lambda class_name: (
            -len(matched_keywords_by_class[class_name]),
            CLASS_PRIORITY.index(class_name),
        ),
    )[0]
    matched_keywords: list[str] = []
    for class_name in matched_classes:
        matched_keywords.extend(matched_keywords_by_class[class_name][:4])
    return primary_class, matched_classes, matched_keywords


def poisoning_budget(count: int, rate: float) -> int:
    return int(math.floor(count * rate))


def load_audio_index() -> dict[tuple[str, str], dict[str, str]]:
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    with AUDIO_INDEX.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mapping[(row["source_dataset"], row["audio_id"])] = row
    return mapping


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_index = load_audio_index()

    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    with UNIFIED_DATASET.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            audio_info = audio_index.get((row["source_dataset"], row["audio_id"]), {})
            primary_class, matched_classes, matched_keywords = classify(row["caption"], row["original_label"])
            counts[primary_class] += 1
            rows.append(
                {
                    **row,
                    "bundle_audio_path": audio_info.get("bundle_audio_path", ""),
                    "original_audio_path": audio_info.get("original_audio_path", ""),
                    "primary_class": primary_class,
                    "matched_classes": matched_classes,
                    "matched_keywords": matched_keywords,
                }
            )

    with OUTPUT_METADATA.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    total = len(rows)
    summary_rows: list[dict[str, object]] = []
    for class_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        summary_rows.append(
            {
                "class_name": class_name,
                "count": count,
                "fraction": round(count / total, 6),
                "poison_5pct": poisoning_budget(count, 0.05),
                "poison_15pct": poisoning_budget(count, 0.15),
                "poison_20pct": poisoning_budget(count, 0.20),
            }
        )

    with OUTPUT_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "count", "fraction", "poison_5pct", "poison_15pct", "poison_20pct"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    OUTPUT_SUMMARY_JSON.write_text(
        json.dumps({"num_samples": total, "classes": summary_rows}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    OUTPUT_README.write_text(
        "\n".join(
            [
                "Classified view of the merged AudioSet + Clotho + ESC-50 dataset.",
                "Each row includes caption fields plus bundle_audio_path/original_audio_path for lookup.",
                "This file is separate from the original merged dataset.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {total} classified rows to {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
