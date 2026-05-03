from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


SOURCE_METADATA = Path("data/full_datasets/wavcaps_freesound/metadata.jsonl")
OUTPUT_DIR = Path("classified_dataset/wavcaps_freesound")
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
            "restaurant",
            "crowd",
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
            "classical-guitar",
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
            "notes",
            "soundtrack",
            "cinematic",
            "bells",
            "chimes",
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
            "pajaros",
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
            "engineer",
            "device",
            "tool",
            "furnace",
            "generator",
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
            "wrapping-paper",
            "smash",
            "bottle-breaking",
            "bottle-smash",
            "percussive",
            "grainy",
            "wood",
            "crash",
            "bang",
            "break",
        },
    ),
    (
        "electronic_synthetic",
        {
            "electronic",
            "synth",
            "sci-fi",
            "glitch",
            "glitchy",
            "sfx",
            "fx",
            "loop",
            "sample",
            "drone",
            "experimental",
            "soundscape",
            "ambient",
            "ambience",
            "atmosphere",
            "noise",
            "white-noise",
            "binaural",
            "generated",
        },
    ),
    (
        "ambience_environment",
        {
            "field-recording",
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
            "balcony",
            "quiet",
            "tranquilo",
            "background",
            "environment",
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


def classify_record(record: dict[str, str]) -> tuple[str, list[str], list[str]]:
    category_tokens = tokenize(record.get("category", ""))
    text_tokens = tokenize(record.get("text", ""))
    all_tokens = category_tokens | text_tokens

    matched_classes: list[str] = []
    matched_keywords_by_class: dict[str, list[str]] = {}
    for class_name, keywords in CLASS_RULES:
        matched = sorted(keyword for keyword in keywords if keyword in all_tokens)
        if matched:
            matched_classes.append(class_name)
            matched_keywords_by_class[class_name] = matched

    if matched_classes:
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
    return "other", ["other"], []


def poisoning_budget(count: int, rate: float) -> int:
    return int(math.floor(count * rate))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classified_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    with SOURCE_METADATA.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            primary_class, matched_classes, matched_keywords = classify_record(record)
            counts[primary_class] += 1
            classified_rows.append(
                {
                    **record,
                    "primary_class": primary_class,
                    "matched_classes": matched_classes,
                    "matched_keywords": matched_keywords,
                }
            )

    with OUTPUT_METADATA.open("w", encoding="utf-8") as handle:
        for row in classified_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    total = len(classified_rows)
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
        json.dumps(
            {
                "source": str(SOURCE_METADATA),
                "num_samples": total,
                "classes": summary_rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    OUTPUT_README.write_text(
        "\n".join(
            [
                "Separate classified WavCaps view for poisoning-rate experiments.",
                "This does not modify the original dataset.",
                "",
                "Files:",
                f"- {OUTPUT_METADATA.name}: original rows plus primary_class, matched_classes, matched_keywords",
                f"- {OUTPUT_SUMMARY_CSV.name}: class counts and poisoning budgets for 5%, 15%, 20%",
                f"- {OUTPUT_SUMMARY_JSON.name}: same summary in JSON format",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote classified metadata to {OUTPUT_METADATA}")
    print(f"Wrote class summary to {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
