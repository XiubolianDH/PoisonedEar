from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset.types import AudioSample, AudioStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed-poisoned manifest from clean + malicious datasets.")
    parser.add_argument("--clean-manifest", required=True, help="Path to clean manifest.jsonl")
    parser.add_argument("--malicious-metadata", required=True, help="Path to malicious_dataset metadata.json")
    parser.add_argument("--poisoned-manifest", required=True, help="Output mixed-poisoned manifest.jsonl")
    parser.add_argument("--wavcaps-poisoned-manifest", required=True, help="Output wavcaps-only poisoned manifest.jsonl")
    return parser.parse_args()


def load_manifest(path: Path) -> list[AudioSample]:
    samples: list[AudioSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            stats_raw = raw.pop("stats", None)
            sample = AudioSample(**raw)
            if stats_raw:
                sample.stats = AudioStats(**stats_raw)
            samples.append(sample)
    return samples


def write_manifest(path: Path, samples: list[AudioSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json(), ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    clean_manifest_path = Path(args.clean_manifest).resolve()
    malicious_metadata_path = Path(args.malicious_metadata).resolve()
    poisoned_manifest_path = Path(args.poisoned_manifest).resolve()
    wavcaps_poisoned_manifest_path = Path(args.wavcaps_poisoned_manifest).resolve()
    repo_root = clean_manifest_path.parents[2]

    clean_samples = load_manifest(clean_manifest_path)
    clean_by_dataset: dict[str, list[AudioSample]] = {}
    wavcaps_by_id: dict[str, AudioSample] = {}
    for sample in clean_samples:
        clean_by_dataset.setdefault(sample.dataset_name, []).append(sample)
        if sample.dataset_name == "wavcaps":
            wavcaps_by_id[sample.sample_id] = sample

    malicious_payload = json.loads(malicious_metadata_path.read_text(encoding="utf-8"))
    malicious_samples_raw = malicious_payload.get("samples", [])

    injected_samples: list[AudioSample] = []
    for row in malicious_samples_raw:
        source_id = str(row["sample_id"])
        base_sample = wavcaps_by_id.get(source_id)
        if base_sample is None:
            continue

        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (malicious_metadata_path.parent / audio_path).resolve()

        poisoned_sample = AudioSample(
            sample_id=f"poison_{source_id}",
            dataset_name="wavcaps",
            split="freesound_poisoned",
            audio_path=str(audio_path.relative_to(repo_root)),
            text=str(row["adversarial_caption"]),
            category=base_sample.category,
            metadata={
                "source": "malicious_dataset",
                "source_sample_id": source_id,
                "clean_caption": row.get("clean_caption", ""),
                "acoustic_description": row.get("acoustic_description", ""),
                "raw": row,
            },
            stats=base_sample.stats,
            is_poisoned=True,
            poison_metadata={
                "attack_method": "cdab",
                "malicious_dataset": str(malicious_metadata_path.relative_to(repo_root)),
                "source_sample_id": source_id,
                "clean_caption": row.get("clean_caption", ""),
                "acoustic_description": row.get("acoustic_description", ""),
            },
        )
        injected_samples.append(poisoned_sample)

    mixed_poisoned = list(clean_samples) + injected_samples

    poisoned_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    wavcaps_poisoned_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(poisoned_manifest_path, mixed_poisoned)

    wavcaps_clean = clean_by_dataset.get("wavcaps", [])
    wavcaps_poisoned = list(wavcaps_clean) + injected_samples
    write_manifest(wavcaps_poisoned_manifest_path, wavcaps_poisoned)

    print(f"Built mixed poisoned manifest: {poisoned_manifest_path}")
    print(f"Total clean samples: {len(clean_samples)}")
    print(f"Injected poisoned wavcaps samples: {len(injected_samples)}")
    print(f"Total mixed samples: {len(mixed_poisoned)}")
    print(f"WavCaps poisoned manifest: {wavcaps_poisoned_manifest_path}")


if __name__ == "__main__":
    main()
