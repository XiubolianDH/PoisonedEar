from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset.audio_stats import attach_stats
from dataset.loaders import load_dataset
from dataset.types import AudioSample, AudioStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a full WavCaps mixed-poisoned manifest from clean + malicious samples.")
    parser.add_argument("--wavcaps-metadata", required=True, help="Path to full WavCaps metadata.jsonl")
    parser.add_argument("--wavcaps-audio-root", required=True, help="Path to full WavCaps audio root")
    parser.add_argument("--malicious-metadata", required=True, help="Path to malicious_dataset metadata.json")
    parser.add_argument("--poisoned-manifest", required=True, help="Output mixed-poisoned manifest.jsonl")
    parser.add_argument("--wavcaps-clean-manifest", required=True, help="Output clean wavcaps manifest.jsonl")
    parser.add_argument("--wavcaps-poisoned-manifest", required=True, help="Output wavcaps poisoned manifest.jsonl")
    return parser.parse_args()


def write_manifest(path: Path, samples: list[AudioSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json(), ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()

    clean_wavcaps = list(
        load_dataset(
            "wavcaps",
            {
                "metadata_path": args.wavcaps_metadata,
                "audio_root": args.wavcaps_audio_root,
                "split": "freesound",
            },
        )
    )
    clean_wavcaps = attach_stats(clean_wavcaps)
    wavcaps_by_id = {sample.sample_id: sample for sample in clean_wavcaps}

    malicious_payload = json.loads(Path(args.malicious_metadata).read_text(encoding="utf-8"))
    injected_samples: list[AudioSample] = []
    for row in malicious_payload.get("samples", []):
        source_id = str(row["sample_id"])
        base_sample = wavcaps_by_id.get(source_id)
        if base_sample is None:
            continue
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (Path(args.malicious_metadata).resolve().parent / audio_path).resolve()

        injected_samples.append(
            AudioSample(
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
                stats=AudioStats(**base_sample.stats.__dict__) if base_sample.stats else None,
                is_poisoned=True,
                poison_metadata={
                    "attack_method": "cdab",
                    "source_sample_id": source_id,
                    "clean_caption": row.get("clean_caption", ""),
                    "acoustic_description": row.get("acoustic_description", ""),
                    "malicious_metadata": str(Path(args.malicious_metadata)),
                },
            )
        )

    mixed_poisoned = list(clean_wavcaps) + injected_samples
    write_manifest(Path(args.wavcaps_clean_manifest), clean_wavcaps)
    write_manifest(Path(args.wavcaps_poisoned_manifest), mixed_poisoned)
    write_manifest(Path(args.poisoned_manifest), mixed_poisoned)

    print(f"Built full clean wavcaps samples: {len(clean_wavcaps)}")
    print(f"Injected poisoned wavcaps samples: {len(injected_samples)}")
    print(f"Total mixed wavcaps samples: {len(mixed_poisoned)}")
    print(f"Poisoned manifest: {args.poisoned_manifest}")


if __name__ == "__main__":
    main()
