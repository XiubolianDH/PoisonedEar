from __future__ import annotations

import json
import math
import random
from pathlib import Path


BASE_CLASSIFIED = Path("classified_dataset/wavcaps_freesound/metadata_classified.jsonl")
BASE_MANIFEST = Path("outputs/manifests/wavcaps_clean.jsonl")
MALICIOUS_METADATA = Path("malicious_dataset/unified_impact_material_10/final_dataset/metadata.json")
OUTPUT_DIR = Path("attack_experiments/wavcaps_impact_material_rates")
TARGET_CLASS = "impact_material"
TARGET_CLASS_SIZE = 51
RATES = [1, 5, 10, 15, 20]
SEED = 7


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classified_rows = load_jsonl(BASE_CLASSIFIED)
    manifest_rows = load_jsonl(BASE_MANIFEST)
    malicious_payload = json.loads(MALICIOUS_METADATA.read_text(encoding="utf-8"))
    malicious_samples = list(malicious_payload["samples"])

    rng = random.Random(SEED)
    ordered_malicious = list(malicious_samples)
    rng.shuffle(ordered_malicious)

    plan_rows: list[dict[str, object]] = []
    for rate in RATES:
        poison_count = min(len(ordered_malicious), int(math.floor(TARGET_CLASS_SIZE * (rate / 100.0))))
        subset = ordered_malicious[:poison_count]

        rate_dir = OUTPUT_DIR / f"rate_{rate:02d}pct"
        rate_dir.mkdir(parents=True, exist_ok=True)

        classified_out = rate_dir / "metadata_classified_poisoned.jsonl"
        manifest_out = rate_dir / "wavcaps_poisoned_manifest.jsonl"
        subset_out = rate_dir / "malicious_subset.json"

        augmented_classified = list(classified_rows)
        for row in subset:
            augmented_classified.append(
                {
                    "sample_id": f"poison_{row['sample_id']}",
                    "audio_path": str((MALICIOUS_METADATA.parent / row["audio_path"]).resolve()),
                    "text": row["adversarial_caption"],
                    "category": f"{TARGET_CLASS}_poison",
                    "split": "poison",
                    "primary_class": TARGET_CLASS,
                    "matched_classes": [TARGET_CLASS],
                    "matched_keywords": [f"{TARGET_CLASS}_poison"],
                    "is_poisoned": True,
                    "poison_metadata": {
                        "source_sample_id": row["sample_id"],
                        "clean_caption": row["clean_caption"],
                        "adversarial_caption": row["adversarial_caption"],
                        "acoustic_description": row.get("acoustic_description", ""),
                        "source_dataset": row["dataset"],
                    },
                }
            )

        with classified_out.open("w", encoding="utf-8") as handle:
            for row in augmented_classified:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        augmented_manifest = list(manifest_rows)
        for row in subset:
            augmented_manifest.append(
                {
                    "sample_id": f"poison_{row['sample_id']}",
                    "dataset_name": "wavcaps",
                    "split": "poison",
                    "audio_path": str((MALICIOUS_METADATA.parent / row["audio_path"]).resolve()),
                    "text": row["adversarial_caption"],
                    "category": f"{TARGET_CLASS}_poison",
                    "metadata": {
                        "source": "malicious_dataset",
                        "raw": row,
                    },
                    "stats": None,
                    "is_poisoned": True,
                    "poison_metadata": {
                        "source_sample_id": row["sample_id"],
                        "clean_caption": row["clean_caption"],
                        "adversarial_caption": row["adversarial_caption"],
                        "acoustic_description": row.get("acoustic_description", ""),
                        "source_dataset": row["dataset"],
                    },
                }
            )

        with manifest_out.open("w", encoding="utf-8") as handle:
            for row in augmented_manifest:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        subset_payload = {
            "dataset_name": f"wavcaps_{TARGET_CLASS}_{rate:02d}pct",
            "poison_rate_pct": rate,
            "num_samples": len(subset),
            "samples": subset,
        }
        subset_out.write_text(json.dumps(subset_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        plan_rows.append(
            {
                "poison_rate_pct": rate,
                "poison_count": len(subset),
                "classified_path": str(classified_out),
                "manifest_path": str(manifest_out),
                "malicious_subset_path": str(subset_out),
            }
        )

    (OUTPUT_DIR / "experiment_plan.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "target_class": TARGET_CLASS,
                "target_class_size": TARGET_CLASS_SIZE,
                "rates": plan_rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote poison-rate experiment files under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
