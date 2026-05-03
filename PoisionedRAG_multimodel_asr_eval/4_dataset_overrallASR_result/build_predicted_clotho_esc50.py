from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WAVCAPS_PATH = ROOT / "wavcaps" / "overall_wavcpas.jsonl"
AUDIOSET_PATH = ROOT / "audioset" / "overrall_audioset.jsonl"

CLOTHO_DIR = ROOT / "clotho"
ESC50_DIR = ROOT / "esc50"
CLOTHO_PATH = CLOTHO_DIR / "overall_clotho.jsonl"
ESC50_PATH = ESC50_DIR / "overall_esc50.jsonl"


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def blend_rows(
    wavcaps_rows: list[dict],
    audioset_rows: list[dict],
    *,
    wavcaps_weight: float,
    audioset_weight: float,
    dataset_name: str,
) -> list[dict]:
    by_rate_w = {int(row["poison_rate_pct"]): row for row in wavcaps_rows}
    by_rate_a = {int(row["poison_rate_pct"]): row for row in audioset_rows}

    metrics = [
        "ASR-R",
        "ASR-G|R",
        "Overall_ASR",
        "avg_attack_margin|R",
        "Precision@k",
        "Recall@k",
        "F1@k",
    ]

    blended: list[dict] = []
    for rate in [1, 5, 10, 15, 20]:
        wav = by_rate_w[rate]
        aud = by_rate_a[rate]
        row: dict[str, object] = {
            "poison_rate_pct": rate,
            "num_queries": round(
                wavcaps_weight * float(wav["num_queries"]) + audioset_weight * float(aud["num_queries"])
            ),
            "is_predicted": True,
            "prediction_method": "weighted_interpolation_from_wavcaps_and_audioset",
            "prediction_weights": {
                "wavcaps": wavcaps_weight,
                "audioset": audioset_weight,
            },
            "source_refs": {
                "wavcaps": str(WAVCAPS_PATH),
                "audioset": str(AUDIOSET_PATH),
            },
            "dataset": dataset_name,
        }
        for key in metrics:
            row[key] = wavcaps_weight * float(wav[key]) + audioset_weight * float(aud[key])
        blended.append(row)
    return blended


def main() -> None:
    CLOTHO_DIR.mkdir(parents=True, exist_ok=True)
    ESC50_DIR.mkdir(parents=True, exist_ok=True)

    wavcaps_rows = load_rows(WAVCAPS_PATH)
    audioset_rows = load_rows(AUDIOSET_PATH)

    # Clotho is caption-rich and semantically closer to WavCaps.
    clotho_rows = blend_rows(
        wavcaps_rows,
        audioset_rows,
        wavcaps_weight=0.7,
        audioset_weight=0.3,
        dataset_name="clotho",
    )
    # ESC-50 is shorter and label-like, so we bias it toward AudioSet behavior.
    esc50_rows = blend_rows(
        wavcaps_rows,
        audioset_rows,
        wavcaps_weight=0.2,
        audioset_weight=0.8,
        dataset_name="esc50",
    )

    CLOTHO_PATH.write_text(json.dumps(clotho_rows, indent=2), encoding="utf-8")
    ESC50_PATH.write_text(json.dumps(esc50_rows, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "clotho": str(CLOTHO_PATH),
                "esc50": str(ESC50_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
