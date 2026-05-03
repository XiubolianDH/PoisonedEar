import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

WAVCAPS_PATH = REPO_ROOT / "malicious_dataset" / "all_classes_attack_metrics_v3" / "all_classes_long_table_v3.json"
AUDIOSET_PATH = REPO_ROOT / "malicious_dataset" / "audioset_all_classes_overall_asr" / "audioset_all_classes_overall_asr_long.json"

OVERALL_CLOTHO_PATH = REPO_ROOT / "4_dataset_overrallASR_result" / "clotho" / "overall_clotho.jsonl"
OVERALL_ESC50_PATH = REPO_ROOT / "4_dataset_overrallASR_result" / "esc50" / "overall_esc50.jsonl"

OUTPUTS = {
    "clotho": {
        "weights": {"wavcaps": 0.7, "audioset": 0.3},
        "dir": REPO_ROOT / "4_dataset_overrallASR_result" / "clotho",
        "overall_ref": OVERALL_CLOTHO_PATH,
    },
    "esc50": {
        "weights": {"wavcaps": 0.2, "audioset": 0.8},
        "dir": REPO_ROOT / "4_dataset_overrallASR_result" / "esc50",
        "overall_ref": OVERALL_ESC50_PATH,
    },
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_lookup(rows):
    lookup = {}
    for row in rows:
        lookup[(row["class_name"], row["poison_rate_pct"])] = row
    return lookup


def average_by_rate(long_rows):
    buckets = defaultdict(list)
    for row in long_rows:
        buckets[row["poison_rate_pct"]].append(row["Overall_ASR"])
    return {
        rate: sum(values) / len(values)
        for rate, values in sorted(buckets.items())
    }


def calibrate_rows(long_rows, overall_by_rate):
    buckets = defaultdict(list)
    for row in long_rows:
        buckets[row["poison_rate_pct"]].append(row)

    calibrated = []
    consistency = []

    for poison_rate_pct, rows in sorted(buckets.items()):
        current_avg = sum(row["Overall_ASR"] for row in rows) / len(rows)
        target_avg = overall_by_rate[poison_rate_pct]
        factor = 1.0 if current_avg == 0 else target_avg / current_avg

        adjusted_rows = []
        for row in rows:
            new_row = dict(row)
            new_row["Overall_ASR_raw"] = row["Overall_ASR"]
            new_row["calibration_factor"] = factor
            new_row["Overall_ASR"] = max(0.0, min(1.0, row["Overall_ASR"] * factor))
            adjusted_rows.append(new_row)

        adjusted_avg = sum(row["Overall_ASR"] for row in adjusted_rows) / len(adjusted_rows)
        consistency.append(
            {
                "dataset": rows[0]["dataset"],
                "poison_rate_pct": poison_rate_pct,
                "raw_avg_from_9_classes": current_avg,
                "overall_reference_value": target_avg,
                "calibration_factor": factor,
                "calibrated_avg_from_9_classes": adjusted_avg,
                "abs_diff_after_calibration": abs(adjusted_avg - target_avg),
            }
        )
        calibrated.extend(adjusted_rows)

    calibrated.sort(key=lambda x: (x["class_name"], x["poison_rate_pct"]))
    return calibrated, consistency


def main():
    wavcaps_rows = load_json(WAVCAPS_PATH)
    audioset_rows = load_json(AUDIOSET_PATH)

    wavcaps_lookup = build_lookup(wavcaps_rows)
    audioset_lookup = build_lookup(audioset_rows)

    class_names = sorted({row["class_name"] for row in wavcaps_rows})
    poison_rates = sorted({row["poison_rate_pct"] for row in wavcaps_rows})

    for dataset_name, cfg in OUTPUTS.items():
        out_dir = cfg["dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        weights = cfg["weights"]
        overall_ref = load_json(cfg["overall_ref"])
        overall_by_rate = {
            row["poison_rate_pct"]: row["Overall_ASR"] for row in overall_ref
        }

        long_rows = []
        grouped_rows = []

        for class_name in class_names:
            class_rows = []
            for poison_rate_pct in poison_rates:
                w = wavcaps_lookup[(class_name, poison_rate_pct)]
                a = audioset_lookup[(class_name, poison_rate_pct)]
                predicted_overall = (
                    weights["wavcaps"] * w["Overall_ASR"]
                    + weights["audioset"] * a["Overall_ASR"]
                )
                predicted_num_queries = round(
                    weights["wavcaps"] * w["num_queries"]
                    + weights["audioset"] * a["num_queries"]
                )
                predicted_poison_count = round(
                    weights["wavcaps"] * w["poison_count"]
                    + weights["audioset"] * a["poison_count"]
                )

                row = {
                    "dataset": dataset_name,
                    "class_name": class_name,
                    "poison_rate_pct": poison_rate_pct,
                    "poison_count": predicted_poison_count,
                    "num_queries": predicted_num_queries,
                    "Overall_ASR": predicted_overall,
                    "is_predicted": True,
                    "prediction_method": "weighted_interpolation_from_wavcaps_and_audioset_per_class",
                    "prediction_weights": weights,
                    "source_refs": {
                        "wavcaps": str(WAVCAPS_PATH),
                        "audioset": str(AUDIOSET_PATH),
                    },
                }
                long_rows.append(row)
                class_rows.append(
                    {
                        "poison_rate_pct": poison_rate_pct,
                        "poison_count": predicted_poison_count,
                        "num_queries": predicted_num_queries,
                        "Overall_ASR": predicted_overall,
                    }
                )

            grouped_rows.append(
                {
                    "dataset": dataset_name,
                    "class_name": class_name,
                    "is_predicted": True,
                    "prediction_method": "weighted_interpolation_from_wavcaps_and_audioset_per_class",
                    "prediction_weights": weights,
                    "source_refs": {
                        "wavcaps": str(WAVCAPS_PATH),
                        "audioset": str(AUDIOSET_PATH),
                    },
                    "results": class_rows,
                }
            )

        long_rows.sort(key=lambda x: (x["class_name"], x["poison_rate_pct"]))
        grouped_rows.sort(key=lambda x: x["class_name"])

        calibrated_long_rows, consistency_rows = calibrate_rows(long_rows, overall_by_rate)

        calibrated_grouped_rows = []
        grouped_lookup = defaultdict(list)
        for row in calibrated_long_rows:
            grouped_lookup[row["class_name"]].append(
                {
                    "poison_rate_pct": row["poison_rate_pct"],
                    "poison_count": row["poison_count"],
                    "num_queries": row["num_queries"],
                    "Overall_ASR": row["Overall_ASR"],
                    "Overall_ASR_raw": row["Overall_ASR_raw"],
                    "calibration_factor": row["calibration_factor"],
                }
            )

        for class_name in class_names:
            calibrated_grouped_rows.append(
                {
                    "dataset": dataset_name,
                    "class_name": class_name,
                    "is_predicted": True,
                    "prediction_method": "weighted_interpolation_from_wavcaps_and_audioset_per_class_then_rate_calibrated_to_dataset_overall",
                    "prediction_weights": weights,
                    "source_refs": {
                        "wavcaps": str(WAVCAPS_PATH),
                        "audioset": str(AUDIOSET_PATH),
                        "overall_reference": str(cfg["overall_ref"]),
                    },
                    "results": grouped_lookup[class_name],
                }
            )

        (out_dir / f"{dataset_name}_all_classes_overall_asr_long_predicted.json").write_text(
            json.dumps(long_rows, indent=2),
            encoding="utf-8",
        )
        (out_dir / f"{dataset_name}_all_classes_overall_asr_grouped_predicted.json").write_text(
            json.dumps(grouped_rows, indent=2),
            encoding="utf-8",
        )
        (out_dir / f"{dataset_name}_all_classes_prediction_consistency.json").write_text(
            json.dumps(consistency_rows, indent=2),
            encoding="utf-8",
        )
        (out_dir / f"{dataset_name}_all_classes_overall_asr_long_predicted_calibrated.json").write_text(
            json.dumps(calibrated_long_rows, indent=2),
            encoding="utf-8",
        )
        (out_dir / f"{dataset_name}_all_classes_overall_asr_grouped_predicted_calibrated.json").write_text(
            json.dumps(calibrated_grouped_rows, indent=2),
            encoding="utf-8",
        )

        print(out_dir / f"{dataset_name}_all_classes_overall_asr_long_predicted.json")
        print(out_dir / f"{dataset_name}_all_classes_overall_asr_grouped_predicted.json")
        print(out_dir / f"{dataset_name}_all_classes_prediction_consistency.json")
        print(out_dir / f"{dataset_name}_all_classes_overall_asr_long_predicted_calibrated.json")
        print(out_dir / f"{dataset_name}_all_classes_overall_asr_grouped_predicted_calibrated.json")


if __name__ == "__main__":
    main()
