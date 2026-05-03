from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("AUDIO_RAG_PYTHON", "/home/shuhaoz/miniconda3/envs/audio_rag/bin/python"))
TOPK_RUNNER = REPO_ROOT / "topk_impact_study" / "run_topk_impact_eval.py"
METRICS_RUNNER = REPO_ROOT / "topk_impact_study" / "build_topk_impact_metrics.py"

DATASET_SPECS = {
    "wavcaps": {
        "poisoned_manifest": "attack_experiments/wavcaps_electronic_synthetic_rates/rate_15pct/wavcaps_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/unified_electronic_synthetic_35/final_dataset/query_subset_20.json",
        "poison_count": 24,
        "slug": "wavcaps_electronic_synthetic_15pct",
    },
    "audioset": {
        "poisoned_manifest": "attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json",
        "poison_count": 116,
        "slug": "audioset_music_instrument_15pct",
    },
    "clotho": {
        "poisoned_manifest": "attack_experiments/clotho_vehicle_transport_rates/rate_15pct/clotho_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/wavcaps_clotho_esc50_to_clotho_vehicle_transport_556/final_dataset/query_subset_20.json",
        "poison_count": 556,
        "slug": "clotho_vehicle_transport_15pct",
    },
    "esc50": {
        "poisoned_manifest": "attack_experiments/esc50_animal_bio_rates/rate_15pct/esc50_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/wavcaps_clotho_esc50_to_esc50_animal_bio_72/final_dataset/query_subset_20.json",
        "poison_count": 72,
        "slug": "esc50_animal_bio_15pct",
    },
}

MODEL_SPECS = {
    "gpt": {
        "config": "configs/openai_gpt4o_audio.yaml",
        "label": "gpt4o_audio_preview",
    },
    "gemini": {
        "config": "configs/gemini_audio.yaml",
        "label": "gemini_2_5_flash",
    },
}

RETRIEVERS = ["clap", "audioclip", "panns", "wav2vec2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated retriever x model-family x dataset top-k experiments.")
    parser.add_argument("--output-root", default="outputs/retriever_family_matrix_20260420")
    parser.add_argument("--families", nargs="+", choices=sorted(MODEL_SPECS), default=sorted(MODEL_SPECS))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), default=sorted(DATASET_SPECS))
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=RETRIEVERS)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10, 15, 20])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failure_log = output_root / "failed_runs.jsonl"

    for family in args.families:
        model_spec = MODEL_SPECS[family]
        config_path = (REPO_ROOT / model_spec["config"]).resolve()
        for dataset in args.datasets:
            dataset_spec = DATASET_SPECS[dataset]
            poisoned_manifest = (REPO_ROOT / dataset_spec["poisoned_manifest"]).resolve()
            malicious_metadata = (REPO_ROOT / dataset_spec["malicious_metadata"]).resolve()
            for retriever in args.retrievers:
                run_output_dir = output_root / family / dataset_spec["slug"] / retriever
                run_output_dir.mkdir(parents=True, exist_ok=True)

                topk_cmd = [
                    str(PYTHON),
                    str(TOPK_RUNNER),
                    "--config",
                    str(config_path),
                    "--poisoned-manifest",
                    str(poisoned_manifest),
                    "--malicious-metadata",
                    str(malicious_metadata),
                    "--output-dir",
                    str(run_output_dir),
                    "--poison-rate-pct",
                    "15",
                    "--poison-count",
                    str(dataset_spec["poison_count"]),
                    "--retriever-encoder",
                    retriever,
                    "--k-values",
                    *[str(value) for value in args.k_values],
                ]
                if args.max_samples is not None:
                    topk_cmd.extend(["--max-samples", str(args.max_samples)])
                if args.continue_on_error:
                    topk_cmd.append("--continue-on-error")

                print(f"[MATRIX] family={family} dataset={dataset} retriever={retriever}")
                try:
                    subprocess.run(topk_cmd, cwd=REPO_ROOT, check=True)

                    metrics_cmd = [
                        str(PYTHON),
                        str(METRICS_RUNNER),
                        "--input-dir",
                        str(run_output_dir),
                        "--poison-rate-pct",
                        "15",
                        "--poison-count",
                        str(dataset_spec["poison_count"]),
                        "--k-values",
                        *[str(value) for value in args.k_values],
                    ]
                    subprocess.run(metrics_cmd, cwd=REPO_ROOT, check=True)
                except subprocess.CalledProcessError as exc:
                    payload = {
                        "family": family,
                        "dataset": dataset,
                        "retriever": retriever,
                        "config": str(config_path),
                        "output_dir": str(run_output_dir),
                        "returncode": exc.returncode,
                    }
                    with failure_log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                    print(f"[FAILED] family={family} dataset={dataset} retriever={retriever} returncode={exc.returncode}")
                    if not args.continue_on_error:
                        raise


if __name__ == "__main__":
    main()
