from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("AUDIO_RAG_PYTHON", "/home/shuhaoz/miniconda3/envs/audio_rag/bin/python"))
RUNNER = REPO_ROOT / "run_cross_dataset_wavcaps_attack_eval.py"

MODEL_SPECS = {
    "gemini_2_5_flash": {
        "config": "configs/gemini_2_5_flash.yaml",
        "model_name": "gemini-2.5-flash",
    },
    "gemini_2_5_pro": {
        "config": "configs/gemini_2_5_pro.yaml",
        "model_name": "gemini-2.5-pro",
    },
}

DATASET_SPECS = {
    "wavcaps": {
        "slug": "wavcaps_electronic_synthetic_15pct",
        "poisoned_manifest": "attack_experiments/wavcaps_electronic_synthetic_rates/rate_15pct/wavcaps_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/unified_electronic_synthetic_35/final_dataset/query_subset_20.json",
        "poison_count": 24,
    },
    "audioset": {
        "slug": "audioset_music_instrument_15pct",
        "poisoned_manifest": "attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json",
        "poison_count": 116,
    },
    "clotho": {
        "slug": "clotho_vehicle_transport_15pct",
        "poisoned_manifest": "attack_experiments/clotho_vehicle_transport_rates/rate_15pct/clotho_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/wavcaps_clotho_esc50_to_clotho_vehicle_transport_556/final_dataset/query_subset_20.json",
        "poison_count": 556,
    },
    "esc50": {
        "slug": "esc50_animal_bio_15pct",
        "poisoned_manifest": "attack_experiments/esc50_animal_bio_rates/rate_15pct/esc50_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/wavcaps_clotho_esc50_to_esc50_animal_bio_72/final_dataset/query_subset_20.json",
        "poison_count": 72,
    },
}

RETRIEVERS = ["clap", "audioclip", "panns", "wav2vec2"]
RETRIEVER_BATCH_SIZE = {
    "audioclip": 1,
    "wav2vec2": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini family attack matrix with fixed k=5 and poison rate 15%.")
    parser.add_argument("--output-root", default="outputs/gemini_family_fixed_k5_15pct_4retrievers_20260421")
    parser.add_argument("--index-cache-dir", default="outputs/shared_retriever_index_cache_15pct_4datasets_4retrievers")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), default=list(DATASET_SPECS))
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=RETRIEVERS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true", help="Rerun combinations even if summary_k05.json already exists.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_failure(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failure_log = output_root / "failed_runs.jsonl"
    index_cache_dir = (REPO_ROOT / args.index_cache_dir).resolve()
    index_cache_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, object]] = []

    for model_key in args.models:
        model_spec = MODEL_SPECS[model_key]
        config_path = (REPO_ROOT / model_spec["config"]).resolve()
        for dataset_key in args.datasets:
            dataset_spec = DATASET_SPECS[dataset_key]
            poisoned_manifest = (REPO_ROOT / dataset_spec["poisoned_manifest"]).resolve()
            malicious_metadata = (REPO_ROOT / dataset_spec["malicious_metadata"]).resolve()
            for retriever in args.retrievers:
                run_dir = output_root / model_key / dataset_spec["slug"] / retriever
                run_dir.mkdir(parents=True, exist_ok=True)
                results_csv = run_dir / "results_k05.csv"
                summary_json = run_dir / "summary_k05.json"
                error_log = run_dir / "errors_k05.jsonl"
                if summary_json.exists() and not args.force_rerun:
                    summary = read_summary(summary_json)
                    aggregate_rows.append(
                        {
                            "model_key": model_key,
                            "model_name": model_spec["model_name"],
                            "dataset": dataset_key,
                            "retriever": retriever,
                            "poison_rate_pct": 15,
                            "poison_count": dataset_spec["poison_count"],
                            "top_k": args.top_k,
                            "num_queries": summary["num_queries"],
                            "overall_asr": summary["overall_asr"],
                            "recall_at_5": summary["recall_at_k"],
                            "summary_json": str(summary_json),
                        }
                    )
                    print(f"[SKIP] model={model_key} dataset={dataset_key} retriever={retriever} already completed")
                    continue

                cmd = [
                    str(PYTHON),
                    str(RUNNER),
                    "--config",
                    str(config_path),
                    "--poisoned-manifest",
                    str(poisoned_manifest),
                    "--malicious-metadata",
                    str(malicious_metadata),
                    "--results-csv",
                    str(results_csv),
                    "--summary-json",
                    str(summary_json),
                    "--error-log",
                    str(error_log),
                    "--top-k",
                    str(args.top_k),
                    "--retriever-encoder",
                    retriever,
                    "--combo-label",
                    f"model={model_spec['model_name']} dataset={dataset_key} retriever={retriever} k={args.top_k}",
                    "--resume",
                    "--index-cache-dir",
                    str(index_cache_dir),
                ]
                if args.max_samples is not None:
                    cmd.extend(["--max-samples", str(args.max_samples)])
                if args.continue_on_error:
                    cmd.append("--continue-on-error")
                if retriever in RETRIEVER_BATCH_SIZE:
                    cmd.extend(["--embedding-batch-size", str(RETRIEVER_BATCH_SIZE[retriever])])

                banner = (
                    "=" * 88
                    + f"\n[RUN-START] model={model_spec['model_name']} "
                    f"dataset={dataset_key} retriever={retriever} "
                    f"poison_rate=15% k={args.top_k}\n"
                    + f"output_dir={run_dir}\n"
                    + "=" * 88
                )
                print(banner, flush=True)
                try:
                    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
                    summary = read_summary(summary_json)
                    aggregate_rows.append(
                        {
                            "model_key": model_key,
                            "model_name": model_spec["model_name"],
                            "dataset": dataset_key,
                            "retriever": retriever,
                            "poison_rate_pct": 15,
                            "poison_count": dataset_spec["poison_count"],
                            "top_k": args.top_k,
                            "num_queries": summary["num_queries"],
                            "overall_asr": summary["overall_asr"],
                            "recall_at_5": summary["recall_at_k"],
                            "summary_json": str(summary_json),
                        }
                    )
                except subprocess.CalledProcessError as exc:
                    append_failure(
                        failure_log,
                        {
                            "model_key": model_key,
                            "model_name": model_spec["model_name"],
                            "dataset": dataset_key,
                            "retriever": retriever,
                            "poison_rate_pct": 15,
                            "top_k": args.top_k,
                            "returncode": exc.returncode,
                            "output_dir": str(run_dir),
                        },
                    )
                    print(f"[FAILED] model={model_key} dataset={dataset_key} retriever={retriever} returncode={exc.returncode}")
                    if not args.continue_on_error:
                        raise

    summary_csv = output_root / "gemini_family_overall_asr_recall_k5.csv"
    summary_json = output_root / "gemini_family_overall_asr_recall_k5.json"
    fieldnames = [
        "model_key",
        "model_name",
        "dataset",
        "retriever",
        "poison_rate_pct",
        "poison_count",
        "top_k",
        "num_queries",
        "overall_asr",
        "recall_at_5",
        "summary_json",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    summary_json.write_text(json.dumps(aggregate_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    print(summary_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
