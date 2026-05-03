from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("AUDIO_RAG_PYTHON", "/home/shuhaoz/miniconda3/envs/audio_rag/bin/python"))
RUNNER = REPO_ROOT / "run_cross_dataset_wavcaps_attack_eval.py"


MODEL_SPECS = {
    "gpt_4o_audio": {
        "provider": "openai",
        "model_name": "gpt-4o-audio-preview",
        "display_name": "gpt-4o-audio",
        "base_config": "configs/openai_gpt4o_audio.yaml",
    },
    "gpt_4o_mini_audio": {
        "provider": "openai",
        "model_name": "gpt-4o-mini-audio-preview",
        "display_name": "GPT-4o-mini-audio",
        "base_config": "configs/openai_gpt4o_mini_audio.yaml",
    },
    "gpt_audio": {
        "provider": "openai",
        "model_name": "gpt-audio",
        "display_name": "GPT-audio",
        "base_config": "configs/openai_gpt4o_audio.yaml",
    },
    "gpt_audio_mini": {
        "provider": "openai",
        "model_name": "gpt-audio-mini",
        "display_name": "GPT-audio-mini",
        "base_config": "configs/openai_gpt4o_mini_audio.yaml",
    },
    "gemini_2_0_flash": {
        "provider": "gemini",
        "model_name": "gemini-2.0-flash",
        "display_name": "Google-gemini-2.0-flash",
        "base_config": "configs/gemini_2_5_flash.yaml",
    },
    "gemini_2_0_flash_lite": {
        "provider": "gemini",
        "model_name": "gemini-2.0-flash-lite",
        "display_name": "Google-gemini-2.0-flash-lite",
        "base_config": "configs/gemini_2_5_flash.yaml",
    },
    "gemini_2_5_flash": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "display_name": "Google-gemini-2.5-flash",
        "base_config": "configs/gemini_2_5_flash.yaml",
    },
    "gemini_2_5_pro": {
        "provider": "gemini",
        "model_name": "gemini-2.5-pro",
        "display_name": "Google-gemini-2.5-pro",
        "base_config": "configs/gemini_2_5_pro.yaml",
    },
    "gemini_2_5_flash_lite": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash-lite",
        "display_name": "Google-gemini-2.5-flash-lite",
        "base_config": "configs/gemini_2_5_flash.yaml",
    },
    "gemini_3_pro_preview": {
        "provider": "gemini",
        "model_name": "gemini-3-pro-preview",
        "display_name": "Google-Gemini-3-pro-preview",
        "base_config": "configs/gemini_2_5_pro.yaml",
    },
    "gemini_3_flash_preview": {
        "provider": "gemini",
        "model_name": "gemini-3-flash-preview",
        "display_name": "Gemini-3-flash-preview",
        "base_config": "configs/gemini_2_5_flash.yaml",
    },
    "qwen3_omni_flash": {
        "provider": "qwen",
        "model_name": "qwen3-omni-flash",
        "display_name": "Qwen3-Omni-Flash",
        "base_config": "configs/openai_gpt4o_mini_audio.yaml",
    },
}


DATASET_SPECS = {
    "wavcaps": {
        "slug": "wavcaps_electronic_synthetic_15pct",
        "poisoned_manifest": "attack_experiments/wavcaps_electronic_synthetic_rates/rate_15pct/wavcaps_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/unified_electronic_synthetic_35/final_dataset/query_subset_20.json",
    },
    "audioset": {
        "slug": "audioset_music_instrument_15pct",
        "poisoned_manifest": "attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json",
    },
    "clotho": {
        "slug": "clotho_vehicle_transport_15pct",
        "poisoned_manifest": "attack_experiments/clotho_vehicle_transport_rates/rate_15pct/clotho_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/wavcaps_clotho_esc50_to_clotho_vehicle_transport_556/final_dataset/query_subset_20.json",
    },
    "esc50": {
        "slug": "esc50_animal_bio_15pct",
        "poisoned_manifest": "attack_experiments/esc50_animal_bio_rates/rate_15pct/esc50_poisoned_manifest.jsonl",
        "malicious_metadata": "malicious_dataset/wavcaps_clotho_esc50_to_esc50_animal_bio_72/final_dataset/query_subset_20.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small multimodel ASR-R / ASR-G|R / Overall_ASR evaluation.")
    parser.add_argument("--output-root", default="outputs/multimodel_asr_small_20260421")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), default=list(DATASET_SPECS))
    parser.add_argument("--retriever", choices=["clap", "audioclip", "panns", "wav2vec2"], default="clap")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def write_config(model_key: str, output_root: Path) -> Path:
    spec = MODEL_SPECS[model_key]
    base_path = REPO_ROOT / spec["base_config"]
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config["models"]["victim_provider"] = spec["provider"]
    config["models"]["victim_model"] = spec["model_name"]
    config["models"]["temperature"] = 0.0
    config["models"]["max_tokens"] = 256
    config.setdefault("retrieval", {})["top_k"] = 5
    config_dir = output_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{model_key}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def append_failure(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_existing_clap_cache(cache_dir: Path) -> None:
    src = (
        Path("/scratch/shuhaoz/CU_Project/PoisionedRAG_gpt_family_eval_20260420")
        / "outputs"
        / "shared_retriever_index_cache_15pct_4datasets_4retrievers"
    )
    if not src.exists():
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = cache_dir / item.name
        if not target.exists():
            try:
                os.symlink(item, target)
            except OSError:
                shutil.copy2(item, target)


def main() -> None:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index_cache_dir = output_root / "index_cache"
    copy_existing_clap_cache(index_cache_dir)
    failure_log = output_root / "failed_runs.jsonl"
    rows: list[dict[str, object]] = []
    configs = {model_key: write_config(model_key, output_root) for model_key in args.models}

    for model_key in args.models:
        model_spec = MODEL_SPECS[model_key]
        for dataset_key in args.datasets:
            dataset_spec = DATASET_SPECS[dataset_key]
            run_dir = output_root / model_key / dataset_spec["slug"] / args.retriever
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_json = run_dir / "summary_k05.json"
            results_csv = run_dir / "results_k05.csv"
            error_log = run_dir / "errors_k05.jsonl"
            if summary_json.exists() and not args.force_rerun:
                summary = read_summary(summary_json)
            else:
                cmd = [
                    str(PYTHON),
                    str(RUNNER),
                    "--config",
                    str(configs[model_key]),
                    "--poisoned-manifest",
                    str((REPO_ROOT / dataset_spec["poisoned_manifest"]).resolve()),
                    "--malicious-metadata",
                    str((REPO_ROOT / dataset_spec["malicious_metadata"]).resolve()),
                    "--results-csv",
                    str(results_csv),
                    "--summary-json",
                    str(summary_json),
                    "--error-log",
                    str(error_log),
                    "--top-k",
                    str(args.top_k),
                    "--retriever-encoder",
                    args.retriever,
                    "--combo-label",
                    f"model={model_spec['display_name']} dataset={dataset_key} retriever={args.retriever} k={args.top_k}",
                    "--resume",
                    "--index-cache-dir",
                    str(index_cache_dir),
                    "--max-samples",
                    str(args.max_samples),
                ]
                if args.continue_on_error:
                    cmd.append("--continue-on-error")
                print(
                    "=" * 88
                    + f"\n[RUN-START] model={model_spec['display_name']} dataset={dataset_key} "
                    f"retriever={args.retriever} samples={args.max_samples} k={args.top_k}\n"
                    + f"output_dir={run_dir}\n"
                    + "=" * 88,
                    flush=True,
                )
                try:
                    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
                    summary = read_summary(summary_json)
                except subprocess.CalledProcessError as exc:
                    append_failure(
                        failure_log,
                        {
                            "model_key": model_key,
                            "model_name": model_spec["model_name"],
                            "dataset": dataset_key,
                            "retriever": args.retriever,
                            "returncode": exc.returncode,
                            "output_dir": str(run_dir),
                        },
                    )
                    print(f"[FAILED] model={model_key} dataset={dataset_key} returncode={exc.returncode}")
                    if not args.continue_on_error:
                        raise
                    continue
                except FileNotFoundError as exc:
                    append_failure(
                        failure_log,
                        {
                            "model_key": model_key,
                            "model_name": model_spec["model_name"],
                            "dataset": dataset_key,
                            "retriever": args.retriever,
                            "error": str(exc),
                            "output_dir": str(run_dir),
                        },
                    )
                    if not args.continue_on_error:
                        raise
                    continue

            rows.append(
                {
                    "model_key": model_key,
                    "display_name": model_spec["display_name"],
                    "provider": model_spec["provider"],
                    "model_name": model_spec["model_name"],
                    "dataset": dataset_key,
                    "dataset_slug": dataset_spec["slug"],
                    "retriever": args.retriever,
                    "top_k": args.top_k,
                    "max_samples": args.max_samples,
                    "num_queries": summary.get("num_queries"),
                    "asr_r": summary.get("asr_r_rate"),
                    "asr_g_given_r": summary.get("asr_g_given_r_rate"),
                    "overall_asr": summary.get("overall_asr"),
                    "avg_malicious_count": summary.get("avg_malicious_count"),
                    "recall_at_5": summary.get("recall_at_k"),
                    "summary_json": str(summary_json),
                }
            )

    summary_csv = output_root / "multimodel_asr_small_summary.csv"
    summary_json = output_root / "multimodel_asr_small_summary.json"
    fieldnames = [
        "model_key",
        "display_name",
        "provider",
        "model_name",
        "dataset",
        "dataset_slug",
        "retriever",
        "top_k",
        "max_samples",
        "num_queries",
        "asr_r",
        "asr_g_given_r",
        "overall_asr",
        "avg_malicious_count",
        "recall_at_5",
        "summary_json",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")
    print(summary_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
