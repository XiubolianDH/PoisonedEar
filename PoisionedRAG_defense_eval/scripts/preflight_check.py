from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defense.config import load_yaml
from defense.data import resolve_legacy_path
from defense.registry import MODEL_SPECS, get_dataset_spec, get_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checker for the defense evaluation pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--generator-backend", choices=["native", "local"], default=None)
    return parser.parse_args()


def check_import(module_name: str) -> str:
    try:
        __import__(module_name)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{exc.__class__.__name__}"
    return "ok"


def main() -> None:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / args.config)
    configured_models = config.get("study", {}).get("models", [])
    models = list(args.models or configured_models or [spec.key for spec in MODEL_SPECS])
    datasets = list(args.datasets or config["study"]["datasets"])
    generator_backend = str(args.generator_backend or config.get("study", {}).get("generator_backend", "native"))

    print("Preflight check")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Generator backend: {generator_backend}")
    print("")

    print("Core imports:")
    for module_name in [
        "numpy",
        "librosa",
        "sklearn",
        "openai",
        "google.generativeai",
        "transformers",
        "torch",
        "sentence_transformers",
    ]:
        print(f"- {module_name}: {check_import(module_name)}")
    print("")

    print("Dataset files:")
    for dataset_key in datasets:
        spec = get_dataset_spec(dataset_key)
        manifest_path = resolve_legacy_path(spec.poisoned_manifest)
        metadata_path = resolve_legacy_path(spec.malicious_metadata)
        print(f"- {dataset_key}: manifest={'ok' if manifest_path.exists() else 'missing'}, metadata={'ok' if metadata_path.exists() else 'missing'}")
    print("")

    if generator_backend == "native":
        print("Model credentials:")
        for model_key in models:
            spec = get_model_spec(model_key)
            if spec.provider == "openai":
                envs = ["OPENAI_API_KEY"]
            elif spec.provider == "gemini":
                envs = ["GEMINI_API_KEY"]
            elif spec.provider == "openai_compatible":
                envs = [spec.api_base_env or "ANYGPT_API_BASE", spec.api_key_env or "ANYGPT_API_KEY"]
            elif spec.provider == "qwen":
                envs = [spec.api_base_env or "QWEN_API_BASE", spec.api_key_env or "QWEN_API_KEY"]
            else:
                envs = []
            statuses = ", ".join(f"{env}={'set' if os.getenv(env) else 'missing'}" for env in envs) or "no-env-required"
            print(f"- {model_key}: {statuses}")
    else:
        print("Model credentials:")
        print("- local backend selected: remote API credentials not required")


if __name__ == "__main__":
    main()
