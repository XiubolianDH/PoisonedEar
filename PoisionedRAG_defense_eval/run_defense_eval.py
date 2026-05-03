from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defense.config import load_yaml
from defense.data import load_manifest, load_query_samples, resolve_legacy_path
from defense.generator import build_model_input
from defense.io_utils import append_jsonl, ensure_dir, utc_timestamp, write_csv, write_json
from defense.methods import DefenseRuntimeConfig, build_defense
from defense.metrics import TextSimilarityScorer, aggregate_results, build_sample_result
from defense.modeling import build_generation_model
from defense.registry import (
    DATASET_SPECS,
    DEFENSE_SPECS,
    MODEL_SPECS,
    dataset_keys,
    defense_keys,
    get_dataset_spec,
    get_model_spec,
    model_keys,
)
from defense.retrieval import AudioOnlyRetriever
from defense.types import SampleResult

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Defense evaluation pipeline for AudioRAG under retrieval poisoning.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-root", default="outputs/full_defense_eval")
    parser.add_argument("--models", nargs="+", choices=model_keys(), default=model_keys())
    parser.add_argument("--datasets", nargs="+", choices=dataset_keys(), default=dataset_keys())
    parser.add_argument("--defenses", nargs="+", choices=defense_keys(), default=defense_keys())
    parser.add_argument("--retriever", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--query-text", default="What is this sound?")
    parser.add_argument("--skip-generation-errors", action="store_true")
    parser.add_argument("--generator-backend", choices=["native", "local"], default=None)
    parser.add_argument("--text-similarity-backend", choices=["auto", "sentence_transformers", "lexical"], default=None)
    parser.add_argument("--write-run-manifest", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume and rerun every sample.")
    return parser.parse_args()


def _config_value(config: dict, *keys: str, default=None):
    cursor = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _fieldnames() -> list[str]:
    return [
        "model_key",
        "dataset_key",
        "defense_key",
        "num_queries",
        "recall_before",
        "recall_after",
        "asr",
        "avg_ppl_clean",
        "avg_ppl_malicious",
        "avg_malicious_count_before",
        "avg_malicious_count_after",
        "delta_asr_vs_no_defense",
    ]


def _summarize_dependency_state(
    generator_backend: str,
    text_similarity_backend: str,
    runtime: DefenseRuntimeConfig,
) -> dict[str, object]:
    required_imports = {
        "retrieval": ["numpy", "librosa", "sklearn"],
        "native_generation": ["openai"],
        "dense_metrics": ["sentence_transformers"],
        "hf_perplexity": ["transformers", "torch"],
    }
    status: dict[str, object] = {}
    for group, modules in required_imports.items():
        result = {}
        for module_name in modules:
            try:
                __import__(module_name)
                result[module_name] = "ok"
            except Exception as exc:  # noqa: BLE001
                result[module_name] = f"unavailable:{exc.__class__.__name__}"
        status[group] = result
    status["generator_backend"] = generator_backend
    status["text_similarity_backend"] = text_similarity_backend
    status["perplexity_backend"] = runtime.ppl_backend
    return status


def _load_jsonl_by_query_id(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows_by_query_id: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            query_id = payload.get("query_id")
            if query_id:
                rows_by_query_id[query_id] = payload
    return rows_by_query_id


def _load_sample_results(path: Path) -> list[SampleResult]:
    rows_by_query_id = _load_jsonl_by_query_id(path)
    return [SampleResult(**payload) for payload in rows_by_query_id.values()]


def _filter_sample_results_by_query_ids(sample_results: list[SampleResult], allowed_query_ids: set[str]) -> list[SampleResult]:
    return [item for item in sample_results if item.query_id in allowed_query_ids]


def _write_progress_snapshot(
    output_root: Path,
    total_units: int,
    completed_units: int,
    current: dict[str, str] | None = None,
) -> None:
    payload = {
        "updated_at_utc": utc_timestamp(),
        "total_units": total_units,
        "completed_units": completed_units,
        "remaining_units": max(total_units - completed_units, 0),
        "percent_complete": (float(completed_units) / float(total_units)) if total_units else 1.0,
        "current": current or {},
    }
    write_json(output_root / "progress.json", payload)


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    config = load_yaml(project_root / args.config)
    generator_backend = str(args.generator_backend or _config_value(config, "study", "generator_backend", default="native"))
    retriever_name = str(args.retriever or _config_value(config, "study", "retriever", default="clap"))
    top_k = int(args.top_k if args.top_k is not None else _config_value(config, "study", "top_k", default=5))
    dataset_sample_limits = dict(_config_value(config, "study", "dataset_sample_limits", default={}) or {})
    dataset_metadata_overrides = dict(_config_value(config, "study", "dataset_metadata_overrides", default={}) or {})
    text_similarity_backend = str(
        args.text_similarity_backend or _config_value(config, "metrics", "text_similarity_backend", default="auto")
    )
    runtime = DefenseRuntimeConfig(
        paraphrase_backend=str(config["defense"]["paraphrase"]["backend"]),
        paraphrase_model=str(config["defense"]["paraphrase"]["model_name"]),
        paraphrase_api_env=str(config["defense"]["paraphrase"]["api_key_env"]),
        ppl_backend=str(config["defense"]["perplexity"]["backend"]),
        ppl_model=str(config["defense"]["perplexity"]["model_name"]),
        ppl_threshold=float(config["defense"]["perplexity"]["threshold"]),
    )

    output_root = ensure_dir((project_root / args.output_root).resolve())
    resume_enabled = not args.no_resume
    similarity_scorer = TextSimilarityScorer(
        str(config["metrics"]["text_similarity_model"]),
        backend=text_similarity_backend,
    )
    all_sample_results = []
    dataset_query_samples: dict[str, list] = {}
    existing_run_results: dict[tuple[str, str, str], list[SampleResult]] = {}
    existing_completed_query_ids: dict[tuple[str, str, str], set[str]] = {}
    completed_progress_units = 0
    total_progress_units = 0
    run_manifest = {
        "started_at_utc": utc_timestamp(),
        "config_path": str((project_root / args.config).resolve()),
        "output_root": str(output_root),
        "generator_backend": generator_backend,
        "text_similarity_backend": similarity_scorer.backend,
        "models": list(args.models),
        "datasets": list(args.datasets),
        "defenses": list(args.defenses),
        "retriever": retriever_name,
        "top_k": top_k,
        "max_samples": args.max_samples,
        "dataset_sample_limits": dataset_sample_limits,
        "dataset_metadata_overrides": dataset_metadata_overrides,
        "dependency_state": _summarize_dependency_state(generator_backend, similarity_scorer.backend, runtime),
        "env_presence": {
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
            "ANYGPT_API_BASE": bool(os.getenv("ANYGPT_API_BASE")),
            "ANYGPT_API_KEY": bool(os.getenv("ANYGPT_API_KEY")),
            "QWEN_API_BASE": bool(os.getenv("QWEN_API_BASE")),
            "QWEN_API_KEY": bool(os.getenv("QWEN_API_KEY")),
        },
        "combination_count": len(args.models) * len(args.datasets) * len(args.defenses),
    }
    if args.write_run_manifest:
        write_json(output_root / "run_manifest.json", run_manifest)

    for dataset_key in args.datasets:
        dataset_spec = get_dataset_spec(dataset_key)
        metadata_override = dataset_metadata_overrides.get(dataset_key)
        metadata_path = resolve_legacy_path(str(metadata_override or dataset_spec.malicious_metadata))
        query_samples = load_query_samples(metadata_path, default_user_query=args.query_text)
        if args.max_samples is not None:
            query_samples = query_samples[: args.max_samples]
        elif dataset_key in dataset_sample_limits:
            query_samples = query_samples[: int(dataset_sample_limits[dataset_key])]
        dataset_query_samples[dataset_key] = query_samples
        total_progress_units += len(query_samples) * len(args.models) * len(args.defenses)

    for dataset_key in args.datasets:
        dataset_spec = get_dataset_spec(dataset_key)
        expected_query_ids = {sample.query_id for sample in dataset_query_samples[dataset_key]}
        for model_key in args.models:
            for defense_key in args.defenses:
                run_dir = output_root / model_key / dataset_spec.slug / defense_key / retriever_name
                result_jsonl = run_dir / "sample_results.jsonl"
                summary_csv = run_dir / "summary.csv"
                run_key = (model_key, dataset_key, defense_key)
                if not resume_enabled:
                    existing_run_results[run_key] = []
                    existing_completed_query_ids[run_key] = set()
                    continue
                existing_results = _filter_sample_results_by_query_ids(
                    _load_sample_results(result_jsonl),
                    expected_query_ids,
                )
                existing_run_results[run_key] = existing_results
                existing_ids = {item.query_id for item in existing_results}
                if summary_csv.exists() and expected_query_ids.issubset(existing_ids):
                    existing_completed_query_ids[run_key] = set(expected_query_ids)
                    completed_progress_units += len(expected_query_ids)
                else:
                    existing_completed_query_ids[run_key] = existing_ids
                    completed_progress_units += len(existing_ids)
                all_sample_results.extend(existing_results)

    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=total_progress_units,
            initial=completed_progress_units,
            desc="Defense eval",
            unit="sample",
            dynamic_ncols=True,
        )
    _write_progress_snapshot(output_root, total_progress_units, completed_progress_units)

    for dataset_key in args.datasets:
        dataset_spec = get_dataset_spec(dataset_key)
        manifest_path = resolve_legacy_path(dataset_spec.poisoned_manifest)
        retrieval_samples = load_manifest(manifest_path)
        query_samples = dataset_query_samples[dataset_key]

        retriever = AudioOnlyRetriever(
            encoder_name=retriever_name,
            checkpoint_overrides=config["retrieval"].get("checkpoint_overrides", {}),
            similarity=str(config["retrieval"].get("similarity", "cosine")),
        )
        retriever.build_index(retrieval_samples, batch_size=args.embedding_batch_size)

        for model_key in args.models:
            model_spec = get_model_spec(model_key)
            try:
                generation_model = build_generation_model(model_spec, backend_override=generator_backend)
            except Exception as exc:  # noqa: BLE001
                append_jsonl(
                    output_root / "model_init_errors.jsonl",
                    {
                        "model_key": model_key,
                        "dataset_key": dataset_key,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                if args.skip_generation_errors:
                    continue
                raise

            for defense_key in args.defenses:
                defense = build_defense(defense_key, runtime)
                run_dir = ensure_dir(output_root / model_key / dataset_spec.slug / defense_key / retriever_name)
                result_jsonl = run_dir / "sample_results.jsonl"
                error_jsonl = run_dir / "errors.jsonl"
                run_key = (model_key, dataset_key, defense_key)
                prior_results = existing_run_results.get(run_key, [])
                completed_query_ids = set(existing_completed_query_ids.get(run_key, set()))
                run_results = list(prior_results)
                write_json(
                    run_dir / "run_info.json",
                    {
                        "model_key": model_key,
                        "dataset_key": dataset_key,
                        "dataset_slug": dataset_spec.slug,
                        "defense_key": defense_key,
                        "retriever": retriever_name,
                        "top_k": top_k,
                        "max_samples": args.max_samples,
                        "dataset_sample_limit": dataset_sample_limits.get(dataset_key),
                        "metadata_path": str(metadata_path),
                        "generator_backend": generator_backend,
                        "text_similarity_backend": similarity_scorer.backend,
                        "resume_enabled": resume_enabled,
                        "resume_completed_queries": len(completed_query_ids),
                        "resume_total_queries": len(query_samples),
                        "started_at_utc": utc_timestamp(),
                    },
                )

                if len(completed_query_ids) == len(query_samples):
                    continue

                for sample in query_samples:
                    if sample.query_id in completed_query_ids:
                        continue
                    if progress_bar is not None:
                        progress_bar.set_postfix_str(
                            f"model={model_key} dataset={dataset_key} defense={defense_key} query={sample.query_id}"
                        )
                    current_progress = {
                        "model_key": model_key,
                        "dataset_key": dataset_key,
                        "defense_key": defense_key,
                        "query_id": sample.query_id,
                    }
                    _write_progress_snapshot(
                        output_root,
                        total_progress_units,
                        completed_progress_units,
                        current=current_progress,
                    )
                    retrieved_before = retriever.retrieve(audio_path=sample.audio_path, top_k=top_k)
                    try:
                        defense_result = defense.apply(sample.user_query, retrieved_before)
                        model_input = build_model_input(
                            audio_path=sample.audio_path,
                            user_query=defense_result.rewritten_query,
                            context_captions=[item.text for item in defense_result.kept_contexts],
                        )
                        output = generation_model.generate(model_input)
                        sample_result = build_sample_result(
                            model_key=model_key,
                            dataset_key=dataset_key,
                            defense_key=defense_key,
                            query_id=sample.query_id,
                            audio_path=sample.audio_path,
                            original_query=sample.user_query,
                            effective_query=defense_result.rewritten_query,
                            clean_caption=sample.clean_caption,
                            adversarial_caption=sample.adversarial_caption,
                            response=output.text,
                            top_k_requested=top_k,
                            total_injected_adversarial=dataset_spec.poison_count,
                            retrieved_before=retrieved_before,
                            retrieved_after=defense_result.kept_contexts,
                            defense_metadata=defense_result.metadata,
                            similarity_scorer=similarity_scorer,
                        )
                        run_results.append(sample_result)
                        all_sample_results.append(sample_result)
                        completed_query_ids.add(sample.query_id)
                        completed_progress_units += 1
                        append_jsonl(result_jsonl, sample_result.to_json())
                    except Exception as exc:  # noqa: BLE001
                        append_jsonl(
                            error_jsonl,
                            {
                                "model_key": model_key,
                                "dataset_key": dataset_key,
                                "defense_key": defense_key,
                                "query_id": sample.query_id,
                                "error_type": exc.__class__.__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        if not args.skip_generation_errors:
                            raise
                    finally:
                        if progress_bar is not None:
                            progress_bar.update(1)
                        _write_progress_snapshot(
                            output_root,
                            total_progress_units,
                            completed_progress_units,
                            current=current_progress,
                        )

                aggregates = aggregate_results(run_results)
                rows = [item.to_json() for item in aggregates]
                write_json(run_dir / "summary.json", rows)
                write_csv(run_dir / "summary.csv", rows, _fieldnames())

    if progress_bar is not None:
        progress_bar.close()

    global_aggregates = aggregate_results(all_sample_results)
    write_json(output_root / "global_summary.json", [item.to_json() for item in global_aggregates])
    global_rows = [item.to_json() for item in global_aggregates]
    write_csv(output_root / "global_summary.csv", global_rows, _fieldnames())
    write_csv(output_root / "leaderboard.csv", global_rows, _fieldnames())
    run_manifest["finished_at_utc"] = utc_timestamp()
    run_manifest["completed_rows"] = len(global_rows)
    write_json(output_root / "run_manifest.json", run_manifest)
    _write_progress_snapshot(output_root, total_progress_units, total_progress_units)


if __name__ == "__main__":
    main()
