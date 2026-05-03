from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from dataset.loaders import load_dataset
from dataset.types import AudioSample
from evaluation.metrics import cosine_similarity, kl_divergence
from models.base import ModelInput, ModelOutput
from models.wrappers import build_victim_model
from rag.context import build_context_string
from retriever.base import AudioClassifier, AudioEmbeddingEncoder, RetrievalHit
from retriever.encoders import build_encoder
from retriever.index import FaissAudioRetriever
from utils.config import ensure_dir, load_yaml_config
from utils.logging import setup_logger


DEFAULT_K_VALUES = [1, 5, 10, 20]
CSV_FIELDS = [
    "sample_id",
    "retriever_type",
    "k",
    "ground_truth",
    "response_rag",
    "response_no_rag",
    "recall@k",
    "retrieval_consistency",
    "acc_rag",
    "acc_no_rag",
    "clap_clean",
    "clap_response_rag",
    "clap_response_no_rag",
    "kl_rag",
    "kl_no_rag",
    "consistency_rag",
    "consistency_no_rag",
]
TEXT_MATCH_THRESHOLD = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable clean WavCaps Audio RAG evaluation.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--results-csv", default="outputs/results_clean_wavcaps.csv", help="Output CSV path.")
    parser.add_argument("--query-text", default="What is this sound?", help="User query used for all samples.")
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES, help="Top-k values to evaluate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for a shorter run.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing results CSV if present.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for model API calls.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=3.0, help="Base backoff for retries.")
    parser.add_argument("--continue-on-error", action="store_true", help="Skip failed samples instead of aborting the full run.")
    parser.add_argument("--error-log", default="outputs/results_clean_wavcaps.errors.jsonl", help="Path to append per-sample errors.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def token_set(text: str) -> set[str]:
    normalized = normalize_text(text)
    return set(normalized.split()) if normalized else set()


def text_overlap_score(reference: str, candidate: str) -> float:
    ref_tokens = token_set(reference)
    cand_tokens = token_set(candidate)
    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0
    intersection = len(ref_tokens & cand_tokens)
    precision = intersection / max(len(cand_tokens), 1)
    recall = intersection / max(len(ref_tokens), 1)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def accuracy_score(reference: str, candidate: str, threshold: float = TEXT_MATCH_THRESHOLD) -> float:
    return float(text_overlap_score(reference, candidate) >= threshold)


def retrieval_recall_at_k(reference: str, hits: list[RetrievalHit], threshold: float = TEXT_MATCH_THRESHOLD) -> float:
    return float(any(text_overlap_score(reference, hit.sample.text) >= threshold for hit in hits))


def retrieval_consistency(reference: str, hits: list[RetrievalHit]) -> float:
    if not hits:
        return 0.0
    scores = [text_overlap_score(reference, hit.sample.text) for hit in hits]
    return float(np.mean(scores))


def build_rag_prompt(prompt_template: str, user_query: str, context: str) -> str:
    return prompt_template.format(user_query=user_query, context=context)


def build_no_rag_prompt(user_query: str) -> str:
    return (
        "You are given the original query audio and the user's question.\n\n"
        "Analyze the audio directly and answer the user's question without any retrieved context.\n\n"
        f"User Query:\n{user_query}\n"
    )


def generate_with_retry(
    victim_model: Any,
    model_input: ModelInput,
    max_retries: int,
    retry_backoff_seconds: float,
    logger: Any,
    request_label: str,
) -> ModelOutput:
    last_exc: Exception | None = None
    total_attempts = max(max_retries, 1)
    for attempt in range(total_attempts):
        started_at = time.time()
        logger.info("Starting model request: %s (attempt %s/%s)", request_label, attempt + 1, total_attempts)
        try:
            output = victim_model.generate(model_input)
            logger.info("Finished model request: %s in %.2fs", request_label, time.time() - started_at)
            return output
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Model request failed: %s in %.2fs with %s: %s",
                request_label,
                time.time() - started_at,
                exc.__class__.__name__,
                exc,
            )
            if attempt + 1 >= total_attempts:
                break
            sleep_seconds = retry_backoff_seconds * (2**attempt)
            logger.info("Retrying %s after %.1fs", request_label, sleep_seconds)
            time.sleep(sleep_seconds)
    assert last_exc is not None
    raise last_exc


def maybe_build_clap_encoder(logger: Any) -> AudioEmbeddingEncoder | None:
    try:
        logger.info("Initializing CLAP evaluation encoder for cross-modal metrics.")
        return build_encoder("clap")
    except Exception as exc:  # noqa: BLE001
        logger.warning("CLAP evaluation metrics disabled because CLAP initialization failed: %s", exc)
        return None


def maybe_build_classifier(logger: Any) -> AudioClassifier | None:
    try:
        logger.info("Initializing PANNs classifier for KL semantic drift metrics.")
        encoder = build_encoder("panns")
        return encoder if isinstance(encoder, AudioClassifier) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("KL semantic drift metrics disabled because PANNs initialization failed: %s", exc)
        return None


def compute_clap_metrics_for_sample(
    clap_encoder: AudioEmbeddingEncoder | None,
    audio_embedding: np.ndarray | None,
    ground_truth_embedding: np.ndarray | None,
    response_rag: str,
    response_no_rag: str,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if clap_encoder is None or audio_embedding is None:
        return None, None, None, None, None
    try:
        clap_clean = None
        if ground_truth_embedding is not None:
            clap_clean = cosine_similarity(audio_embedding, ground_truth_embedding)
        rag_embedding = clap_encoder.encode_text([response_rag])[0]
        no_rag_embedding = clap_encoder.encode_text([response_no_rag])[0]
        clap_response_rag = cosine_similarity(audio_embedding, rag_embedding)
        clap_response_no_rag = cosine_similarity(audio_embedding, no_rag_embedding)
        consistency_rag = cosine_similarity(audio_embedding, rag_embedding)
        consistency_no_rag = cosine_similarity(audio_embedding, no_rag_embedding)
        return (
            float(clap_clean),
            float(clap_response_rag),
            float(clap_response_no_rag),
            float(consistency_rag),
            float(consistency_no_rag),
        )
    except Exception:  # noqa: BLE001
        return None, None, None, None, None


def compute_kl_metrics_for_sample(
    clean_distribution: np.ndarray | None,
    retrieval_distribution: np.ndarray | None = None,
) -> tuple[float | None, float | None]:
    if clean_distribution is None:
        return None, None
    try:
        if retrieval_distribution is not None:
            kl_rag = float(kl_divergence(clean_distribution, retrieval_distribution))
        else:
            kl_rag = 0.0
        kl_no_rag = 0.0
        return kl_rag, kl_no_rag
    except Exception:  # noqa: BLE001
        return None, None


def load_existing_results(path: Path) -> tuple[set[tuple[str, int]], dict[str, dict[str, str]]]:
    completed: set[tuple[str, int]] = set()
    no_rag_cache: dict[str, dict[str, str]] = {}
    if not path.exists():
        return completed, no_rag_cache
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row["sample_id"]
            k_value = int(row["k"])
            completed.add((sample_id, k_value))
            no_rag_cache[sample_id] = {
                "response_no_rag": row["response_no_rag"],
                "acc_no_rag": row["acc_no_rag"],
                "clap_clean": row["clap_clean"],
                "clap_response_no_rag": row["clap_response_no_rag"],
                "kl_no_rag": row["kl_no_rag"],
                "consistency_no_rag": row["consistency_no_rag"],
            }
    return completed, no_rag_cache


def ensure_results_writer(path: Path) -> tuple[Any, csv.DictWriter]:
    ensure_dir(path.parent)
    file_exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def append_error_record(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def prepare_wavcaps_samples(config: dict, logger: Any) -> list[AudioSample]:
    wavcaps_settings = dict(config["data"]["datasets"]["wavcaps"])
    wavcaps_settings["enabled"] = True
    logger.info("Loading wavcaps_freesound metadata from %s", wavcaps_settings["metadata_path"])
    raw_samples = load_dataset(name="wavcaps", settings=wavcaps_settings)
    samples = list(tqdm(raw_samples, desc="Loading WavCaps metadata", unit="sample", dynamic_ncols=True))
    filtered_samples: list[AudioSample] = []
    for sample in tqdm(samples, desc="Checking audio files", unit="audio", dynamic_ncols=True):
        if Path(sample.audio_path).exists():
            filtered_samples.append(sample)
    samples = filtered_samples
    samples.sort(key=lambda sample: sample.sample_id)
    return samples


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger()

    seed = int(config["project"].get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)

    samples = prepare_wavcaps_samples(config=config, logger=logger)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise SystemExit("No WavCaps samples were found.")

    encoder = build_encoder(config["retrieval"]["encoder"], config["retrieval"].get("retriever_checkpoint_overrides"))
    retriever = FaissAudioRetriever(encoder=encoder, similarity=config["retrieval"]["similarity"])
    retriever.build(samples, show_progress=True, progress_desc="Encoding retrieval corpus")
    logger.info("Built %s retriever index for %s WavCaps samples.", encoder.name, len(samples))

    victim_model = build_victim_model(config["models"])
    clap_encoder = maybe_build_clap_encoder(logger)
    classifier = maybe_build_classifier(logger)

    k_values = sorted(set(args.k_values))
    max_k = max(k_values)
    results_path = Path(args.results_csv)
    completed_keys, no_rag_cache = load_existing_results(results_path) if args.resume else (set(), {})
    total_rows = len(samples) * len(k_values)
    remaining_rows = total_rows - len(completed_keys)
    logger.info("Evaluation rows total=%s remaining=%s resume=%s", total_rows, remaining_rows, args.resume)
    if remaining_rows <= 0:
        logger.info("All requested rows are already present in %s", results_path)
        return

    results_handle, results_writer = ensure_results_writer(results_path)
    error_log_path = Path(args.error_log)

    progress = tqdm(total=remaining_rows, desc="Clean WavCaps Eval", unit="row", dynamic_ncols=True)
    try:
        for sample in samples:
            unfinished_ks = [k for k in k_values if (sample.sample_id, k) not in completed_keys]
            if not unfinished_ks:
                continue

            try:
                progress.set_postfix(stage="retrieve", sample=sample.sample_id)
                hits_max = retriever.query(audio_path=sample.audio_path, top_k=max_k)

                clap_audio_embedding = None
                clap_ground_truth_embedding = None
                if clap_encoder is not None:
                    try:
                        clap_audio_embedding = clap_encoder.encode_audio([sample.audio_path])[0]
                        clap_ground_truth_embedding = clap_encoder.encode_text([sample.text])[0]
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("CLAP cache failed for sample=%s: %s", sample.sample_id, exc)
                        clap_audio_embedding = None
                        clap_ground_truth_embedding = None

                clean_distribution = None
                retrieval_distribution_by_k: dict[int, np.ndarray | None] = {}
                if classifier is not None:
                    try:
                        clean_distribution = classifier.predict_proba([sample.audio_path])[0]
                        max_retrieval_probs = classifier.predict_proba([hit.sample.audio_path for hit in hits_max])
                        for k_value in unfinished_ks:
                            retrieval_distribution_by_k[k_value] = max_retrieval_probs[:k_value].mean(axis=0)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("KL cache failed for sample=%s: %s", sample.sample_id, exc)
                        clean_distribution = None
                        retrieval_distribution_by_k = {}

                cached_no_rag = no_rag_cache.get(sample.sample_id)
                if cached_no_rag is None:
                    progress.set_postfix(stage="no-rag", sample=sample.sample_id)
                    no_rag_input = ModelInput(
                        audio_path=sample.audio_path,
                        user_query=args.query_text,
                        retrieved_context="",
                        prompt=build_no_rag_prompt(args.query_text),
                    )
                    no_rag_output = generate_with_retry(
                        victim_model=victim_model,
                        model_input=no_rag_input,
                        max_retries=args.max_retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                        logger=logger,
                        request_label=f"sample={sample.sample_id} mode=no_rag",
                    )
                    response_no_rag = no_rag_output.text
                    acc_no_rag = accuracy_score(sample.text, response_no_rag)
                    (
                        clap_clean,
                        _,
                        clap_response_no_rag,
                        _,
                        consistency_no_rag,
                    ) = compute_clap_metrics_for_sample(
                        clap_encoder=clap_encoder,
                        audio_embedding=clap_audio_embedding,
                        ground_truth_embedding=clap_ground_truth_embedding,
                        response_rag=sample.text,
                        response_no_rag=response_no_rag,
                    )
                    _, kl_no_rag = compute_kl_metrics_for_sample(
                        clean_distribution=clean_distribution,
                        retrieval_distribution=None,
                    )
                    cached_no_rag = {
                        "response_no_rag": response_no_rag,
                        "acc_no_rag": f"{acc_no_rag:.6f}",
                        "clap_response_no_rag": "" if clap_response_no_rag is None else f"{clap_response_no_rag:.6f}",
                        "kl_no_rag": "" if kl_no_rag is None else f"{kl_no_rag:.6f}",
                        "consistency_no_rag": "" if consistency_no_rag is None else f"{consistency_no_rag:.6f}",
                        "clap_clean": "" if clap_clean is None else f"{clap_clean:.6f}",
                    }
                    no_rag_cache[sample.sample_id] = cached_no_rag

                for k_value in unfinished_ks:
                    hits = hits_max[:k_value]
                    context = build_context_string(hits, config["rag"]["context_template"])
                    rag_input = ModelInput(
                        audio_path=sample.audio_path,
                        user_query=args.query_text,
                        retrieved_context=context,
                        prompt=build_rag_prompt(config["rag"]["prompt_template"], args.query_text, context),
                    )
                    progress.set_postfix(stage="rag", sample=sample.sample_id, k=k_value)
                    rag_output = generate_with_retry(
                        victim_model=victim_model,
                        model_input=rag_input,
                        max_retries=args.max_retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                        logger=logger,
                        request_label=f"sample={sample.sample_id} mode=rag k={k_value}",
                    )
                    recall_at_k = retrieval_recall_at_k(sample.text, hits)
                    consistency = retrieval_consistency(sample.text, hits)
                    acc_rag = accuracy_score(sample.text, rag_output.text)
                    (
                        clap_clean,
                        clap_response_rag,
                        _,
                        consistency_rag,
                        _,
                    ) = compute_clap_metrics_for_sample(
                        clap_encoder=clap_encoder,
                        audio_embedding=clap_audio_embedding,
                        ground_truth_embedding=clap_ground_truth_embedding,
                        response_rag=rag_output.text,
                        response_no_rag=cached_no_rag["response_no_rag"],
                    )
                    kl_rag, _ = compute_kl_metrics_for_sample(
                        clean_distribution=clean_distribution,
                        retrieval_distribution=retrieval_distribution_by_k.get(k_value),
                    )

                    row = {
                        "sample_id": sample.sample_id,
                        "retriever_type": encoder.name,
                        "k": k_value,
                        "ground_truth": sample.text,
                        "response_rag": rag_output.text,
                        "response_no_rag": cached_no_rag["response_no_rag"],
                        "recall@k": f"{recall_at_k:.6f}",
                        "retrieval_consistency": f"{consistency:.6f}",
                        "acc_rag": f"{acc_rag:.6f}",
                        "acc_no_rag": cached_no_rag["acc_no_rag"],
                        "clap_clean": cached_no_rag.get("clap_clean") or ("" if clap_clean is None else f"{clap_clean:.6f}"),
                        "clap_response_rag": "" if clap_response_rag is None else f"{clap_response_rag:.6f}",
                        "clap_response_no_rag": cached_no_rag["clap_response_no_rag"],
                        "kl_rag": "" if kl_rag is None else f"{kl_rag:.6f}",
                        "kl_no_rag": cached_no_rag["kl_no_rag"],
                        "consistency_rag": "" if consistency_rag is None else f"{consistency_rag:.6f}",
                        "consistency_no_rag": cached_no_rag["consistency_no_rag"],
                    }
                    results_writer.writerow(row)
                    results_handle.flush()
                    completed_keys.add((sample.sample_id, k_value))
                    progress.update(1)
                    progress.set_postfix(stage="saved", sample=sample.sample_id, k=k_value)
            except Exception as exc:  # noqa: BLE001
                append_error_record(
                    error_log_path,
                    {
                        "sample_id": sample.sample_id,
                        "unfinished_ks": unfinished_ks,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                logger.error("Sample failed: sample=%s ks=%s error=%s", sample.sample_id, unfinished_ks, exc)
                if not args.continue_on_error:
                    raise
    except KeyboardInterrupt:
        logger.warning("Interrupted. Progress is saved in %s and can be resumed with --resume.", results_path)
        raise
    finally:
        progress.close()
        results_handle.close()

    summary_path = results_path.with_suffix(".summary.json")
    summarize_results(results_path=results_path, summary_path=summary_path)
    logger.info("Finished clean WavCaps evaluation. Results saved to %s", results_path)
    logger.info("Summary saved to %s", summary_path)


def summarize_results(results_path: Path, summary_path: Path) -> None:
    rows: list[dict[str, str]] = []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    grouped: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        k_value = int(row["k"])
        bucket = grouped.setdefault(
            k_value,
            {
                "recall@k": [],
                "retrieval_consistency": [],
                "acc_rag": [],
                "acc_no_rag": [],
                "clap_clean": [],
                "clap_response_rag": [],
                "clap_response_no_rag": [],
                "kl_rag": [],
                "kl_no_rag": [],
                "consistency_rag": [],
                "consistency_no_rag": [],
            },
        )
        for key in bucket:
            value = row.get(key, "")
            if value not in ("", None):
                bucket[key].append(float(value))

    summary: dict[str, Any] = {"num_rows": len(rows), "per_k": {}}
    for k_value, metrics in sorted(grouped.items()):
        summary["per_k"][str(k_value)] = {
            name: (float(np.mean(values)) if values else None) for name, values in metrics.items()
        }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
